"""
RTL (Return-to-Launch) Mode Handler

Autonomous return to home position. The drone climbs to RTL altitude,
navigates to home, then lands.
"""

from typing import Tuple, Optional, List, TYPE_CHECKING
from enum import Enum
import time
import math

from .base_mode import FlightModeHandler, FlightMode, FlightCommand

if TYPE_CHECKING:
    from core.drone import Drone
    from core.trajectory.trajectory_generator import TrajectoryPoint


class RTLPhase(Enum):
    """Phases of RTL operation."""
    CLIMB = "climb"              # Climbing to RTL altitude
    TRANSIT = "transit"          # Navigating to home
    DESCEND = "descend"          # Descending over home
    LAND = "land"                # Landing
    COMPLETE = "complete"        # RTL complete


class RTLModeHandler(FlightModeHandler):
    """
    RTL mode - return to launch (home) position.

    In this mode:
    1. Climb to RTL altitude if below
    2. Navigate to home position
    3. Descend and land
    """

    # Default parameters
    DEFAULT_RTL_ALTITUDE = 50.0      # meters
    DEFAULT_LAND_ALTITUDE = 10.0     # meters to switch to LAND
    DEFAULT_SPEED = 8.0              # m/s
    ACCEPTANCE_RADIUS = 3.0          # meters
    CLIMB_RATE = 3.0                 # m/s
    DESCENT_RATE = 2.0               # m/s

    def __init__(self, drone: 'Drone'):
        super().__init__(drone)
        self._rtl_altitude = self.DEFAULT_RTL_ALTITUDE
        self._land_altitude = self.DEFAULT_LAND_ALTITUDE
        self._speed = self.DEFAULT_SPEED

        # Home position
        self._home_position: Optional[Tuple[float, float, float]] = None

        # RTL state
        self._phase = RTLPhase.CLIMB
        self._phase_start_time = 0.0

        # Trajectory
        self._trajectory: List['TrajectoryPoint'] = []
        self._trajectory_start_time = 0.0

        # Distance tracking
        self._distance_to_home = 0.0

    @property
    def mode(self) -> FlightMode:
        return FlightMode.RTL

    def can_enter(self) -> Tuple[bool, str]:
        """Check if RTL mode can be entered."""
        # Require GPS
        if not self._drone.gps_sensor.is_valid():
            return False, "RTL requires valid GPS fix"

        # Require home position
        if not self._drone.home_position:
            return False, "RTL requires home position to be set"

        return True, ""

    def _on_enter(self) -> Tuple[bool, str]:
        """Initialize RTL mode."""
        # Get home position from drone
        if self._drone.home_position:
            self._home_position = (
                self._drone.home_position[0],
                self._drone.home_position[1],
                self._rtl_altitude
            )
        else:
            return False, "Home position not set"

        # Determine starting phase
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return False, "Could not get GPS position"

        current_alt = gps_pos[2]

        if current_alt < self._rtl_altitude - 5:
            # Need to climb first
            self._phase = RTLPhase.CLIMB
        else:
            # Already at or above RTL altitude
            self._phase = RTLPhase.TRANSIT
            self._generate_transit_trajectory()

        self._phase_start_time = time.time()
        self._trajectory = []

        return True, f"RTL initiated - returning to home at {self._rtl_altitude}m"

    def _on_exit(self) -> None:
        """Clean up on exit."""
        self._trajectory = []
        self._phase = RTLPhase.CLIMB

    def update(self, dt: float) -> FlightCommand:
        """
        Update RTL mode.

        Args:
            dt: Time delta since last update

        Returns:
            FlightCommand for RTL navigation
        """
        if not self._home_position:
            return FlightCommand.hold()

        # Get current position
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return FlightCommand.hold()

        # Update distance to home
        self._distance_to_home = self._calculate_horizontal_distance(
            gps_pos, self._home_position
        )

        # Execute current phase
        if self._phase == RTLPhase.CLIMB:
            return self._update_climb(gps_pos, dt)
        elif self._phase == RTLPhase.TRANSIT:
            return self._update_transit(gps_pos, dt)
        elif self._phase == RTLPhase.DESCEND:
            return self._update_descend(gps_pos, dt)
        elif self._phase == RTLPhase.LAND:
            return self._update_land(gps_pos, dt)
        else:
            return FlightCommand.hold()

    def _update_climb(
        self,
        current_pos: Tuple[float, float, float],
        dt: float
    ) -> FlightCommand:
        """Execute climb phase."""
        current_alt = current_pos[2]

        if current_alt >= self._rtl_altitude - 1:
            # Reached RTL altitude, transition to transit
            self._phase = RTLPhase.TRANSIT
            self._phase_start_time = time.time()
            self._generate_transit_trajectory()
            return self._update_transit(current_pos, dt)

        # Continue climbing
        target_alt = min(current_alt + self.CLIMB_RATE * dt, self._rtl_altitude)

        return FlightCommand.position_command(
            current_pos[0],
            current_pos[1],
            target_alt
        )

    def _update_transit(
        self,
        current_pos: Tuple[float, float, float],
        dt: float
    ) -> FlightCommand:
        """Execute transit phase (fly to home)."""
        # Check if reached home horizontally
        if self._distance_to_home <= self.ACCEPTANCE_RADIUS:
            # Reached home, transition to descend
            self._phase = RTLPhase.DESCEND
            self._phase_start_time = time.time()
            return self._update_descend(current_pos, dt)

        # Follow trajectory if available
        if self._trajectory:
            elapsed = time.time() - self._trajectory_start_time
            for point in self._trajectory:
                if point.time >= elapsed:
                    return FlightCommand(
                        position=point.position,
                        velocity=point.velocity,
                        yaw=point.yaw
                    )

        # Direct navigation to home
        bearing = self._calculate_bearing(current_pos, self._home_position)
        return FlightCommand.position_command(
            self._home_position[0],
            self._home_position[1],
            self._rtl_altitude,
            bearing
        )

    def _update_descend(
        self,
        current_pos: Tuple[float, float, float],
        dt: float
    ) -> FlightCommand:
        """Execute descend phase."""
        current_alt = current_pos[2]

        if current_alt <= self._land_altitude:
            # Reached land altitude, transition to land
            self._phase = RTLPhase.LAND
            self._phase_start_time = time.time()
            return self._update_land(current_pos, dt)

        # Continue descending
        target_alt = max(current_alt - self.DESCENT_RATE * dt, self._land_altitude)

        return FlightCommand.position_command(
            self._home_position[0],
            self._home_position[1],
            target_alt
        )

    def _update_land(
        self,
        current_pos: Tuple[float, float, float],
        dt: float
    ) -> FlightCommand:
        """Execute land phase."""
        current_alt = current_pos[2]

        if current_alt <= 0.5:
            # Landed
            self._phase = RTLPhase.COMPLETE
            return FlightCommand.hold()

        # Continue landing
        target_alt = max(current_alt - 1.0 * dt, 0.0)  # 1 m/s descent

        return FlightCommand.position_command(
            self._home_position[0],
            self._home_position[1],
            target_alt
        )

    def _generate_transit_trajectory(self) -> None:
        """Generate trajectory to home."""
        if not self._home_position:
            return

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return

        # Use trajectory generator if available
        if hasattr(self._drone, 'flight_controller'):
            traj_gen = self._drone.flight_controller.trajectory_generator
            self._trajectory = traj_gen.generate_trajectory(
                [(gps_pos[0], gps_pos[1], self._rtl_altitude),
                 (self._home_position[0], self._home_position[1], self._rtl_altitude)],
                speeds=[self._speed, 0.0],
                smooth=True
            )
            self._trajectory_start_time = time.time()

    def set_rtl_altitude(self, altitude: float) -> None:
        """Set RTL altitude."""
        self._rtl_altitude = max(10.0, altitude)

    def set_speed(self, speed: float) -> None:
        """Set RTL navigation speed."""
        self._speed = max(1.0, min(15.0, speed))

    def get_phase(self) -> RTLPhase:
        """Get current RTL phase."""
        return self._phase

    def get_distance_to_home(self) -> float:
        """Get distance to home in meters."""
        return self._distance_to_home

    def get_time_to_home(self) -> float:
        """Get estimated time to home in seconds."""
        if self._speed <= 0:
            return 0.0
        return self._distance_to_home / self._speed

    def is_complete(self) -> bool:
        """Check if RTL is complete."""
        return self._phase == RTLPhase.COMPLETE

    def _calculate_horizontal_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate horizontal distance between two GPS points."""
        lat1, lon1, _ = p1
        lat2, lon2, _ = p2

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return 6371000 * c

    def _calculate_bearing(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate bearing from p1 to p2."""
        lat1 = math.radians(p1[0])
        lat2 = math.radians(p2[0])
        dlon = math.radians(p2[1] - p1[1])

        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360

    def get_status(self) -> dict:
        """Get mode status."""
        status = super().get_status()
        status.update({
            'phase': self._phase.value,
            'home_position': self._home_position,
            'rtl_altitude': self._rtl_altitude,
            'distance_to_home': self._distance_to_home,
            'time_to_home': self.get_time_to_home(),
            'speed': self._speed,
            'complete': self.is_complete(),
            'time_in_phase': time.time() - self._phase_start_time,
        })
        return status
