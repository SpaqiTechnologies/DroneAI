"""
Guided Mode Handler

Single waypoint navigation. The drone navigates to a specified
target position using trajectory-based flight.
"""

from typing import Tuple, Optional, List, Callable, TYPE_CHECKING
import time
import math

from .base_mode import FlightModeHandler, FlightMode, FlightCommand

if TYPE_CHECKING:
    from core.drone import Drone
    from core.trajectory.trajectory_generator import TrajectoryPoint


class GuidedModeHandler(FlightModeHandler):
    """
    Guided mode - single waypoint navigation.

    In this mode:
    - Drone navigates to a target position
    - Trajectory is generated for smooth flight
    - Speed and acceptance radius are configurable
    - Callbacks notify when target is reached
    """

    # Default parameters
    DEFAULT_SPEED = 5.0              # m/s
    DEFAULT_ACCEPTANCE_RADIUS = 2.0  # meters
    MAX_SPEED = 15.0                 # m/s

    def __init__(self, drone: 'Drone'):
        super().__init__(drone)
        self._target_position: Optional[Tuple[float, float, float]] = None
        self._target_yaw: Optional[float] = None
        self._speed = self.DEFAULT_SPEED
        self._acceptance_radius = self.DEFAULT_ACCEPTANCE_RADIUS

        # Trajectory
        self._trajectory: List['TrajectoryPoint'] = []
        self._trajectory_start_time = 0.0
        self._trajectory_index = 0

        # State
        self._target_reached = False
        self._distance_to_target = 0.0

        # Callbacks
        self._target_reached_callbacks: List[Callable[[], None]] = []

    @property
    def mode(self) -> FlightMode:
        return FlightMode.GUIDED

    def can_enter(self) -> Tuple[bool, str]:
        """Check if guided mode can be entered."""
        # Require GPS
        if not self._drone.gps_sensor.is_valid():
            return False, "GUIDED requires valid GPS fix"

        if self._drone.gps_sensor.get_satellites() < 4:
            return False, "GUIDED requires at least 4 satellites"

        # Require armed
        if not self._drone.is_armed:
            return False, "GUIDED requires armed state"

        return True, ""

    def _on_enter(self) -> Tuple[bool, str]:
        """Initialize guided mode."""
        self._target_reached = False
        self._trajectory = []
        return True, ""

    def _on_exit(self) -> None:
        """Clean up on exit."""
        self._trajectory = []

    def update(self, dt: float) -> FlightCommand:
        """
        Update guided mode.

        Args:
            dt: Time delta since last update

        Returns:
            FlightCommand for waypoint navigation
        """
        if not self._target_position:
            # No target set, hold position
            return FlightCommand.hold()

        # Get current position
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return FlightCommand.hold()

        # Calculate distance to target
        self._distance_to_target = self._calculate_distance(gps_pos, self._target_position)

        # Check if target reached
        if self._distance_to_target <= self._acceptance_radius:
            if not self._target_reached:
                self._target_reached = True
                self._notify_target_reached()
            return FlightCommand.position_command(
                self._target_position[0],
                self._target_position[1],
                self._target_position[2],
                self._target_yaw
            )

        # Follow trajectory if available
        if self._trajectory:
            return self._follow_trajectory(dt)

        # Direct navigation (no trajectory)
        return self._direct_navigation(gps_pos)

    def _follow_trajectory(self, dt: float) -> FlightCommand:
        """Follow pre-computed trajectory."""
        if not self._trajectory:
            return FlightCommand.hold()

        # Get elapsed time
        elapsed = time.time() - self._trajectory_start_time

        # Find current trajectory point
        current_point = None
        for i, point in enumerate(self._trajectory):
            if point.time >= elapsed:
                current_point = point
                self._trajectory_index = i
                break

        if current_point is None:
            # Past end of trajectory, use last point
            current_point = self._trajectory[-1]

        return FlightCommand(
            position=current_point.position,
            velocity=current_point.velocity,
            yaw=current_point.yaw if self._target_yaw is None else self._target_yaw
        )

    def _direct_navigation(
        self,
        current_pos: Tuple[float, float, float]
    ) -> FlightCommand:
        """Navigate directly to target without trajectory."""
        if not self._target_position:
            return FlightCommand.hold()

        # Calculate bearing to target
        bearing = self._calculate_bearing(current_pos, self._target_position)

        # Calculate velocity components
        speed = min(self._speed, self._distance_to_target * 0.5)  # Slow down near target
        speed = max(1.0, speed)  # Minimum speed

        bearing_rad = math.radians(bearing)
        vn = speed * math.cos(bearing_rad)
        ve = speed * math.sin(bearing_rad)

        # Vertical velocity
        alt_error = self._target_position[2] - current_pos[2]
        vd = -min(2.0, max(-2.0, alt_error * 0.5))

        return FlightCommand(
            position=self._target_position,
            velocity=(vn, ve, vd),
            yaw=self._target_yaw if self._target_yaw else bearing
        )

    def goto(
        self,
        lat: float,
        lon: float,
        alt: float,
        speed: Optional[float] = None,
        yaw: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Set target position.

        Args:
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude (meters)
            speed: Optional navigation speed (m/s)
            yaw: Optional target yaw (degrees)

        Returns:
            Tuple of (success, message)
        """
        self._target_position = (lat, lon, alt)
        self._target_yaw = yaw
        self._target_reached = False

        if speed is not None:
            self._speed = min(speed, self.MAX_SPEED)

        # Generate trajectory
        self._generate_trajectory()

        return True, f"Navigating to ({lat:.6f}, {lon:.6f}, {alt:.1f}m)"

    def _generate_trajectory(self) -> None:
        """Generate trajectory to target."""
        if not self._target_position:
            return

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return

        # Get trajectory generator from flight controller if available
        if hasattr(self._drone, 'flight_controller'):
            traj_gen = self._drone.flight_controller.trajectory_generator
            self._trajectory = traj_gen.generate_trajectory(
                [gps_pos, self._target_position],
                speeds=[self._speed, 0.0],
                smooth=True
            )
            self._trajectory_start_time = time.time()
            self._trajectory_index = 0

    def set_speed(self, speed: float) -> None:
        """Set navigation speed."""
        self._speed = min(speed, self.MAX_SPEED)

    def set_acceptance_radius(self, radius: float) -> None:
        """Set target acceptance radius."""
        self._acceptance_radius = max(0.5, radius)

    def is_target_reached(self) -> bool:
        """Check if target has been reached."""
        return self._target_reached

    def get_distance_to_target(self) -> float:
        """Get distance to target in meters."""
        return self._distance_to_target

    def get_time_to_target(self) -> float:
        """Get estimated time to target in seconds."""
        if self._distance_to_target <= 0 or self._speed <= 0:
            return 0.0
        return self._distance_to_target / self._speed

    def add_target_reached_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for when target is reached."""
        self._target_reached_callbacks.append(callback)

    def _notify_target_reached(self) -> None:
        """Notify callbacks that target was reached."""
        for callback in self._target_reached_callbacks:
            try:
                callback()
            except Exception:
                pass

    def _calculate_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate 3D distance between two GPS points."""
        # Horizontal distance (Haversine)
        lat1, lon1, alt1 = p1
        lat2, lon2, alt2 = p2

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_dist = 6371000 * c  # Earth radius in meters

        # 3D distance
        vertical_dist = abs(alt2 - alt1)
        return math.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)

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
            'target_position': self._target_position,
            'target_yaw': self._target_yaw,
            'speed': self._speed,
            'acceptance_radius': self._acceptance_radius,
            'target_reached': self._target_reached,
            'distance_to_target': self._distance_to_target,
            'time_to_target': self.get_time_to_target(),
            'trajectory_active': len(self._trajectory) > 0,
            'trajectory_progress': self._trajectory_index / len(self._trajectory) if self._trajectory else 0,
        })
        return status
