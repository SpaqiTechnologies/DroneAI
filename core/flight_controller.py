"""
Flight Controller

Central flight mode management and trajectory execution.
Coordinates between different flight modes and provides a unified
interface for autonomous flight control.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Tuple, TYPE_CHECKING
import time
import math

from core.flight_modes.base_mode import (
    FlightMode,
    FlightModeHandler,
    FlightCommand,
    ModeTransitionError,
)
from core.trajectory.trajectory_generator import (
    TrajectoryGenerator,
    TrajectoryPoint,
)

if TYPE_CHECKING:
    from core.drone import Drone


class ControllerState(Enum):
    """State of the flight controller."""
    IDLE = "idle"              # Not controlling
    ACTIVE = "active"          # Actively controlling flight
    FAILSAFE = "failsafe"      # In failsafe mode
    ERROR = "error"            # Error state


@dataclass
class ControllerStatus:
    """Status information for the flight controller."""
    state: ControllerState
    current_mode: FlightMode
    previous_mode: Optional[FlightMode]
    time_in_mode: float
    target_position: Optional[Tuple[float, float, float]]
    target_reached: bool
    error_message: Optional[str]


class FlightController:
    """
    Central flight mode management and trajectory execution.

    The FlightController:
    - Manages transitions between flight modes
    - Executes trajectory-based navigation
    - Provides position and velocity control
    - Integrates with failsafe system
    """

    # Waypoint acceptance radius (meters)
    DEFAULT_ACCEPTANCE_RADIUS = 2.0

    # Maximum position error before triggering hold (meters)
    MAX_POSITION_ERROR = 50.0

    def __init__(self, drone: 'Drone'):
        """
        Initialize flight controller.

        Args:
            drone: Reference to the drone instance
        """
        self._drone = drone
        self._state = ControllerState.IDLE
        self._current_mode = FlightMode.STABILIZE
        self._previous_mode: Optional[FlightMode] = None
        self._mode_entry_time = time.time()

        # Mode handlers (lazy loaded)
        self._mode_handlers: Dict[FlightMode, FlightModeHandler] = {}

        # Trajectory management
        self._trajectory_generator = TrajectoryGenerator()
        self._current_trajectory: List[TrajectoryPoint] = []
        self._trajectory_index = 0
        self._trajectory_start_time = 0.0

        # Target management
        self._target_position: Optional[Tuple[float, float, float]] = None
        self._target_velocity: Optional[Tuple[float, float, float]] = None
        self._target_yaw: Optional[float] = None
        self._acceptance_radius = self.DEFAULT_ACCEPTANCE_RADIUS

        # State tracking
        self._last_command: Optional[FlightCommand] = None
        self._error_message: Optional[str] = None

        # Callbacks
        self._mode_change_callbacks: List[Callable[[FlightMode, FlightMode], None]] = []
        self._target_reached_callbacks: List[Callable[[Tuple[float, float, float]], None]] = []

    @property
    def current_mode(self) -> FlightMode:
        """Get current flight mode."""
        return self._current_mode

    @property
    def state(self) -> ControllerState:
        """Get controller state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if controller is actively controlling."""
        return self._state == ControllerState.ACTIVE

    @property
    def target_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current target position."""
        return self._target_position

    @property
    def trajectory_generator(self) -> TrajectoryGenerator:
        """Get trajectory generator instance."""
        return self._trajectory_generator

    def get_mode_handler(self, mode: FlightMode) -> Optional[FlightModeHandler]:
        """Get handler for a specific mode."""
        return self._mode_handlers.get(mode)

    def register_mode_handler(
        self,
        mode: FlightMode,
        handler: FlightModeHandler
    ) -> None:
        """Register a mode handler."""
        self._mode_handlers[mode] = handler

    def set_mode(self, mode: FlightMode) -> Tuple[bool, str]:
        """
        Switch to a new flight mode.

        Args:
            mode: Target flight mode

        Returns:
            Tuple of (success, message)
        """
        if mode == self._current_mode:
            return True, f"Already in {mode.value} mode"

        # Check if mode can be entered
        handler = self._mode_handlers.get(mode)
        if handler:
            can_enter, reason = handler.can_enter()
            if not can_enter:
                return False, reason

        # Check if current mode can be exited
        current_handler = self._mode_handlers.get(self._current_mode)
        if current_handler:
            can_exit, reason = current_handler.can_exit()
            if not can_exit:
                return False, reason

        # Perform transition
        old_mode = self._current_mode

        # Exit current mode
        if current_handler:
            current_handler.exit()

        # Enter new mode
        if handler:
            success, message = handler.enter()
            if not success:
                # Rollback to previous mode
                if current_handler:
                    current_handler.enter()
                return False, message

        # Update state
        self._previous_mode = old_mode
        self._current_mode = mode
        self._mode_entry_time = time.time()
        self._state = ControllerState.ACTIVE

        # Clear trajectory on mode change
        self._current_trajectory = []
        self._trajectory_index = 0

        # Notify callbacks
        self._notify_mode_change(old_mode, mode)

        return True, f"Switched from {old_mode.value} to {mode.value}"

    def update(self, dt: float) -> FlightCommand:
        """
        Update flight controller and return command.

        This should be called every control loop iteration.

        Args:
            dt: Time delta since last update (seconds)

        Returns:
            FlightCommand to execute
        """
        if self._state == ControllerState.IDLE:
            return FlightCommand.hold()

        if self._state == ControllerState.ERROR:
            return FlightCommand.hold()

        # Get current mode handler
        handler = self._mode_handlers.get(self._current_mode)
        if handler:
            command = handler.update(dt)
        else:
            # Default behavior based on mode
            command = self._default_mode_update(dt)

        # Check for target reached
        if self._target_position:
            self._check_target_reached()

        self._last_command = command
        return command

    def _default_mode_update(self, dt: float) -> FlightCommand:
        """Default update behavior when no handler is registered."""
        if self._current_mode == FlightMode.STABILIZE:
            return FlightCommand.hold()

        elif self._current_mode == FlightMode.LOITER:
            return self._loiter_update(dt)

        elif self._current_mode == FlightMode.GUIDED:
            return self._guided_update(dt)

        elif self._current_mode == FlightMode.RTL:
            return self._rtl_update(dt)

        elif self._current_mode == FlightMode.LAND:
            return self._land_update(dt)

        return FlightCommand.hold()

    def _loiter_update(self, dt: float) -> FlightCommand:
        """Update for LOITER mode - hold current position."""
        if self._target_position is None:
            # Set current position as target
            state = self._drone.state_estimator.get_state()
            if state:
                gps_pos = self._drone.state_estimator.get_gps_position()
                if gps_pos:
                    self._target_position = (gps_pos[0], gps_pos[1], gps_pos[2])

        if self._target_position:
            return FlightCommand.position_command(
                self._target_position[0],
                self._target_position[1],
                self._target_position[2],
                self._target_yaw
            )

        return FlightCommand.hold()

    def _guided_update(self, dt: float) -> FlightCommand:
        """Update for GUIDED mode - navigate to single waypoint."""
        if self._target_position is None:
            return FlightCommand.hold()

        # Check if using trajectory
        if self._current_trajectory:
            return self._follow_trajectory(dt)

        # Direct position command
        return FlightCommand.position_command(
            self._target_position[0],
            self._target_position[1],
            self._target_position[2],
            self._target_yaw
        )

    def _rtl_update(self, dt: float) -> FlightCommand:
        """Update for RTL mode - return to home."""
        if self._target_position is None:
            # Set home as target
            if self._drone.home_position:
                self._target_position = (
                    self._drone.home_position[0],
                    self._drone.home_position[1],
                    50.0  # Default RTL altitude
                )

        if self._target_position:
            # Check if using trajectory
            if self._current_trajectory:
                return self._follow_trajectory(dt)

            return FlightCommand.position_command(
                self._target_position[0],
                self._target_position[1],
                self._target_position[2],
                self._target_yaw
            )

        return FlightCommand.hold()

    def _land_update(self, dt: float) -> FlightCommand:
        """Update for LAND mode - descend and land."""
        # Get current position
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return FlightCommand.hold()

        # Descend at controlled rate
        current_alt = gps_pos[2]
        target_alt = max(0.0, current_alt - 1.0 * dt)  # 1 m/s descent

        return FlightCommand.position_command(
            gps_pos[0],
            gps_pos[1],
            target_alt
        )

    def _follow_trajectory(self, dt: float) -> FlightCommand:
        """Follow current trajectory."""
        if not self._current_trajectory:
            return FlightCommand.hold()

        # Get elapsed time since trajectory start
        elapsed = time.time() - self._trajectory_start_time

        # Get point at current time
        point = self._trajectory_generator.get_point_at_time(
            self._current_trajectory,
            elapsed
        )

        if point is None:
            return FlightCommand.hold()

        # Create command from trajectory point
        return FlightCommand(
            position=point.position,
            velocity=point.velocity,
            yaw=point.yaw
        )

    def goto(
        self,
        lat: float,
        lon: float,
        alt: float,
        speed: float = 5.0,
        yaw: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Navigate to a single waypoint (GUIDED mode).

        Args:
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude (meters)
            speed: Speed to target (m/s)
            yaw: Target yaw (degrees, None for auto)

        Returns:
            Tuple of (success, message)
        """
        # Switch to GUIDED mode if not already
        if self._current_mode != FlightMode.GUIDED:
            success, msg = self.set_mode(FlightMode.GUIDED)
            if not success:
                return False, msg

        # Set target
        self._target_position = (lat, lon, alt)
        self._target_yaw = yaw

        # Generate trajectory
        gps_pos = self._drone.state_estimator.get_gps_position()
        if gps_pos:
            self._current_trajectory = self._trajectory_generator.generate_trajectory(
                [(gps_pos[0], gps_pos[1], gps_pos[2]), (lat, lon, alt)],
                speeds=[speed, 0.0],
                smooth=True
            )
            self._trajectory_start_time = time.time()

        return True, f"Navigating to ({lat:.6f}, {lon:.6f}, {alt:.1f}m)"

    def return_to_home(self, altitude: float = 50.0) -> Tuple[bool, str]:
        """
        Return to home position (RTL mode).

        Args:
            altitude: RTL altitude (meters)

        Returns:
            Tuple of (success, message)
        """
        if not self._drone.home_position:
            return False, "Home position not set"

        # Switch to RTL mode
        success, msg = self.set_mode(FlightMode.RTL)
        if not success:
            return False, msg

        # Set home as target
        self._target_position = (
            self._drone.home_position[0],
            self._drone.home_position[1],
            altitude
        )

        # Generate trajectory
        gps_pos = self._drone.state_estimator.get_gps_position()
        if gps_pos:
            self._current_trajectory = self._trajectory_generator.generate_trajectory(
                [(gps_pos[0], gps_pos[1], gps_pos[2]),
                 (self._target_position[0], self._target_position[1], altitude)],
                smooth=True
            )
            self._trajectory_start_time = time.time()

        return True, "Returning to home"

    def land(self) -> Tuple[bool, str]:
        """
        Land at current position.

        Returns:
            Tuple of (success, message)
        """
        success, msg = self.set_mode(FlightMode.LAND)
        if not success:
            return False, msg

        return True, "Landing"

    def hold(self) -> Tuple[bool, str]:
        """
        Hold current position (LOITER mode).

        Returns:
            Tuple of (success, message)
        """
        success, msg = self.set_mode(FlightMode.LOITER)
        if not success:
            return False, msg

        # Clear target to capture current position
        self._target_position = None

        return True, "Holding position"

    def set_velocity(
        self,
        vn: float,
        ve: float,
        vd: float,
        yaw_rate: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Set velocity setpoint.

        Args:
            vn: North velocity (m/s)
            ve: East velocity (m/s)
            vd: Down velocity (m/s)
            yaw_rate: Yaw rate (deg/s)

        Returns:
            Tuple of (success, message)
        """
        self._target_velocity = (vn, ve, vd)
        return True, "Velocity setpoint set"

    def set_acceptance_radius(self, radius: float) -> None:
        """Set waypoint acceptance radius."""
        self._acceptance_radius = max(0.5, radius)

    def _check_target_reached(self) -> bool:
        """Check if target position has been reached."""
        if not self._target_position:
            return False

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return False

        distance = self._trajectory_generator.compute_distance_to_target(
            gps_pos,
            self._target_position
        )

        if distance <= self._acceptance_radius:
            # Target reached
            self._notify_target_reached(self._target_position)
            return True

        return False

    def get_distance_to_target(self) -> float:
        """Get distance to current target in meters."""
        if not self._target_position:
            return 0.0

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return 0.0

        return self._trajectory_generator.compute_distance_to_target(
            gps_pos,
            self._target_position
        )

    def get_time_to_target(self, speed: float = 5.0) -> float:
        """Get estimated time to target in seconds."""
        if not self._target_position:
            return 0.0

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return 0.0

        return self._trajectory_generator.compute_time_to_target(
            gps_pos,
            self._target_position,
            speed
        )

    def add_mode_change_callback(
        self,
        callback: Callable[[FlightMode, FlightMode], None]
    ) -> None:
        """Add callback for mode changes."""
        self._mode_change_callbacks.append(callback)

    def add_target_reached_callback(
        self,
        callback: Callable[[Tuple[float, float, float]], None]
    ) -> None:
        """Add callback for target reached events."""
        self._target_reached_callbacks.append(callback)

    def _notify_mode_change(
        self,
        old_mode: FlightMode,
        new_mode: FlightMode
    ) -> None:
        """Notify mode change callbacks."""
        for callback in self._mode_change_callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception:
                pass

    def _notify_target_reached(
        self,
        position: Tuple[float, float, float]
    ) -> None:
        """Notify target reached callbacks."""
        for callback in self._target_reached_callbacks:
            try:
                callback(position)
            except Exception:
                pass

    def activate(self) -> Tuple[bool, str]:
        """Activate the flight controller."""
        self._state = ControllerState.ACTIVE
        return True, "Flight controller activated"

    def deactivate(self) -> Tuple[bool, str]:
        """Deactivate the flight controller."""
        self._state = ControllerState.IDLE
        self._current_trajectory = []
        return True, "Flight controller deactivated"

    def trigger_failsafe(self, action: str = "hold") -> None:
        """Trigger failsafe mode."""
        self._state = ControllerState.FAILSAFE

        if action == "hold":
            self.set_mode(FlightMode.LOITER)
        elif action == "rtl":
            self.return_to_home()
        elif action == "land":
            self.land()

    def get_status(self) -> ControllerStatus:
        """Get controller status."""
        return ControllerStatus(
            state=self._state,
            current_mode=self._current_mode,
            previous_mode=self._previous_mode,
            time_in_mode=time.time() - self._mode_entry_time,
            target_position=self._target_position,
            target_reached=self._check_target_reached(),
            error_message=self._error_message
        )

    def to_dict(self) -> dict:
        """Get controller state as dictionary."""
        return {
            'state': self._state.value,
            'current_mode': self._current_mode.value,
            'previous_mode': self._previous_mode.value if self._previous_mode else None,
            'time_in_mode': time.time() - self._mode_entry_time,
            'target_position': self._target_position,
            'target_velocity': self._target_velocity,
            'target_yaw': self._target_yaw,
            'distance_to_target': self.get_distance_to_target(),
            'trajectory_active': len(self._current_trajectory) > 0,
            'error': self._error_message,
        }
