"""
Base Flight Mode Interface

Defines the abstract interface for all flight modes and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Callable, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from core.drone import Drone


class FlightMode(Enum):
    """Available flight modes."""
    STABILIZE = "stabilize"      # Manual attitude control with auto-leveling
    ALT_HOLD = "alt_hold"        # Altitude hold, manual horizontal
    LOITER = "loiter"            # GPS position hold
    GUIDED = "guided"            # Single waypoint navigation
    AUTO = "auto"                # Autonomous mission execution
    RTL = "rtl"                  # Return to launch
    LAND = "land"                # Automated landing
    ACRO = "acro"                # Acrobatic mode (rate control)

    @property
    def requires_gps(self) -> bool:
        """Check if this mode requires GPS."""
        return self in {
            FlightMode.LOITER,
            FlightMode.GUIDED,
            FlightMode.AUTO,
            FlightMode.RTL,
        }

    @property
    def requires_armed(self) -> bool:
        """Check if this mode requires armed state."""
        return self not in {FlightMode.STABILIZE}

    @property
    def is_autonomous(self) -> bool:
        """Check if this mode is autonomous (no pilot input needed)."""
        return self in {
            FlightMode.AUTO,
            FlightMode.RTL,
            FlightMode.LAND,
        }


class ModeTransitionError(Exception):
    """Raised when a mode transition is not allowed."""
    pass


@dataclass
class FlightCommand:
    """
    Target setpoint for flight controller.

    The flight controller uses these commands to control the drone.
    Not all fields need to be set - None means "don't change".
    """
    # Position target (lat, lon, alt in meters)
    position: Optional[Tuple[float, float, float]] = None

    # Velocity target in NED frame (m/s)
    velocity: Optional[Tuple[float, float, float]] = None

    # Acceleration target in NED frame (m/s^2)
    acceleration: Optional[Tuple[float, float, float]] = None

    # Yaw target (degrees, 0=North, clockwise positive)
    yaw: Optional[float] = None

    # Yaw rate target (deg/s)
    yaw_rate: Optional[float] = None

    # Thrust (0.0-1.0, for manual modes)
    thrust: Optional[float] = None

    # Roll/pitch angles for attitude control (degrees)
    roll: Optional[float] = None
    pitch: Optional[float] = None

    # Command timestamp
    timestamp: float = field(default_factory=time.time)

    # Whether this is a hold command (maintain current position)
    is_hold: bool = False

    def __post_init__(self):
        """Validate command parameters."""
        if self.thrust is not None:
            self.thrust = max(0.0, min(1.0, self.thrust))
        if self.yaw is not None:
            self.yaw = self.yaw % 360

    @classmethod
    def hold(cls) -> 'FlightCommand':
        """Create a hold command (maintain current position)."""
        return cls(is_hold=True)

    @classmethod
    def velocity_command(
        cls,
        vn: float,
        ve: float,
        vd: float,
        yaw_rate: float = 0.0
    ) -> 'FlightCommand':
        """Create a velocity-based command."""
        return cls(velocity=(vn, ve, vd), yaw_rate=yaw_rate)

    @classmethod
    def position_command(
        cls,
        lat: float,
        lon: float,
        alt: float,
        yaw: Optional[float] = None
    ) -> 'FlightCommand':
        """Create a position-based command."""
        return cls(position=(lat, lon, alt), yaw=yaw)


@dataclass
class ModeState:
    """State information for a flight mode."""
    mode: FlightMode
    active: bool = False
    entry_time: float = 0.0
    target_position: Optional[Tuple[float, float, float]] = None
    target_velocity: Optional[Tuple[float, float, float]] = None
    error_message: Optional[str] = None

    @property
    def time_in_mode(self) -> float:
        """Time since mode was entered."""
        if not self.active:
            return 0.0
        return time.time() - self.entry_time


class FlightModeHandler(ABC):
    """
    Abstract base class for flight mode handlers.

    Each flight mode (STABILIZE, LOITER, AUTO, etc.) implements
    this interface to provide mode-specific behavior.
    """

    def __init__(self, drone: 'Drone'):
        """
        Initialize the mode handler.

        Args:
            drone: Reference to the drone instance
        """
        self._drone = drone
        self._state = ModeState(mode=self.mode)
        self._callbacks: List[Callable[[ModeState], None]] = []

    @property
    @abstractmethod
    def mode(self) -> FlightMode:
        """Return the flight mode this handler manages."""
        pass

    @property
    def state(self) -> ModeState:
        """Return current mode state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if this mode is currently active."""
        return self._state.active

    def can_enter(self) -> Tuple[bool, str]:
        """
        Check if the mode can be entered.

        Returns:
            Tuple of (can_enter, reason_if_not)
        """
        # Check GPS requirement
        if self.mode.requires_gps:
            if not self._drone.gps_sensor.is_valid():
                return False, f"{self.mode.value} requires valid GPS"
            if self._drone.gps_sensor.get_satellites() < 4:
                return False, f"{self.mode.value} requires at least 4 satellites"

        # Check armed requirement
        if self.mode.requires_armed and not self._drone.is_armed:
            return False, f"{self.mode.value} requires armed state"

        return True, ""

    def can_exit(self) -> Tuple[bool, str]:
        """
        Check if the mode can be exited.

        Returns:
            Tuple of (can_exit, reason_if_not)
        """
        return True, ""

    def enter(self) -> Tuple[bool, str]:
        """
        Enter this flight mode.

        Returns:
            Tuple of (success, message)
        """
        can_enter, reason = self.can_enter()
        if not can_enter:
            return False, reason

        self._state.active = True
        self._state.entry_time = time.time()
        self._state.error_message = None

        # Call mode-specific entry logic
        success, message = self._on_enter()
        if not success:
            self._state.active = False
            self._state.error_message = message
            return False, message

        self._notify_callbacks()
        return True, f"Entered {self.mode.value} mode"

    def exit(self) -> Tuple[bool, str]:
        """
        Exit this flight mode.

        Returns:
            Tuple of (success, message)
        """
        can_exit, reason = self.can_exit()
        if not can_exit:
            return False, reason

        # Call mode-specific exit logic
        self._on_exit()

        self._state.active = False
        self._notify_callbacks()
        return True, f"Exited {self.mode.value} mode"

    @abstractmethod
    def update(self, dt: float) -> FlightCommand:
        """
        Update the mode and return a flight command.

        This is called every control loop iteration.

        Args:
            dt: Time delta since last update (seconds)

        Returns:
            FlightCommand to execute
        """
        pass

    def _on_enter(self) -> Tuple[bool, str]:
        """
        Mode-specific entry logic. Override in subclasses.

        Returns:
            Tuple of (success, message)
        """
        return True, ""

    def _on_exit(self) -> None:
        """Mode-specific exit logic. Override in subclasses."""
        pass

    def add_callback(self, callback: Callable[[ModeState], None]) -> None:
        """Add a callback for mode state changes."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ModeState], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception:
                pass  # Don't let callback errors affect mode operation

    def set_target_position(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[bool, str]:
        """
        Set target position for the mode.

        Args:
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude (meters)

        Returns:
            Tuple of (success, message)
        """
        self._state.target_position = (lat, lon, alt)
        return True, "Target position set"

    def set_target_velocity(
        self,
        vn: float,
        ve: float,
        vd: float
    ) -> Tuple[bool, str]:
        """
        Set target velocity for the mode.

        Args:
            vn: North velocity (m/s)
            ve: East velocity (m/s)
            vd: Down velocity (m/s)

        Returns:
            Tuple of (success, message)
        """
        self._state.target_velocity = (vn, ve, vd)
        return True, "Target velocity set"

    def get_status(self) -> dict:
        """Get mode status as dictionary."""
        return {
            'mode': self.mode.value,
            'active': self._state.active,
            'time_in_mode': self._state.time_in_mode,
            'target_position': self._state.target_position,
            'target_velocity': self._state.target_velocity,
            'error': self._state.error_message,
        }
