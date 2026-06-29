"""
Takeoff Manager Module

Automated takeoff sequence management with pre-flight checks,
motor spool-up, and controlled climb to target altitude.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Callable, List, TYPE_CHECKING
import time
import math

if TYPE_CHECKING:
    from core.drone import Drone


class TakeoffState(Enum):
    """State of the takeoff sequence."""
    IDLE = "idle"                   # Not in takeoff
    PRE_ARM_CHECK = "pre_arm_check" # Running pre-arm checks
    ARMING = "arming"               # Arming motors
    MOTOR_SPOOL = "motor_spool"     # Motors spooling up
    LIFTOFF = "liftoff"             # Initial liftoff
    CLIMBING = "climbing"           # Climbing to target altitude
    HOVER_CHECK = "hover_check"     # Stabilization check at altitude
    COMPLETE = "complete"           # Takeoff complete
    ABORTED = "aborted"             # Takeoff aborted
    FAILED = "failed"               # Takeoff failed


class TakeoffMode(Enum):
    """Takeoff mode."""
    NORMAL = "normal"               # Standard takeoff
    QUICK = "quick"                 # Reduced checks, faster
    PRECISION = "precision"         # Extra stabilization
    VERTICAL = "vertical"           # Pure vertical climb


@dataclass
class TakeoffConfig:
    """Configuration for takeoff sequence."""
    target_altitude: float = 10.0           # meters
    climb_rate: float = 2.0                 # m/s
    max_climb_rate: float = 5.0             # m/s

    # Timing
    motor_spool_time: float = 2.0           # seconds
    hover_check_time: float = 3.0           # seconds
    arm_timeout: float = 10.0               # seconds

    # Thresholds
    liftoff_altitude: float = 0.5           # meters to detect liftoff
    altitude_tolerance: float = 1.0         # meters for target reached
    position_hold_tolerance: float = 1.0    # meters lateral drift

    # Safety
    min_battery: float = 25.0               # minimum battery %
    max_wind: float = 10.0                  # max wind speed m/s
    require_gps: bool = True
    min_satellites: int = 6

    # Motor test
    motor_test_enabled: bool = False
    motor_test_throttle: float = 0.3        # 0-1


@dataclass
class TakeoffResult:
    """Result of takeoff attempt."""
    success: bool = False
    state: TakeoffState = TakeoffState.IDLE
    message: str = ""
    duration: float = 0.0                   # seconds
    final_altitude: float = 0.0             # meters

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'state': self.state.value,
            'message': self.message,
            'duration': self.duration,
            'final_altitude': self.final_altitude,
        }


class TakeoffManager:
    """
    Manages automated takeoff sequences.

    Handles the complete takeoff process:
    1. Pre-arm safety checks
    2. Motor arming
    3. Motor spool-up
    4. Controlled liftoff
    5. Climb to target altitude
    6. Hover stabilization check
    """

    def __init__(self, drone: 'Drone'):
        self._drone = drone
        self._config = TakeoffConfig()
        self._state = TakeoffState.IDLE
        self._mode = TakeoffMode.NORMAL

        # Timing
        self._start_time = 0.0
        self._state_start_time = 0.0

        # Target
        self._target_altitude = 10.0
        self._initial_position: Optional[Tuple[float, float]] = None

        # Throttle control
        self._throttle = 0.0
        self._target_throttle = 0.0

        # Callbacks
        self._state_callbacks: List[Callable[[TakeoffState], None]] = []
        self._complete_callbacks: List[Callable[[TakeoffResult], None]] = []

        # Result
        self._result = TakeoffResult()

    @property
    def state(self) -> TakeoffState:
        """Get current takeoff state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if takeoff is in progress."""
        return self._state not in [
            TakeoffState.IDLE,
            TakeoffState.COMPLETE,
            TakeoffState.ABORTED,
            TakeoffState.FAILED,
        ]

    def configure(self, config: TakeoffConfig) -> None:
        """Set takeoff configuration."""
        self._config = config

    def start_takeoff(
        self,
        target_altitude: float = 10.0,
        mode: TakeoffMode = TakeoffMode.NORMAL,
    ) -> Tuple[bool, str]:
        """
        Start the takeoff sequence.

        Args:
            target_altitude: Target altitude in meters
            mode: Takeoff mode

        Returns:
            Tuple of (success, message)
        """
        if self.is_active:
            return False, "Takeoff already in progress"

        if self._drone.is_flying:
            return False, "Drone is already flying"

        # Validate altitude
        if target_altitude < 2.0:
            return False, "Target altitude too low (min 2m)"
        if target_altitude > self._config.max_climb_rate * 60:
            return False, f"Target altitude too high (max {self._config.max_climb_rate * 60}m)"

        # Initialize
        self._target_altitude = target_altitude
        self._mode = mode
        self._start_time = time.time()
        self._result = TakeoffResult()

        # Store initial position
        if self._drone.current_position:
            self._initial_position = self._drone.current_position

        # Start with pre-arm checks
        self._transition_to(TakeoffState.PRE_ARM_CHECK)

        return True, f"Starting takeoff to {target_altitude}m"

    def abort(self, reason: str = "User abort") -> Tuple[bool, str]:
        """
        Abort the takeoff sequence.

        Args:
            reason: Reason for abort

        Returns:
            Tuple of (success, message)
        """
        if not self.is_active:
            return False, "No takeoff in progress"

        self._result.success = False
        self._result.message = reason
        self._transition_to(TakeoffState.ABORTED)

        # Emergency actions based on state
        if self._state in [TakeoffState.LIFTOFF, TakeoffState.CLIMBING]:
            # Already airborne, need to land
            if hasattr(self._drone, 'start_landing'):
                self._drone.start_landing()
        elif self._state == TakeoffState.MOTOR_SPOOL:
            # Cut throttle
            self._throttle = 0.0
            if self._drone.is_armed:
                self._drone.disarm("takeoff_abort")

        return True, f"Takeoff aborted: {reason}"

    def update(self, dt: float) -> Optional[Dict[str, float]]:
        """
        Update takeoff state machine.

        Args:
            dt: Time delta since last update

        Returns:
            Control commands dict or None
        """
        if not self.is_active:
            return None

        state_duration = time.time() - self._state_start_time

        if self._state == TakeoffState.PRE_ARM_CHECK:
            return self._update_pre_arm_check(dt, state_duration)
        elif self._state == TakeoffState.ARMING:
            return self._update_arming(dt, state_duration)
        elif self._state == TakeoffState.MOTOR_SPOOL:
            return self._update_motor_spool(dt, state_duration)
        elif self._state == TakeoffState.LIFTOFF:
            return self._update_liftoff(dt, state_duration)
        elif self._state == TakeoffState.CLIMBING:
            return self._update_climbing(dt, state_duration)
        elif self._state == TakeoffState.HOVER_CHECK:
            return self._update_hover_check(dt, state_duration)

        return None

    def _update_pre_arm_check(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Run pre-arm checks."""
        # Run checks
        checks_passed, message = self._run_pre_arm_checks()

        if checks_passed:
            self._transition_to(TakeoffState.ARMING)
        elif state_duration > 5.0:
            self._fail(f"Pre-arm checks failed: {message}")

        return None

    def _update_arming(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Handle arming sequence."""
        if self._drone.is_armed:
            self._transition_to(TakeoffState.MOTOR_SPOOL)
            return None

        # Try to arm
        if state_duration < 0.5:
            # First attempt
            success, message = self._drone.arm(force=False)
            if not success:
                # Try with force if normal arm fails
                success, message = self._drone.arm(force=True)

        # Check timeout
        if state_duration > self._config.arm_timeout:
            self._fail("Arming timeout")

        return None

    def _update_motor_spool(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Handle motor spool-up."""
        # Ramp up throttle
        spool_progress = min(1.0, state_duration / self._config.motor_spool_time)
        self._throttle = spool_progress * 0.4  # 40% hover throttle estimate

        if state_duration >= self._config.motor_spool_time:
            self._transition_to(TakeoffState.LIFTOFF)

        return {
            'throttle': self._throttle,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
        }

    def _update_liftoff(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Handle initial liftoff."""
        current_alt = self._drone.current_altitude

        # Increase throttle gradually
        self._throttle = min(0.7, self._throttle + 0.1 * dt)

        # Check if we've lifted off
        if current_alt >= self._config.liftoff_altitude:
            self._drone._is_flying = True
            self._transition_to(TakeoffState.CLIMBING)

        # Timeout check
        if state_duration > 10.0 and current_alt < self._config.liftoff_altitude:
            self._fail("Liftoff failed - not gaining altitude")

        return {
            'throttle': self._throttle,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
        }

    def _update_climbing(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Handle climb to target altitude."""
        current_alt = self._drone.current_altitude
        alt_error = self._target_altitude - current_alt

        # Check if target reached
        if abs(alt_error) <= self._config.altitude_tolerance:
            self._transition_to(TakeoffState.HOVER_CHECK)
            return self._get_hover_command()

        # Calculate climb rate
        desired_climb = min(self._config.climb_rate, alt_error * 0.5)
        desired_climb = max(0.5, desired_climb)  # Minimum climb rate

        # Throttle adjustment (simplified)
        if alt_error > 0:
            self._throttle = 0.5 + (desired_climb / self._config.max_climb_rate) * 0.3
        else:
            self._throttle = 0.4  # Reduce if overshooting

        # Check lateral drift
        if self._initial_position:
            drift = self._calculate_drift()
            if drift > self._config.position_hold_tolerance * 3:
                # Too much drift, might need to abort
                pass

        return {
            'throttle': self._throttle,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'altitude_target': self._target_altitude,
            'climb_rate': desired_climb,
        }

    def _update_hover_check(self, dt: float, state_duration: float) -> Optional[Dict[str, float]]:
        """Verify stable hover at target altitude."""
        current_alt = self._drone.current_altitude
        alt_error = abs(self._target_altitude - current_alt)

        # Check stability
        if state_duration >= self._config.hover_check_time:
            if alt_error <= self._config.altitude_tolerance * 2:
                self._complete()
            else:
                # Not stable enough, extend check or fail
                if state_duration > self._config.hover_check_time * 2:
                    self._fail("Failed to stabilize at target altitude")

        return self._get_hover_command()

    def _get_hover_command(self) -> Dict[str, float]:
        """Get command to maintain hover."""
        return {
            'throttle': 0.5,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'altitude_target': self._target_altitude,
            'mode': 'position_hold',
        }

    def _run_pre_arm_checks(self) -> Tuple[bool, str]:
        """Run pre-arm safety checks."""
        # Battery check
        if self._drone.battery_level < self._config.min_battery:
            return False, f"Battery too low ({self._drone.battery_level:.1f}%)"

        # GPS check
        if self._config.require_gps:
            if not self._drone.gps_sensor.is_valid():
                return False, "No GPS fix"
            if self._drone.gps_sensor.satellites < self._config.min_satellites:
                return False, f"Insufficient satellites ({self._drone.gps_sensor.satellites})"

        # Wind check
        if hasattr(self._drone, 'wind_sensor'):
            if self._drone.wind_sensor.wind_speed > self._config.max_wind:
                return False, f"Wind too strong ({self._drone.wind_sensor.wind_speed:.1f} m/s)"

        # IMU check
        if hasattr(self._drone, 'imu_sensor'):
            if not self._drone.imu_sensor.is_calibrated():
                return False, "IMU not calibrated"

        # Geofence check
        if hasattr(self._drone, 'check_geofence'):
            geofence_status = self._drone.check_geofence()
            if geofence_status and not geofence_status.get('inside', True):
                return False, "Outside geofence"

        # Quick mode skips some checks
        if self._mode == TakeoffMode.QUICK:
            return True, "Quick checks passed"

        # Additional checks for normal/precision modes
        # Health check
        if hasattr(self._drone, 'get_health_summary'):
            health = self._drone.get_health_summary()
            if health.get('overall_status') == 'critical':
                return False, "Critical health issue"

        return True, "All checks passed"

    def _calculate_drift(self) -> float:
        """Calculate lateral drift from initial position."""
        if not self._initial_position:
            return 0.0

        current = self._drone.current_position
        if not current:
            return 0.0

        # Simple distance calculation
        dlat = current[0] - self._initial_position[0]
        dlon = current[1] - self._initial_position[1]

        # Approximate meters (at typical latitudes)
        dx = dlat * 111000
        dy = dlon * 111000 * math.cos(math.radians(current[0]))

        return math.sqrt(dx*dx + dy*dy)

    def _transition_to(self, new_state: TakeoffState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_start_time = time.time()

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(new_state)
            except Exception:
                pass

    def _complete(self) -> None:
        """Complete the takeoff sequence."""
        self._result.success = True
        self._result.state = TakeoffState.COMPLETE
        self._result.message = "Takeoff complete"
        self._result.duration = time.time() - self._start_time
        self._result.final_altitude = self._drone.current_altitude

        self._transition_to(TakeoffState.COMPLETE)
        self._notify_complete()

    def _fail(self, reason: str) -> None:
        """Handle takeoff failure."""
        self._result.success = False
        self._result.state = TakeoffState.FAILED
        self._result.message = reason
        self._result.duration = time.time() - self._start_time
        self._result.final_altitude = self._drone.current_altitude

        self._transition_to(TakeoffState.FAILED)

        # Safety: disarm if on ground
        if self._drone.current_altitude < 1.0 and self._drone.is_armed:
            self._drone.disarm("takeoff_failed")

        self._notify_complete()

    def _notify_complete(self) -> None:
        """Notify completion callbacks."""
        for callback in self._complete_callbacks:
            try:
                callback(self._result)
            except Exception:
                pass

    def add_state_callback(self, callback: Callable[[TakeoffState], None]) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    def add_complete_callback(self, callback: Callable[[TakeoffResult], None]) -> None:
        """Add callback for takeoff completion."""
        self._complete_callbacks.append(callback)

    def get_progress(self) -> float:
        """Get takeoff progress (0-1)."""
        if self._state == TakeoffState.IDLE:
            return 0.0
        if self._state == TakeoffState.COMPLETE:
            return 1.0
        if self._state == TakeoffState.FAILED:
            return 0.0
        if self._state == TakeoffState.ABORTED:
            return 0.0

        # Progress through states
        state_progress = {
            TakeoffState.PRE_ARM_CHECK: 0.1,
            TakeoffState.ARMING: 0.2,
            TakeoffState.MOTOR_SPOOL: 0.3,
            TakeoffState.LIFTOFF: 0.4,
            TakeoffState.CLIMBING: 0.7,
            TakeoffState.HOVER_CHECK: 0.9,
        }

        base = state_progress.get(self._state, 0.0)

        # Add sub-progress for climbing
        if self._state == TakeoffState.CLIMBING:
            current_alt = self._drone.current_altitude
            climb_progress = min(1.0, current_alt / self._target_altitude)
            base = 0.4 + climb_progress * 0.3

        return base

    def get_status(self) -> Dict[str, Any]:
        """Get takeoff status."""
        return {
            'active': self.is_active,
            'state': self._state.value,
            'mode': self._mode.value,
            'progress': self.get_progress(),
            'target_altitude': self._target_altitude,
            'current_altitude': self._drone.current_altitude,
            'throttle': self._throttle,
            'duration': time.time() - self._start_time if self.is_active else 0,
            'drift': self._calculate_drift() if self._initial_position else 0,
        }

    def get_result(self) -> TakeoffResult:
        """Get takeoff result."""
        return self._result
