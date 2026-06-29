"""
Arming and disarming system for Drone AI application.
Handles pre-arm checks, kill switch, and safety interlocks.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ArmingState(Enum):
    """Current arming state."""
    DISARMED = "disarmed"
    ARMING = "arming"
    ARMED = "armed"
    DISARMING = "disarming"
    KILLED = "killed"  # Emergency stop active


class PreArmCheck(Enum):
    """Pre-arm check types."""
    GPS_LOCK = auto()
    GPS_ACCURACY = auto()
    COMPASS_CALIBRATED = auto()
    COMPASS_HEALTHY = auto()
    ACCELEROMETER_CALIBRATED = auto()
    GYRO_CALIBRATED = auto()
    BAROMETER_HEALTHY = auto()
    BATTERY_VOLTAGE = auto()
    BATTERY_PERCENTAGE = auto()
    RC_CALIBRATED = auto()
    RC_CONNECTED = auto()
    FAILSAFE_CONFIGURED = auto()
    GEOFENCE_CONFIGURED = auto()
    HOME_SET = auto()
    MOTORS_HEALTHY = auto()
    ESC_CALIBRATED = auto()
    PROP_CHECK = auto()  # Manual check
    AIRSPACE_CLEAR = auto()  # Manual check
    SAFETY_SWITCH = auto()
    EKF_HEALTHY = auto()
    VIBRATION_OK = auto()
    STORAGE_AVAILABLE = auto()
    FIRMWARE_VALID = auto()


@dataclass
class CheckResult:
    """Result of a single pre-arm check."""
    check: PreArmCheck
    passed: bool
    message: str
    required: bool = True  # If True, must pass to arm
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArmingStatus:
    """Complete arming status."""
    state: ArmingState
    can_arm: bool
    checks: List[CheckResult]
    failed_required: List[CheckResult]
    failed_optional: List[CheckResult]
    kill_switch_active: bool
    armed_time: Optional[float]
    disarmed_time: Optional[float]


class ArmingManager:
    """
    Manages arming, disarming, and pre-arm safety checks.
    """

    # Checks that must pass to arm
    REQUIRED_CHECKS = {
        PreArmCheck.GPS_LOCK,
        PreArmCheck.COMPASS_HEALTHY,
        PreArmCheck.BATTERY_VOLTAGE,
        PreArmCheck.BATTERY_PERCENTAGE,
        PreArmCheck.RC_CONNECTED,
        PreArmCheck.HOME_SET,
        PreArmCheck.EKF_HEALTHY,
    }

    # Minimum requirements
    MIN_GPS_SATELLITES = 6
    MIN_GPS_ACCURACY = 5.0  # meters
    MIN_BATTERY_VOLTAGE = 3.3  # per cell
    MIN_BATTERY_PERCENTAGE = 20.0
    MIN_RC_SIGNAL = 0.3

    def __init__(self):
        self._state = ArmingState.DISARMED
        self._checks: Dict[PreArmCheck, CheckResult] = {}
        self._callbacks: List[Callable] = []
        self._kill_switch_active = False
        self._armed_time: Optional[float] = None
        self._disarmed_time: Optional[float] = None
        self._arming_timeout = 5.0  # seconds
        self._manual_checks_passed: Dict[PreArmCheck, bool] = {}

        # Props-off mode for testing
        self._props_off_mode = False

    def add_callback(self, callback: Callable):
        """Add callback for arming state changes."""
        self._callbacks.append(callback)

    def set_props_off_mode(self, enabled: bool):
        """Enable/disable props-off mode for safe testing."""
        self._props_off_mode = enabled
        logger.info(f"Props-off mode: {'enabled' if enabled else 'disabled'}")

    def confirm_manual_check(self, check: PreArmCheck, passed: bool):
        """Confirm a manual pre-arm check (e.g., props installed)."""
        self._manual_checks_passed[check] = passed
        logger.info(f"Manual check {check.name}: {'passed' if passed else 'failed'}")

    def run_pre_arm_checks(
        self,
        gps_satellites: int,
        gps_accuracy: float,
        gps_fix: bool,
        compass_healthy: bool,
        compass_calibrated: bool,
        accel_calibrated: bool,
        gyro_calibrated: bool,
        baro_healthy: bool,
        battery_voltage: float,
        battery_percentage: float,
        battery_cells: int,
        rc_connected: bool,
        rc_calibrated: bool,
        rc_signal: float,
        home_set: bool,
        motors_healthy: bool,
        ekf_healthy: bool,
        vibration_ok: bool,
        failsafe_configured: bool,
        geofence_configured: bool,
        safety_switch_off: bool = True,
    ) -> List[CheckResult]:
        """
        Run all pre-arm checks and return results.
        """
        self._checks.clear()

        # GPS checks
        self._add_check(
            PreArmCheck.GPS_LOCK,
            gps_fix and gps_satellites >= self.MIN_GPS_SATELLITES,
            f"GPS: {gps_satellites} satellites, {'fix' if gps_fix else 'no fix'}",
        )

        self._add_check(
            PreArmCheck.GPS_ACCURACY,
            gps_accuracy <= self.MIN_GPS_ACCURACY,
            f"GPS accuracy: {gps_accuracy:.1f}m",
            required=False,
        )

        # Compass checks
        self._add_check(
            PreArmCheck.COMPASS_CALIBRATED,
            compass_calibrated,
            "Compass calibration",
        )

        self._add_check(
            PreArmCheck.COMPASS_HEALTHY,
            compass_healthy,
            "Compass health",
        )

        # IMU checks
        self._add_check(
            PreArmCheck.ACCELEROMETER_CALIBRATED,
            accel_calibrated,
            "Accelerometer calibration",
        )

        self._add_check(
            PreArmCheck.GYRO_CALIBRATED,
            gyro_calibrated,
            "Gyroscope calibration",
        )

        # Barometer check
        self._add_check(
            PreArmCheck.BAROMETER_HEALTHY,
            baro_healthy,
            "Barometer health",
            required=False,
        )

        # Battery checks
        voltage_per_cell = battery_voltage / battery_cells if battery_cells > 0 else 0
        self._add_check(
            PreArmCheck.BATTERY_VOLTAGE,
            voltage_per_cell >= self.MIN_BATTERY_VOLTAGE,
            f"Battery voltage: {voltage_per_cell:.2f}V/cell",
        )

        self._add_check(
            PreArmCheck.BATTERY_PERCENTAGE,
            battery_percentage >= self.MIN_BATTERY_PERCENTAGE,
            f"Battery: {battery_percentage:.0f}%",
        )

        # RC checks
        self._add_check(
            PreArmCheck.RC_CONNECTED,
            rc_connected,
            "RC connection",
        )

        self._add_check(
            PreArmCheck.RC_CALIBRATED,
            rc_calibrated,
            "RC calibration",
            required=False,
        )

        # Home position check
        self._add_check(
            PreArmCheck.HOME_SET,
            home_set,
            "Home position",
        )

        # Motor check
        self._add_check(
            PreArmCheck.MOTORS_HEALTHY,
            motors_healthy,
            "Motor health",
        )

        # EKF check
        self._add_check(
            PreArmCheck.EKF_HEALTHY,
            ekf_healthy,
            "EKF status",
        )

        # Vibration check
        self._add_check(
            PreArmCheck.VIBRATION_OK,
            vibration_ok,
            "Vibration levels",
            required=False,
        )

        # Failsafe configuration
        self._add_check(
            PreArmCheck.FAILSAFE_CONFIGURED,
            failsafe_configured,
            "Failsafe configuration",
        )

        # Geofence configuration
        self._add_check(
            PreArmCheck.GEOFENCE_CONFIGURED,
            geofence_configured,
            "Geofence configuration",
            required=False,
        )

        # Safety switch
        self._add_check(
            PreArmCheck.SAFETY_SWITCH,
            safety_switch_off,
            "Safety switch",
        )

        # Manual checks
        prop_check = self._manual_checks_passed.get(PreArmCheck.PROP_CHECK, False)
        self._add_check(
            PreArmCheck.PROP_CHECK,
            prop_check or self._props_off_mode,
            "Propeller check (manual)",
        )

        airspace_check = self._manual_checks_passed.get(PreArmCheck.AIRSPACE_CLEAR, False)
        self._add_check(
            PreArmCheck.AIRSPACE_CLEAR,
            airspace_check,
            "Airspace clear (manual)",
            required=False,
        )

        return list(self._checks.values())

    def _add_check(self, check: PreArmCheck, passed: bool, message: str, required: bool = None):
        """Add a check result."""
        if required is None:
            required = check in self.REQUIRED_CHECKS

        self._checks[check] = CheckResult(
            check=check,
            passed=passed,
            message=f"{'✓' if passed else '✗'} {message}",
            required=required,
        )

    def can_arm(self) -> bool:
        """Check if drone can be armed based on pre-arm checks."""
        if self._kill_switch_active:
            return False

        # All required checks must pass
        for check, result in self._checks.items():
            if result.required and not result.passed:
                return False

        return True

    def arm(self, force: bool = False) -> tuple[bool, str]:
        """
        Attempt to arm the drone.

        Args:
            force: If True, skip non-critical checks (dangerous!)

        Returns:
            (success, message)
        """
        if self._kill_switch_active:
            return False, "Cannot arm: kill switch active"

        if self._state == ArmingState.ARMED:
            return True, "Already armed"

        if self._state == ArmingState.KILLED:
            return False, "Cannot arm: emergency stop active, reset required"

        if not force and not self.can_arm():
            failed = [r for r in self._checks.values() if r.required and not r.passed]
            messages = [r.message for r in failed]
            return False, f"Pre-arm checks failed: {', '.join(messages)}"

        self._state = ArmingState.ARMING
        logger.info("Arming initiated")

        # In a real implementation, this would send arm command to flight controller
        # and wait for confirmation

        self._state = ArmingState.ARMED
        self._armed_time = time.time()
        self._disarmed_time = None

        logger.warning("DRONE ARMED")
        self._notify_callbacks('armed')

        return True, "Armed successfully"

    def disarm(self, reason: str = "user_request") -> tuple[bool, str]:
        """
        Disarm the drone.

        Args:
            reason: Reason for disarming

        Returns:
            (success, message)
        """
        if self._state == ArmingState.DISARMED:
            return True, "Already disarmed"

        self._state = ArmingState.DISARMING
        logger.info(f"Disarming: {reason}")

        # In a real implementation, this would send disarm command to flight controller

        self._state = ArmingState.DISARMED
        self._disarmed_time = time.time()
        flight_time = self._disarmed_time - self._armed_time if self._armed_time else 0

        logger.info(f"DRONE DISARMED (flight time: {flight_time:.1f}s)")
        self._notify_callbacks('disarmed', {'reason': reason, 'flight_time': flight_time})

        return True, f"Disarmed: {reason}"

    def kill(self, reason: str = "emergency") -> tuple[bool, str]:
        """
        Emergency kill switch - immediately stop all motors.

        This is a one-way operation that requires a reset to clear.

        Args:
            reason: Reason for kill

        Returns:
            (success, message)
        """
        self._kill_switch_active = True
        self._state = ArmingState.KILLED

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        self._notify_callbacks('killed', {'reason': reason})

        # In a real implementation, this would:
        # 1. Immediately cut motor power
        # 2. Disable all motor commands
        # 3. Log the event
        # 4. Alert ground station

        return True, f"Kill switch activated: {reason}"

    def reset_kill_switch(self) -> tuple[bool, str]:
        """
        Reset the kill switch after emergency.

        Requires manual confirmation in a real system.
        """
        if not self._kill_switch_active:
            return True, "Kill switch not active"

        self._kill_switch_active = False
        self._state = ArmingState.DISARMED

        logger.warning("Kill switch reset - drone can be rearmed")
        self._notify_callbacks('kill_reset')

        return True, "Kill switch reset"

    def _notify_callbacks(self, event: str, data: dict = None):
        """Notify callbacks of state changes."""
        for callback in self._callbacks:
            try:
                callback({
                    'event': event,
                    'state': self._state.value,
                    'kill_switch': self._kill_switch_active,
                    'timestamp': time.time(),
                    **(data or {}),
                })
            except Exception as e:
                logger.error(f"Arming callback error: {e}")

    def get_status(self) -> ArmingStatus:
        """Get complete arming status."""
        checks = list(self._checks.values())
        failed_required = [c for c in checks if c.required and not c.passed]
        failed_optional = [c for c in checks if not c.required and not c.passed]

        return ArmingStatus(
            state=self._state,
            can_arm=self.can_arm(),
            checks=checks,
            failed_required=failed_required,
            failed_optional=failed_optional,
            kill_switch_active=self._kill_switch_active,
            armed_time=self._armed_time,
            disarmed_time=self._disarmed_time,
        )

    @property
    def is_armed(self) -> bool:
        """Check if drone is armed."""
        return self._state == ArmingState.ARMED

    @property
    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    @property
    def state(self) -> ArmingState:
        """Get current arming state."""
        return self._state

    def get_flight_time(self) -> float:
        """Get current flight time in seconds."""
        if self._armed_time is None:
            return 0.0

        if self._state == ArmingState.ARMED:
            return time.time() - self._armed_time
        elif self._disarmed_time:
            return self._disarmed_time - self._armed_time
        return 0.0

    def to_dict(self) -> dict:
        """Serialize arming manager state."""
        return {
            'state': self._state.value,
            'can_arm': self.can_arm(),
            'kill_switch_active': self._kill_switch_active,
            'props_off_mode': self._props_off_mode,
            'armed_time': self._armed_time,
            'flight_time': self.get_flight_time(),
            'checks': {
                check.name: {
                    'passed': result.passed,
                    'message': result.message,
                    'required': result.required,
                }
                for check, result in self._checks.items()
            },
        }
