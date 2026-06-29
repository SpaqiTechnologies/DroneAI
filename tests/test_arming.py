"""
Tests for the arming system.
"""

import pytest
from core.arming import (
    ArmingManager,
    ArmingState,
    PreArmCheck,
    CheckResult,
)


class TestArmingManager:
    """Tests for the ArmingManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = ArmingManager()

    def _run_passing_checks(self):
        """Run pre-arm checks with all passing values."""
        return self.manager.run_pre_arm_checks(
            gps_satellites=10,
            gps_accuracy=2.0,
            gps_fix=True,
            compass_healthy=True,
            compass_calibrated=True,
            accel_calibrated=True,
            gyro_calibrated=True,
            baro_healthy=True,
            battery_voltage=12.6,
            battery_percentage=80.0,
            battery_cells=3,
            rc_connected=True,
            rc_calibrated=True,
            rc_signal=0.9,
            home_set=True,
            motors_healthy=True,
            ekf_healthy=True,
            vibration_ok=True,
            failsafe_configured=True,
            geofence_configured=True,
            safety_switch_off=True,
        )

    def test_initial_state_disarmed(self):
        """Test that initial state is disarmed."""
        assert self.manager.state == ArmingState.DISARMED
        assert not self.manager.is_armed
        assert not self.manager.is_killed

    def test_pre_arm_checks_all_pass(self):
        """Test pre-arm checks with all passing values."""
        # Need to confirm manual check for props
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)

        checks = self._run_passing_checks()

        assert len(checks) > 0
        assert self.manager.can_arm()

    def test_pre_arm_check_gps_fail(self):
        """Test pre-arm check with GPS failure."""
        self.manager.run_pre_arm_checks(
            gps_satellites=3,  # Too few satellites
            gps_accuracy=10.0,
            gps_fix=False,
            compass_healthy=True,
            compass_calibrated=True,
            accel_calibrated=True,
            gyro_calibrated=True,
            baro_healthy=True,
            battery_voltage=12.6,
            battery_percentage=80.0,
            battery_cells=3,
            rc_connected=True,
            rc_calibrated=True,
            rc_signal=0.9,
            home_set=True,
            motors_healthy=True,
            ekf_healthy=True,
            vibration_ok=True,
            failsafe_configured=True,
            geofence_configured=True,
            safety_switch_off=True,
        )

        assert not self.manager.can_arm()

    def test_pre_arm_check_low_battery(self):
        """Test pre-arm check with low battery."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self.manager.run_pre_arm_checks(
            gps_satellites=10,
            gps_accuracy=2.0,
            gps_fix=True,
            compass_healthy=True,
            compass_calibrated=True,
            accel_calibrated=True,
            gyro_calibrated=True,
            baro_healthy=True,
            battery_voltage=9.0,  # Low voltage
            battery_percentage=15.0,  # Below 20% threshold
            battery_cells=3,
            rc_connected=True,
            rc_calibrated=True,
            rc_signal=0.9,
            home_set=True,
            motors_healthy=True,
            ekf_healthy=True,
            vibration_ok=True,
            failsafe_configured=True,
            geofence_configured=True,
            safety_switch_off=True,
        )

        assert not self.manager.can_arm()

    def test_arm_success(self):
        """Test successful arming."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()

        success, message = self.manager.arm()

        assert success
        assert self.manager.state == ArmingState.ARMED
        assert self.manager.is_armed

    def test_arm_fails_without_checks(self):
        """Test that arming fails if required checks don't pass."""
        # Run checks with failing conditions (no GPS fix)
        self.manager.run_pre_arm_checks(
            gps_satellites=2,  # Too few
            gps_accuracy=20.0,
            gps_fix=False,  # No fix
            compass_healthy=True,
            compass_calibrated=True,
            accel_calibrated=True,
            gyro_calibrated=True,
            baro_healthy=True,
            battery_voltage=12.6,
            battery_percentage=80.0,
            battery_cells=3,
            rc_connected=False,  # Not connected
            rc_calibrated=True,
            rc_signal=0.9,
            home_set=False,  # Home not set
            motors_healthy=True,
            ekf_healthy=False,  # EKF not healthy
            vibration_ok=True,
            failsafe_configured=True,
            geofence_configured=True,
            safety_switch_off=True,
        )

        success, message = self.manager.arm()

        assert not success
        assert "Pre-arm checks failed" in message
        assert not self.manager.is_armed

    def test_arm_force_mode(self):
        """Test forced arming bypasses checks."""
        # Don't run checks, force arm
        success, message = self.manager.arm(force=True)

        assert success
        assert self.manager.is_armed

    def test_disarm(self):
        """Test disarming."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()
        self.manager.arm()

        success, message = self.manager.disarm("test")

        assert success
        assert self.manager.state == ArmingState.DISARMED
        assert not self.manager.is_armed

    def test_kill_switch(self):
        """Test emergency kill switch."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()
        self.manager.arm()

        success, message = self.manager.kill("emergency")

        assert success
        assert self.manager.state == ArmingState.KILLED
        assert self.manager.is_killed
        assert not self.manager.is_armed

    def test_cannot_arm_after_kill(self):
        """Test that arming is blocked after kill switch."""
        self.manager.kill("test")

        success, message = self.manager.arm(force=True)

        assert not success
        # Message could be about kill switch being active or reset required
        assert "kill switch" in message.lower() or "reset" in message.lower()

    def test_reset_kill_switch(self):
        """Test resetting kill switch."""
        self.manager.kill("test")
        success, message = self.manager.reset_kill_switch()

        assert success
        assert not self.manager.is_killed
        assert self.manager.state == ArmingState.DISARMED

    def test_props_off_mode(self):
        """Test props-off mode for testing."""
        self.manager.set_props_off_mode(True)

        # Run checks without manual prop check confirmation
        self.manager.run_pre_arm_checks(
            gps_satellites=10,
            gps_accuracy=2.0,
            gps_fix=True,
            compass_healthy=True,
            compass_calibrated=True,
            accel_calibrated=True,
            gyro_calibrated=True,
            baro_healthy=True,
            battery_voltage=12.6,
            battery_percentage=80.0,
            battery_cells=3,
            rc_connected=True,
            rc_calibrated=True,
            rc_signal=0.9,
            home_set=True,
            motors_healthy=True,
            ekf_healthy=True,
            vibration_ok=True,
            failsafe_configured=True,
            geofence_configured=True,
            safety_switch_off=True,
        )

        # Should be able to arm because props-off mode bypasses prop check
        assert self.manager.can_arm()

    def test_callback_on_arm(self):
        """Test callback is called on arm."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.manager.add_callback(on_event)
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()
        self.manager.arm()

        assert len(callback_data) > 0
        assert callback_data[0]['event'] == 'armed'

    def test_callback_on_kill(self):
        """Test callback is called on kill."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.manager.add_callback(on_event)
        self.manager.kill("test")

        assert len(callback_data) > 0
        assert callback_data[0]['event'] == 'killed'

    def test_flight_time_tracking(self):
        """Test flight time tracking."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()

        assert self.manager.get_flight_time() == 0.0

        self.manager.arm()
        # Flight time should be > 0 after arming
        import time
        time.sleep(0.1)
        assert self.manager.get_flight_time() > 0

    def test_get_status(self):
        """Test getting arming status."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()

        status = self.manager.get_status()

        assert status.state == ArmingState.DISARMED
        assert status.can_arm
        assert not status.kill_switch_active
        assert len(status.checks) > 0

    def test_to_dict(self):
        """Test serialization."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self._run_passing_checks()

        data = self.manager.to_dict()

        assert 'state' in data
        assert 'can_arm' in data
        assert 'checks' in data
        assert data['state'] == 'disarmed'

    def test_manual_check_confirmation(self):
        """Test manual check confirmation."""
        self.manager.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        self.manager.confirm_manual_check(PreArmCheck.AIRSPACE_CLEAR, True)

        self._run_passing_checks()

        # Both manual checks should pass
        status = self.manager.get_status()
        prop_check = next((c for c in status.checks if c.check == PreArmCheck.PROP_CHECK), None)
        assert prop_check is not None
        assert prop_check.passed
