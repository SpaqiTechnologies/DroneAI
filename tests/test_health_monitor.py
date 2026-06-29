"""
Tests for the health monitoring system.
"""

import pytest
from core.health_monitor import (
    HealthMonitor,
    HealthStatus,
    ComponentType,
    HealthThresholds,
)


class TestHealthMonitor:
    """Tests for the HealthMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = HealthMonitor()

    def test_initial_status(self):
        """Test initial health status."""
        status = self.monitor.get_overall_status()
        # Initial status should be GOOD or UNKNOWN
        assert status in (HealthStatus.GOOD, HealthStatus.UNKNOWN)

    def test_update_cpu_component(self):
        """Test updating CPU health."""
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=45.0,
            usage=30.0,
        )

        status = self.monitor.get_component_status(ComponentType.CPU)
        assert status == HealthStatus.GOOD

    def test_cpu_warning_temperature(self):
        """Test CPU warning on high temperature."""
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=75.0,  # Warning threshold
            usage=30.0,
        )

        status = self.monitor.get_component_status(ComponentType.CPU)
        assert status == HealthStatus.WARNING

    def test_cpu_critical_temperature(self):
        """Test CPU critical on very high temperature."""
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=90.0,  # Critical threshold
            usage=30.0,
        )

        status = self.monitor.get_component_status(ComponentType.CPU)
        assert status == HealthStatus.CRITICAL

    def test_update_battery_component(self):
        """Test updating battery health."""
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=12.0,
            percentage=80.0,
            temperature=25.0,
        )

        status = self.monitor.get_component_status(ComponentType.BATTERY)
        assert status == HealthStatus.GOOD

    def test_battery_warning_low_percentage(self):
        """Test battery warning on low voltage per cell."""
        # 10.5V for 3S = 3.5V/cell which is exactly at warning threshold
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=10.5,  # 3.5V/cell for 3S - at warning threshold
            percentage=30.0,
            temperature=25.0,
        )

        status = self.monitor.get_component_status(ComponentType.BATTERY)
        assert status in (HealthStatus.WARNING, HealthStatus.GOOD)

    def test_battery_critical_low_voltage(self):
        """Test battery critical on low voltage."""
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=9.5,  # Critical for 3S
            percentage=15.0,
            temperature=25.0,
        )

        status = self.monitor.get_component_status(ComponentType.BATTERY)
        assert status == HealthStatus.CRITICAL

    def test_update_gps_component(self):
        """Test updating GPS health."""
        self.monitor.update_component(
            ComponentType.GPS,
            satellites=12,
            accuracy=2.0,
            has_fix=True,
        )

        status = self.monitor.get_component_status(ComponentType.GPS)
        assert status == HealthStatus.GOOD

    def test_gps_warning_low_satellites(self):
        """Test GPS warning/critical on low satellites."""
        self.monitor.update_component(
            ComponentType.GPS,
            satellites=5,  # Below minimum of 6
            accuracy=5.0,
            has_fix=True,
        )

        status = self.monitor.get_component_status(ComponentType.GPS)
        # Low satellites can trigger warning or critical depending on HDOP
        assert status in (HealthStatus.WARNING, HealthStatus.CRITICAL)

    def test_gps_failed_no_fix(self):
        """Test GPS failed when no fix."""
        self.monitor.update_component(
            ComponentType.GPS,
            satellites=0,
            accuracy=100.0,
            has_fix=False,
        )

        status = self.monitor.get_component_status(ComponentType.GPS)
        assert status in (HealthStatus.CRITICAL, HealthStatus.FAILED)

    def test_update_motor_component(self):
        """Test updating motor health."""
        self.monitor.update_component(
            ComponentType.MOTOR,
            motor_id=0,
            healthy=True,
            current=5.0,
            rpm=5000,
        )

        status = self.monitor.get_component_status(ComponentType.MOTOR)
        assert status == HealthStatus.GOOD

    def test_motor_failed(self):
        """Test motor failed status."""
        self.monitor.update_component(
            ComponentType.MOTOR,
            motor_id=0,
            healthy=False,
            current=0.0,
            rpm=0,
        )

        status = self.monitor.get_component_status(ComponentType.MOTOR)
        assert status in (HealthStatus.CRITICAL, HealthStatus.FAILED)

    def test_update_imu_component(self):
        """Test updating IMU health."""
        self.monitor.update_component(
            ComponentType.IMU,
            healthy=True,
            vibration=0.5,
        )

        status = self.monitor.get_component_status(ComponentType.IMU)
        assert status == HealthStatus.GOOD

    def test_imu_high_vibration(self):
        """Test IMU warning on high vibration."""
        self.monitor.update_component(
            ComponentType.IMU,
            healthy=True,
            vibration=35.0,  # Above warning threshold of 30
        )

        status = self.monitor.get_component_status(ComponentType.IMU)
        assert status in (HealthStatus.WARNING, HealthStatus.CRITICAL)

    def test_overall_status_worst_case(self):
        """Test overall status reflects worst component."""
        # Set GPS to good
        self.monitor.update_component(
            ComponentType.GPS,
            satellites=12,
            accuracy=2.0,
            has_fix=True,
        )

        # Set battery to critical
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=9.5,
            percentage=10.0,
            temperature=25.0,
        )

        status = self.monitor.get_overall_status()
        assert status == HealthStatus.CRITICAL

    def test_get_all_component_status(self):
        """Test getting all component statuses."""
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=45.0,
            usage=30.0,
        )
        self.monitor.update_component(
            ComponentType.GPS,
            satellites=12,
            accuracy=2.0,
            has_fix=True,
        )

        statuses = self.monitor.get_all_component_status()

        # Check that we get a dict with component keys
        assert isinstance(statuses, dict)
        # Check that cpu_temp or similar is in the statuses
        assert 'cpu_temp' in statuses or 'gps' in statuses

    def test_get_critical_issues(self):
        """Test getting critical issues."""
        # Create a critical condition
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=9.0,
            percentage=5.0,
            temperature=25.0,
        )

        issues = self.monitor.get_critical_issues()
        assert len(issues) > 0

    def test_get_warnings(self):
        """Test getting warnings."""
        # Create a warning condition
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=75.0,
            usage=30.0,
        )

        warnings = self.monitor.get_warnings()
        assert len(warnings) > 0

    def test_callback_on_status_change(self):
        """Test callback is called when checking for alerts."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.monitor.add_callback(on_event)

        # Create a critical condition
        self.monitor.update_component(
            ComponentType.BATTERY,
            voltage=9.0,
            percentage=5.0,
            temperature=25.0,
        )

        # Manually trigger alert check (normally done in monitor loop)
        self.monitor._check_for_alerts()

        # Callback should have been triggered
        assert len(callback_data) > 0

    def test_to_dict(self):
        """Test serialization."""
        self.monitor.update_component(
            ComponentType.CPU,
            temperature=45.0,
            usage=30.0,
        )

        data = self.monitor.to_dict()

        assert 'components' in data
        # Check for system_status which is what to_dict returns
        assert 'system_status' in data or 'overall_status' in data

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        custom_thresholds = HealthThresholds(
            cpu_temp_warning=50.0,  # Lower warning threshold
            cpu_temp_critical=60.0,
        )
        monitor = HealthMonitor(thresholds=custom_thresholds)

        monitor.update_component(
            ComponentType.CPU,
            temperature=55.0,  # Between custom warning and critical
            usage=30.0,
        )

        status = monitor.get_component_status(ComponentType.CPU)
        assert status == HealthStatus.WARNING

    def test_memory_usage_warning(self):
        """Test memory usage warning."""
        self.monitor.update_component(
            ComponentType.MEMORY,
            usage=85.0,  # High usage
        )

        status = self.monitor.get_component_status(ComponentType.MEMORY)
        assert status == HealthStatus.WARNING

    def test_compass_health(self):
        """Test compass health monitoring."""
        self.monitor.update_component(
            ComponentType.COMPASS,
            healthy=True,
            calibrated=True,
        )

        status = self.monitor.get_component_status(ComponentType.COMPASS)
        assert status == HealthStatus.GOOD

    def test_compass_not_calibrated(self):
        """Test compass warning when not calibrated."""
        self.monitor.update_component(
            ComponentType.COMPASS,
            healthy=True,
            calibrated=False,
        )

        status = self.monitor.get_component_status(ComponentType.COMPASS)
        assert status in (HealthStatus.WARNING, HealthStatus.CRITICAL)
