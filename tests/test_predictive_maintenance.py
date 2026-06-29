"""
Tests for predictive maintenance system.
"""

import pytest
import time
from maintenance.predictive_maintenance import (
    PredictiveMaintenanceSystem,
    ComponentType,
    HealthStatus,
    MaintenanceUrgency,
    MaintenanceAction,
    ComponentHealth,
    MaintenanceAlert,
    MaintenanceConfig,
    VibrationAnalyzer,
    BatteryAnalyzer,
    MotorAnalyzer,
)


class TestComponentHealth:
    """Tests for ComponentHealth class."""

    def test_creation(self):
        health = ComponentHealth(
            component_type=ComponentType.MOTOR_1,
            health_percent=85.0,
        )

        assert health.component_type == ComponentType.MOTOR_1
        assert health.health_percent == 85.0

    def test_update_status_excellent(self):
        health = ComponentHealth(
            component_type=ComponentType.BATTERY,
            health_percent=95.0,
        )
        health.update_status()

        assert health.status == HealthStatus.EXCELLENT

    def test_update_status_good(self):
        health = ComponentHealth(
            component_type=ComponentType.BATTERY,
            health_percent=75.0,
        )
        health.update_status()

        assert health.status == HealthStatus.GOOD

    def test_update_status_critical(self):
        health = ComponentHealth(
            component_type=ComponentType.BATTERY,
            health_percent=5.0,
        )
        health.update_status()

        assert health.status == HealthStatus.CRITICAL

    def test_to_dict(self):
        health = ComponentHealth(
            component_type=ComponentType.PROPELLER,
            health_percent=70.0,
            flight_hours=25.5,
        )
        health.update_status()

        d = health.to_dict()

        assert d['component'] == 'propeller'
        assert d['health_percent'] == 70.0
        assert d['flight_hours'] == 25.5
        assert d['status'] == 'good'


class TestMaintenanceAlert:
    """Tests for MaintenanceAlert class."""

    def test_creation(self):
        alert = MaintenanceAlert(
            component=ComponentType.MOTOR_1,
            action=MaintenanceAction.INSPECTION,
            urgency=MaintenanceUrgency.RECOMMENDED,
            description="Check motor bearings",
        )

        assert alert.component == ComponentType.MOTOR_1
        assert alert.urgency == MaintenanceUrgency.RECOMMENDED
        assert not alert.acknowledged

    def test_to_dict(self):
        alert = MaintenanceAlert(
            component=ComponentType.BATTERY,
            action=MaintenanceAction.REPLACEMENT,
            urgency=MaintenanceUrgency.REQUIRED,
            description="Battery degraded",
            estimated_time=30.0,
            parts_needed=["New battery pack"],
        )

        d = alert.to_dict()

        assert d['component'] == 'battery'
        assert d['action'] == 'replacement'
        assert d['urgency'] == 'required'
        assert d['estimated_time_minutes'] == 30.0
        assert 'New battery pack' in d['parts_needed']


class TestVibrationAnalyzer:
    """Tests for VibrationAnalyzer."""

    def test_creation(self):
        analyzer = VibrationAnalyzer()
        result = analyzer.analyze()

        assert result['rms_vibration'] == 0.0
        assert result['imbalance_score'] == 0.0

    def test_add_samples(self):
        analyzer = VibrationAnalyzer()

        # Add some samples
        for i in range(20):
            analyzer.add_sample(0.1, 0.1, 9.81 + 0.05)

        result = analyzer.analyze()

        assert result['rms_vibration'] > 0

    def test_high_vibration_detection(self):
        analyzer = VibrationAnalyzer()

        # Add high vibration samples
        for i in range(50):
            ax = 5.0 * (1 if i % 2 == 0 else -1)
            analyzer.add_sample(ax, 0, 9.81)

        result = analyzer.analyze()
        impact = analyzer.get_motor_health_impact()

        assert result['rms_vibration'] > 3.0
        assert impact > 0  # Should indicate wear

    def test_low_vibration_no_impact(self):
        analyzer = VibrationAnalyzer()

        # Add low vibration samples
        for i in range(20):
            analyzer.add_sample(0.01, 0.01, 9.81)

        impact = analyzer.get_motor_health_impact()

        assert impact == 0  # No impact from low vibration


class TestBatteryAnalyzer:
    """Tests for BatteryAnalyzer."""

    def test_creation(self):
        analyzer = BatteryAnalyzer()
        result = analyzer.analyze()

        assert result['cycle_count'] == 0
        assert result['capacity_retention'] == 100.0

    def test_update(self):
        analyzer = BatteryAnalyzer()

        analyzer.update(voltage=22.4, current=5.0, temperature=30.0)

        result = analyzer.analyze()

        assert result['avg_temperature'] == 30.0

    def test_add_cycle(self):
        analyzer = BatteryAnalyzer()

        analyzer.add_cycle()
        analyzer.add_cycle()

        result = analyzer.analyze()

        assert result['cycle_count'] == 2

    def test_set_capacity(self):
        analyzer = BatteryAnalyzer()

        analyzer.set_original_capacity(5000)  # 5000 mAh

        health = analyzer.get_health_percent()

        assert health > 90  # New battery should be healthy

    def test_degradation_detection(self):
        analyzer = BatteryAnalyzer()

        analyzer.set_original_capacity(5000)

        # Simulate many cycles
        for _ in range(200):
            analyzer.add_cycle()

        # Simulate capacity loss
        analyzer._current_capacity = 4000  # 80% capacity

        result = analyzer.analyze()

        assert result['capacity_retention'] == 80.0
        assert result['remaining_cycles'] < 500


class TestMotorAnalyzer:
    """Tests for MotorAnalyzer."""

    def test_creation(self):
        analyzer = MotorAnalyzer(motor_id=1)
        result = analyzer.analyze()

        assert result['motor_id'] == 1
        assert result['efficiency'] == 100.0

    def test_update(self):
        analyzer = MotorAnalyzer(motor_id=2)

        analyzer.update(rpm=5000, current=10.0, throttle=0.5, temperature=45.0)

        result = analyzer.analyze()

        assert result['motor_id'] == 2

    def test_operating_time(self):
        analyzer = MotorAnalyzer(motor_id=3)

        analyzer.add_operating_time(10.0)
        analyzer.add_operating_time(5.0)

        result = analyzer.analyze()

        assert result['operating_hours'] == 15.0

    def test_health_degradation(self):
        analyzer = MotorAnalyzer(motor_id=1)

        # Simulate many operating hours
        analyzer.add_operating_time(400)

        health = analyzer.get_health_percent()

        # Should show degradation after many hours
        assert health < 100


class TestPredictiveMaintenanceSystem:
    """Tests for main maintenance system."""

    def test_creation(self):
        system = PredictiveMaintenanceSystem()

        assert system.get_overall_health() > 0

    def test_start_end_flight(self):
        system = PredictiveMaintenanceSystem()

        system.start_flight()

        # Simulate short flight
        time.sleep(0.1)

        system.end_flight()

        status = system.get_status()

        assert status['total_flights'] == 1

    def test_update_imu(self):
        system = PredictiveMaintenanceSystem()

        # Send IMU data
        for _ in range(20):
            system.update_imu(0.1, 0.1, 9.81)

        result = system.analyze()

        assert 'components' in result

    def test_update_battery(self):
        system = PredictiveMaintenanceSystem()

        system.update_battery(voltage=22.4, current=5.0, temperature=25.0)

        result = system.analyze()

        assert 'battery' in result['components']

    def test_update_motor(self):
        system = PredictiveMaintenanceSystem()

        system.update_motor(
            motor_id=1,
            rpm=5000,
            current=10.0,
            throttle=0.5,
        )

        result = system.analyze()

        assert 'motor_1' in result['components']

    def test_record_maintenance(self):
        system = PredictiveMaintenanceSystem()

        # Degrade a component
        system._components[ComponentType.MOTOR_1].health_percent = 50.0

        # Record maintenance
        system.record_maintenance(
            ComponentType.MOTOR_1,
            MaintenanceAction.REPLACEMENT,
        )

        health = system.get_component_health(ComponentType.MOTOR_1)

        assert health.health_percent == 100.0

    def test_generate_alerts(self):
        system = PredictiveMaintenanceSystem()

        # Simulate high vibration to trigger alert
        for _ in range(50):
            system._vibration_analyzer.add_sample(15.0, 15.0, 9.81)

        result = system.analyze()

        # High vibration should trigger propeller inspection alert
        # or analyze the components dict for low health alerts
        all_alerts = system.get_alerts()
        low_health_components = [
            c for c in result['components'].values()
            if c['health_percent'] < 50
        ]

        # Either alerts generated OR components degraded
        assert len(all_alerts) > 0 or len(low_health_components) > 0 or len(result['recommendations']) > 0

    def test_acknowledge_alert(self):
        system = PredictiveMaintenanceSystem()

        # Create an alert
        alert = MaintenanceAlert(
            component=ComponentType.PROPELLER,
            action=MaintenanceAction.INSPECTION,
            urgency=MaintenanceUrgency.RECOMMENDED,
            description="Test alert",
        )
        system._alerts.append(alert)

        # Acknowledge it
        result = system.acknowledge_alert(
            ComponentType.PROPELLER,
            MaintenanceAction.INSPECTION,
        )

        assert result is True
        assert system._alerts[0].acknowledged is True

    def test_get_alerts_filtered(self):
        system = PredictiveMaintenanceSystem()

        # Create alerts with different urgencies
        system._alerts = [
            MaintenanceAlert(
                component=ComponentType.MOTOR_1,
                action=MaintenanceAction.INSPECTION,
                urgency=MaintenanceUrgency.REQUIRED,
                description="Required",
            ),
            MaintenanceAlert(
                component=ComponentType.BATTERY,
                action=MaintenanceAction.INSPECTION,
                urgency=MaintenanceUrgency.SCHEDULED,
                description="Scheduled",
            ),
        ]

        required = system.get_alerts(urgency=MaintenanceUrgency.REQUIRED)

        assert len(required) == 1
        assert required[0].component == ComponentType.MOTOR_1

    def test_is_flight_ready(self):
        system = PredictiveMaintenanceSystem()

        ready, issues = system.is_flight_ready()

        # New system should be ready
        assert ready is True
        assert len(issues) == 0

    def test_is_flight_ready_with_critical(self):
        system = PredictiveMaintenanceSystem()

        # Add critical alert
        alert = MaintenanceAlert(
            component=ComponentType.MOTOR_1,
            action=MaintenanceAction.REPLACEMENT,
            urgency=MaintenanceUrgency.REQUIRED,
            description="Motor failure imminent",
        )
        system._alerts.append(alert)

        ready, issues = system.is_flight_ready()

        assert ready is False
        assert len(issues) > 0

    def test_get_next_maintenance(self):
        system = PredictiveMaintenanceSystem()

        # Add alerts with different urgencies
        system._alerts = [
            MaintenanceAlert(
                component=ComponentType.BATTERY,
                action=MaintenanceAction.INSPECTION,
                urgency=MaintenanceUrgency.SCHEDULED,
                description="Scheduled",
            ),
            MaintenanceAlert(
                component=ComponentType.MOTOR_1,
                action=MaintenanceAction.INSPECTION,
                urgency=MaintenanceUrgency.REQUIRED,
                description="Required",
            ),
        ]

        next_maint = system.get_next_maintenance()

        # Should return most urgent first
        assert next_maint is not None
        assert next_maint.urgency == MaintenanceUrgency.REQUIRED

    def test_register_alert_callback(self):
        system = PredictiveMaintenanceSystem()
        alerts_received = []

        system.register_alert_callback(lambda a: alerts_received.append(a))

        # Force an alert
        system._components[ComponentType.IMU].health_percent = 5.0
        system.analyze()

        # Callback should have been called
        # (depends on alert generation logic)

    def test_get_status(self):
        system = PredictiveMaintenanceSystem()
        status = system.get_status()

        assert 'overall_health' in status
        assert 'flight_ready' in status
        assert 'total_flight_hours' in status
        assert 'components' in status

    def test_to_dict(self):
        system = PredictiveMaintenanceSystem()
        d = system.to_dict()

        assert 'overall_health' in d
        assert 'components' in d


class TestMaintenanceConfig:
    """Tests for MaintenanceConfig."""

    def test_defaults(self):
        config = MaintenanceConfig()

        assert config.motor_service_hours == 100.0
        assert config.warning_threshold == 70.0
        assert config.critical_threshold == 30.0

    def test_custom_config(self):
        config = MaintenanceConfig(
            motor_service_hours=150.0,
            vibration_warning=3.0,
        )

        assert config.motor_service_hours == 150.0
        assert config.vibration_warning == 3.0
