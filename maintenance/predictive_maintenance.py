"""
Predictive Maintenance System

Analyzes sensor data, flight patterns, and component usage to predict
maintenance needs before failures occur. Uses statistical models and
anomaly detection to identify degradation trends.

Features:
- Motor health prediction
- Battery degradation tracking
- Vibration analysis
- Flight hour tracking
- Maintenance scheduling
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque
import statistics


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class ComponentType(Enum):
    """Types of components tracked for maintenance."""
    MOTOR_1 = "motor_1"
    MOTOR_2 = "motor_2"
    MOTOR_3 = "motor_3"
    MOTOR_4 = "motor_4"
    BATTERY = "battery"
    ESC = "esc"
    PROPELLER = "propeller"
    GIMBAL = "gimbal"
    GPS = "gps"
    IMU = "imu"
    BAROMETER = "barometer"
    CAMERA = "camera"
    FRAME = "frame"
    LANDING_GEAR = "landing_gear"


class HealthStatus(Enum):
    """Component health status."""
    EXCELLENT = "excellent"     # 90-100%
    GOOD = "good"               # 70-90%
    FAIR = "fair"               # 50-70%
    DEGRADED = "degraded"       # 30-50%
    POOR = "poor"               # 10-30%
    CRITICAL = "critical"       # 0-10%


class MaintenanceUrgency(Enum):
    """Maintenance urgency level."""
    NONE = "none"               # No maintenance needed
    SCHEDULED = "scheduled"     # Normal scheduled maintenance
    RECOMMENDED = "recommended" # Should be done soon
    REQUIRED = "required"       # Required before next flight
    IMMEDIATE = "immediate"     # Ground immediately


class MaintenanceAction(Enum):
    """Types of maintenance actions."""
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    CALIBRATION = "calibration"
    REPLACEMENT = "replacement"
    REPAIR = "repair"
    FIRMWARE_UPDATE = "firmware_update"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    component_type: ComponentType
    health_percent: float = 100.0
    status: HealthStatus = HealthStatus.EXCELLENT
    flight_hours: float = 0.0
    cycles: int = 0  # On/off cycles
    last_maintenance: float = 0.0  # Timestamp
    estimated_life_remaining: float = 0.0  # Hours
    failure_probability: float = 0.0  # 0-1 probability of failure before next maintenance
    degradation_rate: float = 0.0  # Percent per hour

    # Trend data
    health_history: List[Tuple[float, float]] = field(default_factory=list)

    def update_status(self) -> None:
        """Update status based on health percent."""
        if self.health_percent >= 90:
            self.status = HealthStatus.EXCELLENT
        elif self.health_percent >= 70:
            self.status = HealthStatus.GOOD
        elif self.health_percent >= 50:
            self.status = HealthStatus.FAIR
        elif self.health_percent >= 30:
            self.status = HealthStatus.DEGRADED
        elif self.health_percent >= 10:
            self.status = HealthStatus.POOR
        else:
            self.status = HealthStatus.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component_type.value,
            'health_percent': round(self.health_percent, 1),
            'status': self.status.value,
            'flight_hours': round(self.flight_hours, 1),
            'cycles': self.cycles,
            'estimated_life_remaining': round(self.estimated_life_remaining, 1),
            'failure_probability': round(self.failure_probability, 3),
            'degradation_rate': round(self.degradation_rate, 4),
        }


@dataclass
class MaintenanceAlert:
    """Maintenance alert/recommendation."""
    component: ComponentType
    action: MaintenanceAction
    urgency: MaintenanceUrgency
    description: str
    estimated_time: float = 0.0  # Minutes to complete
    parts_needed: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    due_by: Optional[float] = None  # Timestamp
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component.value,
            'action': self.action.value,
            'urgency': self.urgency.value,
            'description': self.description,
            'estimated_time_minutes': self.estimated_time,
            'parts_needed': self.parts_needed,
            'created_at': self.created_at,
            'due_by': self.due_by,
            'acknowledged': self.acknowledged,
        }


@dataclass
class MaintenanceConfig:
    """Configuration for maintenance predictions."""
    # Service intervals (hours)
    motor_service_hours: float = 100.0
    propeller_replace_hours: float = 50.0
    battery_service_cycles: int = 300
    gimbal_service_hours: float = 200.0
    frame_inspection_hours: float = 500.0

    # Alert thresholds
    warning_threshold: float = 70.0  # Health percent
    critical_threshold: float = 30.0

    # Vibration thresholds (m/s^2)
    vibration_warning: float = 5.0
    vibration_critical: float = 10.0

    # Battery thresholds
    battery_capacity_warning: float = 80.0  # Percent of original
    battery_internal_resistance_warning: float = 0.1  # Ohms increase

    # History settings
    history_window: int = 100  # Data points to keep


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------

class VibrationAnalyzer:
    """
    Analyzes vibration patterns to detect motor/propeller issues.

    Uses FFT-like analysis to detect imbalance and bearing wear.
    """

    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize analyzer.

        Args:
            sample_rate: IMU sample rate in Hz
        """
        self._sample_rate = sample_rate
        self._buffer_size = 256
        self._buffers: Dict[str, deque] = {
            'x': deque(maxlen=self._buffer_size),
            'y': deque(maxlen=self._buffer_size),
            'z': deque(maxlen=self._buffer_size),
        }

        # Analysis results
        self._rms_vibration = 0.0
        self._peak_frequency = 0.0
        self._imbalance_score = 0.0

    def add_sample(self, ax: float, ay: float, az: float) -> None:
        """Add accelerometer sample."""
        # Remove gravity component (assuming z is up when level)
        self._buffers['x'].append(ax)
        self._buffers['y'].append(ay)
        self._buffers['z'].append(az - 9.81)

    def analyze(self) -> Dict[str, float]:
        """
        Analyze vibration data.

        Returns:
            Dict with vibration metrics
        """
        if len(self._buffers['x']) < 10:
            return {
                'rms_vibration': 0.0,
                'peak_vibration': 0.0,
                'imbalance_score': 0.0,
            }

        # Calculate RMS vibration
        x_data = list(self._buffers['x'])
        y_data = list(self._buffers['y'])
        z_data = list(self._buffers['z'])

        x_rms = self._calculate_rms(x_data)
        y_rms = self._calculate_rms(y_data)
        z_rms = self._calculate_rms(z_data)

        self._rms_vibration = math.sqrt(x_rms**2 + y_rms**2 + z_rms**2)

        # Peak vibration
        peak_x = max(abs(v) for v in x_data)
        peak_y = max(abs(v) for v in y_data)
        peak_z = max(abs(v) for v in z_data)
        peak_vibration = max(peak_x, peak_y, peak_z)

        # Imbalance score (based on low-frequency oscillation)
        # Simplified: use variance ratio between axes
        try:
            var_x = statistics.variance(x_data)
            var_y = statistics.variance(y_data)
            var_ratio = max(var_x, var_y) / (min(var_x, var_y) + 0.001)
            self._imbalance_score = min(1.0, (var_ratio - 1) / 5)
        except statistics.StatisticsError:
            self._imbalance_score = 0.0

        return {
            'rms_vibration': self._rms_vibration,
            'peak_vibration': peak_vibration,
            'imbalance_score': self._imbalance_score,
        }

    def _calculate_rms(self, data: List[float]) -> float:
        """Calculate RMS of data."""
        if not data:
            return 0.0
        return math.sqrt(sum(x**2 for x in data) / len(data))

    def get_motor_health_impact(self) -> float:
        """
        Get estimated impact on motor health.

        Returns:
            Degradation factor (0-1, higher = more degradation)
        """
        # High vibration accelerates motor wear
        if self._rms_vibration < 2.0:
            return 0.0
        elif self._rms_vibration < 5.0:
            return 0.1
        elif self._rms_vibration < 10.0:
            return 0.3
        else:
            return 0.5


class BatteryAnalyzer:
    """
    Analyzes battery health and predicts remaining cycles.
    """

    def __init__(self):
        """Initialize analyzer."""
        self._voltage_history: deque = deque(maxlen=1000)
        self._current_history: deque = deque(maxlen=1000)
        self._temperature_history: deque = deque(maxlen=100)

        # Battery state
        self._cycle_count = 0
        self._original_capacity = 0.0
        self._current_capacity = 0.0
        self._internal_resistance = 0.0
        self._original_resistance = 0.0

        # Health metrics
        self._capacity_retention = 100.0
        self._resistance_increase = 0.0

    def update(
        self,
        voltage: float,
        current: float,
        temperature: float = 25.0,
        capacity_mah: Optional[float] = None,
    ) -> None:
        """Update with new battery readings."""
        timestamp = time.time()

        self._voltage_history.append((timestamp, voltage))
        self._current_history.append((timestamp, current))
        self._temperature_history.append((timestamp, temperature))

        if capacity_mah is not None:
            if self._original_capacity == 0:
                self._original_capacity = capacity_mah
            self._current_capacity = capacity_mah

    def add_cycle(self) -> None:
        """Record a charge cycle."""
        self._cycle_count += 1

    def set_original_capacity(self, capacity_mah: float) -> None:
        """Set original battery capacity."""
        self._original_capacity = capacity_mah
        if self._current_capacity == 0:
            self._current_capacity = capacity_mah

    def estimate_internal_resistance(self) -> float:
        """
        Estimate internal resistance from voltage/current data.

        Uses voltage sag under load to estimate IR.
        """
        if len(self._voltage_history) < 10 or len(self._current_history) < 10:
            return 0.0

        # Find voltage drop during high current draw
        voltages = [v for _, v in self._voltage_history]
        currents = [c for _, c in self._current_history]

        # Simple estimation: V_drop / I = R
        # Look for high current events
        high_current_idx = [i for i, c in enumerate(currents) if c > 10]

        if not high_current_idx:
            return self._internal_resistance

        # Average resistance estimate
        resistances = []
        for i in high_current_idx:
            if i > 0 and currents[i] - currents[i-1] > 5:
                v_drop = voltages[i-1] - voltages[i]
                i_delta = currents[i] - currents[i-1]
                if i_delta > 0:
                    resistances.append(v_drop / i_delta)

        if resistances:
            self._internal_resistance = statistics.mean(resistances)

        return self._internal_resistance

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze battery health.

        Returns:
            Dict with health metrics
        """
        # Capacity retention
        if self._original_capacity > 0:
            self._capacity_retention = (
                self._current_capacity / self._original_capacity * 100
            )

        # Resistance increase
        if self._original_resistance > 0:
            self._resistance_increase = (
                (self._internal_resistance - self._original_resistance)
                / self._original_resistance * 100
            )

        # Estimate remaining cycles (simplified Peukert-like model)
        if self._cycle_count > 0:
            degradation_per_cycle = (100 - self._capacity_retention) / self._cycle_count
            remaining_capacity_loss = self._capacity_retention - 70  # 70% is end of life
            if degradation_per_cycle > 0:
                remaining_cycles = int(remaining_capacity_loss / degradation_per_cycle)
            else:
                remaining_cycles = 500  # Default
        else:
            remaining_cycles = 500

        # Temperature analysis
        if self._temperature_history:
            temps = [t for _, t in self._temperature_history]
            max_temp = max(temps)
            avg_temp = statistics.mean(temps)
        else:
            max_temp = 25.0
            avg_temp = 25.0

        return {
            'cycle_count': self._cycle_count,
            'capacity_retention': round(self._capacity_retention, 1),
            'internal_resistance': round(self._internal_resistance, 4),
            'resistance_increase_percent': round(self._resistance_increase, 1),
            'remaining_cycles': remaining_cycles,
            'max_temperature': round(max_temp, 1),
            'avg_temperature': round(avg_temp, 1),
        }

    def get_health_percent(self) -> float:
        """Get overall battery health percentage."""
        # Weight factors
        capacity_weight = 0.6
        resistance_weight = 0.3
        cycle_weight = 0.1

        # Capacity score (100% at full, 0% at 70%)
        capacity_score = max(0, (self._capacity_retention - 70) / 30 * 100)

        # Resistance score (penalize increase)
        resistance_score = max(0, 100 - self._resistance_increase * 5)

        # Cycle score (assuming 500 cycle life)
        cycle_score = max(0, (1 - self._cycle_count / 500) * 100)

        return (
            capacity_weight * capacity_score +
            resistance_weight * resistance_score +
            cycle_weight * cycle_score
        )


class MotorAnalyzer:
    """
    Analyzes motor health based on performance metrics.
    """

    def __init__(self, motor_id: int):
        """Initialize analyzer for specific motor."""
        self._motor_id = motor_id
        self._rpm_history: deque = deque(maxlen=1000)
        self._current_history: deque = deque(maxlen=1000)
        self._temperature_history: deque = deque(maxlen=100)
        self._throttle_history: deque = deque(maxlen=1000)

        # Baseline metrics (set during calibration)
        self._baseline_efficiency = 0.0  # RPM per amp
        self._baseline_response = 0.0  # Time to target RPM

        # Current metrics
        self._current_efficiency = 0.0
        self._response_degradation = 0.0
        self._operating_hours = 0.0

    def update(
        self,
        rpm: float,
        current: float,
        throttle: float,
        temperature: float = 40.0,
    ) -> None:
        """Update with motor telemetry."""
        timestamp = time.time()

        self._rpm_history.append((timestamp, rpm))
        self._current_history.append((timestamp, current))
        self._throttle_history.append((timestamp, throttle))
        self._temperature_history.append((timestamp, temperature))

    def add_operating_time(self, hours: float) -> None:
        """Add operating time."""
        self._operating_hours += hours

    def analyze(self) -> Dict[str, Any]:
        """Analyze motor health."""
        if len(self._rpm_history) < 10:
            return {
                'motor_id': self._motor_id,
                'efficiency': 100.0,
                'operating_hours': round(self._operating_hours, 1),
                'max_temperature': 40.0,
                'avg_temperature': 40.0,
                'current_efficiency': 0.0,
            }

        # Calculate current efficiency
        rpms = [r for _, r in self._rpm_history]
        currents = [c for _, c in self._current_history]

        # RPM per amp at stable throttle
        efficiencies = []
        for i in range(len(rpms)):
            if currents[i] > 1:
                efficiencies.append(rpms[i] / currents[i])

        if efficiencies:
            self._current_efficiency = statistics.mean(efficiencies)

        # Calculate efficiency relative to baseline
        if self._baseline_efficiency > 0:
            efficiency_percent = (
                self._current_efficiency / self._baseline_efficiency * 100
            )
        else:
            efficiency_percent = 100.0
            if self._current_efficiency > 0:
                self._baseline_efficiency = self._current_efficiency

        # Temperature analysis
        if self._temperature_history:
            temps = [t for _, t in self._temperature_history]
            max_temp = max(temps)
            avg_temp = statistics.mean(temps)
        else:
            max_temp = 40.0
            avg_temp = 40.0

        return {
            'motor_id': self._motor_id,
            'efficiency': round(min(100, efficiency_percent), 1),
            'operating_hours': round(self._operating_hours, 1),
            'max_temperature': round(max_temp, 1),
            'avg_temperature': round(avg_temp, 1),
            'current_efficiency': round(self._current_efficiency, 1),
        }

    def get_health_percent(self) -> float:
        """Get motor health percentage."""
        analysis = self.analyze()

        # Efficiency degradation
        efficiency_score = analysis['efficiency']

        # Temperature penalty
        temp_penalty = max(0, (analysis['max_temperature'] - 60) * 2)

        # Hours penalty (assume 500 hour life)
        hours_penalty = min(30, self._operating_hours / 500 * 30)

        return max(0, efficiency_score - temp_penalty - hours_penalty)


# ---------------------------------------------------------------------------
# Main Predictive Maintenance System
# ---------------------------------------------------------------------------

class PredictiveMaintenanceSystem:
    """
    Main predictive maintenance system.

    Integrates all analyzers and generates maintenance recommendations.
    """

    def __init__(self, config: Optional[MaintenanceConfig] = None):
        """Initialize system."""
        self.config = config or MaintenanceConfig()

        # Component health tracking
        self._components: Dict[ComponentType, ComponentHealth] = {}
        self._initialize_components()

        # Analyzers
        self._vibration_analyzer = VibrationAnalyzer()
        self._battery_analyzer = BatteryAnalyzer()
        self._motor_analyzers: Dict[int, MotorAnalyzer] = {
            i: MotorAnalyzer(i) for i in range(1, 5)
        }

        # Alerts
        self._alerts: List[MaintenanceAlert] = []

        # Flight tracking
        self._flight_start_time: Optional[float] = None
        self._total_flight_hours = 0.0
        self._total_flights = 0
        self._total_landings = 0

        # Callbacks
        self._alert_callbacks: List[Callable[[MaintenanceAlert], None]] = []

    def _initialize_components(self) -> None:
        """Initialize component health records."""
        for comp_type in ComponentType:
            self._components[comp_type] = ComponentHealth(
                component_type=comp_type,
            )

    def start_flight(self) -> None:
        """Record flight start."""
        self._flight_start_time = time.time()
        self._total_flights += 1

    def end_flight(self) -> None:
        """Record flight end."""
        if self._flight_start_time:
            flight_hours = (time.time() - self._flight_start_time) / 3600
            self._total_flight_hours += flight_hours

            # Update component flight hours
            for comp in self._components.values():
                comp.flight_hours += flight_hours

            self._flight_start_time = None
            self._total_landings += 1

            # Update component cycles
            for comp_type in [
                ComponentType.MOTOR_1, ComponentType.MOTOR_2,
                ComponentType.MOTOR_3, ComponentType.MOTOR_4,
                ComponentType.BATTERY, ComponentType.LANDING_GEAR,
            ]:
                self._components[comp_type].cycles += 1

    def update_imu(self, ax: float, ay: float, az: float) -> None:
        """Update with IMU data for vibration analysis."""
        self._vibration_analyzer.add_sample(ax, ay, az)

    def update_battery(
        self,
        voltage: float,
        current: float,
        temperature: float = 25.0,
        capacity_mah: Optional[float] = None,
    ) -> None:
        """Update battery status."""
        self._battery_analyzer.update(voltage, current, temperature, capacity_mah)

    def update_motor(
        self,
        motor_id: int,
        rpm: float,
        current: float,
        throttle: float,
        temperature: float = 40.0,
    ) -> None:
        """Update motor telemetry."""
        if motor_id in self._motor_analyzers:
            self._motor_analyzers[motor_id].update(rpm, current, throttle, temperature)

    def record_battery_cycle(self) -> None:
        """Record a battery charge cycle."""
        self._battery_analyzer.add_cycle()
        self._components[ComponentType.BATTERY].cycles += 1

    def record_maintenance(
        self,
        component: ComponentType,
        action: MaintenanceAction,
    ) -> None:
        """Record completed maintenance."""
        if component in self._components:
            self._components[component].last_maintenance = time.time()

            # Reset health based on action
            if action == MaintenanceAction.REPLACEMENT:
                self._components[component].health_percent = 100.0
                self._components[component].flight_hours = 0.0
                self._components[component].cycles = 0
            elif action == MaintenanceAction.REPAIR:
                self._components[component].health_percent = min(
                    100, self._components[component].health_percent + 30
                )
            elif action == MaintenanceAction.CALIBRATION:
                self._components[component].health_percent = min(
                    100, self._components[component].health_percent + 10
                )

            self._components[component].update_status()

        # Remove related alerts
        self._alerts = [
            a for a in self._alerts
            if not (a.component == component and a.action == action)
        ]

    def analyze(self) -> Dict[str, Any]:
        """
        Run full analysis and generate alerts.

        Returns:
            Analysis results
        """
        results = {
            'overall_health': 100.0,
            'components': {},
            'new_alerts': [],
            'recommendations': [],
        }

        # Analyze vibration
        vibration = self._vibration_analyzer.analyze()
        vibration_impact = self._vibration_analyzer.get_motor_health_impact()

        # Analyze battery
        battery = self._battery_analyzer.analyze()
        battery_health = self._battery_analyzer.get_health_percent()

        # Update battery component
        self._components[ComponentType.BATTERY].health_percent = battery_health
        self._components[ComponentType.BATTERY].estimated_life_remaining = (
            battery['remaining_cycles'] * 0.5  # Assume 30 min per cycle
        )
        self._components[ComponentType.BATTERY].update_status()

        # Analyze motors
        min_health = 100.0
        for motor_id, analyzer in self._motor_analyzers.items():
            motor_data = analyzer.analyze()
            motor_health = analyzer.get_health_percent()

            # Apply vibration impact
            motor_health -= vibration_impact * 10

            comp_type = ComponentType(f"motor_{motor_id}")
            self._components[comp_type].health_percent = max(0, motor_health)
            self._components[comp_type].update_status()

            min_health = min(min_health, motor_health)

        # Update other components based on flight hours
        for comp_type, comp in self._components.items():
            if comp_type not in [
                ComponentType.MOTOR_1, ComponentType.MOTOR_2,
                ComponentType.MOTOR_3, ComponentType.MOTOR_4,
                ComponentType.BATTERY,
            ]:
                # Simple degradation model
                self._update_component_health(comp)

            results['components'][comp_type.value] = comp.to_dict()

        # Generate alerts
        new_alerts = self._generate_alerts(vibration, battery)
        results['new_alerts'] = [a.to_dict() for a in new_alerts]

        # Calculate overall health
        health_values = [c.health_percent for c in self._components.values()]
        results['overall_health'] = min(health_values) if health_values else 100.0

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations()

        return results

    def _update_component_health(self, comp: ComponentHealth) -> None:
        """Update component health based on usage."""
        # Service intervals (hours)
        service_intervals = {
            ComponentType.PROPELLER: 50,
            ComponentType.GIMBAL: 200,
            ComponentType.GPS: 1000,
            ComponentType.IMU: 500,
            ComponentType.FRAME: 500,
            ComponentType.LANDING_GEAR: 200,
            ComponentType.ESC: 300,
            ComponentType.CAMERA: 1000,
            ComponentType.BAROMETER: 1000,
        }

        interval = service_intervals.get(comp.component_type, 500)
        hours_since_maintenance = comp.flight_hours

        if comp.last_maintenance > 0:
            # Calculate actual hours since maintenance
            pass  # Would need flight log integration

        # Linear degradation model
        degradation = (hours_since_maintenance / interval) * 30
        comp.health_percent = max(0, 100 - degradation)
        comp.degradation_rate = 30 / interval
        comp.estimated_life_remaining = max(0, (100 - degradation) / comp.degradation_rate)
        comp.update_status()

    def _generate_alerts(
        self,
        vibration: Dict[str, float],
        battery: Dict[str, Any],
    ) -> List[MaintenanceAlert]:
        """Generate maintenance alerts based on analysis."""
        new_alerts = []

        # Vibration alerts
        if vibration['rms_vibration'] > self.config.vibration_critical:
            alert = MaintenanceAlert(
                component=ComponentType.PROPELLER,
                action=MaintenanceAction.REPLACEMENT,
                urgency=MaintenanceUrgency.REQUIRED,
                description="Critical vibration detected - check propellers and motor mounts",
                parts_needed=["Propeller set"],
            )
            if not self._alert_exists(alert):
                new_alerts.append(alert)

        elif vibration['rms_vibration'] > self.config.vibration_warning:
            alert = MaintenanceAlert(
                component=ComponentType.PROPELLER,
                action=MaintenanceAction.INSPECTION,
                urgency=MaintenanceUrgency.RECOMMENDED,
                description="Elevated vibration - inspect propellers for damage",
            )
            if not self._alert_exists(alert):
                new_alerts.append(alert)

        # Battery alerts
        if battery['capacity_retention'] < self.config.battery_capacity_warning:
            alert = MaintenanceAlert(
                component=ComponentType.BATTERY,
                action=MaintenanceAction.REPLACEMENT,
                urgency=MaintenanceUrgency.RECOMMENDED,
                description=f"Battery capacity at {battery['capacity_retention']:.0f}% - consider replacement",
                parts_needed=["New battery pack"],
            )
            if not self._alert_exists(alert):
                new_alerts.append(alert)

        # Component health alerts
        for comp_type, comp in self._components.items():
            if comp.health_percent < self.config.critical_threshold:
                alert = MaintenanceAlert(
                    component=comp_type,
                    action=MaintenanceAction.INSPECTION,
                    urgency=MaintenanceUrgency.REQUIRED,
                    description=f"{comp_type.value} health critical at {comp.health_percent:.0f}%",
                )
                if not self._alert_exists(alert):
                    new_alerts.append(alert)

            elif comp.health_percent < self.config.warning_threshold:
                alert = MaintenanceAlert(
                    component=comp_type,
                    action=MaintenanceAction.INSPECTION,
                    urgency=MaintenanceUrgency.SCHEDULED,
                    description=f"{comp_type.value} health degraded to {comp.health_percent:.0f}%",
                )
                if not self._alert_exists(alert):
                    new_alerts.append(alert)

        # Add new alerts to list
        for alert in new_alerts:
            self._alerts.append(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass

        return new_alerts

    def _alert_exists(self, new_alert: MaintenanceAlert) -> bool:
        """Check if similar alert already exists."""
        for alert in self._alerts:
            if (alert.component == new_alert.component and
                alert.action == new_alert.action and
                not alert.acknowledged):
                return True
        return False

    def _generate_recommendations(self) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        # Sort components by health
        sorted_components = sorted(
            self._components.values(),
            key=lambda c: c.health_percent,
        )

        # Recommend for worst components
        for comp in sorted_components[:3]:
            if comp.health_percent < 50:
                recommendations.append(
                    f"Schedule {comp.component_type.value} maintenance - "
                    f"estimated {comp.estimated_life_remaining:.0f} hours remaining"
                )

        # Flight hours based recommendations
        if self._total_flight_hours > 100:
            recommendations.append(
                "Perform comprehensive inspection - 100+ flight hours accumulated"
            )

        if self._total_landings > 200:
            recommendations.append(
                "Inspect landing gear - 200+ landings completed"
            )

        return recommendations

    def acknowledge_alert(self, component: ComponentType, action: MaintenanceAction) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.component == component and alert.action == action:
                alert.acknowledged = True
                return True
        return False

    def get_alerts(
        self,
        urgency: Optional[MaintenanceUrgency] = None,
        unacknowledged_only: bool = False,
    ) -> List[MaintenanceAlert]:
        """Get alerts filtered by criteria."""
        alerts = self._alerts

        if urgency:
            alerts = [a for a in alerts if a.urgency == urgency]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def register_alert_callback(
        self,
        callback: Callable[[MaintenanceAlert], None],
    ) -> None:
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)

    def get_component_health(self, component: ComponentType) -> ComponentHealth:
        """Get health for specific component."""
        return self._components.get(component, ComponentHealth(component_type=component))

    def get_overall_health(self) -> float:
        """Get overall system health percentage."""
        if not self._components:
            return 100.0

        # Weighted average with emphasis on critical components
        weights = {
            ComponentType.MOTOR_1: 2.0,
            ComponentType.MOTOR_2: 2.0,
            ComponentType.MOTOR_3: 2.0,
            ComponentType.MOTOR_4: 2.0,
            ComponentType.BATTERY: 2.0,
            ComponentType.ESC: 1.5,
            ComponentType.IMU: 1.5,
            ComponentType.GPS: 1.0,
        }

        total_weight = 0.0
        weighted_health = 0.0

        for comp_type, comp in self._components.items():
            weight = weights.get(comp_type, 1.0)
            weighted_health += comp.health_percent * weight
            total_weight += weight

        return weighted_health / total_weight if total_weight > 0 else 100.0

    def get_next_maintenance(self) -> Optional[MaintenanceAlert]:
        """Get most urgent pending maintenance."""
        unacked = [a for a in self._alerts if not a.acknowledged]
        if not unacked:
            return None

        # Sort by urgency
        urgency_order = {
            MaintenanceUrgency.IMMEDIATE: 0,
            MaintenanceUrgency.REQUIRED: 1,
            MaintenanceUrgency.RECOMMENDED: 2,
            MaintenanceUrgency.SCHEDULED: 3,
            MaintenanceUrgency.NONE: 4,
        }

        unacked.sort(key=lambda a: urgency_order.get(a.urgency, 5))
        return unacked[0]

    def is_flight_ready(self) -> Tuple[bool, List[str]]:
        """
        Check if system is ready for flight.

        Returns:
            Tuple of (ready, list of issues)
        """
        issues = []

        # Check for critical alerts
        critical_alerts = [
            a for a in self._alerts
            if a.urgency in [MaintenanceUrgency.IMMEDIATE, MaintenanceUrgency.REQUIRED]
            and not a.acknowledged
        ]

        for alert in critical_alerts:
            issues.append(f"{alert.component.value}: {alert.description}")

        # Check component health
        for comp_type, comp in self._components.items():
            if comp.health_percent < 10:
                issues.append(f"{comp_type.value} critically low at {comp.health_percent:.0f}%")

        return len(issues) == 0, issues

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'overall_health': round(self.get_overall_health(), 1),
            'flight_ready': self.is_flight_ready()[0],
            'total_flight_hours': round(self._total_flight_hours, 1),
            'total_flights': self._total_flights,
            'active_alerts': len([a for a in self._alerts if not a.acknowledged]),
            'components': {
                k.value: v.to_dict() for k, v in self._components.items()
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_status()
