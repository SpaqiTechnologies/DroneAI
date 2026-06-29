"""
Health monitoring system for Drone AI application.
Monitors CPU, memory, sensors, motors, and system health.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of monitored components."""
    CPU = auto()
    MEMORY = auto()
    GPS = auto()
    IMU = auto()
    BAROMETER = auto()
    MAGNETOMETER = auto()
    COMPASS = auto()  # Alias for MAGNETOMETER
    BATTERY = auto()
    MOTOR = auto()
    ESC = auto()
    RC_LINK = auto()
    TELEMETRY = auto()
    STORAGE = auto()
    EKF = auto()
    VIBRATION = auto()
    CAMERA = auto()


@dataclass
class HealthThresholds:
    """Thresholds for health monitoring."""
    # CPU temperature (Celsius)
    cpu_temp_warning: float = 70.0
    cpu_temp_critical: float = 85.0

    # CPU usage (percentage)
    cpu_usage_warning: float = 80.0
    cpu_usage_critical: float = 95.0

    # Memory usage (percentage)
    memory_warning: float = 80.0
    memory_critical: float = 95.0

    # GPS
    gps_min_satellites: int = 6
    gps_hdop_warning: float = 2.0
    gps_hdop_critical: float = 5.0

    # IMU
    imu_temp_min: float = -20.0
    imu_temp_max: float = 60.0
    imu_vibration_warning: float = 30.0  # m/s^2
    imu_vibration_critical: float = 60.0

    # Battery
    battery_voltage_warning: float = 3.5  # per cell
    battery_voltage_critical: float = 3.3
    battery_temp_warning: float = 45.0
    battery_temp_critical: float = 60.0

    # Motors
    motor_temp_warning: float = 60.0
    motor_temp_critical: float = 80.0
    motor_current_warning: float = 20.0  # Amps
    motor_current_critical: float = 30.0

    # EKF
    ekf_variance_warning: float = 0.5
    ekf_variance_critical: float = 1.0

    # Communication
    rc_signal_warning: float = 0.3  # Signal strength 0-1
    rc_signal_critical: float = 0.1
    telemetry_latency_warning: float = 500  # ms
    telemetry_latency_critical: float = 2000


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    component_type: ComponentType
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    value: Any = None
    message: str = ""
    last_update: float = 0.0
    history: List[tuple] = field(default_factory=list)  # (timestamp, status, value)

    def update(self, status: HealthStatus, value: Any = None, message: str = ""):
        """Update component health status."""
        self.status = status
        self.value = value
        self.message = message
        self.last_update = time.time()

        # Keep history (last 100 entries)
        self.history.append((self.last_update, status, value))
        if len(self.history) > 100:
            self.history = self.history[-100:]


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    warnings: List[str]
    errors: List[str]
    timestamp: float


class HealthMonitor:
    """
    Monitors system health including CPU, memory, sensors, and motors.
    Provides real-time health status and alerts.
    """

    def __init__(self, thresholds: Optional[HealthThresholds] = None):
        self.thresholds = thresholds or HealthThresholds()
        self._components: Dict[str, ComponentHealth] = {}
        self._callbacks: List[Callable] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._update_interval = 1.0  # seconds
        self._last_system_health: Optional[SystemHealth] = None

        # Initialize default components
        self._init_default_components()

    def _init_default_components(self):
        """Initialize default monitored components."""
        defaults = [
            (ComponentType.CPU, "cpu_temp", "CPU Temperature"),
            (ComponentType.CPU, "cpu_usage", "CPU Usage"),
            (ComponentType.MEMORY, "memory", "System Memory"),
            (ComponentType.GPS, "gps", "GPS Receiver"),
            (ComponentType.IMU, "imu", "IMU Sensor"),
            (ComponentType.BATTERY, "battery", "Main Battery"),
            (ComponentType.RC_LINK, "rc_link", "RC Link"),
            (ComponentType.TELEMETRY, "telemetry", "Telemetry Link"),
            (ComponentType.EKF, "ekf", "EKF Status"),
            (ComponentType.VIBRATION, "vibration", "Vibration Level"),
        ]

        for comp_type, name, display_name in defaults:
            self._components[name] = ComponentHealth(
                component_type=comp_type,
                name=display_name,
            )

        # Add 4 motors by default
        for i in range(1, 5):
            self._components[f"motor_{i}"] = ComponentHealth(
                component_type=ComponentType.MOTOR,
                name=f"Motor {i}",
            )

    def add_component(self, key: str, component_type: ComponentType, name: str):
        """Add a custom component to monitor."""
        self._components[key] = ComponentHealth(
            component_type=component_type,
            name=name,
        )

    def add_callback(self, callback: Callable):
        """Add a callback for health status changes."""
        self._callbacks.append(callback)

    def update_component(self, component_type: ComponentType, **kwargs):
        """
        Generic method to update any component's health status.
        Routes to the appropriate specific update method based on component type.
        """
        if component_type == ComponentType.CPU:
            temp = kwargs.get('temperature', 45.0)
            usage = kwargs.get('usage', 30.0)
            self.update_cpu(temp, usage)
        elif component_type == ComponentType.MEMORY:
            usage = kwargs.get('usage', 50.0)
            self.update_memory(usage)
        elif component_type == ComponentType.GPS:
            sats = kwargs.get('satellites', 0)
            accuracy = kwargs.get('accuracy', 100.0)
            has_fix = kwargs.get('has_fix', False)
            # Convert accuracy to HDOP approximation
            hdop = accuracy / 2.5 if accuracy < 25 else 10.0
            fix_type = 3 if has_fix else 0
            self.update_gps(sats, hdop, fix_type)
        elif component_type == ComponentType.IMU:
            healthy = kwargs.get('healthy', True)
            vibration = kwargs.get('vibration', 0.0)
            self.update_imu(25.0, vibration, healthy)
        elif component_type == ComponentType.BATTERY:
            voltage = kwargs.get('voltage', 12.0)
            percentage = kwargs.get('percentage', 100.0)
            temp = kwargs.get('temperature', 25.0)
            self.update_battery(voltage, 5.0, percentage, temp, cell_count=3)
        elif component_type == ComponentType.MOTOR:
            motor_id = kwargs.get('motor_id', 1)
            healthy = kwargs.get('healthy', True)
            current = kwargs.get('current', 5.0)
            rpm = kwargs.get('rpm', 5000)
            temp = kwargs.get('temperature', 40.0) if healthy else 100.0
            if not healthy:
                rpm = 0
                current = 0
            self.update_motor(motor_id + 1, rpm, current, temp)
        elif component_type in (ComponentType.MAGNETOMETER, ComponentType.COMPASS):
            healthy = kwargs.get('healthy', True)
            calibrated = kwargs.get('calibrated', True)
            self._update_compass(healthy, calibrated)
        elif component_type == ComponentType.EKF:
            variance = kwargs.get('variance', 0.1)
            self.update_ekf(variance, variance, variance)
        elif component_type == ComponentType.VIBRATION:
            x = kwargs.get('x', 0.0)
            y = kwargs.get('y', 0.0)
            z = kwargs.get('z', 0.0)
            self.update_vibration(x, y, z)
        elif component_type == ComponentType.RC_LINK:
            signal = kwargs.get('signal_strength', 1.0)
            rssi = kwargs.get('rssi', -50)
            lost = kwargs.get('lost', False)
            self.update_rc_link(signal, rssi, lost)
        elif component_type == ComponentType.CAMERA:
            healthy = kwargs.get('healthy', True)
            streaming = kwargs.get('streaming', False)
            self._update_camera(healthy, streaming)

    def _update_compass(self, healthy: bool, calibrated: bool):
        """Update compass/magnetometer health status."""
        if 'compass' not in self._components:
            self._components['compass'] = ComponentHealth(
                component_type=ComponentType.MAGNETOMETER,
                name="Compass",
            )

        if not healthy:
            status = HealthStatus.FAILED
            msg = "Compass unhealthy"
        elif not calibrated:
            status = HealthStatus.CRITICAL
            msg = "Compass not calibrated"
        else:
            status = HealthStatus.GOOD
            msg = "Compass healthy and calibrated"

        self._components['compass'].update(status, {'healthy': healthy, 'calibrated': calibrated}, msg)

    def _update_camera(self, healthy: bool, streaming: bool):
        """Update camera health status."""
        if 'camera' not in self._components:
            self._components['camera'] = ComponentHealth(
                component_type=ComponentType.CAMERA,
                name="Camera",
            )

        if not healthy:
            status = HealthStatus.FAILED
            msg = "Camera error"
        else:
            status = HealthStatus.GOOD
            msg = "Camera streaming" if streaming else "Camera ready"

        self._components['camera'].update(status, {'healthy': healthy, 'streaming': streaming}, msg)

    def start(self):
        """Start the health monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitor started")

    def stop(self):
        """Stop the health monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("Health monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._update_system_health()
                self._check_for_alerts()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self._update_interval)

    def _update_system_health(self):
        """Update system health by collecting all component statuses."""
        # In a real implementation, these would read from actual sensors/system
        # For simulation, we'll use placeholder values

        # CPU (simulated)
        self.update_cpu(
            temperature=45.0,  # Would be read from system
            usage=25.0,
        )

        # Memory (simulated)
        self.update_memory(used_percent=50.0)

    def update_cpu(self, temperature: float, usage: float):
        """Update CPU health status."""
        # Temperature check
        if temperature >= self.thresholds.cpu_temp_critical:
            status = HealthStatus.CRITICAL
            msg = f"CPU temperature critical: {temperature:.1f}°C"
        elif temperature >= self.thresholds.cpu_temp_warning:
            status = HealthStatus.WARNING
            msg = f"CPU temperature high: {temperature:.1f}°C"
        else:
            status = HealthStatus.GOOD
            msg = f"CPU temperature normal: {temperature:.1f}°C"

        self._components["cpu_temp"].update(status, temperature, msg)

        # Usage check
        if usage >= self.thresholds.cpu_usage_critical:
            status = HealthStatus.CRITICAL
            msg = f"CPU usage critical: {usage:.1f}%"
        elif usage >= self.thresholds.cpu_usage_warning:
            status = HealthStatus.WARNING
            msg = f"CPU usage high: {usage:.1f}%"
        else:
            status = HealthStatus.GOOD
            msg = f"CPU usage normal: {usage:.1f}%"

        self._components["cpu_usage"].update(status, usage, msg)

    def update_memory(self, used_percent: float):
        """Update memory health status."""
        if used_percent >= self.thresholds.memory_critical:
            status = HealthStatus.CRITICAL
            msg = f"Memory critical: {used_percent:.1f}%"
        elif used_percent >= self.thresholds.memory_warning:
            status = HealthStatus.WARNING
            msg = f"Memory high: {used_percent:.1f}%"
        else:
            status = HealthStatus.GOOD
            msg = f"Memory normal: {used_percent:.1f}%"

        self._components["memory"].update(status, used_percent, msg)

    def update_gps(self, satellites: int, hdop: float, fix_type: int):
        """Update GPS health status."""
        if fix_type == 0 or satellites < 4:
            status = HealthStatus.FAILED
            msg = f"No GPS fix (sats: {satellites})"
        elif satellites < self.thresholds.gps_min_satellites or hdop > self.thresholds.gps_hdop_critical:
            status = HealthStatus.CRITICAL
            msg = f"GPS poor: {satellites} sats, HDOP {hdop:.1f}"
        elif hdop > self.thresholds.gps_hdop_warning:
            status = HealthStatus.WARNING
            msg = f"GPS degraded: {satellites} sats, HDOP {hdop:.1f}"
        else:
            status = HealthStatus.GOOD
            msg = f"GPS good: {satellites} sats, HDOP {hdop:.1f}"

        self._components["gps"].update(status, {'satellites': satellites, 'hdop': hdop}, msg)

    def update_imu(self, temperature: float, vibration: float, calibrated: bool):
        """Update IMU health status."""
        if not calibrated:
            status = HealthStatus.CRITICAL
            msg = "IMU not calibrated"
        elif temperature < self.thresholds.imu_temp_min or temperature > self.thresholds.imu_temp_max:
            status = HealthStatus.CRITICAL
            msg = f"IMU temperature out of range: {temperature:.1f}°C"
        elif vibration >= self.thresholds.imu_vibration_critical:
            status = HealthStatus.CRITICAL
            msg = f"IMU vibration critical: {vibration:.1f} m/s²"
        elif vibration >= self.thresholds.imu_vibration_warning:
            status = HealthStatus.WARNING
            msg = f"IMU vibration high: {vibration:.1f} m/s²"
        else:
            status = HealthStatus.GOOD
            msg = f"IMU healthy: temp {temperature:.1f}°C, vib {vibration:.1f}"

        self._components["imu"].update(status, {'temp': temperature, 'vibration': vibration}, msg)

    def update_battery(
        self,
        voltage: float,
        current: float,
        percentage: float,
        temperature: float,
        cell_count: int = 4,
    ):
        """Update battery health status."""
        voltage_per_cell = voltage / cell_count

        if voltage_per_cell <= self.thresholds.battery_voltage_critical:
            status = HealthStatus.CRITICAL
            msg = f"Battery voltage critical: {voltage:.2f}V ({voltage_per_cell:.2f}V/cell)"
        elif temperature >= self.thresholds.battery_temp_critical:
            status = HealthStatus.CRITICAL
            msg = f"Battery temperature critical: {temperature:.1f}°C"
        elif voltage_per_cell <= self.thresholds.battery_voltage_warning:
            status = HealthStatus.WARNING
            msg = f"Battery voltage low: {voltage:.2f}V ({voltage_per_cell:.2f}V/cell)"
        elif temperature >= self.thresholds.battery_temp_warning:
            status = HealthStatus.WARNING
            msg = f"Battery temperature high: {temperature:.1f}°C"
        else:
            status = HealthStatus.GOOD
            msg = f"Battery healthy: {voltage:.2f}V, {percentage:.0f}%, {temperature:.1f}°C"

        self._components["battery"].update(
            status,
            {'voltage': voltage, 'current': current, 'percentage': percentage, 'temp': temperature},
            msg
        )

    def update_motor(self, motor_id: int, rpm: int, current: float, temperature: float):
        """Update motor health status."""
        key = f"motor_{motor_id}"
        if key not in self._components:
            self.add_component(key, ComponentType.MOTOR, f"Motor {motor_id}")

        if temperature >= self.thresholds.motor_temp_critical:
            status = HealthStatus.CRITICAL
            msg = f"Motor {motor_id} temp critical: {temperature:.1f}°C"
        elif current >= self.thresholds.motor_current_critical:
            status = HealthStatus.CRITICAL
            msg = f"Motor {motor_id} current critical: {current:.1f}A"
        elif temperature >= self.thresholds.motor_temp_warning:
            status = HealthStatus.WARNING
            msg = f"Motor {motor_id} temp high: {temperature:.1f}°C"
        elif current >= self.thresholds.motor_current_warning:
            status = HealthStatus.WARNING
            msg = f"Motor {motor_id} current high: {current:.1f}A"
        elif rpm == 0:
            status = HealthStatus.WARNING
            msg = f"Motor {motor_id} not spinning"
        else:
            status = HealthStatus.GOOD
            msg = f"Motor {motor_id} healthy: {rpm} RPM, {current:.1f}A, {temperature:.1f}°C"

        self._components[key].update(status, {'rpm': rpm, 'current': current, 'temp': temperature}, msg)

    def update_ekf(self, pos_variance: float, vel_variance: float, yaw_variance: float):
        """Update EKF health status."""
        max_variance = max(pos_variance, vel_variance, yaw_variance)

        if max_variance >= self.thresholds.ekf_variance_critical:
            status = HealthStatus.CRITICAL
            msg = f"EKF diverging: variance {max_variance:.2f}"
        elif max_variance >= self.thresholds.ekf_variance_warning:
            status = HealthStatus.WARNING
            msg = f"EKF variance high: {max_variance:.2f}"
        else:
            status = HealthStatus.GOOD
            msg = f"EKF healthy: variance {max_variance:.2f}"

        self._components["ekf"].update(
            status,
            {'pos': pos_variance, 'vel': vel_variance, 'yaw': yaw_variance},
            msg
        )

    def update_vibration(self, x: float, y: float, z: float):
        """Update vibration health status."""
        magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5

        if magnitude >= self.thresholds.imu_vibration_critical:
            status = HealthStatus.CRITICAL
            msg = f"Vibration critical: {magnitude:.1f} m/s²"
        elif magnitude >= self.thresholds.imu_vibration_warning:
            status = HealthStatus.WARNING
            msg = f"Vibration high: {magnitude:.1f} m/s²"
        else:
            status = HealthStatus.GOOD
            msg = f"Vibration normal: {magnitude:.1f} m/s²"

        self._components["vibration"].update(status, {'x': x, 'y': y, 'z': z, 'mag': magnitude}, msg)

    def update_rc_link(self, signal_strength: float, rssi: int, lost: bool):
        """Update RC link health status."""
        if lost:
            status = HealthStatus.FAILED
            msg = "RC link lost"
        elif signal_strength <= self.thresholds.rc_signal_critical:
            status = HealthStatus.CRITICAL
            msg = f"RC signal critical: {signal_strength * 100:.0f}%"
        elif signal_strength <= self.thresholds.rc_signal_warning:
            status = HealthStatus.WARNING
            msg = f"RC signal weak: {signal_strength * 100:.0f}%"
        else:
            status = HealthStatus.GOOD
            msg = f"RC link good: {signal_strength * 100:.0f}%, RSSI {rssi}"

        self._components["rc_link"].update(status, {'strength': signal_strength, 'rssi': rssi}, msg)

    def update_telemetry(self, latency_ms: float, packet_loss: float, connected: bool):
        """Update telemetry link health status."""
        if not connected:
            status = HealthStatus.FAILED
            msg = "Telemetry disconnected"
        elif latency_ms >= self.thresholds.telemetry_latency_critical:
            status = HealthStatus.CRITICAL
            msg = f"Telemetry latency critical: {latency_ms:.0f}ms"
        elif latency_ms >= self.thresholds.telemetry_latency_warning:
            status = HealthStatus.WARNING
            msg = f"Telemetry latency high: {latency_ms:.0f}ms"
        else:
            status = HealthStatus.GOOD
            msg = f"Telemetry good: {latency_ms:.0f}ms, {packet_loss:.1f}% loss"

        self._components["telemetry"].update(
            status,
            {'latency': latency_ms, 'packet_loss': packet_loss},
            msg
        )

    def _check_for_alerts(self):
        """Check for health alerts and trigger callbacks."""
        for key, component in self._components.items():
            if component.status in (HealthStatus.CRITICAL, HealthStatus.FAILED):
                for callback in self._callbacks:
                    try:
                        callback({
                            'event': 'health_alert',
                            'component': key,
                            'status': component.status.value,
                            'message': component.message,
                            'value': component.value,
                        })
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

    def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        warnings = []
        errors = []
        overall_status = HealthStatus.GOOD

        for key, component in self._components.items():
            if component.status == HealthStatus.FAILED:
                errors.append(component.message)
                overall_status = HealthStatus.FAILED
            elif component.status == HealthStatus.CRITICAL:
                errors.append(component.message)
                if overall_status != HealthStatus.FAILED:
                    overall_status = HealthStatus.CRITICAL
            elif component.status == HealthStatus.WARNING:
                warnings.append(component.message)
                if overall_status == HealthStatus.GOOD:
                    overall_status = HealthStatus.WARNING

        health = SystemHealth(
            status=overall_status,
            components=dict(self._components),
            warnings=warnings,
            errors=errors,
            timestamp=time.time(),
        )

        self._last_system_health = health
        return health

    def get_component(self, key: str) -> Optional[ComponentHealth]:
        """Get a specific component's health status."""
        return self._components.get(key)

    def is_healthy(self) -> bool:
        """Check if system is healthy enough to fly."""
        health = self.get_system_health()
        return health.status in (HealthStatus.GOOD, HealthStatus.WARNING)

    def is_critical(self) -> bool:
        """Check if any component is in critical state."""
        return any(c.status in (HealthStatus.CRITICAL, HealthStatus.FAILED)
                   for c in self._components.values())

    def get_component_status(self, component_type: ComponentType) -> HealthStatus:
        """Get the health status for a specific component type."""
        # Map component types to their keys
        type_to_key = {
            ComponentType.CPU: ["cpu_temp", "cpu_usage"],
            ComponentType.MEMORY: ["memory"],
            ComponentType.GPS: ["gps"],
            ComponentType.IMU: ["imu"],
            ComponentType.BATTERY: ["battery"],
            ComponentType.MOTOR: ["motor_1", "motor_2", "motor_3", "motor_4"],
            ComponentType.MAGNETOMETER: ["compass"],
            ComponentType.COMPASS: ["compass"],
            ComponentType.EKF: ["ekf"],
            ComponentType.VIBRATION: ["vibration"],
            ComponentType.RC_LINK: ["rc_link"],
            ComponentType.TELEMETRY: ["telemetry"],
            ComponentType.CAMERA: ["camera"],
        }

        keys = type_to_key.get(component_type, [])
        worst_status = HealthStatus.UNKNOWN

        status_priority = {
            HealthStatus.FAILED: 4,
            HealthStatus.CRITICAL: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.GOOD: 1,
            HealthStatus.UNKNOWN: 0,
        }

        for key in keys:
            if key in self._components:
                comp_status = self._components[key].status
                if status_priority.get(comp_status, 0) > status_priority.get(worst_status, 0):
                    worst_status = comp_status

        return worst_status

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        health = self.get_system_health()
        return health.status

    def get_all_component_status(self) -> Dict[str, HealthStatus]:
        """Get status for all components."""
        return {key: comp.status for key, comp in self._components.items()}

    def get_critical_issues(self) -> List[str]:
        """Get list of critical issues."""
        issues = []
        for key, comp in self._components.items():
            if comp.status in (HealthStatus.CRITICAL, HealthStatus.FAILED):
                issues.append(comp.message)
        return issues

    def get_warnings(self) -> List[str]:
        """Get list of warnings."""
        warnings = []
        for key, comp in self._components.items():
            if comp.status == HealthStatus.WARNING:
                warnings.append(comp.message)
        return warnings

    def to_dict(self) -> dict:
        """Serialize health monitor state."""
        return {
            'components': {
                key: {
                    'name': c.name,
                    'type': c.component_type.name,
                    'status': c.status.value,
                    'value': c.value,
                    'message': c.message,
                    'last_update': c.last_update,
                }
                for key, c in self._components.items()
            },
            'system_status': self._last_system_health.status.value if self._last_system_health else 'unknown',
            'warnings': self._last_system_health.warnings if self._last_system_health else [],
            'errors': self._last_system_health.errors if self._last_system_health else [],
        }
