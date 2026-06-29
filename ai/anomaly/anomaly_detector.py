"""
Anomaly detection for predictive maintenance.

Monitors sensor data to detect abnormal patterns
that may indicate component degradation or failure.
"""

import time
import math
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque, Tuple, Callable
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies."""
    MOTOR_VIBRATION = "motor_vibration"
    MOTOR_TEMPERATURE = "motor_temperature"
    MOTOR_CURRENT = "motor_current"
    BATTERY_DEGRADATION = "battery_degradation"
    BATTERY_TEMPERATURE = "battery_temperature"
    IMU_DRIFT = "imu_drift"
    GPS_ANOMALY = "gps_anomaly"
    SENSOR_NOISE = "sensor_noise"
    COMMUNICATION_LOSS = "communication_loss"
    FLIGHT_PATTERN = "flight_pattern"
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: int
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    component: str
    description: str
    confidence: float
    value: float
    expected_range: Tuple[float, float]
    deviation: float  # Standard deviations from mean
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.anomaly_id,
            'type': self.anomaly_type.value,
            'severity': self.severity.value,
            'component': self.component,
            'description': self.description,
            'confidence': self.confidence,
            'value': self.value,
            'expected_min': self.expected_range[0],
            'expected_max': self.expected_range[1],
            'deviation': self.deviation,
            'timestamp': self.timestamp,
        }


@dataclass
class SensorBaseline:
    """Baseline statistics for a sensor."""
    sensor_name: str
    mean: float = 0.0
    std: float = 1.0
    min_seen: float = float('inf')
    max_seen: float = float('-inf')
    sample_count: int = 0
    last_update: float = 0.0


@dataclass
class DetectorConfig:
    """Configuration for anomaly detector."""
    window_size: int = 100  # Samples for baseline
    warmup_samples: int = 20  # Min samples before detection
    warning_threshold: float = 2.0  # Std devs
    critical_threshold: float = 3.5  # Std devs
    emergency_threshold: float = 5.0  # Std devs
    baseline_update_rate: float = 0.01  # Learning rate
    min_confidence: float = 0.6


class AnomalyDetector:
    """
    Anomaly detector for drone health monitoring.

    Uses statistical methods to detect anomalies in:
    - Motor performance (vibration, temp, current)
    - Battery health
    - Sensor readings
    - Flight patterns
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize anomaly detector.

        Args:
            config: Detector configuration
        """
        self._config = config or DetectorConfig()
        self._baselines: Dict[str, SensorBaseline] = {}
        self._history: Dict[str, Deque[float]] = {}
        self._anomalies: List[Anomaly] = []
        self._next_id = 1
        self._callbacks: List[Callable[["Anomaly"], None]] = []

        # Define sensor mappings
        self._sensor_types = {
            'motor_vibration': AnomalyType.MOTOR_VIBRATION,
            'motor_temp': AnomalyType.MOTOR_TEMPERATURE,
            'motor_current': AnomalyType.MOTOR_CURRENT,
            'battery_voltage': AnomalyType.BATTERY_DEGRADATION,
            'battery_temp': AnomalyType.BATTERY_TEMPERATURE,
            'imu_accel': AnomalyType.IMU_DRIFT,
            'imu_gyro': AnomalyType.IMU_DRIFT,
            'gps_hdop': AnomalyType.GPS_ANOMALY,
        }

    def update(
        self,
        sensor_name: str,
        value: float,
        component: str = "",
    ) -> Optional[Anomaly]:
        """
        Update with new sensor reading and check for anomaly.

        Args:
            sensor_name: Name of sensor
            value: Sensor value
            component: Component name (e.g., "motor_1")

        Returns:
            Anomaly if detected, None otherwise
        """
        # Initialize baseline if needed
        if sensor_name not in self._baselines:
            self._baselines[sensor_name] = SensorBaseline(
                sensor_name=sensor_name
            )
            self._history[sensor_name] = deque(
                maxlen=self._config.window_size
            )

        baseline = self._baselines[sensor_name]
        history = self._history[sensor_name]

        # Add to history
        history.append(value)
        baseline.sample_count += 1
        baseline.min_seen = min(baseline.min_seen, value)
        baseline.max_seen = max(baseline.max_seen, value)

        # Update baseline statistics
        if len(history) >= self._config.warmup_samples:
            self._update_baseline(baseline, history)

            # Check for anomaly
            anomaly = self._check_anomaly(
                sensor_name,
                value,
                baseline,
                component,
            )

            if anomaly:
                self._anomalies.append(anomaly)
                for cb in list(self._callbacks):
                    try:
                        cb(anomaly)
                    except Exception:
                        pass
                return anomaly

        return None

    def add_callback(self, callback: "Callable[[Anomaly], None]") -> None:
        """Register a callback invoked once per detected anomaly."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: "Callable[[Anomaly], None]") -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def _update_baseline(
        self,
        baseline: SensorBaseline,
        history: Deque[float],
    ) -> None:
        """Update baseline statistics with new data."""
        values = list(history)

        # Calculate statistics
        new_mean = statistics.mean(values)
        new_std = statistics.stdev(values) if len(values) > 1 else 1.0

        # Exponential moving average update
        rate = self._config.baseline_update_rate
        baseline.mean = (1 - rate) * baseline.mean + rate * new_mean
        baseline.std = (1 - rate) * baseline.std + rate * new_std
        baseline.last_update = time.time()

        # Ensure minimum std to avoid division by zero
        baseline.std = max(baseline.std, 0.001)

    def _check_anomaly(
        self,
        sensor_name: str,
        value: float,
        baseline: SensorBaseline,
        component: str,
    ) -> Optional[Anomaly]:
        """Check if value is anomalous."""
        # Calculate deviation
        deviation = abs(value - baseline.mean) / baseline.std

        # Determine severity
        cfg = self._config
        if deviation < cfg.warning_threshold:
            return None

        if deviation >= cfg.emergency_threshold:
            severity = AnomalySeverity.EMERGENCY
        elif deviation >= cfg.critical_threshold:
            severity = AnomalySeverity.CRITICAL
        else:
            severity = AnomalySeverity.WARNING

        # Calculate confidence based on sample count
        confidence = min(
            1.0,
            baseline.sample_count / (cfg.warmup_samples * 2)
        )

        if confidence < cfg.min_confidence:
            return None

        # Get anomaly type
        anomaly_type = self._get_anomaly_type(sensor_name)

        # Expected range (2 std devs)
        expected_range = (
            baseline.mean - 2 * baseline.std,
            baseline.mean + 2 * baseline.std,
        )

        # Create description
        direction = "high" if value > baseline.mean else "low"
        description = (
            f"{sensor_name} value {direction}: "
            f"{value:.2f} (expected {baseline.mean:.2f})"
        )

        anomaly = Anomaly(
            anomaly_id=self._next_id,
            anomaly_type=anomaly_type,
            severity=severity,
            component=component or sensor_name,
            description=description,
            confidence=confidence,
            value=value,
            expected_range=expected_range,
            deviation=deviation,
        )

        self._next_id += 1
        logger.warning(f"Anomaly detected: {description}")

        return anomaly

    def _get_anomaly_type(self, sensor_name: str) -> AnomalyType:
        """Get anomaly type for sensor."""
        for key, atype in self._sensor_types.items():
            if key in sensor_name.lower():
                return atype
        return AnomalyType.UNKNOWN

    def update_batch(
        self,
        readings: Dict[str, float],
        component: str = "",
    ) -> List[Anomaly]:
        """
        Update with multiple sensor readings.

        Args:
            readings: Dict of sensor_name -> value
            component: Component name

        Returns:
            List of detected anomalies
        """
        anomalies = []
        for sensor_name, value in readings.items():
            anomaly = self.update(sensor_name, value, component)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def check_motor_health(
        self,
        motor_id: int,
        rpm: float,
        current: float,
        temperature: float,
        vibration: float,
    ) -> List[Anomaly]:
        """
        Check motor health with multiple metrics.

        Args:
            motor_id: Motor identifier
            rpm: Current RPM
            current: Current draw (amps)
            temperature: Temperature (celsius)
            vibration: Vibration level

        Returns:
            List of detected anomalies
        """
        component = f"motor_{motor_id}"
        readings = {
            f"{component}_rpm": rpm,
            f"{component}_current": current,
            f"{component}_temp": temperature,
            f"{component}_vibration": vibration,
        }
        return self.update_batch(readings, component)

    def check_battery_health(
        self,
        voltage: float,
        current: float,
        temperature: float,
        capacity_remaining: float,
    ) -> List[Anomaly]:
        """
        Check battery health.

        Args:
            voltage: Battery voltage
            current: Current draw
            temperature: Battery temperature
            capacity_remaining: Remaining capacity percent

        Returns:
            List of detected anomalies
        """
        readings = {
            'battery_voltage': voltage,
            'battery_current': current,
            'battery_temp': temperature,
            'battery_capacity': capacity_remaining,
        }
        return self.update_batch(readings, "battery")

    def check_imu_health(
        self,
        accel_magnitude: float,
        gyro_magnitude: float,
        mag_magnitude: float,
    ) -> List[Anomaly]:
        """
        Check IMU sensor health.

        Args:
            accel_magnitude: Accelerometer magnitude
            gyro_magnitude: Gyroscope magnitude
            mag_magnitude: Magnetometer magnitude

        Returns:
            List of detected anomalies
        """
        readings = {
            'imu_accel_mag': accel_magnitude,
            'imu_gyro_mag': gyro_magnitude,
            'imu_mag_mag': mag_magnitude,
        }
        return self.update_batch(readings, "imu")

    def get_recent_anomalies(
        self,
        max_age: float = 300.0,
    ) -> List[Anomaly]:
        """Get anomalies from last N seconds."""
        cutoff = time.time() - max_age
        return [
            a for a in self._anomalies
            if a.timestamp > cutoff
        ]

    def get_anomalies_by_severity(
        self,
        severity: AnomalySeverity,
    ) -> List[Anomaly]:
        """Get anomalies of specific severity."""
        return [
            a for a in self._anomalies
            if a.severity == severity
        ]

    def get_anomalies_by_component(
        self,
        component: str,
    ) -> List[Anomaly]:
        """Get anomalies for specific component."""
        return [
            a for a in self._anomalies
            if component in a.component
        ]

    def get_health_score(self, component: str = "") -> float:
        """
        Get overall health score (0-1).

        Args:
            component: Filter by component (empty = all)

        Returns:
            Health score where 1.0 is healthy
        """
        recent = self.get_recent_anomalies(60.0)

        if component:
            recent = [
                a for a in recent
                if component in a.component
            ]

        if not recent:
            return 1.0

        # Weight by severity
        severity_weights = {
            AnomalySeverity.INFO: 0.05,
            AnomalySeverity.WARNING: 0.15,
            AnomalySeverity.CRITICAL: 0.35,
            AnomalySeverity.EMERGENCY: 0.6,
        }

        total_penalty = sum(
            severity_weights.get(a.severity, 0.1) * a.confidence
            for a in recent
        )

        return max(0.0, 1.0 - min(total_penalty, 1.0))

    def get_baseline(
        self,
        sensor_name: str,
    ) -> Optional[SensorBaseline]:
        """Get baseline for a sensor."""
        return self._baselines.get(sensor_name)

    def reset_baseline(self, sensor_name: str) -> bool:
        """Reset baseline for a sensor."""
        if sensor_name in self._baselines:
            del self._baselines[sensor_name]
            if sensor_name in self._history:
                self._history[sensor_name].clear()
            return True
        return False

    def reset_all(self) -> None:
        """Reset all baselines and history."""
        self._baselines.clear()
        self._history.clear()
        self._anomalies.clear()

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'total_anomalies': len(self._anomalies),
            'sensors_monitored': len(self._baselines),
            'recent_anomalies': len(self.get_recent_anomalies()),
            'health_score': self.get_health_score(),
        }
