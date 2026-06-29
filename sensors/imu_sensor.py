"""
IMU (Inertial Measurement Unit) sensor module for Drone AI application.

Provides accelerometer, gyroscope, and magnetometer data with realistic
simulation of sensor characteristics including noise, bias, and drift.

This is the foundation for:
- GPS + IMU fusion (primary outdoor navigation)
- Dead reckoning (fallback when GPS is lost)
- Attitude estimation (roll, pitch, yaw)
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
from .sensor import Sensor


class IMUCalibrationState(Enum):
    """Calibration state for IMU components."""
    UNCALIBRATED = 0
    CALIBRATING = 1
    CALIBRATED = 2
    FAILED = 3


class IMUHealthStatus(Enum):
    """Health status of IMU sensor."""
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class Vector3:
    """3D vector for sensor measurements."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Vector3':
        """Return normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class EulerAngles:
    """Euler angles for attitude representation."""
    roll: float = 0.0   # Rotation around X-axis (degrees)
    pitch: float = 0.0  # Rotation around Y-axis (degrees)
    yaw: float = 0.0    # Rotation around Z-axis (degrees)

    def to_radians(self) -> 'EulerAngles':
        """Convert to radians."""
        return EulerAngles(
            math.radians(self.roll),
            math.radians(self.pitch),
            math.radians(self.yaw)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw}


@dataclass
class IMUReading:
    """Complete IMU reading with all sensor data."""
    # Accelerometer (m/s²)
    accelerometer: Vector3
    # Gyroscope (deg/s)
    gyroscope: Vector3
    # Magnetometer (µT - microtesla)
    magnetometer: Vector3
    # Computed attitude (degrees)
    attitude: EulerAngles
    # Temperature (°C)
    temperature: float
    # Timestamp
    timestamp: float
    # Validity flags
    accel_valid: bool = True
    gyro_valid: bool = True
    mag_valid: bool = True

    def is_valid(self) -> bool:
        """Check if IMU reading is valid."""
        return self.accel_valid and self.gyro_valid and self.mag_valid

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'accelerometer': self.accelerometer.to_dict(),
            'gyroscope': self.gyroscope.to_dict(),
            'magnetometer': self.magnetometer.to_dict(),
            'attitude': self.attitude.to_dict(),
            'temperature': self.temperature,
            'timestamp': self.timestamp,
            'accel_valid': self.accel_valid,
            'gyro_valid': self.gyro_valid,
            'mag_valid': self.mag_valid,
            'is_valid': self.is_valid()
        }


@dataclass
class IMUCalibration:
    """Calibration data for IMU sensors."""
    # Accelerometer calibration
    accel_offset: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    accel_scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    accel_state: IMUCalibrationState = IMUCalibrationState.UNCALIBRATED

    # Gyroscope calibration
    gyro_offset: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    gyro_state: IMUCalibrationState = IMUCalibrationState.UNCALIBRATED

    # Magnetometer calibration (hard/soft iron)
    mag_offset: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    mag_scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    mag_state: IMUCalibrationState = IMUCalibrationState.UNCALIBRATED

    def is_fully_calibrated(self) -> bool:
        """Check if all sensors are calibrated."""
        return (
            self.accel_state == IMUCalibrationState.CALIBRATED and
            self.gyro_state == IMUCalibrationState.CALIBRATED and
            self.mag_state == IMUCalibrationState.CALIBRATED
        )


class IMUSensor(Sensor):
    """
    Inertial Measurement Unit sensor combining accelerometer, gyroscope, and magnetometer.

    Features:
    - 3-axis accelerometer (±16g range, typical)
    - 3-axis gyroscope (±2000°/s range, typical)
    - 3-axis magnetometer (±4800µT range, typical)
    - Temperature sensor
    - Realistic noise simulation
    - Bias and drift modeling
    - Calibration support

    Coordinate System (NED - North-East-Down):
    - X: Forward (North)
    - Y: Right (East)
    - Z: Down

    At rest on level surface:
    - Accelerometer: (0, 0, 9.81) m/s²
    - Gyroscope: (0, 0, 0) deg/s
    """

    # Sensor specifications (based on typical MEMS IMU like MPU6050/ICM20948)
    ACCEL_RANGE = 16.0  # ±16g
    ACCEL_RESOLUTION = 0.0005  # g/LSB at ±16g
    ACCEL_NOISE_DENSITY = 0.003  # g/√Hz

    GYRO_RANGE = 2000.0  # ±2000 deg/s
    GYRO_RESOLUTION = 0.06  # deg/s/LSB
    GYRO_NOISE_DENSITY = 0.01  # deg/s/√Hz
    GYRO_BIAS_STABILITY = 5.0  # deg/hr

    MAG_RANGE = 4800.0  # ±4800 µT
    MAG_RESOLUTION = 0.15  # µT/LSB
    MAG_NOISE_DENSITY = 0.1  # µT/√Hz

    # Update rate
    DEFAULT_UPDATE_RATE = 100  # Hz

    # Earth's gravity
    GRAVITY = 9.80665  # m/s²

    # Earth's magnetic field (approximate, varies by location)
    # Default: ~50µT total, mostly horizontal in mid-latitudes
    DEFAULT_MAG_FIELD = Vector3(20.0, 0.0, 45.0)  # µT (North, East, Down)

    def __init__(self, update_rate: int = DEFAULT_UPDATE_RATE):
        """
        Initialize IMU sensor.

        Args:
            update_rate: Sensor update rate in Hz (default 100Hz)
        """
        self.type = "imu"
        self._update_rate = update_rate
        self._update_interval = 1.0 / update_rate

        # Raw sensor values (before calibration)
        self._raw_accel = Vector3(0, 0, self.GRAVITY)
        self._raw_gyro = Vector3(0, 0, 0)
        self._raw_mag = Vector3(20.0, 0.0, 45.0)

        # Calibrated sensor values
        self._accel = Vector3(0, 0, self.GRAVITY)
        self._gyro = Vector3(0, 0, 0)
        self._mag = Vector3(20.0, 0.0, 45.0)

        # Attitude (computed from sensor fusion)
        self._attitude = EulerAngles(0, 0, 0)

        # Temperature
        self._temperature = 25.0  # °C

        # Sensor biases (simulated manufacturing imperfections)
        self._accel_bias = Vector3(
            random.gauss(0, 0.02),  # ~20mg bias
            random.gauss(0, 0.02),
            random.gauss(0, 0.02)
        )
        self._gyro_bias = Vector3(
            random.gauss(0, 0.5),  # ~0.5 deg/s bias
            random.gauss(0, 0.5),
            random.gauss(0, 0.5)
        )
        self._mag_bias = Vector3(
            random.gauss(0, 5.0),  # ~5µT hard iron
            random.gauss(0, 5.0),
            random.gauss(0, 5.0)
        )

        # Gyro drift rate (deg/s per second)
        self._gyro_drift_rate = Vector3(
            random.gauss(0, 0.001),
            random.gauss(0, 0.001),
            random.gauss(0, 0.001)
        )

        # Calibration data
        self._calibration = IMUCalibration()

        # State
        self._is_connected = False
        self._last_update_time = 0.0
        self._last_reading: Optional[IMUReading] = None

        # Simulation state
        self._true_attitude = EulerAngles(0, 0, 0)  # True attitude for simulation
        self._angular_velocity = Vector3(0, 0, 0)  # True angular velocity
        self._linear_acceleration = Vector3(0, 0, 0)  # True acceleration (body frame)

        # Health monitoring
        self._consecutive_errors = 0
        self._health_status = IMUHealthStatus.GOOD

        # Vibration level (0-1, affects noise)
        self._vibration_level = 0.0

    def start(self):
        """Initialize and start IMU sensor."""
        self._is_connected = True
        self._last_update_time = time.time()
        self._consecutive_errors = 0
        self._health_status = IMUHealthStatus.GOOD

    def stop(self):
        """Stop IMU sensor."""
        self._is_connected = False

    def update(self) -> IMUReading:
        """
        Fetch latest IMU measurement.

        Returns:
            IMUReading with accelerometer, gyroscope, magnetometer data
        """
        if not self._is_connected:
            return self._create_invalid_reading()

        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else self._update_interval

        # Simulate sensor readings
        self._simulate_sensors(dt)

        # Apply calibration
        self._apply_calibration()

        # Compute attitude from sensors
        self._compute_attitude(dt)

        # Create reading
        reading = IMUReading(
            accelerometer=Vector3(self._accel.x, self._accel.y, self._accel.z),
            gyroscope=Vector3(self._gyro.x, self._gyro.y, self._gyro.z),
            magnetometer=Vector3(self._mag.x, self._mag.y, self._mag.z),
            attitude=EulerAngles(self._attitude.roll, self._attitude.pitch, self._attitude.yaw),
            temperature=self._temperature,
            timestamp=current_time,
            accel_valid=True,
            gyro_valid=True,
            mag_valid=True
        )

        self._last_reading = reading
        self._last_update_time = current_time
        return reading

    def measure(self) -> IMUReading:
        """Alias for update() to match sensor interface."""
        return self.update()

    def _simulate_sensors(self, dt: float):
        """
        Simulate realistic sensor behavior.

        Args:
            dt: Time delta since last update
        """
        # Update gyro drift over time
        self._gyro_bias = Vector3(
            self._gyro_bias.x + self._gyro_drift_rate.x * dt,
            self._gyro_bias.y + self._gyro_drift_rate.y * dt,
            self._gyro_bias.z + self._gyro_drift_rate.z * dt
        )

        # Noise scaling based on vibration
        noise_scale = 1.0 + self._vibration_level * 5.0

        # Simulate accelerometer
        # At rest: measures gravity in body frame
        # When moving: measures acceleration + gravity
        true_accel = self._linear_acceleration
        gravity_body = self._rotate_to_body_frame(Vector3(0, 0, self.GRAVITY))

        self._raw_accel = Vector3(
            true_accel.x + gravity_body.x + self._accel_bias.x * self.GRAVITY +
            random.gauss(0, self.ACCEL_NOISE_DENSITY * noise_scale),
            true_accel.y + gravity_body.y + self._accel_bias.y * self.GRAVITY +
            random.gauss(0, self.ACCEL_NOISE_DENSITY * noise_scale),
            true_accel.z + gravity_body.z + self._accel_bias.z * self.GRAVITY +
            random.gauss(0, self.ACCEL_NOISE_DENSITY * noise_scale)
        )

        # Simulate gyroscope
        self._raw_gyro = Vector3(
            self._angular_velocity.x + self._gyro_bias.x +
            random.gauss(0, self.GYRO_NOISE_DENSITY * noise_scale),
            self._angular_velocity.y + self._gyro_bias.y +
            random.gauss(0, self.GYRO_NOISE_DENSITY * noise_scale),
            self._angular_velocity.z + self._gyro_bias.z +
            random.gauss(0, self.GYRO_NOISE_DENSITY * noise_scale)
        )

        # Simulate magnetometer
        # Rotate earth's magnetic field to body frame
        mag_body = self._rotate_to_body_frame(self.DEFAULT_MAG_FIELD)

        self._raw_mag = Vector3(
            mag_body.x + self._mag_bias.x +
            random.gauss(0, self.MAG_NOISE_DENSITY * noise_scale),
            mag_body.y + self._mag_bias.y +
            random.gauss(0, self.MAG_NOISE_DENSITY * noise_scale),
            mag_body.z + self._mag_bias.z +
            random.gauss(0, self.MAG_NOISE_DENSITY * noise_scale)
        )

        # Simulate temperature variation
        self._temperature += random.gauss(0, 0.01)
        self._temperature = max(min(self._temperature, 85.0), -40.0)

    def _rotate_to_body_frame(self, vec: Vector3) -> Vector3:
        """
        Rotate a vector from world frame to body frame using current attitude.

        Args:
            vec: Vector in world (NED) frame

        Returns:
            Vector in body frame
        """
        # Convert attitude to radians
        roll = math.radians(self._true_attitude.roll)
        pitch = math.radians(self._true_attitude.pitch)
        yaw = math.radians(self._true_attitude.yaw)

        # Rotation matrix (ZYX Euler)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Apply rotation
        x = (cy*cp)*vec.x + (sy*cp)*vec.y + (-sp)*vec.z
        y = (cy*sp*sr - sy*cr)*vec.x + (sy*sp*sr + cy*cr)*vec.y + (cp*sr)*vec.z
        z = (cy*sp*cr + sy*sr)*vec.x + (sy*sp*cr - cy*sr)*vec.y + (cp*cr)*vec.z

        return Vector3(x, y, z)

    def _apply_calibration(self):
        """Apply calibration corrections to raw sensor data."""
        cal = self._calibration

        # Apply accelerometer calibration
        if cal.accel_state == IMUCalibrationState.CALIBRATED:
            self._accel = Vector3(
                (self._raw_accel.x - cal.accel_offset.x) * cal.accel_scale.x,
                (self._raw_accel.y - cal.accel_offset.y) * cal.accel_scale.y,
                (self._raw_accel.z - cal.accel_offset.z) * cal.accel_scale.z
            )
        else:
            self._accel = self._raw_accel

        # Apply gyroscope calibration
        if cal.gyro_state == IMUCalibrationState.CALIBRATED:
            self._gyro = Vector3(
                self._raw_gyro.x - cal.gyro_offset.x,
                self._raw_gyro.y - cal.gyro_offset.y,
                self._raw_gyro.z - cal.gyro_offset.z
            )
        else:
            self._gyro = self._raw_gyro

        # Apply magnetometer calibration
        if cal.mag_state == IMUCalibrationState.CALIBRATED:
            self._mag = Vector3(
                (self._raw_mag.x - cal.mag_offset.x) * cal.mag_scale.x,
                (self._raw_mag.y - cal.mag_offset.y) * cal.mag_scale.y,
                (self._raw_mag.z - cal.mag_offset.z) * cal.mag_scale.z
            )
        else:
            self._mag = self._raw_mag

    def _compute_attitude(self, dt: float):
        """
        Compute attitude from sensor data using complementary filter.

        Args:
            dt: Time delta
        """
        # Compute roll and pitch from accelerometer
        accel_roll = math.degrees(math.atan2(self._accel.y, self._accel.z))
        accel_pitch = math.degrees(math.atan2(
            -self._accel.x,
            math.sqrt(self._accel.y**2 + self._accel.z**2)
        ))

        # Compute yaw from magnetometer (tilt-compensated)
        roll_rad = math.radians(self._attitude.roll)
        pitch_rad = math.radians(self._attitude.pitch)

        mag_x = (self._mag.x * math.cos(pitch_rad) +
                 self._mag.z * math.sin(pitch_rad))
        mag_y = (self._mag.x * math.sin(roll_rad) * math.sin(pitch_rad) +
                 self._mag.y * math.cos(roll_rad) -
                 self._mag.z * math.sin(roll_rad) * math.cos(pitch_rad))

        mag_yaw = math.degrees(math.atan2(-mag_y, mag_x))

        # Complementary filter: trust gyro for short-term, accel/mag for long-term
        alpha = 0.98  # Filter coefficient (higher = trust gyro more)

        # Integrate gyroscope
        gyro_roll = self._attitude.roll + self._gyro.x * dt
        gyro_pitch = self._attitude.pitch + self._gyro.y * dt
        gyro_yaw = self._attitude.yaw + self._gyro.z * dt

        # Apply complementary filter
        self._attitude = EulerAngles(
            alpha * gyro_roll + (1 - alpha) * accel_roll,
            alpha * gyro_pitch + (1 - alpha) * accel_pitch,
            alpha * gyro_yaw + (1 - alpha) * mag_yaw
        )

        # Normalize yaw to 0-360
        self._attitude.yaw = self._attitude.yaw % 360

    def _create_invalid_reading(self) -> IMUReading:
        """Create an invalid IMU reading for error states."""
        return IMUReading(
            accelerometer=Vector3(float('nan'), float('nan'), float('nan')),
            gyroscope=Vector3(float('nan'), float('nan'), float('nan')),
            magnetometer=Vector3(float('nan'), float('nan'), float('nan')),
            attitude=EulerAngles(float('nan'), float('nan'), float('nan')),
            temperature=float('nan'),
            timestamp=time.time(),
            accel_valid=False,
            gyro_valid=False,
            mag_valid=False
        )

    # ==================== Calibration Methods ====================

    def calibrate_gyroscope(self, samples: int = 100) -> bool:
        """
        Calibrate gyroscope by measuring bias at rest.

        Args:
            samples: Number of samples to average

        Returns:
            True if calibration successful
        """
        if not self._is_connected:
            return False

        self._calibration.gyro_state = IMUCalibrationState.CALIBRATING

        # Collect samples
        sum_gyro = Vector3(0, 0, 0)
        for _ in range(samples):
            sum_gyro = Vector3(
                sum_gyro.x + self._raw_gyro.x,
                sum_gyro.y + self._raw_gyro.y,
                sum_gyro.z + self._raw_gyro.z
            )
            time.sleep(1.0 / self._update_rate)

        # Compute average bias
        self._calibration.gyro_offset = Vector3(
            sum_gyro.x / samples,
            sum_gyro.y / samples,
            sum_gyro.z / samples
        )

        self._calibration.gyro_state = IMUCalibrationState.CALIBRATED
        return True

    def calibrate_accelerometer(self, samples: int = 100) -> bool:
        """
        Calibrate accelerometer assuming level surface.

        Args:
            samples: Number of samples to average

        Returns:
            True if calibration successful
        """
        if not self._is_connected:
            return False

        self._calibration.accel_state = IMUCalibrationState.CALIBRATING

        # Collect samples
        sum_accel = Vector3(0, 0, 0)
        for _ in range(samples):
            sum_accel = Vector3(
                sum_accel.x + self._raw_accel.x,
                sum_accel.y + self._raw_accel.y,
                sum_accel.z + self._raw_accel.z
            )
            time.sleep(1.0 / self._update_rate)

        # Compute average
        avg_accel = Vector3(
            sum_accel.x / samples,
            sum_accel.y / samples,
            sum_accel.z / samples
        )

        # At rest and level, should read (0, 0, 9.81)
        self._calibration.accel_offset = Vector3(
            avg_accel.x - 0,
            avg_accel.y - 0,
            avg_accel.z - self.GRAVITY
        )
        self._calibration.accel_scale = Vector3(1, 1, 1)

        self._calibration.accel_state = IMUCalibrationState.CALIBRATED
        return True

    def calibrate_magnetometer_simple(self, samples: int = 100) -> bool:
        """
        Simple magnetometer calibration (hard iron only).

        Args:
            samples: Number of samples

        Returns:
            True if calibration successful
        """
        if not self._is_connected:
            return False

        self._calibration.mag_state = IMUCalibrationState.CALIBRATING

        # For simple calibration, just measure current offset
        sum_mag = Vector3(0, 0, 0)
        for _ in range(samples):
            sum_mag = Vector3(
                sum_mag.x + self._raw_mag.x,
                sum_mag.y + self._raw_mag.y,
                sum_mag.z + self._raw_mag.z
            )
            time.sleep(1.0 / self._update_rate)

        avg_mag = Vector3(
            sum_mag.x / samples,
            sum_mag.y / samples,
            sum_mag.z / samples
        )

        # Estimate hard iron offset
        expected_magnitude = self.DEFAULT_MAG_FIELD.magnitude()
        current_magnitude = avg_mag.magnitude()

        if current_magnitude > 0:
            scale = expected_magnitude / current_magnitude
            self._calibration.mag_scale = Vector3(scale, scale, scale)

        self._calibration.mag_state = IMUCalibrationState.CALIBRATED
        return True

    def set_calibration_complete(self):
        """Mark all calibrations as complete (for testing/simulation)."""
        self._calibration.accel_state = IMUCalibrationState.CALIBRATED
        self._calibration.gyro_state = IMUCalibrationState.CALIBRATED
        self._calibration.mag_state = IMUCalibrationState.CALIBRATED

    # ==================== Getters ====================

    def get_accelerometer(self) -> Vector3:
        """Get calibrated accelerometer reading (m/s²)."""
        return self._accel

    def get_gyroscope(self) -> Vector3:
        """Get calibrated gyroscope reading (deg/s)."""
        return self._gyro

    def get_magnetometer(self) -> Vector3:
        """Get calibrated magnetometer reading (µT)."""
        return self._mag

    def get_attitude(self) -> EulerAngles:
        """Get computed attitude (roll, pitch, yaw in degrees)."""
        return self._attitude

    def get_temperature(self) -> float:
        """Get IMU temperature (°C)."""
        return self._temperature

    def get_calibration(self) -> IMUCalibration:
        """Get calibration data."""
        return self._calibration

    def get_health_status(self) -> IMUHealthStatus:
        """Get IMU health status."""
        return self._health_status

    def is_calibrated(self) -> bool:
        """Check if IMU is fully calibrated."""
        return self._calibration.is_fully_calibrated()

    def is_gyro_calibrated(self) -> bool:
        """Check if gyroscope is calibrated."""
        return self._calibration.gyro_state == IMUCalibrationState.CALIBRATED

    def is_accel_calibrated(self) -> bool:
        """Check if accelerometer is calibrated."""
        return self._calibration.accel_state == IMUCalibrationState.CALIBRATED

    def is_mag_calibrated(self) -> bool:
        """Check if magnetometer is calibrated."""
        return self._calibration.mag_state == IMUCalibrationState.CALIBRATED

    def is_valid(self) -> bool:
        """Check if last reading is valid."""
        if self._last_reading is None:
            return False
        return self._last_reading.is_valid()

    # ==================== Setters for Simulation ====================

    def set_true_attitude(self, roll: float, pitch: float, yaw: float):
        """Set true attitude for simulation (degrees)."""
        self._true_attitude = EulerAngles(roll, pitch, yaw)

    def set_angular_velocity(self, roll_rate: float, pitch_rate: float, yaw_rate: float):
        """Set angular velocity for simulation (deg/s)."""
        self._angular_velocity = Vector3(roll_rate, pitch_rate, yaw_rate)

    def set_linear_acceleration(self, ax: float, ay: float, az: float):
        """Set linear acceleration for simulation (m/s²)."""
        self._linear_acceleration = Vector3(ax, ay, az)

    def set_vibration_level(self, level: float):
        """Set vibration level (0.0 to 1.0)."""
        self._vibration_level = max(0.0, min(1.0, level))

    def set_temperature(self, temp: float):
        """Set IMU temperature for simulation (°C)."""
        self._temperature = max(-40.0, min(85.0, temp))

    def simulate_failure(self, component: str = 'all'):
        """
        Simulate sensor failure.

        Args:
            component: 'accel', 'gyro', 'mag', or 'all'
        """
        if component in ('accel', 'all'):
            self._accel_bias = Vector3(100, 100, 100)  # Huge bias
        if component in ('gyro', 'all'):
            self._gyro_bias = Vector3(1000, 1000, 1000)
        if component in ('mag', 'all'):
            self._mag_bias = Vector3(10000, 10000, 10000)
        self._health_status = IMUHealthStatus.FAILED

    def simulate_recovery(self):
        """Reset sensor to normal operation."""
        self._accel_bias = Vector3(
            random.gauss(0, 0.02),
            random.gauss(0, 0.02),
            random.gauss(0, 0.02)
        )
        self._gyro_bias = Vector3(
            random.gauss(0, 0.5),
            random.gauss(0, 0.5),
            random.gauss(0, 0.5)
        )
        self._mag_bias = Vector3(
            random.gauss(0, 5.0),
            random.gauss(0, 5.0),
            random.gauss(0, 5.0)
        )
        self._health_status = IMUHealthStatus.GOOD

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize IMU data to dictionary."""
        return {
            'type': self.type,
            'accelerometer': self._accel.to_dict(),
            'gyroscope': self._gyro.to_dict(),
            'magnetometer': self._mag.to_dict(),
            'attitude': self._attitude.to_dict(),
            'temperature': self._temperature,
            'calibration': {
                'accel': self._calibration.accel_state.name,
                'gyro': self._calibration.gyro_state.name,
                'mag': self._calibration.mag_state.name,
                'fully_calibrated': self._calibration.is_fully_calibrated()
            },
            'health': self._health_status.value,
            'vibration_level': self._vibration_level,
            'is_valid': self.is_valid(),
            'update_rate': self._update_rate
        }
