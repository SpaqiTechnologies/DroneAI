"""
State Estimator module using Extended Kalman Filter (EKF) for sensor fusion.

Fuses GPS and IMU data to provide robust position, velocity, and attitude estimation.
Provides dead reckoning capability when GPS is unavailable.

Navigation Redundancy Strategy:
1. GPS + IMU fusion (primary outdoor) - This module
2. Dead reckoning fallback when GPS lost - This module
3. Future: Visual odometry, optical flow integration
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Callable
import logging

# NumPy-free matrix operations for portability
from sensors.imu_sensor import IMUReading, Vector3, EulerAngles
from sensors.gps_sensor import GPSReading

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    """Current navigation mode based on sensor availability."""
    GPS_IMU_FUSION = "gps_imu_fusion"      # Full fusion (best accuracy)
    DEAD_RECKONING = "dead_reckoning"      # IMU only (drifts over time)
    GPS_ONLY = "gps_only"                   # GPS only (no smoothing)
    DEGRADED = "degraded"                   # Limited sensors available
    FAILED = "failed"                       # No valid navigation


class EstimatorHealth(Enum):
    """Health status of the state estimator."""
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class NavigationState:
    """
    Complete navigation state estimate.

    Coordinate frame: NED (North-East-Down)
    - Position: meters from home position
    - Velocity: m/s in NED frame
    - Attitude: degrees (roll, pitch, yaw)
    """
    # Position (meters from home in NED frame)
    position_north: float = 0.0
    position_east: float = 0.0
    position_down: float = 0.0  # Negative = above home

    # Velocity (m/s in NED frame)
    velocity_north: float = 0.0
    velocity_east: float = 0.0
    velocity_down: float = 0.0

    # Attitude (degrees)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0  # Heading, 0 = North

    # Accelerometer bias estimates (m/s²)
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0

    # Gyro bias estimates (deg/s)
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0

    # Uncertainty (standard deviations)
    position_uncertainty: float = 10.0  # meters
    velocity_uncertainty: float = 1.0   # m/s
    attitude_uncertainty: float = 5.0   # degrees

    # Metadata
    timestamp: float = 0.0
    navigation_mode: NavigationMode = NavigationMode.FAILED
    gps_available: bool = False
    imu_available: bool = False

    def get_position_ned(self) -> Tuple[float, float, float]:
        """Get position as (North, East, Down) tuple."""
        return (self.position_north, self.position_east, self.position_down)

    def get_velocity_ned(self) -> Tuple[float, float, float]:
        """Get velocity as (North, East, Down) tuple."""
        return (self.velocity_north, self.velocity_east, self.velocity_down)

    def get_attitude(self) -> EulerAngles:
        """Get attitude as EulerAngles."""
        return EulerAngles(self.roll, self.pitch, self.yaw)

    def get_speed(self) -> float:
        """Get horizontal speed (m/s)."""
        return math.sqrt(self.velocity_north**2 + self.velocity_east**2)

    def get_altitude_agl(self) -> float:
        """Get altitude above ground level (positive up)."""
        return -self.position_down

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'position': {
                'north': self.position_north,
                'east': self.position_east,
                'down': self.position_down,
                'altitude_agl': self.get_altitude_agl()
            },
            'velocity': {
                'north': self.velocity_north,
                'east': self.velocity_east,
                'down': self.velocity_down,
                'speed': self.get_speed()
            },
            'attitude': {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw
            },
            'uncertainty': {
                'position': self.position_uncertainty,
                'velocity': self.velocity_uncertainty,
                'attitude': self.attitude_uncertainty
            },
            'mode': self.navigation_mode.value,
            'gps_available': self.gps_available,
            'imu_available': self.imu_available,
            'timestamp': self.timestamp
        }


@dataclass
class EKFConfig:
    """Configuration parameters for the EKF."""
    # Process noise (how much we trust the prediction model)
    process_noise_accel: float = 0.5      # m/s² - accelerometer noise
    process_noise_gyro: float = 0.1       # deg/s - gyroscope noise
    process_noise_accel_bias: float = 0.001  # m/s²/s - accel bias random walk
    process_noise_gyro_bias: float = 0.0001  # deg/s/s - gyro bias random walk

    # Measurement noise (how much we trust measurements)
    gps_position_noise: float = 2.5       # meters - GPS position noise
    gps_velocity_noise: float = 0.5       # m/s - GPS velocity noise

    # Dead reckoning limits
    max_dead_reckoning_time: float = 30.0  # seconds before declaring failed
    dead_reckoning_drift_rate: float = 0.5  # m/s position uncertainty growth

    # GPS quality thresholds
    min_gps_satellites: int = 4
    max_gps_accuracy: float = 10.0        # meters - reject if worse

    # Innovation gate (reject outliers)
    innovation_gate: float = 5.0          # standard deviations


class Matrix:
    """Simple matrix class for EKF operations (NumPy-free)."""

    def __init__(self, rows: int, cols: int, data: Optional[List[List[float]]] = None):
        self.rows = rows
        self.cols = cols
        if data:
            self.data = [row[:] for row in data]
        else:
            self.data = [[0.0] * cols for _ in range(rows)]

    @staticmethod
    def identity(n: int) -> 'Matrix':
        """Create identity matrix."""
        m = Matrix(n, n)
        for i in range(n):
            m.data[i][i] = 1.0
        return m

    @staticmethod
    def diagonal(values: List[float]) -> 'Matrix':
        """Create diagonal matrix from list of values."""
        n = len(values)
        m = Matrix(n, n)
        for i in range(n):
            m.data[i][i] = values[i]
        return m

    def __getitem__(self, key: Tuple[int, int]) -> float:
        return self.data[key[0]][key[1]]

    def __setitem__(self, key: Tuple[int, int], value: float):
        self.data[key[0]][key[1]] = value

    def __add__(self, other: 'Matrix') -> 'Matrix':
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __mul__(self, other) -> 'Matrix':
        if isinstance(other, Matrix):
            # Matrix multiplication
            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
        else:
            # Scalar multiplication
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result

    def transpose(self) -> 'Matrix':
        """Return transpose of matrix."""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def inverse_3x3(self) -> 'Matrix':
        """Compute inverse of 3x3 matrix (for measurement update)."""
        if self.rows != 3 or self.cols != 3:
            raise ValueError("inverse_3x3 only works on 3x3 matrices")

        a = self.data
        det = (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1]) -
               a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0]) +
               a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))

        if abs(det) < 1e-10:
            # Matrix is singular, return identity scaled large
            return Matrix.identity(3) * 1e6

        inv_det = 1.0 / det
        result = Matrix(3, 3)
        result.data[0][0] = (a[1][1]*a[2][2] - a[1][2]*a[2][1]) * inv_det
        result.data[0][1] = (a[0][2]*a[2][1] - a[0][1]*a[2][2]) * inv_det
        result.data[0][2] = (a[0][1]*a[1][2] - a[0][2]*a[1][1]) * inv_det
        result.data[1][0] = (a[1][2]*a[2][0] - a[1][0]*a[2][2]) * inv_det
        result.data[1][1] = (a[0][0]*a[2][2] - a[0][2]*a[2][0]) * inv_det
        result.data[1][2] = (a[0][2]*a[1][0] - a[0][0]*a[1][2]) * inv_det
        result.data[2][0] = (a[1][0]*a[2][1] - a[1][1]*a[2][0]) * inv_det
        result.data[2][1] = (a[0][1]*a[2][0] - a[0][0]*a[2][1]) * inv_det
        result.data[2][2] = (a[0][0]*a[1][1] - a[0][1]*a[1][0]) * inv_det
        return result


class StateEstimator:
    """
    Extended Kalman Filter for GPS + IMU sensor fusion.

    State vector (15 elements):
    [0-2]   Position NED (meters from home)
    [3-5]   Velocity NED (m/s)
    [6-8]   Attitude (roll, pitch, yaw in degrees)
    [9-11]  Accelerometer bias (m/s²)
    [12-14] Gyroscope bias (deg/s)

    Provides:
    - GPS + IMU fusion for accurate position/velocity
    - Attitude estimation from gyro + accelerometer
    - Automatic dead reckoning when GPS is lost
    - Bias estimation to improve accuracy over time
    """

    STATE_SIZE = 15

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize the state estimator.

        Args:
            config: EKF configuration parameters
        """
        self.config = config or EKFConfig()

        # State vector
        self._state = [0.0] * self.STATE_SIZE

        # State covariance matrix (uncertainty)
        self._covariance = Matrix.identity(self.STATE_SIZE)
        # Initialize with reasonable uncertainties
        for i in range(3):  # Position
            self._covariance[i, i] = 100.0  # 10m std dev
        for i in range(3, 6):  # Velocity
            self._covariance[i, i] = 1.0  # 1m/s std dev
        for i in range(6, 9):  # Attitude
            self._covariance[i, i] = 25.0  # 5 deg std dev
        for i in range(9, 12):  # Accel bias
            self._covariance[i, i] = 0.01  # 0.1 m/s² std dev
        for i in range(12, 15):  # Gyro bias
            self._covariance[i, i] = 1.0  # 1 deg/s std dev

        # Process noise matrix
        self._Q = Matrix.diagonal([
            self.config.process_noise_accel**2,  # pos_n
            self.config.process_noise_accel**2,  # pos_e
            self.config.process_noise_accel**2,  # pos_d
            self.config.process_noise_accel**2,  # vel_n
            self.config.process_noise_accel**2,  # vel_e
            self.config.process_noise_accel**2,  # vel_d
            self.config.process_noise_gyro**2,   # roll
            self.config.process_noise_gyro**2,   # pitch
            self.config.process_noise_gyro**2,   # yaw
            self.config.process_noise_accel_bias**2,  # accel_bias_x
            self.config.process_noise_accel_bias**2,  # accel_bias_y
            self.config.process_noise_accel_bias**2,  # accel_bias_z
            self.config.process_noise_gyro_bias**2,   # gyro_bias_x
            self.config.process_noise_gyro_bias**2,   # gyro_bias_y
            self.config.process_noise_gyro_bias**2,   # gyro_bias_z
        ])

        # Home position (GPS coordinates)
        self._home_lat = 0.0
        self._home_lon = 0.0
        self._home_alt = 0.0
        self._home_set = False

        # Timing
        self._last_imu_time = 0.0
        self._last_gps_time = 0.0
        self._last_update_time = 0.0

        # Navigation mode tracking
        self._navigation_mode = NavigationMode.FAILED
        self._gps_available = False
        self._imu_available = False
        self._dead_reckoning_start_time = 0.0

        # Health monitoring
        self._health = EstimatorHealth.GOOD
        self._gps_loss_count = 0
        self._innovation_failures = 0

        # Callbacks
        self._mode_change_callbacks: List[Callable[[NavigationMode], None]] = []

        # Initialized flag
        self._initialized = False

    def set_home_position(self, lat: float, lon: float, alt: float = 0.0):
        """
        Set the home position as origin for NED frame.

        Args:
            lat: Home latitude (degrees)
            lon: Home longitude (degrees)
            alt: Home altitude MSL (meters)
        """
        self._home_lat = lat
        self._home_lon = lon
        self._home_alt = alt
        self._home_set = True

        # Reset state to home
        self._state[0:6] = [0.0] * 6  # Position and velocity at origin
        self._initialized = True

        logger.info(f"Home position set: {lat:.6f}, {lon:.6f}, {alt:.1f}m")

    def predict(self, imu: IMUReading):
        """
        EKF prediction step using IMU data.

        Propagates state forward using accelerometer and gyroscope.

        Args:
            imu: IMU reading with accelerometer and gyroscope data
        """
        if not self._initialized or not self._home_set:
            return

        current_time = imu.timestamp
        if self._last_imu_time == 0:
            self._last_imu_time = current_time
            return

        dt = current_time - self._last_imu_time
        if dt <= 0 or dt > 1.0:  # Skip invalid dt
            self._last_imu_time = current_time
            return

        self._last_imu_time = current_time
        self._imu_available = imu.is_valid()

        if not imu.is_valid():
            self._update_navigation_mode()
            return

        # Get corrected IMU data (subtract estimated biases)
        accel_x = imu.accelerometer.x - self._state[9]
        accel_y = imu.accelerometer.y - self._state[10]
        accel_z = imu.accelerometer.z - self._state[11]

        gyro_x = imu.gyroscope.x - self._state[12]
        gyro_y = imu.gyroscope.y - self._state[13]
        gyro_z = imu.gyroscope.z - self._state[14]

        # Current attitude
        roll = math.radians(self._state[6])
        pitch = math.radians(self._state[7])
        yaw = math.radians(self._state[8])

        # Rotate acceleration from body to NED frame
        # Remove gravity component (accelerometer measures gravity as up)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Rotation matrix body to NED
        accel_n = (cy*cp)*accel_x + (cy*sp*sr - sy*cr)*accel_y + (cy*sp*cr + sy*sr)*accel_z
        accel_e = (sy*cp)*accel_x + (sy*sp*sr + cy*cr)*accel_y + (sy*sp*cr - cy*sr)*accel_z
        accel_d = (-sp)*accel_x + (cp*sr)*accel_y + (cp*cr)*accel_z

        # Remove gravity (g is in +Down direction in NED)
        accel_d -= 9.80665

        # State prediction (simple integration)
        # Position += velocity * dt + 0.5 * accel * dt²
        self._state[0] += self._state[3] * dt + 0.5 * accel_n * dt**2
        self._state[1] += self._state[4] * dt + 0.5 * accel_e * dt**2
        self._state[2] += self._state[5] * dt + 0.5 * accel_d * dt**2

        # Velocity += accel * dt
        self._state[3] += accel_n * dt
        self._state[4] += accel_e * dt
        self._state[5] += accel_d * dt

        # Attitude += gyro * dt (simplified, ignores coupling)
        self._state[6] += gyro_x * dt
        self._state[7] += gyro_y * dt
        self._state[8] += gyro_z * dt

        # Normalize angles
        self._state[6] = self._normalize_angle(self._state[6])
        self._state[7] = self._normalize_angle(self._state[7])
        self._state[8] = self._state[8] % 360

        # Covariance prediction: P = F * P * F' + Q
        # Simplified: just add process noise scaled by dt
        for i in range(self.STATE_SIZE):
            self._covariance[i, i] += self._Q[i, i] * dt

        # In dead reckoning, uncertainty grows faster
        if not self._gps_available:
            drift = self.config.dead_reckoning_drift_rate * dt
            for i in range(3):  # Position uncertainty
                self._covariance[i, i] += drift**2

        self._last_update_time = current_time
        self._update_navigation_mode()

    def update_gps(self, gps: GPSReading):
        """
        EKF measurement update using GPS data.

        Corrects state estimate using GPS position.

        Args:
            gps: GPS reading with position data
        """
        if not self._initialized:
            return

        current_time = gps.timestamp

        # Check GPS validity
        if not gps.is_valid():
            self._gps_available = False
            self._update_navigation_mode()
            return

        if gps.satellites_used < self.config.min_gps_satellites:
            self._gps_available = False
            self._update_navigation_mode()
            return

        if gps.accuracy_horizontal > self.config.max_gps_accuracy:
            self._gps_available = False
            self._update_navigation_mode()
            return

        # Set home if not set
        if not self._home_set:
            self.set_home_position(gps.latitude, gps.longitude, gps.altitude)

        # Convert GPS to NED position
        pos_n, pos_e = self._gps_to_ned(gps.latitude, gps.longitude)
        pos_d = -(gps.altitude - self._home_alt)  # Down is negative altitude

        # Measurement vector
        z = [pos_n, pos_e, pos_d]

        # Predicted measurement (from state)
        z_pred = [self._state[0], self._state[1], self._state[2]]

        # Innovation (measurement residual)
        innovation = [z[i] - z_pred[i] for i in range(3)]

        # Measurement noise (from GPS accuracy)
        R = Matrix.diagonal([
            max(gps.accuracy_horizontal, self.config.gps_position_noise)**2,
            max(gps.accuracy_horizontal, self.config.gps_position_noise)**2,
            max(gps.accuracy_vertical, self.config.gps_position_noise * 1.5)**2
        ])

        # Observation matrix H (maps state to measurement)
        # Only position is observed: H = [I_3x3, 0_3x12]
        H = Matrix(3, self.STATE_SIZE)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Innovation covariance: S = H * P * H' + R
        HP = H * self._covariance
        S = HP * H.transpose() + R

        # Check innovation gate (reject outliers)
        S_inv = S.inverse_3x3()
        mahal_dist = 0.0
        for i in range(3):
            for j in range(3):
                mahal_dist += innovation[i] * S_inv[i, j] * innovation[j]
        mahal_dist = math.sqrt(mahal_dist)

        if mahal_dist > self.config.innovation_gate:
            self._innovation_failures += 1
            if self._innovation_failures > 5:
                logger.warning(f"GPS innovation gate failed: {mahal_dist:.1f} > {self.config.innovation_gate}")
            return

        self._innovation_failures = 0

        # Kalman gain: K = P * H' * S^-1
        PHt = self._covariance * H.transpose()
        K = Matrix(self.STATE_SIZE, 3)
        for i in range(self.STATE_SIZE):
            for j in range(3):
                for k in range(3):
                    K[i, j] += PHt[i, k] * S_inv[k, j]

        # State update: x = x + K * innovation
        for i in range(self.STATE_SIZE):
            for j in range(3):
                self._state[i] += K[i, j] * innovation[j]

        # Covariance update: P = (I - K * H) * P
        KH = K * H
        I_KH = Matrix.identity(self.STATE_SIZE) - KH
        self._covariance = I_KH * self._covariance

        # Update GPS state
        self._gps_available = True
        self._last_gps_time = current_time
        self._gps_loss_count = 0
        self._dead_reckoning_start_time = 0

        self._last_update_time = current_time
        self._update_navigation_mode()

    def _gps_to_ned(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to NED position relative to home.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)

        Returns:
            (north, east) position in meters
        """
        # Earth radius
        R = 6371000.0  # meters

        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        home_lat_rad = math.radians(self._home_lat)
        home_lon_rad = math.radians(self._home_lon)

        # Calculate NED position
        north = R * (lat_rad - home_lat_rad)
        east = R * math.cos(home_lat_rad) * (lon_rad - home_lon_rad)

        return (north, east)

    def _ned_to_gps(self, north: float, east: float) -> Tuple[float, float]:
        """
        Convert NED position to GPS coordinates.

        Args:
            north: North position (meters)
            east: East position (meters)

        Returns:
            (latitude, longitude) in degrees
        """
        R = 6371000.0

        home_lat_rad = math.radians(self._home_lat)

        lat_rad = home_lat_rad + north / R
        lon_rad = math.radians(self._home_lon) + east / (R * math.cos(home_lat_rad))

        return (math.degrees(lat_rad), math.degrees(lon_rad))

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -180 to 180 degrees."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _update_navigation_mode(self):
        """Update navigation mode based on sensor availability."""
        old_mode = self._navigation_mode

        if self._gps_available and self._imu_available:
            self._navigation_mode = NavigationMode.GPS_IMU_FUSION
            self._health = EstimatorHealth.GOOD
        elif self._imu_available and not self._gps_available:
            # Dead reckoning
            if self._dead_reckoning_start_time == 0:
                self._dead_reckoning_start_time = time.time()
                self._gps_loss_count += 1

            dr_time = time.time() - self._dead_reckoning_start_time

            if dr_time < self.config.max_dead_reckoning_time:
                self._navigation_mode = NavigationMode.DEAD_RECKONING
                if dr_time < 10:
                    self._health = EstimatorHealth.WARNING
                else:
                    self._health = EstimatorHealth.CRITICAL
            else:
                self._navigation_mode = NavigationMode.DEGRADED
                self._health = EstimatorHealth.CRITICAL
        elif self._gps_available and not self._imu_available:
            self._navigation_mode = NavigationMode.GPS_ONLY
            self._health = EstimatorHealth.WARNING
        else:
            self._navigation_mode = NavigationMode.FAILED
            self._health = EstimatorHealth.FAILED

        # Notify callbacks if mode changed
        if old_mode != self._navigation_mode:
            logger.info(f"Navigation mode changed: {old_mode.value} -> {self._navigation_mode.value}")
            for callback in self._mode_change_callbacks:
                try:
                    callback(self._navigation_mode)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")

    def get_state(self) -> NavigationState:
        """
        Get current navigation state estimate.

        Returns:
            NavigationState with position, velocity, attitude
        """
        # Calculate uncertainties from covariance diagonal
        pos_uncertainty = math.sqrt(
            self._covariance[0, 0] + self._covariance[1, 1] + self._covariance[2, 2]
        ) / math.sqrt(3)

        vel_uncertainty = math.sqrt(
            self._covariance[3, 3] + self._covariance[4, 4] + self._covariance[5, 5]
        ) / math.sqrt(3)

        att_uncertainty = math.sqrt(
            self._covariance[6, 6] + self._covariance[7, 7] + self._covariance[8, 8]
        ) / math.sqrt(3)

        return NavigationState(
            position_north=self._state[0],
            position_east=self._state[1],
            position_down=self._state[2],
            velocity_north=self._state[3],
            velocity_east=self._state[4],
            velocity_down=self._state[5],
            roll=self._state[6],
            pitch=self._state[7],
            yaw=self._state[8],
            accel_bias_x=self._state[9],
            accel_bias_y=self._state[10],
            accel_bias_z=self._state[11],
            gyro_bias_x=self._state[12],
            gyro_bias_y=self._state[13],
            gyro_bias_z=self._state[14],
            position_uncertainty=pos_uncertainty,
            velocity_uncertainty=vel_uncertainty,
            attitude_uncertainty=att_uncertainty,
            timestamp=self._last_update_time,
            navigation_mode=self._navigation_mode,
            gps_available=self._gps_available,
            imu_available=self._imu_available
        )

    def get_gps_position(self) -> Tuple[float, float, float]:
        """
        Get current position as GPS coordinates.

        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        lat, lon = self._ned_to_gps(self._state[0], self._state[1])
        alt = self._home_alt - self._state[2]  # Convert down to altitude
        return (lat, lon, alt)

    def get_navigation_mode(self) -> NavigationMode:
        """Get current navigation mode."""
        return self._navigation_mode

    def get_health(self) -> EstimatorHealth:
        """Get estimator health status."""
        return self._health

    def get_dead_reckoning_time(self) -> float:
        """Get time spent in dead reckoning mode (seconds)."""
        if self._dead_reckoning_start_time == 0:
            return 0.0
        return time.time() - self._dead_reckoning_start_time

    def is_gps_available(self) -> bool:
        """Check if GPS is currently available."""
        return self._gps_available

    def is_initialized(self) -> bool:
        """Check if estimator is initialized."""
        return self._initialized

    def add_mode_change_callback(self, callback: Callable[[NavigationMode], None]):
        """Add callback for navigation mode changes."""
        self._mode_change_callbacks.append(callback)

    def reset(self):
        """Reset the state estimator."""
        self._state = [0.0] * self.STATE_SIZE
        self._covariance = Matrix.identity(self.STATE_SIZE)
        for i in range(3):
            self._covariance[i, i] = 100.0
        for i in range(3, 6):
            self._covariance[i, i] = 1.0
        for i in range(6, 9):
            self._covariance[i, i] = 25.0

        self._navigation_mode = NavigationMode.FAILED
        self._gps_available = False
        self._imu_available = False
        self._dead_reckoning_start_time = 0
        self._last_imu_time = 0
        self._last_gps_time = 0
        self._initialized = self._home_set

    def to_dict(self) -> dict:
        """Serialize estimator state to dictionary."""
        state = self.get_state()
        return {
            'state': state.to_dict(),
            'home': {
                'lat': self._home_lat,
                'lon': self._home_lon,
                'alt': self._home_alt,
                'set': self._home_set
            },
            'health': self._health.value,
            'dead_reckoning_time': self.get_dead_reckoning_time(),
            'gps_loss_count': self._gps_loss_count,
            'initialized': self._initialized
        }
