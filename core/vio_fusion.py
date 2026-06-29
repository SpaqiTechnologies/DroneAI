"""
Visual-Inertial Odometry (VIO) fusion module for Drone AI application.

Combines visual odometry with IMU data for robust pose estimation
in GPS-denied environments.

Navigation Redundancy Strategy:
2. Visual SLAM / VIO (primary indoor/GPS-denied) - This module

Algorithm: Loosely-coupled VIO with complementary filter
- IMU provides high-rate attitude and short-term motion
- Visual odometry provides drift-free position updates
- Scale correction from other sensors (barometer, rangefinder)
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Callable
from sensors.visual_odometry import VOReading, VOState, VOQuality, VOPose
from sensors.imu_sensor import IMUReading, EulerAngles


class VIOState(Enum):
    """VIO system state."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    VISION_ONLY = "vision_only"
    IMU_ONLY = "imu_only"
    DEGRADED = "degraded"
    FAILED = "failed"


class VIOHealth(Enum):
    """VIO health status."""
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class VIOState3D:
    """Complete VIO state estimate."""
    # Position (meters, NED frame)
    position_north: float = 0.0
    position_east: float = 0.0
    position_down: float = 0.0

    # Velocity (m/s, NED frame)
    velocity_north: float = 0.0
    velocity_east: float = 0.0
    velocity_down: float = 0.0

    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Biases
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0

    # Scale factor for monocular VO
    scale: float = 1.0

    # Uncertainties
    position_uncertainty: float = 0.0
    velocity_uncertainty: float = 0.0
    attitude_uncertainty: float = 0.0

    # Availability flags
    vision_available: bool = False
    imu_available: bool = False

    def get_position(self) -> Tuple[float, float, float]:
        """Get position as (north, east, down)."""
        return (self.position_north, self.position_east, self.position_down)

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity as (north, east, down)."""
        return (self.velocity_north, self.velocity_east, self.velocity_down)

    def get_attitude(self) -> EulerAngles:
        """Get attitude as EulerAngles."""
        return EulerAngles(roll=self.roll, pitch=self.pitch, yaw=self.yaw)

    def get_speed(self) -> float:
        """Get horizontal speed."""
        return math.sqrt(self.velocity_north**2 + self.velocity_east**2)

    def get_altitude(self) -> float:
        """Get altitude (positive up)."""
        return -self.position_down

    def to_dict(self) -> dict:
        return {
            'position': {
                'north': self.position_north,
                'east': self.position_east,
                'down': self.position_down,
                'altitude': self.get_altitude()
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
                'yaw': self.yaw,
                'roll_deg': math.degrees(self.roll),
                'pitch_deg': math.degrees(self.pitch),
                'yaw_deg': math.degrees(self.yaw)
            },
            'scale': self.scale,
            'uncertainties': {
                'position': self.position_uncertainty,
                'velocity': self.velocity_uncertainty,
                'attitude': self.attitude_uncertainty
            },
            'vision_available': self.vision_available,
            'imu_available': self.imu_available
        }


@dataclass
class VIOConfig:
    """VIO fusion configuration."""
    # Process noise
    accel_noise: float = 0.1  # m/s^2
    gyro_noise: float = 0.01  # rad/s
    accel_bias_noise: float = 0.001
    gyro_bias_noise: float = 0.0001

    # Measurement noise
    vo_position_noise: float = 0.05  # meters
    vo_orientation_noise: float = 0.02  # radians
    vo_velocity_noise: float = 0.1  # m/s

    # Complementary filter gains
    position_gain: float = 0.1  # How much to trust VO position
    velocity_gain: float = 0.2  # How much to trust VO velocity
    attitude_gain: float = 0.05  # How much to trust VO attitude (prefer IMU)

    # Scale estimation
    scale_gain: float = 0.01  # Scale correction rate
    min_scale: float = 0.5
    max_scale: float = 2.0

    # Timeouts
    vo_timeout: float = 0.5  # seconds without VO before degraded
    imu_timeout: float = 0.1  # seconds without IMU before degraded

    # Quality thresholds
    min_vo_confidence: float = 0.3
    min_features: int = 20


class VIOFusion:
    """
    Visual-Inertial Odometry fusion system.

    Combines:
    - Visual Odometry: Position, orientation (drift-free but noisy)
    - IMU: Acceleration, angular rate (low noise but drifts)

    Uses complementary filtering approach:
    - High-pass filter on IMU (short-term motion)
    - Low-pass filter on VO (long-term position)
    """

    def __init__(self, config: Optional[VIOConfig] = None):
        """
        Initialize VIO fusion system.

        Args:
            config: Configuration parameters
        """
        self._config = config or VIOConfig()

        # System state
        self._system_state = VIOState.IDLE
        self._health = VIOHealth.FAILED

        # State estimate
        self._state = VIOState3D()

        # Last sensor readings
        self._last_vo_reading: Optional[VOReading] = None
        self._last_imu_reading: Optional[IMUReading] = None

        # Timing
        self._last_vo_time = 0.0
        self._last_imu_time = 0.0
        self._last_update_time = 0.0

        # IMU integration state
        self._imu_velocity = [0.0, 0.0, 0.0]  # Integrated velocity
        self._imu_position = [0.0, 0.0, 0.0]  # Integrated position

        # VO reference frame
        self._vo_origin_set = False
        self._vo_origin = (0.0, 0.0, 0.0)

        # Scale estimation
        self._scale_samples: List[float] = []
        self._scale_initialized = False

        # Callbacks
        self._state_change_callbacks: List[Callable[[VIOState], None]] = []

        # Statistics
        self._vo_updates = 0
        self._imu_updates = 0

    def initialize(self, initial_yaw: float = 0.0):
        """
        Initialize VIO system.

        Args:
            initial_yaw: Initial heading (radians)
        """
        self._system_state = VIOState.INITIALIZING
        self._state = VIOState3D(yaw=initial_yaw)
        self._imu_velocity = [0.0, 0.0, 0.0]
        self._imu_position = [0.0, 0.0, 0.0]
        self._vo_origin_set = False
        self._scale_samples.clear()
        self._scale_initialized = False
        self._vo_updates = 0
        self._imu_updates = 0

    def process_imu(self, imu: IMUReading):
        """
        Process IMU measurement (high rate).

        Args:
            imu: IMU reading with acceleration and angular rates
        """
        if not imu.is_valid():
            return

        current_time = time.time()
        dt = current_time - self._last_imu_time if self._last_imu_time > 0 else 0.01

        if dt <= 0 or dt > 1.0:
            dt = 0.01

        self._last_imu_reading = imu
        self._last_imu_time = current_time
        self._imu_updates += 1

        # Update attitude from IMU (complementary filter with VO)
        self._update_attitude_from_imu(imu, dt)

        # Integrate acceleration to get velocity
        self._integrate_acceleration(imu, dt)

        # Update state availability
        self._state.imu_available = True

        # Update system state
        self._update_system_state()

        self._last_update_time = current_time

    def process_vo(self, vo: VOReading):
        """
        Process visual odometry measurement (lower rate).

        Args:
            vo: Visual odometry reading with pose
        """
        current_time = time.time()

        # Check validity
        if not vo.is_valid() or vo.confidence < self._config.min_vo_confidence:
            # VO lost - continue with IMU only
            self._state.vision_available = False
            self._update_system_state()
            return

        self._last_vo_reading = vo
        self._last_vo_time = current_time
        self._vo_updates += 1

        # Set VO origin on first valid reading
        if not self._vo_origin_set:
            self._vo_origin = vo.pose.get_position()
            self._vo_origin_set = True

        # Convert VO pose to NED frame
        vo_pos = vo.pose.get_position()
        vo_vel = vo.pose.get_velocity()

        # Relative position from origin
        rel_x = vo_pos[0] - self._vo_origin[0]
        rel_y = vo_pos[1] - self._vo_origin[1]
        rel_z = vo_pos[2] - self._vo_origin[2]

        # Apply scale
        scale = self._state.scale
        scaled_x = rel_x * scale
        scaled_y = rel_y * scale
        scaled_z = rel_z * scale

        # Transform from VO body frame to NED
        # Assuming VO x=forward, y=right, z=down (similar to NED)
        cos_yaw = math.cos(self._state.yaw)
        sin_yaw = math.sin(self._state.yaw)

        vo_north = scaled_x * cos_yaw - scaled_y * sin_yaw
        vo_east = scaled_x * sin_yaw + scaled_y * cos_yaw
        vo_down = scaled_z

        # Complementary filter: fuse with IMU-integrated position
        alpha = self._config.position_gain * vo.confidence

        self._state.position_north = (1 - alpha) * self._state.position_north + alpha * vo_north
        self._state.position_east = (1 - alpha) * self._state.position_east + alpha * vo_east
        self._state.position_down = (1 - alpha) * self._state.position_down + alpha * vo_down

        # Fuse velocity
        beta = self._config.velocity_gain * vo.confidence

        vo_vel_north = vo_vel[0] * scale * cos_yaw - vo_vel[1] * scale * sin_yaw
        vo_vel_east = vo_vel[0] * scale * sin_yaw + vo_vel[1] * scale * cos_yaw
        vo_vel_down = vo_vel[2] * scale

        self._state.velocity_north = (1 - beta) * self._state.velocity_north + beta * vo_vel_north
        self._state.velocity_east = (1 - beta) * self._state.velocity_east + beta * vo_vel_east
        self._state.velocity_down = (1 - beta) * self._state.velocity_down + beta * vo_vel_down

        # Reset IMU integration to prevent drift
        self._imu_velocity = [self._state.velocity_north, self._state.velocity_east, self._state.velocity_down]
        self._imu_position = [self._state.position_north, self._state.position_east, self._state.position_down]

        # Update uncertainties based on VO quality
        self._update_uncertainties(vo)

        # Update state availability
        self._state.vision_available = True

        # Update system state
        self._update_system_state()

        self._last_update_time = current_time

    def _update_attitude_from_imu(self, imu: IMUReading, dt: float):
        """Update attitude using IMU gyroscope."""
        # Get gyro rates (deg/s -> rad/s)
        wx = math.radians(imu.gyroscope.x) - self._state.gyro_bias_x
        wy = math.radians(imu.gyroscope.y) - self._state.gyro_bias_y
        wz = math.radians(imu.gyroscope.z) - self._state.gyro_bias_z

        # Integrate angular rates
        self._state.roll += wx * dt
        self._state.pitch += wy * dt
        self._state.yaw += wz * dt

        # Normalize angles
        while self._state.yaw > math.pi:
            self._state.yaw -= 2 * math.pi
        while self._state.yaw < -math.pi:
            self._state.yaw += 2 * math.pi

        # Complementary filter with accelerometer for roll/pitch
        # (gravity provides absolute reference)
        if imu.accelerometer:
            ax = imu.accelerometer.x
            ay = imu.accelerometer.y
            az = imu.accelerometer.z

            accel_roll = math.atan2(ay, az)
            accel_pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))

            # High-pass on gyro, low-pass on accelerometer
            alpha = 0.98  # Gyro weight
            self._state.roll = alpha * self._state.roll + (1 - alpha) * accel_roll
            self._state.pitch = alpha * self._state.pitch + (1 - alpha) * accel_pitch

    def _integrate_acceleration(self, imu: IMUReading, dt: float):
        """Integrate acceleration to get velocity and position."""
        if not imu.accelerometer:
            return

        # Get acceleration in body frame (m/s^2)
        ax = imu.accelerometer.x - self._state.accel_bias_x
        ay = imu.accelerometer.y - self._state.accel_bias_y
        az = imu.accelerometer.z - self._state.accel_bias_z

        # Remove gravity (assuming az includes gravity)
        az -= 9.81

        # Transform to NED frame
        cr = math.cos(self._state.roll)
        sr = math.sin(self._state.roll)
        cp = math.cos(self._state.pitch)
        sp = math.sin(self._state.pitch)
        cy = math.cos(self._state.yaw)
        sy = math.sin(self._state.yaw)

        # Rotation matrix (body to NED)
        an = ax * (cp * cy) + ay * (sr * sp * cy - cr * sy) + az * (cr * sp * cy + sr * sy)
        ae = ax * (cp * sy) + ay * (sr * sp * sy + cr * cy) + az * (cr * sp * sy - sr * cy)
        ad = ax * (-sp) + ay * (sr * cp) + az * (cr * cp)

        # Integrate to velocity
        self._imu_velocity[0] += an * dt
        self._imu_velocity[1] += ae * dt
        self._imu_velocity[2] += ad * dt

        # Apply damping to prevent runaway integration
        damping = 0.99
        self._imu_velocity[0] *= damping
        self._imu_velocity[1] *= damping
        self._imu_velocity[2] *= damping

        # Integrate to position
        self._imu_position[0] += self._imu_velocity[0] * dt
        self._imu_position[1] += self._imu_velocity[1] * dt
        self._imu_position[2] += self._imu_velocity[2] * dt

        # Update state (will be corrected by VO)
        self._state.velocity_north = self._imu_velocity[0]
        self._state.velocity_east = self._imu_velocity[1]
        self._state.velocity_down = self._imu_velocity[2]

        self._state.position_north = self._imu_position[0]
        self._state.position_east = self._imu_position[1]
        self._state.position_down = self._imu_position[2]

    def _update_uncertainties(self, vo: VOReading):
        """Update uncertainty estimates based on VO quality."""
        base_pos_uncertainty = self._config.vo_position_noise
        base_vel_uncertainty = self._config.vo_velocity_noise
        base_att_uncertainty = self._config.vo_orientation_noise

        # Scale by confidence (lower confidence = higher uncertainty)
        confidence = max(0.1, vo.confidence)
        self._state.position_uncertainty = base_pos_uncertainty / confidence
        self._state.velocity_uncertainty = base_vel_uncertainty / confidence
        self._state.attitude_uncertainty = base_att_uncertainty / confidence

    def _update_system_state(self):
        """Update overall system state based on sensor availability."""
        current_time = time.time()

        vo_age = current_time - self._last_vo_time if self._last_vo_time > 0 else float('inf')
        imu_age = current_time - self._last_imu_time if self._last_imu_time > 0 else float('inf')

        vo_valid = vo_age < self._config.vo_timeout and self._state.vision_available
        imu_valid = imu_age < self._config.imu_timeout and self._state.imu_available

        old_state = self._system_state

        if vo_valid and imu_valid:
            self._system_state = VIOState.RUNNING
            self._health = VIOHealth.GOOD
        elif vo_valid and not imu_valid:
            self._system_state = VIOState.VISION_ONLY
            self._health = VIOHealth.MARGINAL
        elif not vo_valid and imu_valid:
            self._system_state = VIOState.IMU_ONLY
            self._health = VIOHealth.POOR
        else:
            self._system_state = VIOState.FAILED
            self._health = VIOHealth.FAILED

        # Trigger callbacks on state change
        if old_state != self._system_state:
            for callback in self._state_change_callbacks:
                callback(self._system_state)

    def correct_scale_from_height(self, known_height: float):
        """
        Correct scale using known height (from barometer/rangefinder).

        Args:
            known_height: Actual height above ground (meters)
        """
        current_height = abs(self._state.position_down)

        if current_height > 0.1:
            new_scale = known_height / current_height
            new_scale = max(self._config.min_scale, min(self._config.max_scale, new_scale))

            # Gradual correction
            self._state.scale = (1 - self._config.scale_gain) * self._state.scale + self._config.scale_gain * new_scale

            self._scale_samples.append(new_scale)
            if len(self._scale_samples) > 100:
                self._scale_samples.pop(0)

            self._scale_initialized = True

    def add_state_change_callback(self, callback: Callable[[VIOState], None]):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    # ==================== Getters ====================

    def get_state(self) -> VIOState3D:
        """Get current state estimate."""
        return self._state

    def get_position(self) -> Tuple[float, float, float]:
        """Get position (north, east, down)."""
        return self._state.get_position()

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity (north, east, down)."""
        return self._state.get_velocity()

    def get_attitude(self) -> EulerAngles:
        """Get attitude."""
        return self._state.get_attitude()

    def get_system_state(self) -> VIOState:
        """Get system state."""
        return self._system_state

    def get_health(self) -> VIOHealth:
        """Get health status."""
        return self._health

    def is_running(self) -> bool:
        """Check if VIO is running with both sensors."""
        return self._system_state == VIOState.RUNNING

    def is_valid(self) -> bool:
        """Check if VIO has valid output."""
        return self._system_state in [VIOState.RUNNING, VIOState.VISION_ONLY, VIOState.IMU_ONLY]

    def get_scale(self) -> float:
        """Get current scale estimate."""
        return self._state.scale

    def is_scale_initialized(self) -> bool:
        """Check if scale has been initialized."""
        return self._scale_initialized

    # ==================== Setters ====================

    def set_initial_position(self, north: float, east: float, down: float):
        """Set initial position."""
        self._state.position_north = north
        self._state.position_east = east
        self._state.position_down = down
        self._imu_position = [north, east, down]

    def set_initial_yaw(self, yaw: float):
        """Set initial heading."""
        self._state.yaw = yaw

    def reset(self):
        """Reset VIO system."""
        self._system_state = VIOState.IDLE
        self._health = VIOHealth.FAILED
        self._state = VIOState3D()
        self._imu_velocity = [0.0, 0.0, 0.0]
        self._imu_position = [0.0, 0.0, 0.0]
        self._vo_origin_set = False
        self._scale_samples.clear()
        self._scale_initialized = False

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize VIO state."""
        return {
            'system_state': self._system_state.value,
            'health': self._health.value,
            'state': self._state.to_dict(),
            'scale_initialized': self._scale_initialized,
            'vo_updates': self._vo_updates,
            'imu_updates': self._imu_updates,
            'is_running': self.is_running(),
            'is_valid': self.is_valid()
        }
