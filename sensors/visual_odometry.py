"""
Visual Odometry sensor module for Drone AI application.

Provides visual-based motion estimation using camera images.
Foundation for Visual SLAM and VIO (Visual-Inertial Odometry).

Navigation Redundancy Strategy:
2. Visual SLAM / VIO (primary indoor/GPS-denied) - This module

Typical implementations: ORB-SLAM, VINS-Mono, ROVIO
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
from .sensor import Sensor


class VOState(Enum):
    """Visual Odometry system state."""
    OFF = "off"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    RELOCALIZING = "relocalizing"


class VOQuality(Enum):
    """Quality level of visual odometry estimate."""
    INVALID = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXCELLENT = 4


class FeatureType(Enum):
    """Type of visual features being tracked."""
    ORB = "orb"
    FAST = "fast"
    SIFT = "sift"
    HARRIS = "harris"


@dataclass
class Feature2D:
    """A 2D feature point in image coordinates."""
    x: float  # Pixel x coordinate
    y: float  # Pixel y coordinate
    id: int = 0  # Feature ID for tracking
    response: float = 0.0  # Feature response/strength
    octave: int = 0  # Scale octave
    age: int = 0  # Frames tracked

    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'id': self.id,
            'response': self.response,
            'age': self.age
        }


@dataclass
class Feature3D:
    """A 3D landmark point in world coordinates."""
    x: float  # World X (meters)
    y: float  # World Y (meters)
    z: float  # World Z (meters)
    id: int = 0
    observations: int = 0  # Number of times observed
    uncertainty: float = 1.0  # Position uncertainty

    def distance_from_origin(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'id': self.id,
            'observations': self.observations,
            'uncertainty': self.uncertainty
        }


@dataclass
class VOPose:
    """Visual odometry pose estimate."""
    # Position (meters, body frame relative to start)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Orientation (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Angular velocity (rad/s)
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def get_position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def get_velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    def get_speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def to_dict(self) -> dict:
        return {
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'orientation': {
                'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw,
                'roll_deg': math.degrees(self.roll),
                'pitch_deg': math.degrees(self.pitch),
                'yaw_deg': math.degrees(self.yaw)
            },
            'velocity': {'x': self.vx, 'y': self.vy, 'z': self.vz},
            'speed': self.get_speed()
        }


@dataclass
class VOReading:
    """Complete visual odometry measurement."""
    pose: VOPose
    timestamp: float

    # Quality metrics
    quality: VOQuality
    confidence: float  # 0-1

    # Feature tracking info
    features_tracked: int
    features_new: int
    features_lost: int
    inlier_ratio: float  # Ratio of inlier matches

    # Map info
    landmarks_visible: int
    landmarks_total: int

    # Timing
    processing_time: float  # seconds

    # State
    state: VOState

    def is_valid(self) -> bool:
        """Check if reading is valid for navigation."""
        return (
            self.state == VOState.TRACKING and
            self.quality != VOQuality.INVALID and
            self.confidence >= 0.3 and
            self.features_tracked >= 20
        )

    def to_dict(self) -> dict:
        return {
            'pose': self.pose.to_dict(),
            'timestamp': self.timestamp,
            'quality': self.quality.name,
            'confidence': self.confidence,
            'features_tracked': self.features_tracked,
            'features_new': self.features_new,
            'features_lost': self.features_lost,
            'inlier_ratio': self.inlier_ratio,
            'landmarks_visible': self.landmarks_visible,
            'landmarks_total': self.landmarks_total,
            'processing_time': self.processing_time,
            'state': self.state.value,
            'is_valid': self.is_valid()
        }


class VisualOdometrySensor(Sensor):
    """
    Visual Odometry sensor for camera-based motion estimation.

    Simulates a monocular/stereo visual odometry system that tracks
    features across frames to estimate camera motion.

    Specifications (typical drone camera):
    - Resolution: 640x480 to 1920x1080
    - Frame rate: 30-60 fps
    - FOV: 60-120 degrees
    - Feature count: 100-500 per frame

    Limitations:
    - Requires textured environment
    - Scale drift in monocular mode
    - Affected by motion blur
    - Fails in low light or featureless areas
    """

    # Default parameters
    DEFAULT_FRAME_RATE = 30
    DEFAULT_RESOLUTION = (640, 480)
    DEFAULT_FOV = 90.0  # degrees
    MIN_FEATURES = 20
    TARGET_FEATURES = 200
    MAX_FEATURES = 500

    # Quality thresholds
    QUALITY_EXCELLENT_THRESHOLD = 150
    QUALITY_HIGH_THRESHOLD = 100
    QUALITY_MEDIUM_THRESHOLD = 50
    QUALITY_LOW_THRESHOLD = 20

    def __init__(
        self,
        frame_rate: int = DEFAULT_FRAME_RATE,
        resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        fov: float = DEFAULT_FOV,
        stereo: bool = False,
        baseline: float = 0.1  # Stereo baseline in meters
    ):
        """
        Initialize visual odometry sensor.

        Args:
            frame_rate: Camera frame rate (Hz)
            resolution: Image resolution (width, height)
            fov: Field of view (degrees)
            stereo: True for stereo camera
            baseline: Stereo baseline (meters)
        """
        self.type = "visual_odometry"
        self._frame_rate = frame_rate
        self._frame_interval = 1.0 / frame_rate
        self._resolution = resolution
        self._fov = fov
        self._stereo = stereo
        self._baseline = baseline

        # Sensor state
        self._state = VOState.OFF
        self._is_connected = False

        # Current pose estimate
        self._pose = VOPose()

        # Feature tracking
        self._features_2d: List[Feature2D] = []
        self._landmarks: Dict[int, Feature3D] = {}
        self._next_feature_id = 0
        self._features_tracked = 0
        self._features_new = 0
        self._features_lost = 0

        # Quality metrics
        self._quality = VOQuality.INVALID
        self._confidence = 0.0
        self._inlier_ratio = 0.0

        # Timing
        self._last_update_time = 0.0
        self._frame_count = 0

        # Pose history for velocity estimation
        self._pose_history: List[Tuple[float, VOPose]] = []
        self._max_history = 10

        # Last reading
        self._last_reading: Optional[VOReading] = None

        # Simulation parameters
        self._true_position = (0.0, 0.0, 0.0)
        self._true_velocity = (0.0, 0.0, 0.0)
        self._true_attitude = (0.0, 0.0, 0.0)  # roll, pitch, yaw

        # Environmental factors
        self._texture_quality = 1.0  # 0-1, affects feature detection
        self._lighting_level = 1.0  # 0-1, affects tracking
        self._motion_blur = 0.0  # 0-1, affects quality

        # Noise parameters
        self._position_noise = 0.01  # meters
        self._rotation_noise = 0.005  # radians
        self._drift_rate = 0.001  # drift per second

        # Scale factor (for monocular, scale is unknown)
        self._scale_estimate = 1.0
        self._scale_initialized = False

        # Lost tracking counter
        self._lost_count = 0

    def start(self):
        """Initialize and start visual odometry."""
        self._is_connected = True
        self._state = VOState.INITIALIZING
        self._last_update_time = time.time()
        self._frame_count = 0

        # Initialize with some features
        self._initialize_features()

        self._state = VOState.TRACKING

    def stop(self):
        """Stop visual odometry."""
        self._is_connected = False
        self._state = VOState.OFF
        self._features_2d.clear()
        self._landmarks.clear()

    def update(self) -> VOReading:
        """
        Process a new frame and update pose estimate.

        Returns:
            VOReading with pose and quality data
        """
        if not self._is_connected:
            return self._create_invalid_reading()

        start_time = time.time()
        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else self._frame_interval

        if dt <= 0:
            dt = self._frame_interval

        # Simulate feature tracking
        self._track_features(dt)

        # Estimate motion from features
        self._estimate_motion(dt)

        # Update quality assessment
        self._assess_quality()

        # Check for tracking loss
        if self._features_tracked < self.MIN_FEATURES:
            self._lost_count += 1
            if self._lost_count > 5:
                self._state = VOState.LOST
        else:
            self._lost_count = 0
            self._state = VOState.TRACKING

        processing_time = time.time() - start_time

        # Create reading
        reading = VOReading(
            pose=VOPose(
                x=self._pose.x, y=self._pose.y, z=self._pose.z,
                roll=self._pose.roll, pitch=self._pose.pitch, yaw=self._pose.yaw,
                vx=self._pose.vx, vy=self._pose.vy, vz=self._pose.vz,
                wx=self._pose.wx, wy=self._pose.wy, wz=self._pose.wz
            ),
            timestamp=current_time,
            quality=self._quality,
            confidence=self._confidence,
            features_tracked=self._features_tracked,
            features_new=self._features_new,
            features_lost=self._features_lost,
            inlier_ratio=self._inlier_ratio,
            landmarks_visible=len([f for f in self._features_2d if f.age > 1]),
            landmarks_total=len(self._landmarks),
            processing_time=processing_time,
            state=self._state
        )

        self._last_reading = reading
        self._last_update_time = current_time
        self._frame_count += 1

        return reading

    def measure(self) -> VOReading:
        """Alias for update()."""
        return self.update()

    def _initialize_features(self):
        """Initialize feature set for first frame."""
        self._features_2d.clear()
        num_features = int(self.TARGET_FEATURES * self._texture_quality * self._lighting_level)
        num_features = max(self.MIN_FEATURES, min(self.MAX_FEATURES, num_features))

        for _ in range(num_features):
            feature = Feature2D(
                x=random.uniform(0, self._resolution[0]),
                y=random.uniform(0, self._resolution[1]),
                id=self._next_feature_id,
                response=random.uniform(0.5, 1.0),
                octave=random.randint(0, 3),
                age=0
            )
            self._features_2d.append(feature)
            self._next_feature_id += 1

        self._features_tracked = len(self._features_2d)
        self._features_new = self._features_tracked

    def _track_features(self, dt: float):
        """Simulate feature tracking between frames."""
        # Calculate tracking success based on conditions
        track_probability = (
            0.95 *  # Base tracking success
            self._texture_quality *
            self._lighting_level *
            (1.0 - self._motion_blur * 0.5)
        )

        tracked = []
        lost = 0

        for feature in self._features_2d:
            if random.random() < track_probability:
                # Feature tracked successfully
                # Simulate feature motion in image based on camera motion
                dx = self._true_velocity[0] * dt * 100  # pixels
                dy = self._true_velocity[1] * dt * 100
                dz = self._true_velocity[2] * dt * 50  # Depth affects motion less

                # Add some noise
                noise_x = random.gauss(0, 1.0)
                noise_y = random.gauss(0, 1.0)

                new_x = feature.x + dx + noise_x
                new_y = feature.y + dy + noise_y

                # Check if still in frame
                if 0 <= new_x < self._resolution[0] and 0 <= new_y < self._resolution[1]:
                    feature.x = new_x
                    feature.y = new_y
                    feature.age += 1
                    tracked.append(feature)
                else:
                    lost += 1
            else:
                lost += 1

        # Add new features to maintain count
        target = int(self.TARGET_FEATURES * self._texture_quality)
        new_count = max(0, target - len(tracked))

        new_features = []
        for _ in range(min(new_count, 50)):  # Max 50 new features per frame
            feature = Feature2D(
                x=random.uniform(0, self._resolution[0]),
                y=random.uniform(0, self._resolution[1]),
                id=self._next_feature_id,
                response=random.uniform(0.5, 1.0),
                octave=random.randint(0, 3),
                age=0
            )
            new_features.append(feature)
            self._next_feature_id += 1

        self._features_2d = tracked + new_features
        self._features_tracked = len(tracked)
        self._features_new = len(new_features)
        self._features_lost = lost

        # Calculate inlier ratio (simulated)
        self._inlier_ratio = self._features_tracked / max(1, self._features_tracked + lost) * 0.9

    def _estimate_motion(self, dt: float):
        """Estimate camera motion from feature tracks."""
        if self._features_tracked < self.MIN_FEATURES:
            return

        # Simulate pose estimation with noise
        # In real system, this would use PnP, essential matrix, etc.

        # Get true motion
        true_dx = self._true_velocity[0] * dt
        true_dy = self._true_velocity[1] * dt
        true_dz = self._true_velocity[2] * dt

        # Add noise and drift
        noise_scale = 1.0 - (self._features_tracked / self.MAX_FEATURES) * 0.5
        drift = self._drift_rate * dt

        dx = true_dx + random.gauss(0, self._position_noise * noise_scale) + drift
        dy = true_dy + random.gauss(0, self._position_noise * noise_scale) + drift
        dz = true_dz + random.gauss(0, self._position_noise * noise_scale) + drift

        # Apply scale (monocular VO has unknown scale)
        if not self._stereo and not self._scale_initialized:
            # Scale is ambiguous in monocular
            self._scale_estimate = 1.0 + random.gauss(0, 0.1)
            self._scale_initialized = True

        scale = self._scale_estimate if not self._stereo else 1.0

        # Update position (transform by current orientation)
        cos_yaw = math.cos(self._pose.yaw)
        sin_yaw = math.sin(self._pose.yaw)

        self._pose.x += (dx * cos_yaw - dy * sin_yaw) * scale
        self._pose.y += (dx * sin_yaw + dy * cos_yaw) * scale
        self._pose.z += dz * scale

        # Update orientation
        self._pose.roll = self._true_attitude[0] + random.gauss(0, self._rotation_noise)
        self._pose.pitch = self._true_attitude[1] + random.gauss(0, self._rotation_noise)
        self._pose.yaw += self._true_attitude[2] * dt + random.gauss(0, self._rotation_noise)

        # Normalize yaw
        while self._pose.yaw > math.pi:
            self._pose.yaw -= 2 * math.pi
        while self._pose.yaw < -math.pi:
            self._pose.yaw += 2 * math.pi

        # Estimate velocity from pose history
        self._update_velocity(dt)

    def _update_velocity(self, dt: float):
        """Update velocity estimate from pose history."""
        current_time = time.time()
        self._pose_history.append((current_time, VOPose(
            x=self._pose.x, y=self._pose.y, z=self._pose.z,
            roll=self._pose.roll, pitch=self._pose.pitch, yaw=self._pose.yaw
        )))

        while len(self._pose_history) > self._max_history:
            self._pose_history.pop(0)

        if len(self._pose_history) >= 2:
            t0, p0 = self._pose_history[0]
            t1, p1 = self._pose_history[-1]
            dt_hist = t1 - t0

            if dt_hist > 0.01:
                self._pose.vx = (p1.x - p0.x) / dt_hist
                self._pose.vy = (p1.y - p0.y) / dt_hist
                self._pose.vz = (p1.z - p0.z) / dt_hist

                # Angular velocity
                dyaw = p1.yaw - p0.yaw
                if dyaw > math.pi:
                    dyaw -= 2 * math.pi
                elif dyaw < -math.pi:
                    dyaw += 2 * math.pi
                self._pose.wz = dyaw / dt_hist

    def _assess_quality(self):
        """Assess tracking quality based on conditions."""
        # Base quality from feature count
        if self._features_tracked >= self.QUALITY_EXCELLENT_THRESHOLD:
            base_quality = 4
        elif self._features_tracked >= self.QUALITY_HIGH_THRESHOLD:
            base_quality = 3
        elif self._features_tracked >= self.QUALITY_MEDIUM_THRESHOLD:
            base_quality = 2
        elif self._features_tracked >= self.QUALITY_LOW_THRESHOLD:
            base_quality = 1
        else:
            base_quality = 0

        # Adjust for conditions
        condition_factor = self._texture_quality * self._lighting_level * (1 - self._motion_blur * 0.3)

        adjusted_quality = base_quality * condition_factor
        if adjusted_quality >= 3.5:
            self._quality = VOQuality.EXCELLENT
        elif adjusted_quality >= 2.5:
            self._quality = VOQuality.HIGH
        elif adjusted_quality >= 1.5:
            self._quality = VOQuality.MEDIUM
        elif adjusted_quality >= 0.5:
            self._quality = VOQuality.LOW
        else:
            self._quality = VOQuality.INVALID

        # Calculate confidence
        feature_conf = min(1.0, self._features_tracked / self.TARGET_FEATURES)
        inlier_conf = self._inlier_ratio
        self._confidence = (feature_conf + inlier_conf) / 2 * condition_factor

    def _create_invalid_reading(self) -> VOReading:
        """Create invalid reading for error states."""
        return VOReading(
            pose=VOPose(),
            timestamp=time.time(),
            quality=VOQuality.INVALID,
            confidence=0.0,
            features_tracked=0,
            features_new=0,
            features_lost=0,
            inlier_ratio=0.0,
            landmarks_visible=0,
            landmarks_total=0,
            processing_time=0.0,
            state=VOState.OFF
        )

    # ==================== Getters ====================

    def get_pose(self) -> VOPose:
        """Get current pose estimate."""
        return self._pose

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z)."""
        return self._pose.get_position()

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity estimate (vx, vy, vz)."""
        return self._pose.get_velocity()

    def get_speed(self) -> float:
        """Get speed magnitude."""
        return self._pose.get_speed()

    def get_state(self) -> VOState:
        """Get tracking state."""
        return self._state

    def get_quality(self) -> VOQuality:
        """Get quality level."""
        return self._quality

    def get_confidence(self) -> float:
        """Get confidence (0-1)."""
        return self._confidence

    def get_feature_count(self) -> int:
        """Get number of tracked features."""
        return self._features_tracked

    def get_frame_count(self) -> int:
        """Get number of processed frames."""
        return self._frame_count

    def is_tracking(self) -> bool:
        """Check if actively tracking."""
        return self._state == VOState.TRACKING

    def is_valid(self) -> bool:
        """Check if last reading is valid."""
        if self._last_reading is None:
            return False
        return self._last_reading.is_valid()

    # ==================== Setters for Simulation ====================

    def set_true_position(self, x: float, y: float, z: float):
        """Set true position for simulation."""
        self._true_position = (x, y, z)

    def set_true_velocity(self, vx: float, vy: float, vz: float):
        """Set true velocity for simulation (m/s)."""
        self._true_velocity = (vx, vy, vz)

    def set_true_attitude(self, roll: float, pitch: float, yaw: float):
        """Set true attitude for simulation (radians)."""
        self._true_attitude = (roll, pitch, yaw)

    def set_texture_quality(self, quality: float):
        """Set environmental texture quality (0-1)."""
        self._texture_quality = max(0.0, min(1.0, quality))

    def set_lighting_level(self, level: float):
        """Set lighting level (0-1)."""
        self._lighting_level = max(0.0, min(1.0, level))

    def set_motion_blur(self, blur: float):
        """Set motion blur level (0-1)."""
        self._motion_blur = max(0.0, min(1.0, blur))

    def simulate_low_texture(self):
        """Simulate low texture environment (e.g., white wall)."""
        self._texture_quality = 0.2

    def simulate_low_light(self):
        """Simulate low light conditions."""
        self._lighting_level = 0.3

    def simulate_fast_motion(self):
        """Simulate fast motion causing blur."""
        self._motion_blur = 0.7

    def simulate_normal_conditions(self):
        """Reset to normal conditions."""
        self._texture_quality = 1.0
        self._lighting_level = 1.0
        self._motion_blur = 0.0

    def reset_pose(self):
        """Reset pose to origin."""
        self._pose = VOPose()
        self._pose_history.clear()

    # ==================== Scale Correction ====================

    def set_scale(self, scale: float):
        """Set scale factor (for monocular VO correction)."""
        self._scale_estimate = scale
        self._scale_initialized = True

    def correct_scale_from_height(self, known_height: float):
        """Correct scale using known height above ground."""
        if abs(self._pose.z) > 0.1:
            self._scale_estimate = known_height / abs(self._pose.z)
            # Apply correction to position
            self._pose.x *= self._scale_estimate
            self._pose.y *= self._scale_estimate
            self._pose.z = known_height

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize sensor data."""
        return {
            'type': self.type,
            'state': self._state.value,
            'pose': self._pose.to_dict(),
            'quality': self._quality.name,
            'confidence': self._confidence,
            'features_tracked': self._features_tracked,
            'frame_count': self._frame_count,
            'stereo': self._stereo,
            'scale': self._scale_estimate,
            'is_tracking': self.is_tracking(),
            'is_valid': self.is_valid(),
            'conditions': {
                'texture_quality': self._texture_quality,
                'lighting_level': self._lighting_level,
                'motion_blur': self._motion_blur
            }
        }
