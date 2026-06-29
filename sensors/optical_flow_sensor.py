"""
Optical Flow sensor module for Drone AI application.

Provides ground-relative velocity estimation using visual motion detection.
Used for low-altitude position hold and GPS-denied navigation.

Navigation Redundancy Strategy:
3. Optical Flow + Rangefinder (low altitude hold) - This module

Typical sensors: PX4FLOW, PMW3901, PAA5100
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from .sensor import Sensor


class OpticalFlowQuality(Enum):
    """Quality level of optical flow measurement."""
    INVALID = 0      # No valid measurement
    LOW = 1          # Marginal quality, use with caution
    MEDIUM = 2       # Acceptable quality
    HIGH = 3         # Good quality
    EXCELLENT = 4    # Excellent quality


class OpticalFlowState(Enum):
    """Operational state of optical flow sensor."""
    OFF = "off"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class OpticalFlowReading:
    """
    Optical flow measurement data.

    Flow is measured in radians/second (angular rate of image motion).
    Ground velocity is computed using height above ground.

    Coordinate frame: Body frame (X forward, Y right)
    """
    # Raw flow rates (rad/s) - angular motion of image
    flow_x: float  # Flow in X direction (forward/backward motion)
    flow_y: float  # Flow in Y direction (left/right motion)

    # Integrated flow (radians) since last reading
    integrated_x: float
    integrated_y: float

    # Ground-relative velocity (m/s) - requires height input
    velocity_x: float  # Forward velocity
    velocity_y: float  # Right velocity

    # Quality metrics
    quality: OpticalFlowQuality
    quality_percent: int  # 0-100%

    # Height used for velocity calculation
    height_agl: float  # meters above ground

    # Timing
    timestamp: float
    integration_time: float  # seconds since last reading

    # Sensor data
    gyro_compensated: bool  # True if gyro rotation has been removed
    surface_valid: bool     # True if surface is suitable for flow

    def is_valid(self) -> bool:
        """Check if reading is valid for navigation."""
        return (
            self.quality != OpticalFlowQuality.INVALID and
            self.quality_percent >= 20 and
            self.surface_valid and
            self.height_agl > 0.1  # Need minimum height for velocity calc
        )

    def get_velocity(self) -> Tuple[float, float]:
        """Get ground velocity as (vx, vy) tuple in m/s."""
        return (self.velocity_x, self.velocity_y)

    def get_speed(self) -> float:
        """Get horizontal ground speed in m/s."""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'flow_x': self.flow_x,
            'flow_y': self.flow_y,
            'integrated_x': self.integrated_x,
            'integrated_y': self.integrated_y,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'quality': self.quality.name,
            'quality_percent': self.quality_percent,
            'height_agl': self.height_agl,
            'timestamp': self.timestamp,
            'integration_time': self.integration_time,
            'gyro_compensated': self.gyro_compensated,
            'surface_valid': self.surface_valid,
            'speed': self.get_speed(),
            'is_valid': self.is_valid()
        }


class OpticalFlowSensor(Sensor):
    """
    Optical Flow sensor for ground-relative velocity estimation.

    Measures the visual motion of the ground surface to estimate
    horizontal velocity. Combined with a height sensor (ultrasonic/lidar),
    provides accurate velocity hold at low altitudes.

    Specifications (based on PMW3901/PX4FLOW):
    - Field of view: ~42 degrees
    - Resolution: 35x35 pixels (internal)
    - Update rate: 50-200 Hz
    - Effective range: 0.1m to 10m height
    - Max flow rate: ~7.4 rad/s (typical)

    Limitations:
    - Requires textured surface (fails on water, uniform surfaces)
    - Affected by lighting conditions
    - Accuracy decreases with height
    - Cannot measure absolute position, only velocity
    """

    # Sensor specifications
    FOV_DEGREES = 42.0           # Field of view
    FOV_RADIANS = math.radians(FOV_DEGREES)
    MAX_FLOW_RATE = 7.4          # rad/s maximum measurable flow
    MIN_HEIGHT = 0.1             # meters - minimum effective height
    MAX_HEIGHT = 10.0            # meters - maximum effective height
    UPDATE_RATE = 100            # Hz default update rate

    # Quality thresholds
    QUALITY_EXCELLENT_THRESHOLD = 80
    QUALITY_HIGH_THRESHOLD = 60
    QUALITY_MEDIUM_THRESHOLD = 40
    QUALITY_LOW_THRESHOLD = 20

    def __init__(self, update_rate: int = UPDATE_RATE):
        """
        Initialize optical flow sensor.

        Args:
            update_rate: Sensor update rate in Hz
        """
        self.type = "optical_flow"
        self._update_rate = update_rate
        self._update_interval = 1.0 / update_rate

        # Sensor state
        self._state = OpticalFlowState.OFF
        self._is_connected = False

        # Raw flow measurements (rad/s)
        self._flow_x = 0.0
        self._flow_y = 0.0

        # Integrated flow (radians)
        self._integrated_x = 0.0
        self._integrated_y = 0.0

        # Computed velocity (m/s)
        self._velocity_x = 0.0
        self._velocity_y = 0.0

        # Height above ground (from external sensor)
        self._height_agl = 1.0

        # Quality metrics
        self._quality = OpticalFlowQuality.INVALID
        self._quality_percent = 0
        self._surface_valid = True

        # Gyro compensation
        self._gyro_compensated = False
        self._gyro_rate_x = 0.0  # Body rotation rate for compensation
        self._gyro_rate_y = 0.0

        # Timing
        self._last_update_time = 0.0
        self._last_reading: Optional[OpticalFlowReading] = None

        # Simulation: true velocity for generating realistic flow
        self._true_velocity_x = 0.0
        self._true_velocity_y = 0.0

        # Noise parameters
        self._flow_noise = 0.05    # rad/s noise std dev
        self._quality_variation = 10  # Quality percent variation

        # Surface simulation
        self._surface_texture = 1.0  # 0-1, affects quality

    def start(self):
        """Initialize and start optical flow sensor."""
        self._is_connected = True
        self._state = OpticalFlowState.INITIALIZING
        self._last_update_time = time.time()

        # Brief initialization
        self._state = OpticalFlowState.READY

    def stop(self):
        """Stop optical flow sensor."""
        self._is_connected = False
        self._state = OpticalFlowState.OFF

    def update(self, height_agl: Optional[float] = None) -> OpticalFlowReading:
        """
        Get latest optical flow measurement.

        Args:
            height_agl: Height above ground in meters (required for velocity)

        Returns:
            OpticalFlowReading with flow and velocity data
        """
        if not self._is_connected:
            return self._create_invalid_reading()

        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else self._update_interval

        if dt <= 0:
            dt = self._update_interval

        # Update height if provided
        if height_agl is not None:
            self._height_agl = max(0.0, height_agl)

        # Simulate optical flow measurement
        self._simulate_flow(dt)

        # Compute velocity from flow and height
        self._compute_velocity()

        # Update quality assessment
        self._assess_quality()

        # Create reading
        reading = OpticalFlowReading(
            flow_x=self._flow_x,
            flow_y=self._flow_y,
            integrated_x=self._integrated_x,
            integrated_y=self._integrated_y,
            velocity_x=self._velocity_x,
            velocity_y=self._velocity_y,
            quality=self._quality,
            quality_percent=self._quality_percent,
            height_agl=self._height_agl,
            timestamp=current_time,
            integration_time=dt,
            gyro_compensated=self._gyro_compensated,
            surface_valid=self._surface_valid,
        )

        self._last_reading = reading
        self._last_update_time = current_time

        # Update state
        if reading.is_valid():
            self._state = OpticalFlowState.ACTIVE
        else:
            self._state = OpticalFlowState.DEGRADED

        return reading

    def measure(self) -> OpticalFlowReading:
        """Alias for update() to match sensor interface."""
        return self.update()

    def _simulate_flow(self, dt: float):
        """
        Simulate optical flow based on true velocity and height.

        Flow rate = velocity / height (simplified pinhole model)
        """
        # Check if we're in valid height range
        if self._height_agl < self.MIN_HEIGHT:
            self._surface_valid = False
            self._flow_x = 0.0
            self._flow_y = 0.0
            return

        if self._height_agl > self.MAX_HEIGHT:
            # Reduce quality at high altitude, flow becomes unreliable
            self._surface_valid = self._surface_texture > 0.5

        # Calculate true flow from velocity and height
        # flow = velocity / height (rad/s)
        if self._height_agl > 0:
            true_flow_x = self._true_velocity_x / self._height_agl
            true_flow_y = self._true_velocity_y / self._height_agl
        else:
            true_flow_x = 0.0
            true_flow_y = 0.0

        # Clamp to sensor limits
        true_flow_x = max(-self.MAX_FLOW_RATE, min(self.MAX_FLOW_RATE, true_flow_x))
        true_flow_y = max(-self.MAX_FLOW_RATE, min(self.MAX_FLOW_RATE, true_flow_y))

        # Add measurement noise (increases with height)
        height_factor = min(1.0, self._height_agl / 5.0)  # More noise at height
        noise_scale = self._flow_noise * (1.0 + height_factor)

        self._flow_x = true_flow_x + random.gauss(0, noise_scale)
        self._flow_y = true_flow_y + random.gauss(0, noise_scale)

        # Apply gyro compensation if available
        if self._gyro_compensated:
            # Remove rotation-induced flow (gyro rates are in deg/s, convert to rad/s)
            self._flow_x -= math.radians(self._gyro_rate_x)
            self._flow_y -= math.radians(self._gyro_rate_y)

        # Integrate flow
        self._integrated_x = self._flow_x * dt
        self._integrated_y = self._flow_y * dt

        # Check surface validity based on texture
        self._surface_valid = (
            self._surface_texture > 0.2 and
            self._height_agl >= self.MIN_HEIGHT
        )

    def _compute_velocity(self):
        """Compute ground velocity from flow and height."""
        if not self._surface_valid or self._height_agl < self.MIN_HEIGHT:
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            return

        # velocity = flow * height (m/s)
        self._velocity_x = self._flow_x * self._height_agl
        self._velocity_y = self._flow_y * self._height_agl

    def _assess_quality(self):
        """Assess measurement quality based on conditions."""
        if not self._surface_valid:
            self._quality = OpticalFlowQuality.INVALID
            self._quality_percent = 0
            return

        # Base quality from surface texture
        base_quality = self._surface_texture * 100

        # Height penalty (quality decreases with height)
        if self._height_agl > self.MAX_HEIGHT:
            height_penalty = 50
        elif self._height_agl > 5.0:
            height_penalty = (self._height_agl - 5.0) * 10
        else:
            height_penalty = 0

        # Low height penalty (too close to ground)
        if self._height_agl < 0.3:
            height_penalty += (0.3 - self._height_agl) * 100

        # Flow rate penalty (high flow = motion blur)
        flow_magnitude = math.sqrt(self._flow_x**2 + self._flow_y**2)
        if flow_magnitude > self.MAX_FLOW_RATE * 0.8:
            flow_penalty = 30
        elif flow_magnitude > self.MAX_FLOW_RATE * 0.5:
            flow_penalty = 15
        else:
            flow_penalty = 0

        # Compute final quality
        quality_raw = base_quality - height_penalty - flow_penalty
        quality_raw += random.gauss(0, self._quality_variation)
        self._quality_percent = int(max(0, min(100, quality_raw)))

        # Map to quality enum
        if self._quality_percent >= self.QUALITY_EXCELLENT_THRESHOLD:
            self._quality = OpticalFlowQuality.EXCELLENT
        elif self._quality_percent >= self.QUALITY_HIGH_THRESHOLD:
            self._quality = OpticalFlowQuality.HIGH
        elif self._quality_percent >= self.QUALITY_MEDIUM_THRESHOLD:
            self._quality = OpticalFlowQuality.MEDIUM
        elif self._quality_percent >= self.QUALITY_LOW_THRESHOLD:
            self._quality = OpticalFlowQuality.LOW
        else:
            self._quality = OpticalFlowQuality.INVALID

    def _create_invalid_reading(self) -> OpticalFlowReading:
        """Create an invalid reading for error states."""
        return OpticalFlowReading(
            flow_x=0.0,
            flow_y=0.0,
            integrated_x=0.0,
            integrated_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            quality=OpticalFlowQuality.INVALID,
            quality_percent=0,
            height_agl=0.0,
            timestamp=time.time(),
            integration_time=0.0,
            gyro_compensated=False,
            surface_valid=False,
        )

    # ==================== Getters ====================

    def get_flow(self) -> Tuple[float, float]:
        """Get current flow rates (rad/s)."""
        return (self._flow_x, self._flow_y)

    def get_velocity(self) -> Tuple[float, float]:
        """Get ground velocity (m/s)."""
        return (self._velocity_x, self._velocity_y)

    def get_speed(self) -> float:
        """Get horizontal ground speed (m/s)."""
        return math.sqrt(self._velocity_x**2 + self._velocity_y**2)

    def get_quality(self) -> OpticalFlowQuality:
        """Get current quality level."""
        return self._quality

    def get_quality_percent(self) -> int:
        """Get quality as percentage (0-100)."""
        return self._quality_percent

    def get_state(self) -> OpticalFlowState:
        """Get sensor state."""
        return self._state

    def is_valid(self) -> bool:
        """Check if last reading is valid."""
        if self._last_reading is None:
            return False
        return self._last_reading.is_valid()

    # ==================== Setters for Simulation ====================

    def set_true_velocity(self, vx: float, vy: float):
        """Set true ground velocity for simulation (m/s)."""
        self._true_velocity_x = vx
        self._true_velocity_y = vy

    def set_height(self, height_agl: float):
        """Set height above ground (m)."""
        self._height_agl = max(0.0, height_agl)

    def set_surface_texture(self, texture: float):
        """Set surface texture quality (0.0 to 1.0)."""
        self._surface_texture = max(0.0, min(1.0, texture))

    def set_gyro_rates(self, rate_x: float, rate_y: float):
        """
        Set gyro rotation rates for compensation (deg/s).

        Args:
            rate_x: Roll rate
            rate_y: Pitch rate
        """
        self._gyro_rate_x = rate_x
        self._gyro_rate_y = rate_y
        self._gyro_compensated = True

    def enable_gyro_compensation(self, enabled: bool = True):
        """Enable or disable gyro compensation."""
        self._gyro_compensated = enabled

    def simulate_surface_loss(self):
        """Simulate loss of surface texture (e.g., over water)."""
        self._surface_texture = 0.0
        self._surface_valid = False

    def simulate_surface_recovery(self):
        """Restore normal surface texture."""
        self._surface_texture = 1.0
        self._surface_valid = True

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize sensor data to dictionary."""
        return {
            'type': self.type,
            'state': self._state.value,
            'flow': {
                'x': self._flow_x,
                'y': self._flow_y,
            },
            'velocity': {
                'x': self._velocity_x,
                'y': self._velocity_y,
                'speed': self.get_speed(),
            },
            'quality': {
                'level': self._quality.name,
                'percent': self._quality_percent,
            },
            'height_agl': self._height_agl,
            'surface_valid': self._surface_valid,
            'gyro_compensated': self._gyro_compensated,
            'is_valid': self.is_valid(),
            'update_rate': self._update_rate,
        }
