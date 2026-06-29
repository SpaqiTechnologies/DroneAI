"""
GPS sensor module for Drone AI application.
Handles GPS position tracking with accuracy metrics and satellite information.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from .sensor import Sensor


@dataclass
class GPSReading:
    """Represents a GPS reading with position and metadata."""
    latitude: float
    longitude: float
    altitude: float  # meters above sea level
    accuracy_horizontal: float  # meters
    accuracy_vertical: float  # meters
    satellites_used: int
    fix_type: int  # 0=no fix, 1=2D fix, 2=3D fix
    timestamp: float
    speed: float  # m/s
    heading: float  # degrees (0-360)

    def is_valid(self) -> bool:
        """Check if GPS reading is valid."""
        return (
            self.fix_type >= 1
            and self.satellites_used >= 4
            and self.accuracy_horizontal < 50.0  # Less than 50m horizontal accuracy
        )


class GPSSensor(Sensor):
    """
    GPS sensor for position tracking with accuracy metrics.
    Simulates GPS receiver behavior with realistic characteristics.
    """

    # GPS fix types
    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2

    def __init__(self):
        self.type = "gps"
        self._latitude = 0.0
        self._longitude = 0.0
        self._altitude = 0.0
        self._accuracy_horizontal = 5.0  # meters
        self._accuracy_vertical = 10.0  # meters
        self._satellites_used = 8
        self._fix_type = self.FIX_3D
        self._speed = 0.0
        self._heading = 0.0
        self._last_reading: Optional[GPSReading] = None
        self._last_update_time = 0.0
        self._is_connected = True
        self._signal_quality = 1.0  # 0.0 to 1.0

    def start(self):
        """Initialize GPS sensor."""
        self._is_connected = True
        self._last_update_time = time.time()

    def stop(self):
        """Stop GPS sensor."""
        self._is_connected = False

    def update(self) -> GPSReading:
        """Fetch latest GPS measurement."""
        if not self._is_connected:
            return self._create_invalid_reading()

        # Simulate GPS drift and accuracy variations
        self._simulate_gps_behavior()

        reading = GPSReading(
            latitude=self._latitude,
            longitude=self._longitude,
            altitude=self._altitude,
            accuracy_horizontal=self._accuracy_horizontal,
            accuracy_vertical=self._accuracy_vertical,
            satellites_used=self._satellites_used,
            fix_type=self._fix_type,
            timestamp=time.time(),
            speed=self._speed,
            heading=self._heading,
        )

        self._last_reading = reading
        self._last_update_time = time.time()
        return reading

    def measure(self) -> GPSReading:
        """Alias for update() to match sensor interface."""
        return self.update()

    def _simulate_gps_behavior(self):
        """Simulate realistic GPS behavior including drift and accuracy changes."""
        # Simulate satellite count variation (4-12 satellites)
        self._satellites_used = max(0, min(12, self._satellites_used + random.randint(-1, 1)))

        # Update fix type based on satellites
        if self._satellites_used < 3:
            self._fix_type = self.NO_FIX
        elif self._satellites_used < 4:
            self._fix_type = self.FIX_2D
        else:
            self._fix_type = self.FIX_3D

        # Simulate accuracy based on satellite count and signal quality
        base_accuracy = 3.0 + (12 - self._satellites_used) * 2.0
        self._accuracy_horizontal = base_accuracy / self._signal_quality
        self._accuracy_vertical = self._accuracy_horizontal * 1.5

        # Add small random drift to position (simulates GPS noise)
        if self._fix_type >= self.FIX_2D:
            drift_factor = self._accuracy_horizontal / 111000  # Convert meters to degrees
            self._latitude += random.gauss(0, drift_factor * 0.1)
            self._longitude += random.gauss(0, drift_factor * 0.1)

    def _create_invalid_reading(self) -> GPSReading:
        """Create an invalid GPS reading for error states."""
        return GPSReading(
            latitude=float('nan'),
            longitude=float('nan'),
            altitude=float('nan'),
            accuracy_horizontal=float('inf'),
            accuracy_vertical=float('inf'),
            satellites_used=0,
            fix_type=self.NO_FIX,
            timestamp=time.time(),
            speed=0.0,
            heading=0.0,
        )

    def is_valid(self) -> bool:
        """Check if last reading is valid."""
        if self._last_reading is None:
            return False
        return self._last_reading.is_valid()

    def has_fix(self) -> bool:
        """Check if GPS has a valid fix."""
        return self._fix_type >= self.FIX_2D

    def has_3d_fix(self) -> bool:
        """Check if GPS has a 3D fix (includes altitude)."""
        return self._fix_type == self.FIX_3D

    def get_position(self) -> Tuple[float, float]:
        """Get current position as (latitude, longitude) tuple."""
        return (self._latitude, self._longitude)

    def get_altitude(self) -> float:
        """Get current altitude in meters."""
        return self._altitude

    def get_accuracy(self) -> Tuple[float, float]:
        """Get accuracy as (horizontal, vertical) in meters."""
        return (self._accuracy_horizontal, self._accuracy_vertical)

    def get_satellites(self) -> int:
        """Get number of satellites used."""
        return self._satellites_used

    def get_speed(self) -> float:
        """Get current speed in m/s."""
        return self._speed

    def get_heading(self) -> float:
        """Get current heading in degrees (0-360)."""
        return self._heading

    def get_signal_quality(self) -> float:
        """Get GPS signal quality (0.0 to 1.0)."""
        return self._signal_quality

    # Setters for testing and simulation
    def set_position(self, latitude: float, longitude: float, altitude: float = 0.0):
        """Set GPS position for testing."""
        self._latitude = latitude
        self._longitude = longitude
        self._altitude = altitude

    def set_accuracy(self, horizontal: float, vertical: float = None):
        """Set GPS accuracy for testing."""
        self._accuracy_horizontal = horizontal
        self._accuracy_vertical = vertical if vertical else horizontal * 1.5

    def set_satellites(self, count: int):
        """Set satellite count for testing."""
        self._satellites_used = max(0, min(12, count))
        # Update fix type based on satellites
        if self._satellites_used < 3:
            self._fix_type = self.NO_FIX
        elif self._satellites_used < 4:
            self._fix_type = self.FIX_2D
        else:
            self._fix_type = self.FIX_3D

    def set_fix_type(self, fix_type: int):
        """Set fix type for testing."""
        self._fix_type = fix_type

    def set_signal_quality(self, quality: float):
        """Set signal quality for testing (0.0 to 1.0)."""
        self._signal_quality = max(0.0, min(1.0, quality))

    def set_speed(self, speed: float):
        """Set speed for testing."""
        self._speed = max(0.0, speed)

    def set_heading(self, heading: float):
        """Set heading for testing (0-360)."""
        self._heading = heading % 360

    def simulate_signal_loss(self):
        """Simulate GPS signal loss for testing."""
        self._satellites_used = 0
        self._fix_type = self.NO_FIX
        self._signal_quality = 0.0

    def simulate_signal_recovery(self):
        """Simulate GPS signal recovery for testing."""
        self._satellites_used = 8
        self._fix_type = self.FIX_3D
        self._signal_quality = 1.0

    def to_dict(self) -> dict:
        """Serialize GPS data to dictionary."""
        return {
            'type': self.type,
            'latitude': self._latitude,
            'longitude': self._longitude,
            'altitude': self._altitude,
            'accuracy_horizontal': self._accuracy_horizontal,
            'accuracy_vertical': self._accuracy_vertical,
            'satellites_used': self._satellites_used,
            'fix_type': self._fix_type,
            'speed': self._speed,
            'heading': self._heading,
            'signal_quality': self._signal_quality,
            'is_valid': self.is_valid(),
        }
