"""
Sensor manager for Drone AI application.
Manages all sensor connections and data collection.
"""

from typing import Dict, List, Optional, Tuple
import logging

from . import battery_sensor, wind_sensor, ultrasonic_sensor

# Import Drone class for type hints (optional)
from core.drone import Drone
from .sensor import Sensor

logger = logging.getLogger(__name__)


class SensorManager:
    """
    Manages all sensor connections, registration, and data collection for the drone.
    Provides connection status and measured values.
    """

    def __init__(self):
        self._sensors: Dict[str, Sensor] = {}
        self._sensor_values: Dict[str, object] = {}
        self._last_update = 0.0

        # Initialize sensors
        self.register_sensor(battery_sensor.BatterySensor())
        self.register_sensor(wind_sensor.WindSensor())
        self.register_sensor(ultrasonic_sensor.UltrasonicSensor())

    def register_sensor(self, sensor: Sensor):
        """Register a sensor by its type."""
        if not isinstance(sensor, Sensor):
            raise ValueError(f"Sensor must inherit from Sensor base class")

        sensor_type = sensor.type
        if sensor_type in self._sensors:
            raise ValueError(f"Sensor type '{sensor_type}' already registered")

        self._sensors[sensor_type] = sensor
        logger.info(f"Registered sensor: {sensor_type}")

    def start(self):
        """Start all sensors."""
        for sensor in self._sensors.values():
            try:
                sensor.start()
            except Exception as e:
                logger.error(f"Failed to start sensor {sensor.type}: {str(e)}")
                raise

    def stop(self):
        """Stop all sensors."""
        for sensor in reversed(self._sensors.values()):
            try:
                sensor.stop()
            except Exception as e:
                logger.error(f"Failed to stop sensor {sensor.type}: {str(e)}")

    def update(self, timestamp: float):
        """Update all sensors and store last values."""
        self._last_update = timestamp
        for sensor in self._sensors.values():
            try:
                value = sensor.measure()
                self._sensor_values[sensor.type] = value
            except Exception as e:
                logger.warning(f"Sensor {sensor.type} update failed: {str(e)}")
                # Keep last known good value by default behavior of _sensor_values

    def get(self, sensor_type: str) -> Optional[object]:
        """Get the latest measured value for a sensor type."""
        return self._sensor_values.get(sensor_type)

    def get_all(self) -> Dict[str, object]:
        """Get all latest sensor values."""
        return dict(self._sensor_values.items())

    def set_calibration(self, sensor_type: str, cal_data):
        """Set calibration data for a sensor."""
        if sensor_type not in self._sensors:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        self._sensors[sensor_type].set_calibration(cal_data)

    def get_config(self, sensor_type: str):
        """Get sensor configuration."""
        if sensor_type not in self._sensors:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        return self._sensors[sensor_type].get_config()

    def reset(self):
        """Reset sensor values (not hardware)."""
        self._sensor_values.clear()
