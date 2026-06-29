"""
Battery sensor module for Drone AI application.
Handles battery level data collection.
"""

import random

from .sensor import Sensor


class BatterySensor(Sensor):
    """
    Simulates battery sensor data collection.
    """

    def __init__(self):
        self.type = "battery"
        self.battery_level = 100.0  # Percentage
        self.voltage = 12.6  # 3S LiPo, fully charged
        self.current = 0.0  # Amps

    def start(self):
        """Start the sensor."""
        pass

    def stop(self):
        """Stop the sensor."""
        pass

    def update(self):
        """Fetch latest measurement."""
        self.battery_level = random.uniform(0, 100)
        self._update_electrical_state()
        return self.battery_level

    def measure(self):
        """Alias for update() to match sensor interface."""
        return self.update()

    def is_valid(self):
        """True if last reading is valid."""
        return 0 <= self.battery_level <= 100

    def get_battery_level(self) -> float:
        """Get current battery level in percentage."""
        # In a real implementation, this would read from an actual sensor
        # For simulation, we'll return a random value between 0 and 100
        self.battery_level = random.uniform(0, 100)
        self._update_electrical_state()
        return self.battery_level

    def get_percent(self) -> float:
        """Get the last battery level without mutating it."""
        return self.battery_level

    def get_voltage(self) -> float:
        """Get estimated pack voltage from battery level."""
        self._update_electrical_state()
        return self.voltage

    def get_current(self) -> float:
        """Get estimated current draw."""
        return self.current

    def set_battery_level(self, level: float):
        """Set battery level for testing purposes."""
        self.battery_level = max(0.0, min(100.0, level))
        self._update_electrical_state()

    def _update_electrical_state(self):
        """Update simulated voltage/current values from percentage."""
        # Approximate 3S LiPo discharge curve: 9.9V empty to 12.6V full.
        self.voltage = 9.9 + (self.battery_level / 100.0) * 2.7
        self.current = 8.0 if self.battery_level > 0 else 0.0
