"""
Battery sensor module for Drone AI application.
Handles battery level data collection.
"""

import random


class BatterySensor:
    """
    Simulates battery sensor data collection.
    """

    def __init__(self):
        self.battery_level = 100.0  # Percentage

    def get_battery_level(self) -> float:
        """Get current battery level in percentage."""
        # In a real implementation, this would read from an actual sensor
        # For simulation, we'll return a random value between 0 and 100
        self.battery_level = random.uniform(0, 100)
        return self.battery_level

    def set_battery_level(self, level: float):
        """Set battery level for testing purposes."""
        self.battery_level = level
