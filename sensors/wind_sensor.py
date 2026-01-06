"""
Wind sensor module for Drone AI application.
Handles wind speed and direction data collection.
"""

import random


class WindSensor:
    """
    Simulates wind sensor data collection.
    """

    def __init__(self):
        self.wind_speed = 0.0  # m/s
        self.wind_direction = 0.0  # degrees

    def get_wind_speed(self) -> float:
        """Get current wind speed in m/s."""
        # In a real implementation, this would read from an actual sensor
        # For simulation, we'll return a random value between 0 and 20 m/s
        self.wind_speed = random.uniform(0, 20)
        return self.wind_speed

    def get_wind_direction(self) -> float:
        """Get current wind direction in degrees."""
        # In a real implementation, this would read from an actual sensor
        # For simulation, we'll return a random value between 0 and 360 degrees
        self.wind_direction = random.uniform(0, 360)
        return self.wind_direction

    def set_wind_speed(self, speed: float):
        """Set wind speed for testing purposes."""
        self.wind_speed = speed

    def set_wind_direction(self, direction: float):
        """Set wind direction for testing purposes."""
        self.wind_direction = direction
