"""
Unit tests for Drone AI application.
Tests the Battery-Aware Path Planner with various edge cases.
"""

import unittest
from unittest.mock import patch, MagicMock
from core.drone import Drone
from sensors.wind_sensor import WindSensor
from sensors.battery_sensor import BatterySensor
from sensors.sensor_manager import SensorManager
from core.exceptions import SensorConnectionLostException, LowBatteryException


class TestDrone(unittest.TestCase):
    """Test cases for the Drone class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.drone = Drone()
        self.drone.set_home_position(52.52, 13.405)  # Berlin coordinates
        self.drone.set_current_position(52.53, 13.415)  # Near Berlin

    def test_normal_path_planning(self):
        """Test normal path planning with good conditions."""
        # Set normal wind and battery levels
        self.drone.wind_sensor.set_wind_speed(5.0)
        self.drone.battery_sensor.set_battery_level(80.0)

        # Calculate return path
        path = self.drone.calculate_return_to_home()

        # Should return a path (list of waypoints)
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

    def test_high_wind_conditions(self):
        """Test path planning with high wind conditions."""
        # Set up a scenario with high wind using mocking
        with patch.object(self.drone.wind_sensor, "get_wind_speed", return_value=15.0):
            with patch.object(
                self.drone.battery_sensor, "get_battery_level", return_value=80.0
            ):
                # Calculate return path
                path = self.drone.calculate_return_to_home()

                # Should return a path even with high wind
                self.assertIsInstance(path, list)
                self.assertGreater(len(path), 0)

    def test_low_battery_conditions(self):
        """Test path planning with low battery conditions."""
        # Set low battery level
        self.drone.battery_sensor.set_battery_level(5.0)
        self.drone.wind_sensor.set_wind_speed(5.0)

        # Should raise LowBatteryException
        with self.assertRaises(LowBatteryException):
            self.drone.calculate_return_to_home()

    def test_sensor_connection_lost(self):
        """Test handling of sensor connection lost."""
        # Simulate sensor failure
        self.drone.sensor_manager.simulate_sensor_failure()

        # Should raise SensorConnectionLostException
        with self.assertRaises(SensorConnectionLostException):
            self.drone.update_sensors()

    def test_gps_drift_simulation(self):
        """Test path planning with simulated GPS drift."""
        # Set up a scenario with GPS drift (different positions)
        self.drone.set_current_position(52.525, 13.410)  # Slightly different position
        # Use mocking to set wind and battery levels
        with patch.object(self.drone.wind_sensor, "get_wind_speed", return_value=5.0):
            with patch.object(
                self.drone.battery_sensor, "get_battery_level", return_value=80.0
            ):
                # Calculate return path
                path = self.drone.calculate_return_to_home()

                # Should return a path
                self.assertIsInstance(path, list)
                self.assertGreater(len(path), 0)

    def test_critical_low_battery(self):
        """Test path planning with critically low battery."""
        # Set up a scenario where sensors will return critically low battery
        # We'll mock the battery sensor to return a very low value
        with patch.object(
            self.drone.battery_sensor, "get_battery_level", return_value=0.0
        ):
            self.drone.wind_sensor.set_wind_speed(5.0)

            # Should raise LowBatteryException
            with self.assertRaises(LowBatteryException):
                self.drone.calculate_return_to_home()


if __name__ == "__main__":
    unittest.main()
