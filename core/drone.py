"""
Main Drone class for managing drone operations and path planning.
"""

import math
from typing import Tuple, List
from sensors.sensor_manager import SensorManager
from sensors.wind_sensor import WindSensor
from sensors.battery_sensor import BatterySensor
from core.exceptions import SensorConnectionLostException, LowBatteryException


class Drone:
    """
    Main drone class that manages flight operations and path planning.
    """

    def __init__(self):
        self.sensor_manager = SensorManager()
        self.current_position = (0.0, 0.0)  # (latitude, longitude)
        self.home_position = (0.0, 0.0)  # (latitude, longitude)
        self.battery_level = 100.0  # Percentage
        self._battery_level_override = None  # For testing purposes

    def update_sensors(self):
        """Update all sensor readings."""
        # Check if sensors are connected
        if not self.sensor_manager.check_sensor_connection():
            raise SensorConnectionLostException("Sensor connection lost")

        try:
            self.wind_speed = self.wind_sensor.get_wind_speed()
            self.wind_direction = self.wind_sensor.get_wind_direction()

            # Use overridden battery level for testing, otherwise get from sensor
            if self._battery_level_override is not None:
                self.battery_level = self._battery_level_override
            else:
                self.battery_level = self.battery_sensor.get_battery_level()
        except Exception as e:
            print(f"Error updating sensors: {e}")
            raise SensorConnectionLostException("Failed to connect to sensors")

    def set_home_position(self, latitude: float, longitude: float):
        """Set the home position for the drone."""
        self.home_position = (latitude, longitude)

    def set_current_position(self, latitude: float, longitude: float):
        """Set the current position of the drone."""
        self.current_position = (latitude, longitude)

    def set_battery_level_override(self, level: float):
        """Set battery level override for testing purposes."""
        self._battery_level_override = level

    def calculate_return_to_home(self) -> List[Tuple[float, float]]:
        """
        Calculate the safest return-to-home route based on current wind speed and battery percentage.

        Returns:
            List of waypoints (latitude, longitude) for the return path
        """
        # Update sensor data first
        self.update_sensors()

        # Check for low battery condition after updating sensors
        if self.battery_level < 10:
            raise LowBatteryException("Battery level critically low for safe flight")

        # Calculate distance to home
        distance_to_home = self._calculate_distance(
            self.current_position, self.home_position
        )

        # Calculate safe return path considering wind
        safe_path = self._calculate_safe_path(distance_to_home)

        return safe_path

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two GPS coordinates using Haversine formula."""
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # Earth radius in meters
        R = 6371000

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        # Haversine formula
        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(
            lat1_rad
        ) * math.cos(lat2_rad) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_safe_path(
        self, distance_to_home: float
    ) -> List[Tuple[float, float]]:
        """
        Calculate a safe path considering wind conditions and battery level.

        Args:
            distance_to_home: Distance to home in meters

        Returns:
            List of waypoints for the return path
        """
        # Base path is direct to home
        waypoints = [self.home_position]

        # Add ultrasonic obstacle checks globally
        home_to_current = self._calculate_distance(
            self.home_position, self.current_position
        )
        
        # Safety: if ultrasonic is valid and distance < min_clearance, trigger avoidance
        ultra_dist_m = self.sensor_manager.get("ultrasonic").get_distance()
        if not math.isnan(ultra_dist_m) and ultra_dist_m < 1.5:  # 1.5m clearance
            waypoints = self._add_obstacle_avoid_waypoint(
                waypoints, self.current_position, ultra_dist_m
            )

        # Keep existing wind/battery behavior
        if self.wind_speed > 10:  # High wind condition
            waypoints = self._add_wind_detour(waypoints)

        if self.battery_level < 20:
            waypoints = self._reduce_path_for_battery(waypoints)

        return waypoints

    def _add_wind_detour(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Add a detour to avoid strong wind conditions."""
        # For simplicity, we'll just add a waypoint that's 100m away from the direct path
        # In a real implementation, this would be more sophisticated
        detour_waypoint = self._calculate_detour_point(waypoints[0])
        waypoints.insert(0, detour_waypoint)
        return waypoints

    def _calculate_detour_point(
        self, target_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Calculate a detour point to avoid wind."""
        # Simple detour: move 100m perpendicular to the wind direction
        # This is a simplified implementation
        lat, lon = target_point
        # In a real implementation, this would use proper geodesic calculations
        return (lat + 0.001, lon + 0.001)

    def _reduce_path_for_battery(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Reduce path length to conserve battery when battery is low."""
        # For simplicity, we'll reduce the path to just the home position
        # In a real implementation, this would be more sophisticated
        return [waypoints[-1]]  # Return only the final waypoint (home)
