"""
Survey Pattern Generator

Generates mission waypoints for common survey patterns:
- Grid/Lawnmower pattern for area coverage
- Orbit pattern around a point of interest
- Corridor survey along a path
- Expanding square search pattern
"""

from typing import List, Tuple, Optional
import math

from .waypoint import Waypoint, WaypointType, WaypointActionItem, WaypointAction
from .mission import Mission, MissionType


class SurveyPatternGenerator:
    """
    Generates survey mission patterns.

    Supports various patterns for mapping, inspection, and search operations.
    """

    # Earth radius in meters
    EARTH_RADIUS = 6371000.0

    def generate_grid(
        self,
        center: Tuple[float, float],
        width: float,
        height: float,
        altitude: float,
        line_spacing: float,
        overshoot: float = 20.0,
        speed: float = 5.0,
        direction: float = 0.0,
        camera_trigger: bool = True,
        name: str = "Grid Survey"
    ) -> Mission:
        """
        Generate a grid/lawnmower survey pattern.

        Args:
            center: Center point (lat, lon)
            width: Survey width in meters (perpendicular to lines)
            height: Survey height in meters (along lines)
            altitude: Flight altitude in meters
            line_spacing: Spacing between survey lines in meters
            overshoot: Extra distance at line ends for turns
            speed: Flight speed in m/s
            direction: Survey direction in degrees (0 = North-South lines)
            camera_trigger: Add photo triggers at each waypoint
            name: Mission name

        Returns:
            Mission with grid pattern waypoints
        """
        mission = Mission(
            name=name,
            mission_type=MissionType.SURVEY,
            default_speed=speed,
            default_altitude=altitude,
        )

        # Calculate number of lines
        num_lines = max(1, int(width / line_spacing) + 1)

        # Calculate line endpoints
        waypoints = []
        direction_rad = math.radians(direction)
        perpendicular_rad = direction_rad + math.pi / 2

        for i in range(num_lines):
            # Offset from center for this line
            offset = (i - (num_lines - 1) / 2) * line_spacing

            # Calculate line start and end points
            # Move perpendicular to direction for line offset
            offset_lat = offset * math.cos(perpendicular_rad) / 111320
            offset_lon = offset * math.sin(perpendicular_rad) / (111320 * math.cos(math.radians(center[0])))

            line_center_lat = center[0] + offset_lat
            line_center_lon = center[1] + offset_lon

            # Calculate line endpoints along direction
            half_length = (height / 2 + overshoot)
            start_lat = line_center_lat - half_length * math.cos(direction_rad) / 111320
            start_lon = line_center_lon - half_length * math.sin(direction_rad) / (111320 * math.cos(math.radians(center[0])))
            end_lat = line_center_lat + half_length * math.cos(direction_rad) / 111320
            end_lon = line_center_lon + half_length * math.sin(direction_rad) / (111320 * math.cos(math.radians(center[0])))

            # Alternate direction for efficiency
            if i % 2 == 0:
                waypoints.append((start_lat, start_lon))
                waypoints.append((end_lat, end_lon))
            else:
                waypoints.append((end_lat, end_lon))
                waypoints.append((start_lat, start_lon))

        # Create waypoints
        for i, (lat, lon) in enumerate(waypoints):
            wp = Waypoint(
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                sequence=i,
                speed=speed,
                pass_through=True,
                waypoint_type=WaypointType.NAVIGATE,
            )

            if camera_trigger:
                wp.actions.append(WaypointActionItem.take_photo())

            mission.add_waypoint(wp)

        return mission

    def generate_orbit(
        self,
        center: Tuple[float, float],
        radius: float,
        altitude: float,
        num_points: int = 12,
        turns: float = 1.0,
        direction: int = 1,
        speed: float = 5.0,
        camera_trigger: bool = True,
        face_center: bool = True,
        name: str = "Orbit Survey"
    ) -> Mission:
        """
        Generate an orbit pattern around a point of interest.

        Args:
            center: Center point to orbit (lat, lon)
            radius: Orbit radius in meters
            altitude: Flight altitude in meters
            num_points: Number of waypoints per orbit
            turns: Number of complete orbits
            direction: 1 for clockwise, -1 for counter-clockwise
            speed: Flight speed in m/s
            camera_trigger: Add photo triggers
            face_center: Point camera toward center (set yaw)
            name: Mission name

        Returns:
            Mission with orbit pattern waypoints
        """
        mission = Mission(
            name=name,
            mission_type=MissionType.ORBIT,
            default_speed=speed,
            default_altitude=altitude,
        )

        total_points = int(num_points * turns)
        angle_step = direction * 2 * math.pi / num_points

        for i in range(total_points):
            angle = i * angle_step

            # Calculate point on circle
            dlat = radius * math.cos(angle) / 111320
            dlon = radius * math.sin(angle) / (111320 * math.cos(math.radians(center[0])))

            lat = center[0] + dlat
            lon = center[1] + dlon

            # Calculate yaw to face center
            yaw = None
            if face_center:
                yaw = (math.degrees(angle) + 180) % 360

            wp = Waypoint(
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                sequence=i,
                speed=speed,
                pass_through=True,
                yaw=yaw,
                waypoint_type=WaypointType.NAVIGATE,
            )

            if camera_trigger:
                wp.actions.append(WaypointActionItem.take_photo())

            mission.add_waypoint(wp)

        return mission

    def generate_corridor(
        self,
        path: List[Tuple[float, float]],
        altitude: float,
        corridor_width: float,
        speed: float = 5.0,
        camera_trigger: bool = True,
        name: str = "Corridor Survey"
    ) -> Mission:
        """
        Generate a corridor survey along a path.

        Creates parallel lines along the path for coverage.

        Args:
            path: List of points defining the corridor center line
            altitude: Flight altitude in meters
            corridor_width: Width of corridor in meters
            speed: Flight speed in m/s
            camera_trigger: Add photo triggers
            name: Mission name

        Returns:
            Mission with corridor pattern waypoints
        """
        mission = Mission(
            name=name,
            mission_type=MissionType.CORRIDOR,
            default_speed=speed,
            default_altitude=altitude,
        )

        if len(path) < 2:
            return mission

        # Generate waypoints along center line first
        for i, (lat, lon) in enumerate(path):
            wp = Waypoint(
                latitude=lat,
                longitude=lon,
                altitude=altitude,
                sequence=i,
                speed=speed,
                pass_through=True,
            )

            if camera_trigger:
                wp.actions.append(WaypointActionItem.take_photo())

            mission.add_waypoint(wp)

        # If corridor width > 0, add parallel lines
        if corridor_width > 0:
            # Add offset lines (simplified - just offset each point)
            offset = corridor_width / 2

            # Right side line (reversed for efficiency)
            for i in range(len(path) - 1, -1, -1):
                lat, lon = path[i]

                # Calculate perpendicular offset
                if i < len(path) - 1:
                    next_lat, next_lon = path[i + 1]
                    bearing = math.atan2(next_lon - lon, next_lat - lat)
                elif i > 0:
                    prev_lat, prev_lon = path[i - 1]
                    bearing = math.atan2(lon - prev_lon, lat - prev_lat)
                else:
                    bearing = 0

                perp_bearing = bearing + math.pi / 2
                dlat = offset * math.cos(perp_bearing) / 111320
                dlon = offset * math.sin(perp_bearing) / (111320 * math.cos(math.radians(lat)))

                wp = Waypoint(
                    latitude=lat + dlat,
                    longitude=lon + dlon,
                    altitude=altitude,
                    sequence=len(mission.waypoints),
                    speed=speed,
                    pass_through=True,
                )

                if camera_trigger:
                    wp.actions.append(WaypointActionItem.take_photo())

                mission.add_waypoint(wp)

        return mission

    def generate_expanding_square(
        self,
        center: Tuple[float, float],
        altitude: float,
        initial_leg: float,
        expansion: float,
        max_legs: int = 20,
        speed: float = 5.0,
        direction: int = 1,
        name: str = "Expanding Square Search"
    ) -> Mission:
        """
        Generate an expanding square search pattern.

        Used for search and rescue operations.

        Args:
            center: Starting point (lat, lon)
            altitude: Flight altitude in meters
            initial_leg: Initial leg length in meters
            expansion: How much to expand each leg in meters
            max_legs: Maximum number of legs
            speed: Flight speed in m/s
            direction: 1 for clockwise, -1 for counter-clockwise
            name: Mission name

        Returns:
            Mission with expanding square pattern
        """
        mission = Mission(
            name=name,
            mission_type=MissionType.SEARCH,
            default_speed=speed,
            default_altitude=altitude,
        )

        # Start at center
        current_lat = center[0]
        current_lon = center[1]

        # Add starting point
        wp = Waypoint(
            latitude=current_lat,
            longitude=current_lon,
            altitude=altitude,
            sequence=0,
            speed=speed,
        )
        mission.add_waypoint(wp)

        # Generate expanding square
        leg_length = initial_leg
        heading = 0  # Start heading north

        for leg_num in range(max_legs):
            # Move in current direction
            dlat = leg_length * math.cos(math.radians(heading)) / 111320
            dlon = leg_length * math.sin(math.radians(heading)) / (111320 * math.cos(math.radians(current_lat)))

            current_lat += dlat
            current_lon += dlon

            wp = Waypoint(
                latitude=current_lat,
                longitude=current_lon,
                altitude=altitude,
                sequence=leg_num + 1,
                speed=speed,
                pass_through=True,
            )
            mission.add_waypoint(wp)

            # Turn 90 degrees
            heading = (heading + 90 * direction) % 360

            # Expand leg every two legs
            if leg_num % 2 == 1:
                leg_length += expansion

        return mission

    def generate_sector_search(
        self,
        center: Tuple[float, float],
        altitude: float,
        radius: float,
        num_sectors: int = 6,
        speed: float = 5.0,
        name: str = "Sector Search"
    ) -> Mission:
        """
        Generate a sector search pattern.

        Divides area into pie-slice sectors for systematic search.

        Args:
            center: Center point (lat, lon)
            altitude: Flight altitude in meters
            radius: Search radius in meters
            num_sectors: Number of sectors to divide into
            speed: Flight speed in m/s
            name: Mission name

        Returns:
            Mission with sector search pattern
        """
        mission = Mission(
            name=name,
            mission_type=MissionType.SEARCH,
            default_speed=speed,
            default_altitude=altitude,
        )

        angle_step = 2 * math.pi / num_sectors

        for i in range(num_sectors):
            angle = i * angle_step

            # Go to center
            wp_center = Waypoint(
                latitude=center[0],
                longitude=center[1],
                altitude=altitude,
                sequence=len(mission.waypoints),
                speed=speed,
            )
            mission.add_waypoint(wp_center)

            # Go to edge of sector
            dlat = radius * math.cos(angle) / 111320
            dlon = radius * math.sin(angle) / (111320 * math.cos(math.radians(center[0])))

            wp_edge = Waypoint(
                latitude=center[0] + dlat,
                longitude=center[1] + dlon,
                altitude=altitude,
                sequence=len(mission.waypoints),
                speed=speed,
            )
            mission.add_waypoint(wp_edge)

        # Return to center
        wp_final = Waypoint(
            latitude=center[0],
            longitude=center[1],
            altitude=altitude,
            sequence=len(mission.waypoints),
            speed=speed,
        )
        mission.add_waypoint(wp_final)

        return mission

    def generate_crosshatch(
        self,
        center: Tuple[float, float],
        width: float,
        height: float,
        altitude: float,
        line_spacing: float,
        speed: float = 5.0,
        camera_trigger: bool = True,
        name: str = "Crosshatch Survey"
    ) -> Mission:
        """
        Generate a crosshatch pattern (grid in two directions).

        Provides better coverage with overlapping passes.

        Args:
            center: Center point (lat, lon)
            width: Survey width in meters
            height: Survey height in meters
            altitude: Flight altitude in meters
            line_spacing: Spacing between lines in meters
            speed: Flight speed in m/s
            camera_trigger: Add photo triggers
            name: Mission name

        Returns:
            Mission with crosshatch pattern
        """
        # Generate first direction (0 degrees)
        mission1 = self.generate_grid(
            center=center,
            width=width,
            height=height,
            altitude=altitude,
            line_spacing=line_spacing,
            speed=speed,
            direction=0,
            camera_trigger=camera_trigger,
            name=name,
        )

        # Generate second direction (90 degrees)
        mission2 = self.generate_grid(
            center=center,
            width=height,  # Swap width/height
            height=width,
            altitude=altitude,
            line_spacing=line_spacing,
            speed=speed,
            direction=90,
            camera_trigger=camera_trigger,
            name=name,
        )

        # Combine missions
        mission = Mission(
            name=name,
            mission_type=MissionType.SURVEY,
            default_speed=speed,
            default_altitude=altitude,
        )

        # Add all waypoints
        for wp in mission1.waypoints:
            wp_copy = wp.copy()
            wp_copy.sequence = len(mission.waypoints)
            mission.add_waypoint(wp_copy)

        for wp in mission2.waypoints:
            wp_copy = wp.copy()
            wp_copy.sequence = len(mission.waypoints)
            mission.add_waypoint(wp_copy)

        return mission
