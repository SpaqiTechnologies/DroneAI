"""
Search pattern generator for search and rescue operations.

Generates optimal search patterns for locating targets
in various terrain and conditions.
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search patterns."""
    EXPANDING_SQUARE = "expanding_square"
    SECTOR = "sector"
    PARALLEL = "parallel"
    CREEPING_LINE = "creeping_line"
    CONTOUR = "contour"
    SPIRAL = "spiral"
    GRID = "grid"


@dataclass
class SearchPattern:
    """A generated search pattern."""
    pattern_type: SearchType
    waypoints: List[Tuple[float, float, float]]  # lat, lon, alt
    center: Tuple[float, float]
    area_sqm: float
    total_distance: float  # meters
    estimated_time: float  # seconds
    track_spacing: float  # meters
    coverage_probability: float


@dataclass
class SearchConfig:
    """Configuration for search pattern generation."""
    altitude: float = 30.0  # meters AGL
    sweep_width: float = 50.0  # meters
    track_spacing: float = 40.0  # meters (< sweep_width)
    speed: float = 8.0  # m/s
    search_radius: float = 500.0  # meters
    sector_angle: float = 90.0  # degrees for sector search
    legs: int = 8  # for expanding square


class SearchPatternGenerator:
    """
    Generates search patterns for SAR operations.

    Supports multiple pattern types optimized for
    different scenarios.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize search pattern generator.

        Args:
            config: Search configuration
        """
        self._config = config or SearchConfig()

    def generate(
        self,
        pattern_type: SearchType,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate a search pattern.

        Args:
            pattern_type: Type of pattern
            center: Center point (lat, lon)
            **kwargs: Additional pattern parameters

        Returns:
            SearchPattern with waypoints
        """
        generators = {
            SearchType.EXPANDING_SQUARE: self._expanding_square,
            SearchType.SECTOR: self._sector,
            SearchType.PARALLEL: self._parallel,
            SearchType.CREEPING_LINE: self._creeping_line,
            SearchType.SPIRAL: self._spiral,
            SearchType.GRID: self._grid,
        }

        generator = generators.get(pattern_type)
        if not generator:
            logger.warning(f"Unknown pattern: {pattern_type}")
            generator = self._expanding_square

        return generator(center, **kwargs)

    def _expanding_square(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate expanding square pattern.

        Starts at center and spirals outward in squares.
        Good for point last seen.
        """
        cfg = self._config
        legs = kwargs.get('legs', cfg.legs)
        spacing = kwargs.get('spacing', cfg.track_spacing)

        waypoints = []
        lat, lon = center
        alt = cfg.altitude

        # Start at center
        current_lat = lat
        current_lon = lon
        waypoints.append((current_lat, current_lon, alt))

        # Generate expanding square
        directions = [
            (0, 1),   # East
            (-1, 0),  # South
            (0, -1),  # West
            (1, 0),   # North
        ]

        leg_length = spacing
        for leg in range(legs):
            direction = directions[leg % 4]

            # Calculate distance in degrees
            dlat = (direction[0] * leg_length) / 111000
            dlon = (direction[1] * leg_length) / (
                111000 * math.cos(math.radians(lat))
            )

            current_lat += dlat
            current_lon += dlon
            waypoints.append((current_lat, current_lon, alt))

            # Increase leg length every 2 legs
            if leg % 2 == 1:
                leg_length += spacing

        # Calculate metrics
        total_dist = self._calculate_total_distance(waypoints)
        area = math.pi * (leg_length ** 2)  # Approximate

        return SearchPattern(
            pattern_type=SearchType.EXPANDING_SQUARE,
            waypoints=waypoints,
            center=center,
            area_sqm=area,
            total_distance=total_dist,
            estimated_time=total_dist / cfg.speed,
            track_spacing=spacing,
            coverage_probability=self._calc_coverage(spacing),
        )

    def _sector(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate sector search pattern.

        Radiates outward from center point like pie slices.
        Good when direction is uncertain.
        """
        cfg = self._config
        radius = kwargs.get('radius', cfg.search_radius)
        angle = kwargs.get('angle', cfg.sector_angle)
        spacing = cfg.track_spacing

        waypoints = []
        lat, lon = center
        alt = cfg.altitude

        # Number of radial lines
        num_lines = int(angle / 15) + 1  # Every 15 degrees
        start_angle = -angle / 2

        for i in range(num_lines):
            current_angle = start_angle + (i * angle / (num_lines - 1))
            angle_rad = math.radians(current_angle)

            if i % 2 == 0:
                # Outbound
                waypoints.append((lat, lon, alt))
                end_lat = lat + (radius / 111000) * math.cos(angle_rad)
                end_lon = lon + (radius / 111000) * math.sin(angle_rad) / (
                    math.cos(math.radians(lat))
                )
                waypoints.append((end_lat, end_lon, alt))
            else:
                # Inbound
                start_lat = lat + (radius / 111000) * math.cos(angle_rad)
                start_lon = lon + (radius / 111000) * math.sin(angle_rad) / (
                    math.cos(math.radians(lat))
                )
                waypoints.append((start_lat, start_lon, alt))
                waypoints.append((lat, lon, alt))

        total_dist = self._calculate_total_distance(waypoints)
        area = (math.pi * radius ** 2) * (angle / 360)

        return SearchPattern(
            pattern_type=SearchType.SECTOR,
            waypoints=waypoints,
            center=center,
            area_sqm=area,
            total_distance=total_dist,
            estimated_time=total_dist / cfg.speed,
            track_spacing=spacing,
            coverage_probability=self._calc_coverage(spacing),
        )

    def _parallel(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate parallel track pattern.

        Straight parallel lines across search area.
        Most efficient for large areas.
        """
        cfg = self._config
        width = kwargs.get('width', cfg.search_radius * 2)
        height = kwargs.get('height', cfg.search_radius)
        heading = kwargs.get('heading', 0.0)  # degrees

        waypoints = []
        lat, lon = center
        alt = cfg.altitude
        spacing = cfg.track_spacing

        # Number of tracks
        num_tracks = int(width / spacing) + 1
        heading_rad = math.radians(heading)

        for i in range(num_tracks):
            # Offset from center
            offset = (i - num_tracks // 2) * spacing

            # Calculate start and end points
            perp_angle = heading_rad + math.pi / 2

            # Start point
            start_lat = lat + (offset / 111000) * math.cos(perp_angle)
            start_lon = lon + (offset / 111000) * math.sin(perp_angle) / (
                math.cos(math.radians(lat))
            )

            if i % 2 == 0:
                # Forward
                s_lat = start_lat - (height / 2 / 111000) * math.cos(heading_rad)
                s_lon = start_lon - (height / 2 / 111000) * math.sin(heading_rad) / (
                    math.cos(math.radians(lat))
                )
                e_lat = start_lat + (height / 2 / 111000) * math.cos(heading_rad)
                e_lon = start_lon + (height / 2 / 111000) * math.sin(heading_rad) / (
                    math.cos(math.radians(lat))
                )
            else:
                # Reverse
                e_lat = start_lat - (height / 2 / 111000) * math.cos(heading_rad)
                e_lon = start_lon - (height / 2 / 111000) * math.sin(heading_rad) / (
                    math.cos(math.radians(lat))
                )
                s_lat = start_lat + (height / 2 / 111000) * math.cos(heading_rad)
                s_lon = start_lon + (height / 2 / 111000) * math.sin(heading_rad) / (
                    math.cos(math.radians(lat))
                )

            waypoints.append((s_lat, s_lon, alt))
            waypoints.append((e_lat, e_lon, alt))

        total_dist = self._calculate_total_distance(waypoints)
        area = width * height

        return SearchPattern(
            pattern_type=SearchType.PARALLEL,
            waypoints=waypoints,
            center=center,
            area_sqm=area,
            total_distance=total_dist,
            estimated_time=total_dist / cfg.speed,
            track_spacing=spacing,
            coverage_probability=self._calc_coverage(spacing),
        )

    def _creeping_line(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """Generate creeping line pattern (similar to parallel)."""
        return self._parallel(center, **kwargs)

    def _spiral(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate spiral search pattern.

        Circular spiral from center outward.
        Good for small areas.
        """
        cfg = self._config
        radius = kwargs.get('radius', cfg.search_radius)
        spacing = cfg.track_spacing

        waypoints = []
        lat, lon = center
        alt = cfg.altitude

        # Generate spiral points
        angle = 0.0
        current_radius = 0.0
        points_per_revolution = 12

        while current_radius <= radius:
            angle_rad = math.radians(angle)

            point_lat = lat + (current_radius / 111000) * math.cos(angle_rad)
            point_lon = lon + (current_radius / 111000) * math.sin(angle_rad) / (
                math.cos(math.radians(lat))
            )

            waypoints.append((point_lat, point_lon, alt))

            angle += 360 / points_per_revolution
            current_radius += spacing / points_per_revolution

        total_dist = self._calculate_total_distance(waypoints)
        area = math.pi * radius ** 2

        return SearchPattern(
            pattern_type=SearchType.SPIRAL,
            waypoints=waypoints,
            center=center,
            area_sqm=area,
            total_distance=total_dist,
            estimated_time=total_dist / cfg.speed,
            track_spacing=spacing,
            coverage_probability=self._calc_coverage(spacing),
        )

    def _grid(
        self,
        center: Tuple[float, float],
        **kwargs,
    ) -> SearchPattern:
        """
        Generate grid search pattern.

        Covers area in two perpendicular passes.
        Most thorough coverage.
        """
        # First pass (NS)
        pattern1 = self._parallel(center, heading=0, **kwargs)

        # Second pass (EW)
        pattern2 = self._parallel(center, heading=90, **kwargs)

        # Combine waypoints
        waypoints = pattern1.waypoints + pattern2.waypoints

        return SearchPattern(
            pattern_type=SearchType.GRID,
            waypoints=waypoints,
            center=center,
            area_sqm=pattern1.area_sqm,
            total_distance=pattern1.total_distance + pattern2.total_distance,
            estimated_time=(
                pattern1.estimated_time + pattern2.estimated_time
            ),
            track_spacing=self._config.track_spacing,
            coverage_probability=min(
                0.99,
                pattern1.coverage_probability * 1.5
            ),
        )

    def _calculate_total_distance(
        self,
        waypoints: List[Tuple[float, float, float]],
    ) -> float:
        """Calculate total distance of waypoint path."""
        total = 0.0
        for i in range(1, len(waypoints)):
            lat1, lon1, _ = waypoints[i - 1]
            lat2, lon2, _ = waypoints[i]
            total += self._haversine(lat1, lon1, lat2, lon2)
        return total

    def _haversine(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) *
            math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calc_coverage(self, spacing: float) -> float:
        """Calculate probability of detection."""
        sweep = self._config.sweep_width
        # Coverage factor (simplif ied)
        if spacing >= sweep:
            return 0.5
        ratio = spacing / sweep
        return min(0.95, 1 - (ratio ** 2) / 2)

    def recommend_pattern(
        self,
        scenario: str,
    ) -> SearchType:
        """
        Recommend search pattern based on scenario.

        Args:
            scenario: Description like "point_last_seen",
                     "large_area", "person_lost"

        Returns:
            Recommended SearchType
        """
        scenario = scenario.lower()

        if "point" in scenario or "last_seen" in scenario:
            return SearchType.EXPANDING_SQUARE
        elif "large" in scenario or "wide" in scenario:
            return SearchType.PARALLEL
        elif "direction" in scenario or "heading" in scenario:
            return SearchType.SECTOR
        elif "thorough" in scenario or "detailed" in scenario:
            return SearchType.GRID
        elif "small" in scenario:
            return SearchType.SPIRAL
        else:
            return SearchType.EXPANDING_SQUARE
