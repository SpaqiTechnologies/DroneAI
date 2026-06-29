"""
Aerial mapping application for orthomosaic generation.

Automates flight patterns and image capture for
creating aerial maps and 3D models.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

from ..base_application import BaseApplication

logger = logging.getLogger(__name__)


@dataclass
class MappingConfig:
    """Configuration for aerial mapping."""
    altitude: float = 50.0  # meters AGL
    overlap_front: float = 75.0  # percent
    overlap_side: float = 65.0  # percent
    camera_fov: float = 84.0  # degrees
    capture_interval: float = 2.0  # seconds
    flight_speed: float = 5.0  # m/s
    boundary: List[Tuple[float, float]] = field(
        default_factory=list
    )


@dataclass
class CapturedImage:
    """A captured image with metadata."""
    image_id: int
    timestamp: float
    position: Tuple[float, float, float]  # lat, lon, alt
    heading: float
    pitch: float
    roll: float
    file_path: Optional[str] = None


class AerialMapper(BaseApplication):
    """
    Aerial mapping application.

    Generates flight paths for area coverage and
    coordinates image capture for photogrammetry.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        """
        Initialize aerial mapper.

        Args:
            config: Mapping configuration
        """
        super().__init__("AerialMapper")
        self._config = config or MappingConfig()
        self._waypoints: List[Tuple[float, float, float]] = []
        self._current_waypoint = 0
        self._images: List[CapturedImage] = []
        self._last_capture = 0.0
        self._area_sqm = 0.0
        self._coverage_percent = 0.0

    def set_boundary(
        self,
        boundary: List[Tuple[float, float]],
    ) -> None:
        """
        Set mapping boundary polygon.

        Args:
            boundary: List of (lat, lon) vertices
        """
        self._config.boundary = boundary
        self._area_sqm = self._calculate_area(boundary)

    def _do_initialize(self) -> bool:
        """Initialize mapping mission."""
        if len(self._config.boundary) < 3:
            self._add_error("Boundary must have at least 3 points")
            return False

        # Generate flight path
        self._waypoints = self._generate_lawnmower_path()

        if not self._waypoints:
            self._add_error("Failed to generate flight path")
            return False

        logger.info(
            f"Generated {len(self._waypoints)} waypoints "
            f"for {self._area_sqm:.0f} sqm area"
        )

        self._update_progress(
            percent=0,
            total_steps=len(self._waypoints),
            message="Ready to start mapping",
        )

        return True

    def _do_start(self) -> bool:
        """Start mapping mission."""
        self._current_waypoint = 0
        self._last_capture = time.time()
        self._update_progress(message="Mapping in progress")
        return True

    def _do_update(self, dt: float) -> None:
        """Update mapping progress."""
        current_time = time.time()

        # Check if should capture
        if current_time - self._last_capture >= self._config.capture_interval:
            self._capture_image()
            self._last_capture = current_time

        # Update coverage estimate
        if self._waypoints:
            self._coverage_percent = (
                self._current_waypoint / len(self._waypoints)
            ) * 100

            self._update_progress(
                percent=self._coverage_percent,
                current_step=self._current_waypoint,
                message=f"Waypoint {self._current_waypoint + 1}/"
                        f"{len(self._waypoints)}",
            )

    def _capture_image(self) -> None:
        """Capture an image at current position."""
        # In real implementation, would trigger camera
        image = CapturedImage(
            image_id=len(self._images) + 1,
            timestamp=time.time(),
            position=(0.0, 0.0, self._config.altitude),
            heading=0.0,
            pitch=-90.0,  # Nadir
            roll=0.0,
        )
        self._images.append(image)
        self._add_data(image)

    def _check_completion(self) -> bool:
        """Check if mapping is complete."""
        return self._current_waypoint >= len(self._waypoints)

    def _do_complete(self) -> None:
        """Finalize mapping mission."""
        logger.info(
            f"Mapping complete: {len(self._images)} images captured"
        )

    def _generate_lawnmower_path(
        self,
    ) -> List[Tuple[float, float, float]]:
        """Generate lawnmower pattern flight path."""
        boundary = self._config.boundary
        if len(boundary) < 3:
            return []

        # Calculate bounding box
        lats = [p[0] for p in boundary]
        lons = [p[1] for p in boundary]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Calculate line spacing based on overlap
        ground_width = self._ground_coverage_width()
        spacing = ground_width * (1 - self._config.overlap_side / 100)

        # Convert spacing to degrees (approximate)
        spacing_deg = spacing / 111000  # ~111km per degree

        waypoints = []
        alt = self._config.altitude
        current_lon = min_lon
        row = 0

        while current_lon <= max_lon:
            if row % 2 == 0:
                # South to north
                start_lat = min_lat
                end_lat = max_lat
            else:
                # North to south
                start_lat = max_lat
                end_lat = min_lat

            waypoints.append((start_lat, current_lon, alt))
            waypoints.append((end_lat, current_lon, alt))

            current_lon += spacing_deg
            row += 1

        return waypoints

    def _ground_coverage_width(self) -> float:
        """Calculate ground coverage width at current altitude."""
        fov_rad = math.radians(self._config.camera_fov)
        return 2 * self._config.altitude * math.tan(fov_rad / 2)

    def _calculate_area(
        self,
        boundary: List[Tuple[float, float]],
    ) -> float:
        """Calculate polygon area in square meters."""
        if len(boundary) < 3:
            return 0.0

        # Shoelace formula with lat/lon to meters conversion
        n = len(boundary)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            lat1, lon1 = boundary[i]
            lat2, lon2 = boundary[j]

            # Convert to approximate meters
            x1 = lon1 * 111000 * math.cos(math.radians(lat1))
            y1 = lat1 * 111000
            x2 = lon2 * 111000 * math.cos(math.radians(lat2))
            y2 = lat2 * 111000

            area += x1 * y2 - x2 * y1

        return abs(area) / 2

    def advance_waypoint(self) -> bool:
        """
        Advance to next waypoint.

        Call this when drone reaches current waypoint.
        """
        if self._current_waypoint < len(self._waypoints):
            self._current_waypoint += 1
            return True
        return False

    def get_current_waypoint(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """Get current target waypoint."""
        if self._current_waypoint < len(self._waypoints):
            return self._waypoints[self._current_waypoint]
        return None

    def get_all_waypoints(
        self,
    ) -> List[Tuple[float, float, float]]:
        """Get all waypoints in the mission."""
        return self._waypoints.copy()

    def get_images(self) -> List[CapturedImage]:
        """Get all captured images."""
        return self._images.copy()

    @property
    def image_count(self) -> int:
        """Get number of captured images."""
        return len(self._images)

    @property
    def area(self) -> float:
        """Get mapping area in square meters."""
        return self._area_sqm

    @property
    def coverage(self) -> float:
        """Get coverage percentage."""
        return self._coverage_percent

    def get_stats(self) -> Dict:
        """Get mapping statistics."""
        return {
            **self.get_results(),
            'area_sqm': self._area_sqm,
            'coverage_percent': self._coverage_percent,
            'image_count': len(self._images),
            'waypoint_count': len(self._waypoints),
            'current_waypoint': self._current_waypoint,
        }
