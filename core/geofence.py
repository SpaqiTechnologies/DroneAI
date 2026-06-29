"""
Advanced geofence system for Drone AI application.
Supports polygon boundaries, altitude limits, and warning zones.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Callable
import logging
import time

logger = logging.getLogger(__name__)


class GeofenceType(Enum):
    """Types of geofence boundaries."""
    CIRCULAR = auto()
    POLYGON = auto()
    CYLINDER = auto()  # Circular with altitude limits


class GeofenceAction(Enum):
    """Actions when geofence is breached."""
    NONE = "none"
    WARN = "warn"
    LOITER = "loiter"
    RTH = "return_to_home"
    LAND = "land"
    STOP = "stop"  # Hard stop at boundary


class GeofenceZone(Enum):
    """Current zone relative to geofence."""
    INSIDE = "inside"
    WARNING = "warning"  # Inside but near boundary
    OUTSIDE = "outside"


@dataclass
class GeofencePoint:
    """A point in the geofence boundary."""
    latitude: float
    longitude: float


@dataclass
class GeofenceConfig:
    """Configuration for a geofence."""
    name: str
    fence_type: GeofenceType
    enabled: bool = True

    # For circular/cylinder geofence
    center: Optional[Tuple[float, float]] = None  # (lat, lon)
    radius: float = 100.0  # meters

    # For polygon geofence
    vertices: List[GeofencePoint] = field(default_factory=list)

    # Altitude limits (for all types)
    min_altitude: float = 0.0  # meters AGL
    max_altitude: float = 120.0  # meters AGL (typical legal limit)

    # Warning zone (percentage of boundary)
    warning_threshold: float = 0.8  # Warn at 80% of boundary

    # Actions
    warning_action: GeofenceAction = GeofenceAction.WARN
    breach_action: GeofenceAction = GeofenceAction.RTH

    # Inclusion vs exclusion
    is_inclusion: bool = True  # True = must stay inside, False = must stay outside


@dataclass
class GeofenceStatus:
    """Current geofence status."""
    zone: GeofenceZone
    distance_to_boundary: float  # meters (negative = inside)
    nearest_point: Optional[Tuple[float, float]] = None
    altitude_status: str = "ok"  # "ok", "too_low", "too_high", "warning_low", "warning_high"
    breached_fences: List[str] = field(default_factory=list)
    warning_fences: List[str] = field(default_factory=list)


class GeofenceManager:
    """
    Manages multiple geofences with warning zones and altitude limits.
    """

    # Earth radius in meters
    EARTH_RADIUS = 6371000.0

    def __init__(self):
        self._fences: dict[str, GeofenceConfig] = {}
        self._callbacks: List[Callable] = []
        self._last_status: Optional[GeofenceStatus] = None
        self._breach_start_times: dict[str, float] = {}
        self._warning_last_times: dict[str, float] = {}  # Rate limit warnings
        self._warning_interval: float = 5.0  # Seconds between warning callbacks

    def add_fence(self, config: GeofenceConfig):
        """Add a geofence configuration."""
        if config.fence_type == GeofenceType.POLYGON and len(config.vertices) < 3:
            raise ValueError("Polygon geofence requires at least 3 vertices")

        if config.fence_type in (GeofenceType.CIRCULAR, GeofenceType.CYLINDER) and not config.center:
            raise ValueError("Circular/cylinder geofence requires a center point")

        self._fences[config.name] = config
        logger.info(f"Added geofence: {config.name} ({config.fence_type.name})")

    def remove_fence(self, name: str):
        """Remove a geofence by name."""
        if name in self._fences:
            del self._fences[name]
            logger.info(f"Removed geofence: {name}")

    def enable_fence(self, name: str, enabled: bool = True):
        """Enable or disable a geofence."""
        if name in self._fences:
            self._fences[name].enabled = enabled

    def add_callback(self, callback: Callable):
        """Add a callback for geofence events."""
        self._callbacks.append(callback)

    def check_position(
        self,
        latitude: float,
        longitude: float,
        altitude: float = 0.0,
    ) -> GeofenceStatus:
        """
        Check if a position is within all geofences.

        Returns GeofenceStatus with zone, distance, and any breaches.
        """
        breached = []
        warnings = []
        min_distance = float('inf')
        overall_zone = GeofenceZone.INSIDE
        altitude_status = "ok"

        for name, fence in self._fences.items():
            if not fence.enabled:
                continue

            # Check horizontal boundary
            if fence.fence_type == GeofenceType.POLYGON:
                inside, distance, nearest = self._check_polygon(
                    latitude, longitude, fence.vertices
                )
            else:  # CIRCULAR or CYLINDER
                inside, distance, nearest = self._check_circular(
                    latitude, longitude, fence.center, fence.radius
                )

            # For inclusion fences, we want to be inside
            # For exclusion fences, we want to be outside
            if fence.is_inclusion:
                in_bounds = inside
                boundary_distance = fence.radius - distance if fence.fence_type != GeofenceType.POLYGON else distance
            else:
                in_bounds = not inside
                boundary_distance = distance

            # Check warning zone
            warning_distance = fence.radius * fence.warning_threshold if fence.fence_type != GeofenceType.POLYGON else 0
            in_warning = (fence.is_inclusion and distance > warning_distance) or \
                         (not fence.is_inclusion and distance < fence.radius * (1 - fence.warning_threshold))

            # Check altitude limits
            alt_breach = False
            if altitude < fence.min_altitude:
                altitude_status = "too_low"
                alt_breach = True
            elif altitude > fence.max_altitude:
                altitude_status = "too_high"
                alt_breach = True
            elif altitude < fence.min_altitude + (fence.max_altitude - fence.min_altitude) * 0.1:
                altitude_status = "warning_low"
            elif altitude > fence.max_altitude - (fence.max_altitude - fence.min_altitude) * 0.1:
                altitude_status = "warning_high"

            # Determine breach/warning status
            if not in_bounds or alt_breach:
                breached.append(name)
                overall_zone = GeofenceZone.OUTSIDE
                self._handle_breach(name, fence, latitude, longitude, altitude)
            elif in_warning or altitude_status.startswith("warning"):
                warnings.append(name)
                if overall_zone == GeofenceZone.INSIDE:
                    overall_zone = GeofenceZone.WARNING
                self._handle_warning(name, fence, latitude, longitude, altitude)

            min_distance = min(min_distance, abs(boundary_distance))

        status = GeofenceStatus(
            zone=overall_zone,
            distance_to_boundary=min_distance if min_distance != float('inf') else 0,
            altitude_status=altitude_status,
            breached_fences=breached,
            warning_fences=warnings,
        )

        self._last_status = status
        return status

    def _check_circular(
        self,
        lat: float,
        lon: float,
        center: Tuple[float, float],
        radius: float,
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """
        Check if point is inside circular geofence.

        Returns (is_inside, distance_from_center, nearest_boundary_point)
        """
        distance = self._haversine_distance(lat, lon, center[0], center[1])
        inside = distance <= radius

        # Calculate nearest point on boundary
        if distance > 0:
            bearing = self._bearing(center[0], center[1], lat, lon)
            nearest = self._destination_point(center[0], center[1], bearing, radius)
        else:
            nearest = (center[0] + radius / self.EARTH_RADIUS * 180 / math.pi, center[1])

        return inside, distance, nearest

    def _check_polygon(
        self,
        lat: float,
        lon: float,
        vertices: List[GeofencePoint],
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """
        Check if point is inside polygon geofence using ray casting algorithm.

        Returns (is_inside, min_distance_to_edge, nearest_point_on_edge)
        """
        n = len(vertices)
        inside = False

        # Ray casting algorithm
        j = n - 1
        for i in range(n):
            vi = vertices[i]
            vj = vertices[j]

            if ((vi.longitude > lon) != (vj.longitude > lon)) and \
               (lat < (vj.latitude - vi.latitude) * (lon - vi.longitude) / (vj.longitude - vi.longitude) + vi.latitude):
                inside = not inside
            j = i

        # Find minimum distance to any edge
        min_dist = float('inf')
        nearest = (vertices[0].latitude, vertices[0].longitude)

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            dist, point = self._point_to_segment_distance(
                lat, lon,
                v1.latitude, v1.longitude,
                v2.latitude, v2.longitude
            )

            if dist < min_dist:
                min_dist = dist
                nearest = point

        return inside, min_dist, nearest

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters using Haversine formula."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS * c

    def _bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in radians."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)

        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

        return math.atan2(x, y)

    def _destination_point(self, lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
        """Calculate destination point given start, bearing, and distance."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        d = distance / self.EARTH_RADIUS

        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(d) +
            math.cos(lat_rad) * math.sin(d) * math.cos(bearing)
        )
        new_lon = lon_rad + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat_rad),
            math.cos(d) - math.sin(lat_rad) * math.sin(new_lat)
        )

        return math.degrees(new_lat), math.degrees(new_lon)

    def _point_to_segment_distance(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate minimum distance from point to line segment.

        Returns (distance_in_meters, nearest_point_on_segment)
        """
        # Convert to approximate meters for calculation
        scale = math.cos(math.radians(px)) * 111320  # meters per degree at latitude

        dx = (x2 - x1) * scale
        dy = (y2 - y1) * scale
        px_scaled = (px - x1) * scale
        py_scaled = (py - y1) * scale

        if dx == 0 and dy == 0:
            # Segment is a point
            return self._haversine_distance(px, py, x1, y1), (x1, y1)

        t = max(0, min(1, (px_scaled * dx + py_scaled * dy) / (dx * dx + dy * dy)))

        nearest_x = x1 + t * (x2 - x1)
        nearest_y = y1 + t * (y2 - y1)

        return self._haversine_distance(px, py, nearest_x, nearest_y), (nearest_x, nearest_y)

    def _handle_breach(self, name: str, fence: GeofenceConfig, lat: float, lon: float, alt: float):
        """Handle a geofence breach."""
        if name not in self._breach_start_times:
            self._breach_start_times[name] = time.time()
            logger.warning(f"GEOFENCE BREACH: {name} at ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")

            for callback in self._callbacks:
                try:
                    callback({
                        'event': 'breach',
                        'fence_name': name,
                        'action': fence.breach_action.value,
                        'position': (lat, lon, alt),
                    })
                except Exception as e:
                    logger.error(f"Geofence callback error: {e}")

    def _handle_warning(self, name: str, fence: GeofenceConfig, lat: float, lon: float, alt: float):
        """Handle a geofence warning with rate limiting."""
        current_time = time.time()
        last_warning = self._warning_last_times.get(name, 0)

        # Rate limit warnings to avoid spam
        if current_time - last_warning < self._warning_interval:
            return

        self._warning_last_times[name] = current_time
        logger.debug(f"Geofence warning: {name} at ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")

        for callback in self._callbacks:
            try:
                callback({
                    'event': 'warning',
                    'fence_name': name,
                    'action': fence.warning_action.value,
                    'position': (lat, lon, alt),
                })
            except Exception as e:
                logger.error(f"Geofence callback error: {e}")

    def clear_breach(self, name: str):
        """Clear a breach record (when drone returns to safe zone)."""
        if name in self._breach_start_times:
            del self._breach_start_times[name]
            logger.info(f"Geofence breach cleared: {name}")
        # Also clear warning timer
        if name in self._warning_last_times:
            del self._warning_last_times[name]

    def get_fence(self, name: str) -> Optional[GeofenceConfig]:
        """Get a geofence configuration by name."""
        return self._fences.get(name)

    def get_all_fences(self) -> dict[str, GeofenceConfig]:
        """Get all geofence configurations."""
        return dict(self._fences)

    def get_status(self) -> Optional[GeofenceStatus]:
        """Get last geofence status."""
        return self._last_status

    def create_home_geofence(
        self,
        home_lat: float,
        home_lon: float,
        radius: float = 500.0,
        max_altitude: float = 120.0,
    ) -> GeofenceConfig:
        """Create a standard home-centered geofence."""
        config = GeofenceConfig(
            name="home_zone",
            fence_type=GeofenceType.CYLINDER,
            center=(home_lat, home_lon),
            radius=radius,
            min_altitude=0.0,
            max_altitude=max_altitude,
            warning_threshold=0.8,
            warning_action=GeofenceAction.WARN,
            breach_action=GeofenceAction.RTH,
            is_inclusion=True,
        )
        self.add_fence(config)
        return config

    def create_no_fly_zone(
        self,
        name: str,
        vertices: List[Tuple[float, float]],
    ) -> GeofenceConfig:
        """Create an exclusion zone (no-fly zone)."""
        config = GeofenceConfig(
            name=name,
            fence_type=GeofenceType.POLYGON,
            vertices=[GeofencePoint(lat, lon) for lat, lon in vertices],
            warning_action=GeofenceAction.WARN,
            breach_action=GeofenceAction.STOP,
            is_inclusion=False,  # Must stay OUTSIDE
        )
        self.add_fence(config)
        return config

    def to_dict(self) -> dict:
        """Serialize geofence manager state."""
        return {
            'fences': {
                name: {
                    'type': fence.fence_type.name,
                    'enabled': fence.enabled,
                    'center': fence.center,
                    'radius': fence.radius,
                    'vertices': [(v.latitude, v.longitude) for v in fence.vertices],
                    'min_altitude': fence.min_altitude,
                    'max_altitude': fence.max_altitude,
                    'is_inclusion': fence.is_inclusion,
                }
                for name, fence in self._fences.items()
            },
            'status': {
                'zone': self._last_status.zone.value if self._last_status else None,
                'breached': self._last_status.breached_fences if self._last_status else [],
                'warnings': self._last_status.warning_fences if self._last_status else [],
            } if self._last_status else None,
        }
