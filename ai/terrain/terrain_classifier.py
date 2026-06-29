"""
Terrain classification for landing zone detection.

Analyzes imagery to identify safe landing zones
based on terrain type, flatness, and obstacles.
"""

import time
import random
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TerrainType(Enum):
    """Types of terrain."""
    GRASS = "grass"
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    GRAVEL = "gravel"
    SAND = "sand"
    WATER = "water"
    VEGETATION = "vegetation"
    BUILDING = "building"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


class LandingZoneQuality(Enum):
    """Quality rating for landing zones."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    UNSUITABLE = "unsuitable"


@dataclass
class LandingZone:
    """A potential landing zone."""
    zone_id: int
    center: Tuple[float, float]  # pixel coords
    radius: float  # pixels
    terrain_type: TerrainType
    quality: LandingZoneQuality
    confidence: float
    flatness_score: float  # 0-1
    obstacle_score: float  # 0-1 (1 = no obstacles)
    size_score: float  # 0-1
    overall_score: float  # weighted combination
    gps_coords: Optional[Tuple[float, float]] = None
    altitude_agl: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'zone_id': self.zone_id,
            'center': self.center,
            'radius': self.radius,
            'terrain': self.terrain_type.value,
            'quality': self.quality.value,
            'confidence': self.confidence,
            'flatness': self.flatness_score,
            'obstacle_free': self.obstacle_score,
            'size': self.size_score,
            'score': self.overall_score,
            'gps': self.gps_coords,
        }


@dataclass
class TerrainRegion:
    """A classified terrain region."""
    terrain_type: TerrainType
    confidence: float
    bbox: Tuple[int, int, int, int]
    area: float
    center: Tuple[float, float]


@dataclass
class ClassifierConfig:
    """Configuration for terrain classifier."""
    min_zone_radius: float = 2.0  # meters
    max_zone_radius: float = 10.0  # meters
    min_confidence: float = 0.6
    flatness_weight: float = 0.4
    obstacle_weight: float = 0.4
    size_weight: float = 0.2
    grid_resolution: int = 32  # pixels
    landing_terrain_types: List[TerrainType] = field(
        default_factory=lambda: [
            TerrainType.GRASS,
            TerrainType.CONCRETE,
            TerrainType.ASPHALT,
            TerrainType.GRAVEL,
            TerrainType.SAND,
        ]
    )


class TerrainClassifier:
    """
    Terrain classifier for landing zone detection.

    Analyzes downward-facing camera images to:
    - Classify terrain types
    - Detect flat areas
    - Identify obstacles
    - Score landing zones
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize terrain classifier.

        Args:
            config: Classifier configuration
        """
        self._config = config or ClassifierConfig()
        self._model = None
        self._is_simulation = True
        self._next_zone_id = 1
        self._frame_count = 0

        # Terrain colors (BGR for simulation)
        self._terrain_colors = {
            TerrainType.GRASS: ([35, 100, 35], [90, 255, 150]),
            TerrainType.CONCRETE: ([140, 140, 140], [200, 200, 200]),
            TerrainType.ASPHALT: ([30, 30, 30], [80, 80, 80]),
            TerrainType.WATER: ([100, 80, 0], [255, 150, 50]),
            TerrainType.SAND: ([80, 170, 200], [150, 220, 255]),
        }

    def classify_image(
        self,
        image: Any,
        depth_map: Optional[Any] = None,
    ) -> Tuple[List[TerrainRegion], Optional[Any]]:
        """
        Classify terrain in image.

        Args:
            image: Input image (numpy array)
            depth_map: Optional depth information

        Returns:
            Tuple of (regions, segmentation_mask)
        """
        self._frame_count += 1

        if self._is_simulation:
            return self._simulate_classification(image)

        # Real classification would go here
        return [], None

    def _simulate_classification(
        self,
        image: Any,
    ) -> Tuple[List[TerrainRegion], None]:
        """Generate simulated terrain classification."""
        regions = []

        # Get image size
        if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = 640, 480

        # Generate random terrain regions
        num_regions = random.randint(2, 5)
        grid_w = w // 2
        grid_h = h // 2

        for i in range(num_regions):
            terrain = random.choice(list(TerrainType))
            conf = random.uniform(0.6, 0.95)

            # Random position
            x1 = random.randint(0, w - grid_w)
            y1 = random.randint(0, h - grid_h)
            x2 = x1 + grid_w
            y2 = y1 + grid_h

            region = TerrainRegion(
                terrain_type=terrain,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                area=(x2 - x1) * (y2 - y1),
                center=((x1 + x2) / 2, (y1 + y2) / 2),
            )
            regions.append(region)

        return regions, None

    def find_landing_zones(
        self,
        image: Any,
        depth_map: Optional[Any] = None,
        drone_altitude: float = 10.0,
        camera_fov: float = 60.0,
    ) -> List[LandingZone]:
        """
        Find potential landing zones in image.

        Args:
            image: Downward-facing camera image
            depth_map: Optional depth information
            drone_altitude: Current altitude in meters
            camera_fov: Camera field of view in degrees

        Returns:
            List of landing zones sorted by score
        """
        # Get terrain classification
        regions, _ = self.classify_image(image, depth_map)

        # Filter to landable terrain
        landable_regions = [
            r for r in regions
            if r.terrain_type in self._config.landing_terrain_types
        ]

        if not landable_regions:
            return []

        # Convert regions to landing zones
        zones = []
        meters_per_pixel = self._calc_meters_per_pixel(
            drone_altitude, camera_fov, image
        )

        for region in landable_regions:
            zone = self._region_to_landing_zone(
                region,
                meters_per_pixel,
            )
            if zone:
                zones.append(zone)

        # Sort by overall score
        zones.sort(key=lambda z: z.overall_score, reverse=True)

        return zones

    def _calc_meters_per_pixel(
        self,
        altitude: float,
        fov: float,
        image: Any,
    ) -> float:
        """Calculate ground meters per pixel."""
        if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            img_width = image.shape[1]
        else:
            img_width = 640

        # Ground width visible at altitude
        fov_rad = math.radians(fov)
        ground_width = 2 * altitude * math.tan(fov_rad / 2)

        return ground_width / img_width

    def _region_to_landing_zone(
        self,
        region: TerrainRegion,
        meters_per_pixel: float,
    ) -> Optional[LandingZone]:
        """Convert terrain region to landing zone."""
        # Check minimum size
        width_m = (region.bbox[2] - region.bbox[0]) * meters_per_pixel
        height_m = (region.bbox[3] - region.bbox[1]) * meters_per_pixel
        radius_m = min(width_m, height_m) / 2

        if radius_m < self._config.min_zone_radius:
            return None

        # Calculate scores
        flatness = self._estimate_flatness(region)
        obstacles = self._estimate_obstacle_score(region)
        size_score = min(1.0, radius_m / self._config.max_zone_radius)

        # Overall score
        cfg = self._config
        overall = (
            cfg.flatness_weight * flatness +
            cfg.obstacle_weight * obstacles +
            cfg.size_weight * size_score
        )

        # Quality rating
        quality = self._score_to_quality(overall)

        zone = LandingZone(
            zone_id=self._next_zone_id,
            center=region.center,
            radius=radius_m / meters_per_pixel,  # In pixels
            terrain_type=region.terrain_type,
            quality=quality,
            confidence=region.confidence,
            flatness_score=flatness,
            obstacle_score=obstacles,
            size_score=size_score,
            overall_score=overall,
            metadata={'meters_per_pixel': meters_per_pixel},
        )

        self._next_zone_id += 1
        return zone

    def _estimate_flatness(self, region: TerrainRegion) -> float:
        """Estimate flatness score for region."""
        # In simulation, terrain type determines flatness
        flatness_map = {
            TerrainType.CONCRETE: 0.95,
            TerrainType.ASPHALT: 0.9,
            TerrainType.GRASS: 0.7,
            TerrainType.GRAVEL: 0.6,
            TerrainType.SAND: 0.5,
        }
        base = flatness_map.get(region.terrain_type, 0.5)
        # Add some randomness
        return min(1.0, max(0.0, base + random.uniform(-0.1, 0.1)))

    def _estimate_obstacle_score(self, region: TerrainRegion) -> float:
        """Estimate obstacle-free score for region."""
        # In simulation, assume mostly clear
        return random.uniform(0.7, 1.0)

    def _score_to_quality(self, score: float) -> LandingZoneQuality:
        """Convert score to quality rating."""
        if score >= 0.85:
            return LandingZoneQuality.EXCELLENT
        elif score >= 0.7:
            return LandingZoneQuality.GOOD
        elif score >= 0.55:
            return LandingZoneQuality.ACCEPTABLE
        elif score >= 0.4:
            return LandingZoneQuality.MARGINAL
        else:
            return LandingZoneQuality.UNSUITABLE

    def get_best_landing_zone(
        self,
        image: Any,
        **kwargs,
    ) -> Optional[LandingZone]:
        """Get the best landing zone from image."""
        zones = self.find_landing_zones(image, **kwargs)
        return zones[0] if zones else None

    def is_zone_suitable(
        self,
        zone: LandingZone,
        min_quality: LandingZoneQuality = LandingZoneQuality.ACCEPTABLE,
    ) -> bool:
        """Check if zone meets minimum quality."""
        quality_order = [
            LandingZoneQuality.UNSUITABLE,
            LandingZoneQuality.MARGINAL,
            LandingZoneQuality.ACCEPTABLE,
            LandingZoneQuality.GOOD,
            LandingZoneQuality.EXCELLENT,
        ]
        zone_level = quality_order.index(zone.quality)
        min_level = quality_order.index(min_quality)
        return zone_level >= min_level

    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        return {
            'is_simulation': self._is_simulation,
            'frame_count': self._frame_count,
            'zones_generated': self._next_zone_id - 1,
        }
