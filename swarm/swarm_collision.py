"""
Swarm collision avoidance for inter-drone safety.

Detects potential collisions between drones and
computes avoidance maneuvers.
"""

import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

from .drone_registry import DroneInfo, DronePosition, DroneVelocity

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Collision threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CollisionThreat:
    """Information about a potential collision."""
    other_drone_id: str
    distance: float
    time_to_collision: float
    closest_approach: float
    threat_level: ThreatLevel
    relative_velocity: Tuple[float, float, float]
    avoidance_vector: Tuple[float, float, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'other_drone_id': self.other_drone_id,
            'distance': self.distance,
            'time_to_collision': self.time_to_collision,
            'closest_approach': self.closest_approach,
            'threat_level': self.threat_level.value,
            'relative_velocity': self.relative_velocity,
            'avoidance_vector': self.avoidance_vector,
        }


@dataclass
class AvoidanceConfig:
    """Configuration for collision avoidance."""
    min_separation: float = 5.0  # meters
    warning_distance: float = 20.0  # meters
    critical_time: float = 3.0  # seconds
    avoidance_gain: float = 2.0  # strength of avoidance
    vertical_preference: float = 0.3  # prefer vertical avoidance
    max_avoidance_speed: float = 5.0  # m/s


class SwarmCollisionAvoidance:
    """
    Inter-drone collision detection and avoidance.

    Uses a cooperative approach where all drones
    share position/velocity and compute avoidance.
    """

    # Earth radius for distance calculations
    EARTH_RADIUS = 6371000.0

    def __init__(self, config: Optional[AvoidanceConfig] = None):
        """
        Initialize collision avoidance.

        Args:
            config: Avoidance configuration
        """
        self._config = config or AvoidanceConfig()
        self._threats: Dict[str, List[CollisionThreat]] = {}
        self._last_check: Dict[str, float] = {}

    @property
    def config(self) -> AvoidanceConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def check_collisions(
        self,
        drone_id: str,
        own_position: DronePosition,
        own_velocity: DroneVelocity,
        other_drones: List[DroneInfo],
    ) -> List[CollisionThreat]:
        """
        Check for potential collisions with other drones.

        Args:
            drone_id: This drone's ID
            own_position: This drone's position
            own_velocity: This drone's velocity
            other_drones: List of other drone info

        Returns:
            List of collision threats
        """
        threats = []

        for other in other_drones:
            if other.drone_id == drone_id:
                continue
            if not other.is_online:
                continue

            threat = self._evaluate_threat(
                own_position,
                own_velocity,
                other.position,
                other.velocity,
                other.drone_id,
            )

            if threat.threat_level != ThreatLevel.NONE:
                threats.append(threat)

        # Sort by threat level and time to collision
        threats.sort(
            key=lambda t: (
                -list(ThreatLevel).index(t.threat_level),
                t.time_to_collision,
            )
        )

        self._threats[drone_id] = threats
        self._last_check[drone_id] = time.time()

        return threats

    def _evaluate_threat(
        self,
        pos1: DronePosition,
        vel1: DroneVelocity,
        pos2: DronePosition,
        vel2: DroneVelocity,
        other_id: str,
    ) -> CollisionThreat:
        """Evaluate collision threat between two drones."""
        # Calculate relative position
        dx, dy, dz = self._compute_relative_position(pos1, pos2)
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate relative velocity
        dvx = vel1.vx - vel2.vx
        dvy = vel1.vy - vel2.vy
        dvz = vel1.vz - vel2.vz
        rel_speed = math.sqrt(dvx**2 + dvy**2 + dvz**2)

        # Time to closest approach (CPA)
        ttc, cpa_dist = self._compute_cpa(
            dx, dy, dz,
            dvx, dvy, dvz,
        )

        # Determine threat level
        threat_level = self._classify_threat(
            distance, ttc, cpa_dist
        )

        # Compute avoidance vector if needed
        avoidance = (0.0, 0.0, 0.0)
        if threat_level not in [ThreatLevel.NONE, ThreatLevel.LOW]:
            avoidance = self._compute_avoidance(
                dx, dy, dz,
                distance,
                threat_level,
            )

        return CollisionThreat(
            other_drone_id=other_id,
            distance=distance,
            time_to_collision=ttc,
            closest_approach=cpa_dist,
            threat_level=threat_level,
            relative_velocity=(dvx, dvy, dvz),
            avoidance_vector=avoidance,
        )

    def _compute_relative_position(
        self,
        pos1: DronePosition,
        pos2: DronePosition,
    ) -> Tuple[float, float, float]:
        """
        Compute relative position in meters.

        Returns:
            (north, east, down) relative position
        """
        lat1 = math.radians(pos1.lat)
        lat2 = math.radians(pos2.lat)
        dlon = math.radians(pos2.lon - pos1.lon)
        dlat = lat2 - lat1

        # North-East-Down coordinates
        north = dlat * self.EARTH_RADIUS
        east = dlon * self.EARTH_RADIUS * math.cos(lat1)
        down = pos1.alt - pos2.alt

        return (north, east, down)

    def _compute_cpa(
        self,
        dx: float,
        dy: float,
        dz: float,
        dvx: float,
        dvy: float,
        dvz: float,
    ) -> Tuple[float, float]:
        """
        Compute time to closest point of approach.

        Returns:
            (time_to_cpa, cpa_distance)
        """
        # Relative position and velocity magnitudes
        d_dot_v = dx * dvx + dy * dvy + dz * dvz
        v_sq = dvx**2 + dvy**2 + dvz**2

        if v_sq < 1e-6:
            # Stationary relative to each other
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            return (float('inf'), dist)

        # Time to CPA
        ttc = -d_dot_v / v_sq

        if ttc < 0:
            # Already passed CPA, moving apart
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            return (float('inf'), dist)

        # Position at CPA
        cpa_x = dx + dvx * ttc
        cpa_y = dy + dvy * ttc
        cpa_z = dz + dvz * ttc
        cpa_dist = math.sqrt(cpa_x**2 + cpa_y**2 + cpa_z**2)

        return (ttc, cpa_dist)

    def _classify_threat(
        self,
        distance: float,
        ttc: float,
        cpa_dist: float,
    ) -> ThreatLevel:
        """Classify threat level based on distance and timing."""
        cfg = self._config

        # Already too close
        if distance < cfg.min_separation:
            return ThreatLevel.CRITICAL

        # Will get too close soon
        if cpa_dist < cfg.min_separation:
            if ttc < cfg.critical_time:
                return ThreatLevel.CRITICAL
            elif ttc < cfg.critical_time * 2:
                return ThreatLevel.HIGH
            elif ttc < cfg.critical_time * 4:
                return ThreatLevel.MEDIUM

        # Approaching warning zone
        if distance < cfg.warning_distance:
            if ttc < cfg.critical_time * 2:
                return ThreatLevel.MEDIUM
            else:
                return ThreatLevel.LOW

        return ThreatLevel.NONE

    def _compute_avoidance(
        self,
        dx: float,
        dy: float,
        dz: float,
        distance: float,
        threat_level: ThreatLevel,
    ) -> Tuple[float, float, float]:
        """
        Compute avoidance velocity vector.

        Returns:
            (vx, vy, vz) avoidance velocity in m/s
        """
        cfg = self._config

        if distance < 1e-3:
            # Nearly on top of each other, move up
            return (0.0, 0.0, -cfg.max_avoidance_speed)

        # Normalize relative position (points away from other)
        inv_dist = 1.0 / distance
        nx = -dx * inv_dist
        ny = -dy * inv_dist
        nz = -dz * inv_dist

        # Strength based on threat level
        strength_map = {
            ThreatLevel.CRITICAL: 1.0,
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.MEDIUM: 0.4,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.NONE: 0.0,
        }
        strength = strength_map.get(threat_level, 0.0)

        # Scale by inverse distance for urgency
        urgency = cfg.min_separation / max(distance, cfg.min_separation)
        strength *= urgency * cfg.avoidance_gain

        # Prefer vertical separation
        vertical_boost = 1.0 + cfg.vertical_preference
        nz *= vertical_boost

        # Renormalize
        mag = math.sqrt(nx**2 + ny**2 + nz**2)
        if mag > 1e-6:
            nx /= mag
            ny /= mag
            nz /= mag

        # Scale to max speed
        speed = min(
            strength * cfg.max_avoidance_speed,
            cfg.max_avoidance_speed,
        )

        return (nx * speed, ny * speed, nz * speed)

    def get_combined_avoidance(
        self,
        drone_id: str,
    ) -> Tuple[float, float, float]:
        """
        Get combined avoidance vector for all threats.

        Args:
            drone_id: Drone to get avoidance for

        Returns:
            Combined (vx, vy, vz) avoidance velocity
        """
        threats = self._threats.get(drone_id, [])
        if not threats:
            return (0.0, 0.0, 0.0)

        # Sum weighted avoidance vectors
        total_x = 0.0
        total_y = 0.0
        total_z = 0.0

        for threat in threats:
            weight = 1.0 / max(threat.time_to_collision, 0.1)
            ax, ay, az = threat.avoidance_vector
            total_x += ax * weight
            total_y += ay * weight
            total_z += az * weight

        # Normalize total weight
        total_weight = sum(
            1.0 / max(t.time_to_collision, 0.1) for t in threats
        )
        if total_weight > 0:
            total_x /= total_weight
            total_y /= total_weight
            total_z /= total_weight

        # Clamp to max speed
        mag = math.sqrt(total_x**2 + total_y**2 + total_z**2)
        max_speed = self._config.max_avoidance_speed
        if mag > max_speed:
            scale = max_speed / mag
            total_x *= scale
            total_y *= scale
            total_z *= scale

        return (total_x, total_y, total_z)

    def get_threats(
        self,
        drone_id: str,
    ) -> List[CollisionThreat]:
        """Get current threats for a drone."""
        return self._threats.get(drone_id, []).copy()

    def get_highest_threat(
        self,
        drone_id: str,
    ) -> Optional[CollisionThreat]:
        """Get the most severe threat for a drone."""
        threats = self._threats.get(drone_id, [])
        if not threats:
            return None
        return threats[0]  # Already sorted by severity

    def is_safe(self, drone_id: str) -> bool:
        """Check if drone has no significant threats."""
        threats = self._threats.get(drone_id, [])
        for threat in threats:
            if threat.threat_level in [
                ThreatLevel.HIGH,
                ThreatLevel.CRITICAL,
            ]:
                return False
        return True

    def clear_threats(self, drone_id: str) -> None:
        """Clear recorded threats for a drone."""
        self._threats.pop(drone_id, None)
        self._last_check.pop(drone_id, None)
