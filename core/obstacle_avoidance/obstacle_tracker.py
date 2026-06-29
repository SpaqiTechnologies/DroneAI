"""
Obstacle Tracker Module

Multi-sensor fusion for obstacle detection and tracking.
Fuses data from LiDAR, Camera, and Ultrasonic sensors.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
import time
import math
import uuid


class ObstacleType(Enum):
    """Type of detected obstacle."""
    UNKNOWN = "unknown"
    STATIC = "static"           # Stationary obstacle (building, tree, etc.)
    DYNAMIC = "dynamic"         # Moving obstacle (drone, bird, vehicle)
    GROUND = "ground"           # Ground/terrain
    PERSON = "person"           # Human
    VEHICLE = "vehicle"         # Ground vehicle
    AIRCRAFT = "aircraft"       # Other aircraft


class ObstacleSource(Enum):
    """Source sensor for obstacle detection."""
    LIDAR = "lidar"
    CAMERA = "camera"
    ULTRASONIC = "ultrasonic"
    FUSED = "fused"


@dataclass
class ObstaclePosition:
    """3D position of an obstacle."""
    x: float = 0.0      # meters, forward positive
    y: float = 0.0      # meters, right positive
    z: float = 0.0      # meters, down positive (NED frame)

    def distance_to(self, other: 'ObstaclePosition') -> float:
        """Calculate 3D distance to another position."""
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def horizontal_distance_to(self, other: 'ObstaclePosition') -> float:
        """Calculate horizontal distance to another position."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx*dx + dy*dy)

    def bearing_to(self, other: 'ObstaclePosition') -> float:
        """Calculate bearing to another position in degrees."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.degrees(math.atan2(dy, dx))

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)


@dataclass
class ObstacleVelocity:
    """3D velocity of an obstacle."""
    vx: float = 0.0     # m/s, forward positive
    vy: float = 0.0     # m/s, right positive
    vz: float = 0.0     # m/s, down positive

    @property
    def speed(self) -> float:
        """Calculate total speed."""
        return math.sqrt(self.vx*self.vx + self.vy*self.vy + self.vz*self.vz)

    @property
    def horizontal_speed(self) -> float:
        """Calculate horizontal speed."""
        return math.sqrt(self.vx*self.vx + self.vy*self.vy)

    @property
    def heading(self) -> float:
        """Calculate heading in degrees."""
        return math.degrees(math.atan2(self.vy, self.vx))


@dataclass
class TrackedObstacle:
    """A tracked obstacle with position, velocity, and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    position: ObstaclePosition = field(default_factory=ObstaclePosition)
    velocity: ObstacleVelocity = field(default_factory=ObstacleVelocity)

    # Size estimates
    width: float = 1.0          # meters
    height: float = 1.0         # meters
    depth: float = 1.0          # meters

    # Classification
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN
    source: ObstacleSource = ObstacleSource.FUSED

    # Tracking state
    confidence: float = 0.5     # 0-1
    age: float = 0.0            # seconds since first detection
    last_seen: float = 0.0      # timestamp of last observation
    hit_count: int = 0          # number of times detected
    miss_count: int = 0         # consecutive misses

    # Threat assessment
    threat_level: float = 0.0   # 0-1, higher = more threatening
    time_to_collision: float = float('inf')  # seconds

    def is_stale(self, max_age: float = 2.0) -> bool:
        """Check if obstacle data is stale."""
        return (time.time() - self.last_seen) > max_age

    def is_moving(self, threshold: float = 0.5) -> bool:
        """Check if obstacle is moving."""
        return self.velocity.speed > threshold

    def predict_position(self, dt: float) -> ObstaclePosition:
        """Predict position after dt seconds."""
        return ObstaclePosition(
            x=self.position.x + self.velocity.vx * dt,
            y=self.position.y + self.velocity.vy * dt,
            z=self.position.z + self.velocity.vz * dt,
        )

    def get_bounding_sphere_radius(self) -> float:
        """Get bounding sphere radius."""
        return math.sqrt(self.width**2 + self.height**2 + self.depth**2) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'position': {
                'x': self.position.x,
                'y': self.position.y,
                'z': self.position.z,
            },
            'velocity': {
                'vx': self.velocity.vx,
                'vy': self.velocity.vy,
                'vz': self.velocity.vz,
                'speed': self.velocity.speed,
            },
            'size': {
                'width': self.width,
                'height': self.height,
                'depth': self.depth,
            },
            'type': self.obstacle_type.value,
            'source': self.source.value,
            'confidence': self.confidence,
            'threat_level': self.threat_level,
            'time_to_collision': self.time_to_collision if self.time_to_collision < float('inf') else None,
            'is_moving': self.is_moving(),
            'age': self.age,
        }


class ObstacleTracker:
    """
    Multi-sensor obstacle tracking system.

    Fuses obstacle detections from LiDAR, Camera, and Ultrasonic sensors
    to maintain a consistent world model of nearby obstacles.
    """

    # Tracking parameters
    ASSOCIATION_DISTANCE = 2.0      # meters, max distance for track association
    CONFIRMATION_HITS = 3           # hits needed to confirm track
    DELETION_MISSES = 5             # consecutive misses to delete track
    MAX_TRACKS = 50                 # maximum tracked obstacles
    STALE_TIMEOUT = 3.0             # seconds before track is stale

    # Safety margins
    SAFETY_RADIUS = 2.0             # meters, minimum safe distance

    def __init__(self):
        self._tracks: Dict[str, TrackedObstacle] = {}
        self._last_update_time = time.time()

        # Sensor weights for fusion
        self._sensor_weights = {
            ObstacleSource.LIDAR: 0.4,
            ObstacleSource.CAMERA: 0.35,
            ObstacleSource.ULTRASONIC: 0.25,
        }

        # Track statistics
        self._total_detections = 0
        self._confirmed_tracks = 0

    def update(self, dt: float) -> None:
        """
        Update all tracks (prediction step).

        Args:
            dt: Time delta since last update
        """
        current_time = time.time()

        # Predict track positions
        for track in self._tracks.values():
            # Predict position
            track.position = track.predict_position(dt)
            track.age += dt

            # Decay confidence over time
            track.confidence *= 0.99

            # Increment miss count if not seen recently
            if (current_time - track.last_seen) > 0.5:
                track.miss_count += 1

        # Remove stale tracks
        stale_ids = [
            track_id for track_id, track in self._tracks.items()
            if track.is_stale(self.STALE_TIMEOUT) or track.miss_count > self.DELETION_MISSES
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

        self._last_update_time = current_time

    def process_lidar_detections(
        self,
        detections: List[Dict[str, float]],
        drone_position: Tuple[float, float, float] = (0, 0, 0)
    ) -> None:
        """
        Process LiDAR obstacle detections.

        Args:
            detections: List of detection dicts with 'x', 'y', 'z', 'distance'
            drone_position: Current drone position for reference
        """
        for detection in detections:
            position = ObstaclePosition(
                x=detection.get('x', 0),
                y=detection.get('y', 0),
                z=detection.get('z', 0),
            )

            # Estimate size from point cloud density if available
            size = detection.get('size', 1.0)

            self._add_detection(
                position=position,
                source=ObstacleSource.LIDAR,
                confidence=0.8,
                width=size,
                height=size,
                depth=size,
            )

    def process_camera_detections(
        self,
        detections: List[Dict[str, Any]],
        camera_matrix: Optional[List[List[float]]] = None
    ) -> None:
        """
        Process camera obstacle detections.

        Args:
            detections: List of detection dicts from camera sensor
            camera_matrix: Optional camera intrinsic matrix
        """
        for detection in detections:
            # Convert 2D detection to 3D position estimate
            distance = detection.get('distance', 10.0)
            center_x = detection.get('center_x', 320)
            center_y = detection.get('center_y', 240)

            # Simple pinhole camera projection (assuming 640x480, 60 degree FOV)
            fov_h = 60.0
            fov_v = 45.0
            img_w = 640
            img_h = 480

            angle_h = (center_x - img_w/2) / (img_w/2) * (fov_h/2)
            angle_v = (center_y - img_h/2) / (img_h/2) * (fov_v/2)

            # Convert to 3D position
            x = distance * math.cos(math.radians(angle_h))
            y = distance * math.sin(math.radians(angle_h))
            z = distance * math.tan(math.radians(angle_v))

            position = ObstaclePosition(x=x, y=y, z=z)

            # Determine obstacle type from detection
            detection_type = detection.get('type', 'obstacle')
            if detection_type == 'person':
                obstacle_type = ObstacleType.PERSON
            elif detection_type == 'vehicle':
                obstacle_type = ObstacleType.VEHICLE
            elif detection_type == 'aircraft':
                obstacle_type = ObstacleType.AIRCRAFT
            else:
                obstacle_type = ObstacleType.UNKNOWN

            # Get size from bounding box
            width = detection.get('width', 50) / img_w * distance * 2
            height = detection.get('height', 50) / img_h * distance * 2

            self._add_detection(
                position=position,
                source=ObstacleSource.CAMERA,
                confidence=detection.get('confidence', 0.7),
                obstacle_type=obstacle_type,
                width=width,
                height=height,
                depth=width,
            )

    def process_ultrasonic_detection(
        self,
        distance: float,
        direction: str = 'forward'
    ) -> None:
        """
        Process ultrasonic sensor detection.

        Args:
            distance: Distance to obstacle in meters
            direction: Direction of sensor (forward, left, right, down)
        """
        if distance <= 0 or distance > 10.0:
            return

        # Convert direction to position
        if direction == 'forward':
            position = ObstaclePosition(x=distance, y=0, z=0)
        elif direction == 'left':
            position = ObstaclePosition(x=0, y=-distance, z=0)
        elif direction == 'right':
            position = ObstaclePosition(x=0, y=distance, z=0)
        elif direction == 'down':
            position = ObstaclePosition(x=0, y=0, z=distance)
        elif direction == 'back':
            position = ObstaclePosition(x=-distance, y=0, z=0)
        else:
            position = ObstaclePosition(x=distance, y=0, z=0)

        self._add_detection(
            position=position,
            source=ObstacleSource.ULTRASONIC,
            confidence=0.6,
            width=0.5,  # Ultrasonic has poor size estimation
            height=0.5,
            depth=0.5,
        )

    def _add_detection(
        self,
        position: ObstaclePosition,
        source: ObstacleSource,
        confidence: float = 0.5,
        obstacle_type: ObstacleType = ObstacleType.UNKNOWN,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        velocity: Optional[ObstacleVelocity] = None,
    ) -> Optional[TrackedObstacle]:
        """Add a detection and associate with existing track or create new."""
        self._total_detections += 1

        # Find nearest existing track
        nearest_track = None
        min_distance = self.ASSOCIATION_DISTANCE

        for track in self._tracks.values():
            dist = position.distance_to(track.position)
            if dist < min_distance:
                min_distance = dist
                nearest_track = track

        current_time = time.time()

        if nearest_track is not None:
            # Update existing track with weighted fusion
            weight = self._sensor_weights.get(source, 0.3)
            old_weight = 1.0 - weight

            # Fuse position
            nearest_track.position = ObstaclePosition(
                x=old_weight * nearest_track.position.x + weight * position.x,
                y=old_weight * nearest_track.position.y + weight * position.y,
                z=old_weight * nearest_track.position.z + weight * position.z,
            )

            # Estimate velocity from position change
            dt = current_time - nearest_track.last_seen
            if dt > 0 and dt < 1.0:
                dx = position.x - nearest_track.position.x
                dy = position.y - nearest_track.position.y
                dz = position.z - nearest_track.position.z

                alpha = 0.3  # Velocity smoothing
                nearest_track.velocity = ObstacleVelocity(
                    vx=alpha * dx/dt + (1-alpha) * nearest_track.velocity.vx,
                    vy=alpha * dy/dt + (1-alpha) * nearest_track.velocity.vy,
                    vz=alpha * dz/dt + (1-alpha) * nearest_track.velocity.vz,
                )

            # Update metadata
            nearest_track.confidence = min(1.0, nearest_track.confidence + 0.1)
            nearest_track.last_seen = current_time
            nearest_track.hit_count += 1
            nearest_track.miss_count = 0
            nearest_track.source = ObstacleSource.FUSED

            # Update type if we got better classification
            if obstacle_type != ObstacleType.UNKNOWN:
                nearest_track.obstacle_type = obstacle_type

            # Determine if static or dynamic
            if nearest_track.is_moving():
                nearest_track.obstacle_type = ObstacleType.DYNAMIC

            return nearest_track
        else:
            # Create new track if under limit
            if len(self._tracks) >= self.MAX_TRACKS:
                # Remove lowest confidence track
                min_conf_id = min(
                    self._tracks.keys(),
                    key=lambda k: self._tracks[k].confidence
                )
                del self._tracks[min_conf_id]

            new_track = TrackedObstacle(
                position=position,
                velocity=velocity if velocity else ObstacleVelocity(),
                width=width,
                height=height,
                depth=depth,
                obstacle_type=obstacle_type,
                source=source,
                confidence=confidence * 0.5,  # Start with lower confidence
                last_seen=current_time,
                hit_count=1,
            )

            self._tracks[new_track.id] = new_track
            return new_track

    def get_obstacles(
        self,
        confirmed_only: bool = True,
        max_distance: Optional[float] = None,
    ) -> List[TrackedObstacle]:
        """
        Get list of tracked obstacles.

        Args:
            confirmed_only: Only return confirmed tracks
            max_distance: Optional maximum distance filter

        Returns:
            List of tracked obstacles
        """
        obstacles = []

        for track in self._tracks.values():
            # Filter unconfirmed
            if confirmed_only and track.hit_count < self.CONFIRMATION_HITS:
                continue

            # Filter by distance
            if max_distance is not None:
                dist = math.sqrt(
                    track.position.x**2 +
                    track.position.y**2 +
                    track.position.z**2
                )
                if dist > max_distance:
                    continue

            obstacles.append(track)

        # Sort by distance
        obstacles.sort(
            key=lambda o: math.sqrt(o.position.x**2 + o.position.y**2 + o.position.z**2)
        )

        return obstacles

    def get_nearest_obstacle(
        self,
        direction: Optional[Tuple[float, float, float]] = None
    ) -> Optional[TrackedObstacle]:
        """
        Get nearest obstacle, optionally in a specific direction.

        Args:
            direction: Optional direction vector (x, y, z) to search

        Returns:
            Nearest obstacle or None
        """
        obstacles = self.get_obstacles(confirmed_only=True)
        if not obstacles:
            return None

        if direction is None:
            return obstacles[0]

        # Find nearest in direction
        dx, dy, dz = direction
        dir_mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dir_mag == 0:
            return obstacles[0]

        dx, dy, dz = dx/dir_mag, dy/dir_mag, dz/dir_mag

        best_obstacle = None
        best_score = float('inf')

        for obstacle in obstacles:
            ox, oy, oz = obstacle.position.x, obstacle.position.y, obstacle.position.z
            dist = math.sqrt(ox*ox + oy*oy + oz*oz)
            if dist == 0:
                continue

            # Calculate alignment with direction
            alignment = (ox*dx + oy*dy + oz*dz) / dist

            # Score: lower is better (close + aligned)
            if alignment > 0:  # Only consider obstacles in front
                score = dist / (alignment + 0.1)
                if score < best_score:
                    best_score = score
                    best_obstacle = obstacle

        return best_obstacle

    def get_obstacles_in_cone(
        self,
        direction: Tuple[float, float, float],
        cone_angle: float = 30.0,
        max_distance: float = 50.0,
    ) -> List[TrackedObstacle]:
        """
        Get obstacles within a cone in a given direction.

        Args:
            direction: Direction vector (x, y, z)
            cone_angle: Half-angle of cone in degrees
            max_distance: Maximum distance to search

        Returns:
            List of obstacles in cone
        """
        dx, dy, dz = direction
        dir_mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dir_mag == 0:
            return []

        dx, dy, dz = dx/dir_mag, dy/dir_mag, dz/dir_mag
        cos_threshold = math.cos(math.radians(cone_angle))

        result = []
        for obstacle in self.get_obstacles(confirmed_only=True):
            ox, oy, oz = obstacle.position.x, obstacle.position.y, obstacle.position.z
            dist = math.sqrt(ox*ox + oy*oy + oz*oz)

            if dist > max_distance or dist == 0:
                continue

            # Calculate alignment
            alignment = (ox*dx + oy*dy + oz*dz) / dist

            if alignment >= cos_threshold:
                result.append(obstacle)

        return result

    def update_threat_levels(
        self,
        drone_velocity: Tuple[float, float, float] = (0, 0, 0),
        safety_radius: float = 2.0,
    ) -> None:
        """
        Update threat levels for all tracked obstacles.

        Args:
            drone_velocity: Current drone velocity (vx, vy, vz)
            safety_radius: Safety radius around drone
        """
        vx, vy, vz = drone_velocity

        for track in self._tracks.values():
            # Distance-based threat
            dist = math.sqrt(
                track.position.x**2 +
                track.position.y**2 +
                track.position.z**2
            )

            # Relative velocity (closing rate)
            rel_vx = track.velocity.vx - vx
            rel_vy = track.velocity.vy - vy
            rel_vz = track.velocity.vz - vz

            # Closing speed (positive = approaching)
            if dist > 0:
                closing_speed = -(
                    track.position.x * rel_vx +
                    track.position.y * rel_vy +
                    track.position.z * rel_vz
                ) / dist
            else:
                closing_speed = 0

            # Time to collision
            if closing_speed > 0:
                track.time_to_collision = (dist - safety_radius) / closing_speed
            else:
                track.time_to_collision = float('inf')

            # Calculate threat level
            # Based on: distance, closing speed, time to collision, confidence
            threat = 0.0

            # Distance threat (closer = higher threat)
            if dist < safety_radius:
                threat += 0.5
            elif dist < safety_radius * 2:
                threat += 0.3 * (1 - (dist - safety_radius) / safety_radius)
            else:
                threat += 0.1 * max(0, 1 - dist / 20)

            # Closing speed threat
            if closing_speed > 0:
                threat += 0.3 * min(1, closing_speed / 10)

            # Time to collision threat
            if track.time_to_collision < 5:
                threat += 0.2 * (1 - track.time_to_collision / 5)

            # Weight by confidence
            track.threat_level = min(1.0, threat * track.confidence)

    def clear(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        confirmed = sum(
            1 for t in self._tracks.values()
            if t.hit_count >= self.CONFIRMATION_HITS
        )

        return {
            'total_tracks': len(self._tracks),
            'confirmed_tracks': confirmed,
            'total_detections': self._total_detections,
            'moving_obstacles': sum(1 for t in self._tracks.values() if t.is_moving()),
            'high_threat': sum(1 for t in self._tracks.values() if t.threat_level > 0.7),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        obstacles = self.get_obstacles(confirmed_only=True)

        return {
            'active': True,
            'track_count': len(self._tracks),
            'confirmed_count': len(obstacles),
            'nearest_distance': min(
                (math.sqrt(o.position.x**2 + o.position.y**2 + o.position.z**2)
                 for o in obstacles),
                default=None
            ),
            'max_threat': max(
                (o.threat_level for o in obstacles),
                default=0.0
            ),
            'obstacles': [o.to_dict() for o in obstacles[:10]],  # Top 10
        }
