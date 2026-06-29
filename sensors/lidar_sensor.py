"""
LiDAR sensor module for Drone AI application.

Provides 3D point cloud data for obstacle detection and SLAM.
Used for GPS-denied navigation and precise mapping.

Navigation Redundancy Strategy:
4. LiDAR SLAM (if equipped) - This module

Typical sensors: Velodyne VLP-16, Livox Mid-360, Ouster OS1
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
from .sensor import Sensor


class LiDARState(Enum):
    """Operational state of LiDAR sensor."""
    OFF = "off"
    INITIALIZING = "initializing"
    READY = "ready"
    SCANNING = "scanning"
    DEGRADED = "degraded"
    FAILED = "failed"


class LiDARType(Enum):
    """Type of LiDAR sensor."""
    SPINNING = "spinning"      # Traditional rotating LiDAR (Velodyne)
    SOLID_STATE = "solid_state"  # Non-mechanical (Livox)
    FLASH = "flash"            # Single-shot 3D (short range)


@dataclass
class Point3D:
    """A single 3D point from LiDAR scan."""
    x: float  # Forward (meters)
    y: float  # Right (meters)
    z: float  # Down (meters)
    intensity: float = 0.0  # Reflectivity (0-255)
    ring: int = 0  # Which laser ring (for multi-beam)
    timestamp: float = 0.0

    def distance(self) -> float:
        """Distance from sensor origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return as (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'intensity': self.intensity,
            'ring': self.ring,
            'distance': self.distance()
        }


@dataclass
class PointCloud:
    """Collection of 3D points from a LiDAR scan."""
    points: List[Point3D] = field(default_factory=list)
    timestamp: float = 0.0
    frame_id: int = 0

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def add_point(self, point: Point3D):
        """Add a point to the cloud."""
        self.points.append(point)

    def get_points_in_range(self, min_dist: float, max_dist: float) -> List[Point3D]:
        """Get points within distance range."""
        return [p for p in self.points if min_dist <= p.distance() <= max_dist]

    def get_nearest_point(self) -> Optional[Point3D]:
        """Get the closest point to sensor."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.distance())

    def get_min_distance(self) -> float:
        """Get minimum distance to any point."""
        if not self.points:
            return float('inf')
        return min(p.distance() for p in self.points)

    def get_centroid(self) -> Optional[Tuple[float, float, float]]:
        """Calculate centroid of point cloud."""
        if not self.points:
            return None
        n = len(self.points)
        cx = sum(p.x for p in self.points) / n
        cy = sum(p.y for p in self.points) / n
        cz = sum(p.z for p in self.points) / n
        return (cx, cy, cz)

    def downsample(self, voxel_size: float) -> 'PointCloud':
        """Downsample using voxel grid filter."""
        if not self.points:
            return PointCloud(timestamp=self.timestamp, frame_id=self.frame_id)

        # Use voxel grid to group points
        voxels: Dict[Tuple[int, int, int], List[Point3D]] = {}

        for point in self.points:
            vx = int(point.x / voxel_size)
            vy = int(point.y / voxel_size)
            vz = int(point.z / voxel_size)
            key = (vx, vy, vz)

            if key not in voxels:
                voxels[key] = []
            voxels[key].append(point)

        # Take centroid of each voxel
        downsampled = PointCloud(timestamp=self.timestamp, frame_id=self.frame_id)
        for points_in_voxel in voxels.values():
            n = len(points_in_voxel)
            avg_x = sum(p.x for p in points_in_voxel) / n
            avg_y = sum(p.y for p in points_in_voxel) / n
            avg_z = sum(p.z for p in points_in_voxel) / n
            avg_intensity = sum(p.intensity for p in points_in_voxel) / n

            downsampled.add_point(Point3D(
                x=avg_x, y=avg_y, z=avg_z,
                intensity=avg_intensity,
                ring=points_in_voxel[0].ring,
                timestamp=self.timestamp
            ))

        return downsampled

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'point_count': len(self.points),
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'min_distance': self.get_min_distance(),
            'centroid': self.get_centroid()
        }


@dataclass
class LiDARReading:
    """Complete LiDAR measurement including point cloud and metadata."""
    point_cloud: PointCloud
    timestamp: float
    scan_duration: float  # Time to complete scan

    # Scan statistics
    points_per_second: int
    valid_points: int
    invalid_points: int  # Points filtered out (too close, too far, etc.)

    # Quality metrics
    coverage_percent: float  # Percentage of expected FOV covered
    noise_level: float  # Estimated noise in measurements

    # Sensor state
    state: LiDARState
    motor_rpm: float  # For spinning LiDAR
    temperature: float  # Sensor temperature
    drone_pose: Optional[Tuple[float, float, float, float]] = None  # x, y, z, yaw for simulation

    def is_valid(self) -> bool:
        """Check if reading is valid for navigation."""
        return (
            self.state == LiDARState.SCANNING and
            self.valid_points >= 100 and
            self.coverage_percent >= 50.0
        )

    def get_obstacle_distance(self) -> float:
        """Get distance to nearest obstacle."""
        return self.point_cloud.get_min_distance()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'point_cloud': self.point_cloud.to_dict(),
            'timestamp': self.timestamp,
            'scan_duration': self.scan_duration,
            'points_per_second': self.points_per_second,
            'valid_points': self.valid_points,
            'invalid_points': self.invalid_points,
            'coverage_percent': self.coverage_percent,
            'noise_level': self.noise_level,
            'state': self.state.value,
            'motor_rpm': self.motor_rpm,
            'temperature': self.temperature,
            'is_valid': self.is_valid(),
            'obstacle_distance': self.get_obstacle_distance()
        }


@dataclass
class Obstacle:
    """Detected obstacle from point cloud analysis."""
    position: Tuple[float, float, float]  # Center (x, y, z)
    size: Tuple[float, float, float]  # Bounding box (width, depth, height)
    distance: float  # Distance from sensor
    confidence: float  # Detection confidence (0-1)
    point_count: int  # Number of points in cluster

    def to_dict(self) -> dict:
        return {
            'position': self.position,
            'size': self.size,
            'distance': self.distance,
            'confidence': self.confidence,
            'point_count': self.point_count
        }


class LiDARSensor(Sensor):
    """
    LiDAR sensor for 3D point cloud generation and obstacle detection.

    Simulates a typical drone-mounted LiDAR with configurable parameters.
    Supports both spinning and solid-state LiDAR types.

    Specifications (based on Livox Mid-360 / VLP-16):
    - Range: 0.5m to 120m (configurable)
    - FOV: 360° horizontal, ±15° vertical (spinning)
    - Points per second: 100,000 to 600,000
    - Update rate: 10-20 Hz
    - Accuracy: ±2-3 cm

    Uses:
    - Obstacle detection and avoidance
    - SLAM (Simultaneous Localization and Mapping)
    - Terrain following
    - 3D mapping
    """

    # Default specifications
    DEFAULT_MIN_RANGE = 0.5      # meters
    DEFAULT_MAX_RANGE = 100.0    # meters
    DEFAULT_H_FOV = 360.0        # degrees (horizontal)
    DEFAULT_V_FOV = 30.0         # degrees (vertical, ±15°)
    DEFAULT_H_RESOLUTION = 0.2  # degrees
    DEFAULT_V_RESOLUTION = 2.0  # degrees
    DEFAULT_UPDATE_RATE = 10    # Hz
    DEFAULT_POINTS_PER_SEC = 300000

    # Noise parameters
    RANGE_NOISE_STD = 0.02      # meters
    ANGLE_NOISE_STD = 0.01      # degrees

    def __init__(
        self,
        lidar_type: LiDARType = LiDARType.SPINNING,
        min_range: float = DEFAULT_MIN_RANGE,
        max_range: float = DEFAULT_MAX_RANGE,
        h_fov: float = DEFAULT_H_FOV,
        v_fov: float = DEFAULT_V_FOV,
        h_resolution: float = DEFAULT_H_RESOLUTION,
        v_resolution: float = DEFAULT_V_RESOLUTION,
        update_rate: int = DEFAULT_UPDATE_RATE,
        points_per_second: int = DEFAULT_POINTS_PER_SEC
    ):
        """
        Initialize LiDAR sensor.

        Args:
            lidar_type: Type of LiDAR (spinning, solid_state, flash)
            min_range: Minimum detection range (m)
            max_range: Maximum detection range (m)
            h_fov: Horizontal field of view (degrees)
            v_fov: Vertical field of view (degrees)
            h_resolution: Horizontal angular resolution (degrees)
            v_resolution: Vertical angular resolution (degrees)
            update_rate: Scan update rate (Hz)
            points_per_second: Maximum points per second
        """
        self.type = "lidar"
        self._lidar_type = lidar_type

        # Range parameters
        self._min_range = min_range
        self._max_range = max_range

        # FOV parameters
        self._h_fov = h_fov
        self._v_fov = v_fov
        self._h_resolution = h_resolution
        self._v_resolution = v_resolution

        # Update parameters
        self._update_rate = update_rate
        self._update_interval = 1.0 / update_rate
        self._points_per_second = points_per_second

        # Calculate number of beams
        self._h_beams = int(h_fov / h_resolution)
        self._v_beams = int(v_fov / v_resolution)

        # Sensor state
        self._state = LiDARState.OFF
        self._is_connected = False
        self._motor_rpm = 0.0
        self._temperature = 25.0

        # Timing
        self._last_update_time = 0.0
        self._frame_count = 0

        # Last reading
        self._last_reading: Optional[LiDARReading] = None
        self._last_point_cloud: Optional[PointCloud] = None

        # Simulation: environment obstacles
        self._simulated_obstacles: List[Dict] = []
        self._ground_height = 0.0  # Height of ground plane

        # Drone pose for simulation
        self._drone_x = 0.0
        self._drone_y = 0.0
        self._drone_z = 0.0  # Height above ground
        self._drone_yaw = 0.0  # Heading in radians

        # Noise parameters
        self._range_noise = self.RANGE_NOISE_STD
        self._angle_noise = self.ANGLE_NOISE_STD

    def start(self):
        """Initialize and start LiDAR sensor."""
        self._is_connected = True
        self._state = LiDARState.INITIALIZING
        self._last_update_time = time.time()

        # Simulate motor spin-up for spinning LiDAR
        if self._lidar_type == LiDARType.SPINNING:
            self._motor_rpm = 600.0  # Typical RPM

        self._state = LiDARState.READY

    def stop(self):
        """Stop LiDAR sensor."""
        self._is_connected = False
        self._state = LiDARState.OFF
        self._motor_rpm = 0.0

    def update(self) -> LiDARReading:
        """
        Perform a LiDAR scan and return point cloud.

        Returns:
            LiDARReading with point cloud and metadata
        """
        if not self._is_connected:
            return self._create_invalid_reading()

        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else self._update_interval

        if dt <= 0:
            dt = self._update_interval

        self._state = LiDARState.SCANNING

        # Generate point cloud
        point_cloud = self._simulate_scan()

        # Calculate statistics
        valid_points = len(point_cloud)
        expected_points = int(self._points_per_second * dt)
        coverage = min(100.0, (valid_points / max(1, expected_points)) * 100)

        # Create reading
        reading = LiDARReading(
            point_cloud=point_cloud,
            timestamp=current_time,
            scan_duration=dt,
            points_per_second=int(valid_points / dt) if dt > 0 else 0,
            valid_points=valid_points,
            invalid_points=0,
            coverage_percent=coverage,
            noise_level=self._range_noise,
            state=self._state,
            motor_rpm=self._motor_rpm,
            temperature=self._temperature + random.gauss(0, 0.5),
            drone_pose=(self._drone_x, self._drone_y, self._drone_z, self._drone_yaw),
        )

        self._last_reading = reading
        self._last_point_cloud = point_cloud
        self._last_update_time = current_time
        self._frame_count += 1

        return reading

    def measure(self) -> LiDARReading:
        """Alias for update() to match sensor interface."""
        return self.update()

    def _simulate_scan(self) -> PointCloud:
        """
        Simulate a LiDAR scan based on environment.

        Uses ray casting to detect obstacles and ground.
        """
        point_cloud = PointCloud(timestamp=time.time(), frame_id=self._frame_count)

        # Calculate angular steps
        h_step = self._h_resolution
        v_step = self._v_resolution

        # Reduce point density for simulation performance
        h_skip = max(1, self._h_beams // 180)  # Max 180 horizontal rays
        v_skip = max(1, self._v_beams // 16)   # Max 16 vertical rays

        v_start = -self._v_fov / 2
        h_start = -self._h_fov / 2

        ring = 0
        for v_idx in range(0, self._v_beams, v_skip):
            v_angle = v_start + v_idx * v_step
            v_rad = math.radians(v_angle)

            for h_idx in range(0, self._h_beams, h_skip):
                h_angle = h_start + h_idx * h_step
                h_rad = math.radians(h_angle) + self._drone_yaw

                # Add angle noise
                h_rad += math.radians(random.gauss(0, self._angle_noise))
                v_rad += math.radians(random.gauss(0, self._angle_noise))

                # Ray cast to find intersection
                hit_distance = self._ray_cast(h_rad, v_rad)

                if hit_distance is not None and self._min_range <= hit_distance <= self._max_range:
                    # Add range noise
                    noisy_distance = hit_distance + random.gauss(0, self._range_noise)
                    noisy_distance = max(self._min_range, noisy_distance)

                    # Convert to Cartesian coordinates (body frame)
                    x = noisy_distance * math.cos(v_rad) * math.cos(h_rad - self._drone_yaw)
                    y = noisy_distance * math.cos(v_rad) * math.sin(h_rad - self._drone_yaw)
                    z = noisy_distance * math.sin(v_rad)

                    # Random intensity based on distance and angle
                    intensity = 255 * (1 - hit_distance / self._max_range) * random.uniform(0.8, 1.0)

                    point_cloud.add_point(Point3D(
                        x=x, y=y, z=z,
                        intensity=intensity,
                        ring=ring,
                        timestamp=time.time()
                    ))

            ring += 1

        return point_cloud

    def _ray_cast(self, h_angle: float, v_angle: float) -> Optional[float]:
        """
        Cast a ray and find intersection with obstacles or ground.

        Args:
            h_angle: Horizontal angle (radians, world frame)
            v_angle: Vertical angle (radians, negative = down)

        Returns:
            Distance to intersection, or None if no hit
        """
        # Ray direction in world frame
        dx = math.cos(v_angle) * math.cos(h_angle)
        dy = math.cos(v_angle) * math.sin(h_angle)
        dz = math.sin(v_angle)  # Positive = up

        min_distance = None

        # Check intersection with ground plane
        if dz < -0.01:  # Ray pointing downward
            # Ground is at z = -drone_z (relative to drone)
            t = -self._drone_z / dz
            if self._min_range < t < self._max_range:
                min_distance = t

        # Check intersection with simulated obstacles
        for obstacle in self._simulated_obstacles:
            obs_x = obstacle['x'] - self._drone_x
            obs_y = obstacle['y'] - self._drone_y
            obs_z = obstacle['z'] - self._drone_z
            obs_radius = obstacle.get('radius', 1.0)

            # Simple sphere intersection
            t = self._ray_sphere_intersect(
                0, 0, 0,  # Ray origin (drone position)
                dx, dy, dz,  # Ray direction
                obs_x, obs_y, obs_z, obs_radius  # Sphere
            )

            if t is not None and self._min_range < t < self._max_range:
                if min_distance is None or t < min_distance:
                    min_distance = t

        return min_distance

    def _ray_sphere_intersect(
        self,
        ox: float, oy: float, oz: float,  # Ray origin
        dx: float, dy: float, dz: float,  # Ray direction
        cx: float, cy: float, cz: float,  # Sphere center
        r: float  # Sphere radius
    ) -> Optional[float]:
        """
        Calculate ray-sphere intersection.

        Returns distance to intersection or None.
        """
        # Vector from ray origin to sphere center
        ocx = ox - cx
        ocy = oy - cy
        ocz = oz - cz

        # Quadratic formula coefficients
        a = dx*dx + dy*dy + dz*dz
        b = 2 * (ocx*dx + ocy*dy + ocz*dz)
        c = ocx*ocx + ocy*ocy + ocz*ocz - r*r

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None

        # Return nearest positive intersection
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None

    def _create_invalid_reading(self) -> LiDARReading:
        """Create an invalid reading for error states."""
        return LiDARReading(
            point_cloud=PointCloud(),
            timestamp=time.time(),
            scan_duration=0.0,
            points_per_second=0,
            valid_points=0,
            invalid_points=0,
            coverage_percent=0.0,
            noise_level=0.0,
            state=LiDARState.FAILED,
            motor_rpm=0.0,
            temperature=0.0
        )

    # ==================== Obstacle Detection ====================

    def detect_obstacles(self, min_points: int = 10, cluster_distance: float = 0.5) -> List[Obstacle]:
        """
        Detect obstacles from current point cloud using simple clustering.

        Args:
            min_points: Minimum points to form an obstacle
            cluster_distance: Maximum distance between points in cluster

        Returns:
            List of detected obstacles
        """
        if self._last_point_cloud is None or len(self._last_point_cloud) == 0:
            return []

        # Simple Euclidean clustering
        points = list(self._last_point_cloud.points)
        clusters: List[List[Point3D]] = []
        used = [False] * len(points)

        for i, point in enumerate(points):
            if used[i]:
                continue

            # Start new cluster
            cluster = [point]
            used[i] = True
            queue = [i]

            while queue:
                current_idx = queue.pop(0)
                current = points[current_idx]

                for j, other in enumerate(points):
                    if used[j]:
                        continue

                    # Check distance
                    dist = math.sqrt(
                        (current.x - other.x)**2 +
                        (current.y - other.y)**2 +
                        (current.z - other.z)**2
                    )

                    if dist <= cluster_distance:
                        cluster.append(other)
                        used[j] = True
                        queue.append(j)

            if len(cluster) >= min_points:
                clusters.append(cluster)

        # Convert clusters to obstacles
        obstacles = []
        for cluster in clusters:
            # Calculate bounding box
            xs = [p.x for p in cluster]
            ys = [p.y for p in cluster]
            zs = [p.z for p in cluster]

            center = (
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs)
            )

            size = (
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs)
            )

            distance = math.sqrt(center[0]**2 + center[1]**2 + center[2]**2)
            confidence = min(1.0, len(cluster) / 100)

            obstacles.append(Obstacle(
                position=center,
                size=size,
                distance=distance,
                confidence=confidence,
                point_count=len(cluster)
            ))

        # Sort by distance
        obstacles.sort(key=lambda o: o.distance)
        return obstacles

    def get_nearest_obstacle_distance(self) -> float:
        """Get distance to nearest obstacle."""
        if self._last_point_cloud is None:
            return float('inf')
        return self._last_point_cloud.get_min_distance()

    def check_obstacle_in_direction(
        self,
        direction_angle: float,
        cone_angle: float = 30.0,
        max_distance: float = 10.0
    ) -> Optional[float]:
        """
        Check for obstacles in a specific direction.

        Args:
            direction_angle: Heading angle to check (degrees, 0=forward)
            cone_angle: Search cone half-angle (degrees)
            max_distance: Maximum search distance (meters)

        Returns:
            Distance to nearest obstacle in that direction, or None
        """
        if self._last_point_cloud is None:
            return None

        dir_rad = math.radians(direction_angle)
        cone_rad = math.radians(cone_angle)

        min_dist = None

        for point in self._last_point_cloud.points:
            # Calculate angle to point in horizontal plane
            point_angle = math.atan2(point.y, point.x)
            angle_diff = abs(point_angle - dir_rad)

            # Normalize angle difference
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            # Check if within cone
            if angle_diff <= cone_rad:
                dist = math.sqrt(point.x**2 + point.y**2)
                if dist <= max_distance:
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

        return min_dist

    # ==================== Getters ====================

    def get_point_cloud(self) -> Optional[PointCloud]:
        """Get last point cloud."""
        return self._last_point_cloud

    def get_state(self) -> LiDARState:
        """Get sensor state."""
        return self._state

    def get_frame_count(self) -> int:
        """Get number of scans completed."""
        return self._frame_count

    def get_point_count(self) -> int:
        """Get number of points in last reading."""
        if self._last_reading is None:
            return 0
        return len(self._last_reading.point_cloud)

    def is_valid(self) -> bool:
        """Check if last reading is valid."""
        if self._last_reading is None:
            return False
        return self._last_reading.is_valid()

    # ==================== Setters for Simulation ====================

    def set_drone_pose(self, x: float, y: float, z: float, yaw: float = 0.0):
        """
        Set drone position for simulation.

        Args:
            x: X position (meters, world frame)
            y: Y position (meters, world frame)
            z: Z position (height above ground, meters)
            yaw: Heading angle (radians)
        """
        self._drone_x = x
        self._drone_y = y
        self._drone_z = z
        self._drone_yaw = yaw

    def add_obstacle(self, x: float, y: float, z: float, radius: float = 1.0):
        """
        Add a spherical obstacle to the simulation.

        Args:
            x, y, z: Obstacle center in world frame (meters)
            radius: Obstacle radius (meters)
        """
        self._simulated_obstacles.append({
            'x': x, 'y': y, 'z': z, 'radius': radius
        })

    def clear_obstacles(self):
        """Remove all simulated obstacles."""
        self._simulated_obstacles.clear()

    def set_ground_height(self, height: float):
        """Set ground plane height (meters)."""
        self._ground_height = height

    def set_noise(self, range_noise: float, angle_noise: float):
        """
        Set noise parameters.

        Args:
            range_noise: Range measurement noise std dev (meters)
            angle_noise: Angle measurement noise std dev (degrees)
        """
        self._range_noise = range_noise
        self._angle_noise = angle_noise

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize sensor data to dictionary."""
        return {
            'type': self.type,
            'lidar_type': self._lidar_type.value,
            'state': self._state.value,
            'frame_count': self._frame_count,
            'motor_rpm': self._motor_rpm,
            'temperature': self._temperature,
            'min_range': self._min_range,
            'max_range': self._max_range,
            'h_fov': self._h_fov,
            'v_fov': self._v_fov,
            'update_rate': self._update_rate,
            'last_reading': self._last_reading.to_dict() if self._last_reading else None,
            'is_valid': self.is_valid(),
            'nearest_obstacle': self.get_nearest_obstacle_distance()
        }
