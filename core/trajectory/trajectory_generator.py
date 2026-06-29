"""
Trajectory Generator

Generates smooth trajectories for drone navigation using Bezier curves
and velocity-based motion planning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import time


@dataclass
class TrajectoryPoint:
    """A point along a trajectory with position, velocity, and time."""
    # Position (lat, lon, alt) or (x, y, z) in local frame
    position: Tuple[float, float, float]

    # Velocity (vn, ve, vd) or (vx, vy, vz)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Acceleration (an, ae, ad) or (ax, ay, az)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Yaw angle (degrees)
    yaw: float = 0.0

    # Yaw rate (deg/s)
    yaw_rate: float = 0.0

    # Time along trajectory (seconds from start)
    time: float = 0.0

    # Whether this is a waypoint (vs interpolated point)
    is_waypoint: bool = False


@dataclass
class TrajectorySegment:
    """A segment of a trajectory between two waypoints."""
    start: TrajectoryPoint
    end: TrajectoryPoint
    duration: float  # seconds
    distance: float  # meters
    max_speed: float  # m/s
    points: List[TrajectoryPoint] = field(default_factory=list)


class TrajectoryGenerator:
    """
    Generates smooth trajectories for drone navigation.

    Supports:
    - Linear interpolation for simple paths
    - Bezier curves for smooth cornering
    - Velocity profiling with acceleration limits
    - Time-optimal paths
    """

    # Earth radius for distance calculations (meters)
    EARTH_RADIUS = 6371000.0

    def __init__(
        self,
        max_speed: float = 10.0,          # m/s
        max_acceleration: float = 3.0,     # m/s^2
        max_vertical_speed: float = 3.0,   # m/s
        max_yaw_rate: float = 45.0,        # deg/s
        lookahead_time: float = 2.0,       # seconds
        dt: float = 0.1                    # trajectory point spacing
    ):
        """
        Initialize trajectory generator.

        Args:
            max_speed: Maximum horizontal speed (m/s)
            max_acceleration: Maximum acceleration (m/s^2)
            max_vertical_speed: Maximum vertical speed (m/s)
            max_yaw_rate: Maximum yaw rate (deg/s)
            lookahead_time: How far ahead to look for path smoothing
            dt: Time step for trajectory points
        """
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_vertical_speed = max_vertical_speed
        self.max_yaw_rate = max_yaw_rate
        self.lookahead_time = lookahead_time
        self.dt = dt

    def generate_trajectory(
        self,
        waypoints: List[Tuple[float, float, float]],
        speeds: Optional[List[float]] = None,
        smooth: bool = True
    ) -> List[TrajectoryPoint]:
        """
        Generate a trajectory through waypoints.

        Args:
            waypoints: List of (lat, lon, alt) positions
            speeds: Optional speed at each waypoint (m/s)
            smooth: Whether to use smooth cornering

        Returns:
            List of trajectory points
        """
        if len(waypoints) < 2:
            return []

        if speeds is None:
            speeds = [self.max_speed] * len(waypoints)

        trajectory = []
        total_time = 0.0

        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            speed = min(speeds[i], speeds[i + 1], self.max_speed)

            # Generate segment
            segment = self._generate_segment(
                start, end, speed, total_time, smooth
            )

            # Add points (skip first point after first segment to avoid duplicates)
            if i == 0:
                trajectory.extend(segment.points)
            else:
                trajectory.extend(segment.points[1:])

            total_time += segment.duration

        return trajectory

    def _generate_segment(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        speed: float,
        start_time: float,
        smooth: bool
    ) -> TrajectorySegment:
        """Generate a trajectory segment between two points."""
        # Calculate distance
        distance = self._haversine_distance(start, end)

        # Calculate duration with trapezoidal velocity profile
        duration = self._calculate_duration(distance, speed)

        # Generate points
        points = []
        num_points = max(2, int(duration / self.dt) + 1)

        for i in range(num_points):
            t = i * self.dt
            if t > duration:
                t = duration

            # Interpolation factor (0-1)
            if duration > 0:
                alpha = t / duration
            else:
                alpha = 1.0

            # Apply smoothing if enabled
            if smooth:
                alpha = self._smooth_step(alpha)

            # Interpolate position
            pos = self._interpolate_position(start, end, alpha)

            # Calculate velocity
            if duration > 0:
                vel = self._calculate_velocity(start, end, t, duration, speed)
            else:
                vel = (0.0, 0.0, 0.0)

            # Calculate yaw (point toward next waypoint)
            yaw = self._calculate_bearing(start, end)

            point = TrajectoryPoint(
                position=pos,
                velocity=vel,
                yaw=yaw,
                time=start_time + t,
                is_waypoint=(i == 0 or i == num_points - 1)
            )
            points.append(point)

        return TrajectorySegment(
            start=points[0],
            end=points[-1],
            duration=duration,
            distance=distance,
            max_speed=speed,
            points=points
        )

    def _haversine_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate distance between two GPS points in meters."""
        lat1, lon1, alt1 = p1
        lat2, lon2, alt2 = p2

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        # Haversine formula
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_distance = self.EARTH_RADIUS * c

        # Include altitude difference
        vertical_distance = abs(alt2 - alt1)
        total_distance = math.sqrt(
            horizontal_distance ** 2 + vertical_distance ** 2
        )

        return total_distance

    def _calculate_duration(self, distance: float, speed: float) -> float:
        """
        Calculate duration for a segment with trapezoidal velocity profile.

        Uses acceleration limits to calculate realistic travel time.
        """
        if distance <= 0 or speed <= 0:
            return 0.0

        # Time to accelerate to max speed
        accel_time = speed / self.max_acceleration
        accel_distance = 0.5 * self.max_acceleration * accel_time ** 2

        # Check if we can reach max speed
        if 2 * accel_distance >= distance:
            # Triangle profile (can't reach max speed)
            return 2 * math.sqrt(distance / self.max_acceleration)
        else:
            # Trapezoidal profile
            cruise_distance = distance - 2 * accel_distance
            cruise_time = cruise_distance / speed
            return 2 * accel_time + cruise_time

    def _smooth_step(self, t: float) -> float:
        """Smooth step function for cornering (cubic ease in/out)."""
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return t * t * (3 - 2 * t)

    def _interpolate_position(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        alpha: float
    ) -> Tuple[float, float, float]:
        """Linearly interpolate between two GPS positions."""
        lat = start[0] + alpha * (end[0] - start[0])
        lon = start[1] + alpha * (end[1] - start[1])
        alt = start[2] + alpha * (end[2] - start[2])
        return (lat, lon, alt)

    def _calculate_velocity(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        t: float,
        duration: float,
        max_speed: float
    ) -> Tuple[float, float, float]:
        """Calculate velocity at time t along segment."""
        if duration <= 0:
            return (0.0, 0.0, 0.0)

        # Calculate direction
        bearing = self._calculate_bearing(start, end)
        bearing_rad = math.radians(bearing)

        # Calculate vertical component
        alt_diff = end[2] - start[2]
        horizontal_dist = self._haversine_distance(
            (start[0], start[1], 0),
            (end[0], end[1], 0)
        )

        # Calculate speed profile (trapezoidal)
        accel_time = max_speed / self.max_acceleration
        if t < accel_time:
            # Accelerating
            speed = self.max_acceleration * t
        elif t > duration - accel_time:
            # Decelerating
            speed = self.max_acceleration * (duration - t)
        else:
            # Cruising
            speed = max_speed

        speed = max(0.0, min(speed, max_speed))

        # Calculate velocity components
        if horizontal_dist > 0:
            climb_angle = math.atan2(alt_diff, horizontal_dist)
        else:
            climb_angle = math.pi / 2 if alt_diff > 0 else -math.pi / 2

        horizontal_speed = speed * math.cos(climb_angle)
        vertical_speed = speed * math.sin(climb_angle)

        # Limit vertical speed
        vertical_speed = max(-self.max_vertical_speed,
                            min(self.max_vertical_speed, vertical_speed))

        # NED velocity components
        vn = horizontal_speed * math.cos(bearing_rad)
        ve = horizontal_speed * math.sin(bearing_rad)
        vd = -vertical_speed  # Down is positive in NED

        return (vn, ve, vd)

    def _calculate_bearing(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float]
    ) -> float:
        """Calculate bearing from start to end (degrees, 0=North)."""
        lat1 = math.radians(start[0])
        lat2 = math.radians(end[0])
        dlon = math.radians(end[1] - start[1])

        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360

    def get_point_at_time(
        self,
        trajectory: List[TrajectoryPoint],
        t: float
    ) -> Optional[TrajectoryPoint]:
        """Get trajectory point at specific time."""
        if not trajectory:
            return None

        # Find surrounding points
        for i, point in enumerate(trajectory):
            if point.time >= t:
                if i == 0:
                    return trajectory[0]

                # Interpolate between points
                prev = trajectory[i - 1]
                alpha = (t - prev.time) / (point.time - prev.time)
                return TrajectoryPoint(
                    position=self._interpolate_position(
                        prev.position, point.position, alpha
                    ),
                    velocity=self._interpolate_velocity(
                        prev.velocity, point.velocity, alpha
                    ),
                    yaw=prev.yaw + alpha * (point.yaw - prev.yaw),
                    time=t
                )

        # Past end of trajectory
        return trajectory[-1]

    def _interpolate_velocity(
        self,
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
        alpha: float
    ) -> Tuple[float, float, float]:
        """Interpolate between two velocity vectors."""
        return (
            v1[0] + alpha * (v2[0] - v1[0]),
            v1[1] + alpha * (v2[1] - v1[1]),
            v1[2] + alpha * (v2[2] - v1[2])
        )

    def compute_distance_to_target(
        self,
        current: Tuple[float, float, float],
        target: Tuple[float, float, float]
    ) -> float:
        """Compute 3D distance from current position to target."""
        return self._haversine_distance(current, target)

    def compute_time_to_target(
        self,
        current: Tuple[float, float, float],
        target: Tuple[float, float, float],
        speed: float
    ) -> float:
        """Compute estimated time to reach target at given speed."""
        distance = self._haversine_distance(current, target)
        if speed <= 0:
            return float('inf')
        return self._calculate_duration(distance, speed)
