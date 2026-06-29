"""
Dynamic Replanner Module

Real-time path replanning using VFH+ (Vector Field Histogram Plus)
algorithm for obstacle avoidance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import math
import time

from .obstacle_tracker import TrackedObstacle, ObstaclePosition, ObstacleTracker
from .collision_detector import CollisionDetector, CollisionPrediction, DroneState
from .avoidance_strategies import (
    AvoidanceStrategySelector,
    AvoidanceCommand,
    AvoidanceAction,
)


@dataclass
class ReplanResult:
    """Result of path replanning."""
    success: bool = False
    velocity: Tuple[float, float, float] = (0, 0, 0)
    heading: float = 0.0

    # Original target
    target_position: Optional[Tuple[float, float, float]] = None

    # Avoidance status
    avoiding: bool = False
    avoidance_action: Optional[AvoidanceAction] = None

    # Path information
    path_blocked: bool = False
    detour_required: bool = False

    # Debug info
    histogram: Optional[List[float]] = None
    selected_sector: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'velocity': self.velocity,
            'heading': self.heading,
            'avoiding': self.avoiding,
            'avoidance_action': self.avoidance_action.value if self.avoidance_action else None,
            'path_blocked': self.path_blocked,
            'detour_required': self.detour_required,
        }


class VFHPlusReplanner:
    """
    Vector Field Histogram Plus (VFH+) implementation for
    real-time obstacle avoidance and path replanning.

    VFH+ builds a polar histogram of obstacle density around the drone
    and selects the best free direction toward the goal.
    """

    # VFH+ parameters
    NUM_SECTORS = 72                # 5-degree sectors (360/72)
    SECTOR_ANGLE = 5.0              # degrees per sector
    ACTIVE_WINDOW_SIZE = 33         # sectors (165 degrees)

    # Distance thresholds
    MAX_RANGE = 20.0                # meters
    SAFETY_DISTANCE = 2.0           # meters

    # Weights for direction selection
    GOAL_WEIGHT = 5.0               # Weight for goal direction
    SMOOTH_WEIGHT = 2.0             # Weight for smooth turning
    WIDE_WEIGHT = 1.0               # Weight for wide openings

    # Speed control
    MAX_SPEED = 5.0                 # m/s
    MIN_SPEED = 0.5                 # m/s

    def __init__(
        self,
        obstacle_tracker: ObstacleTracker,
        collision_detector: CollisionDetector,
    ):
        self._tracker = obstacle_tracker
        self._detector = collision_detector
        self._strategy_selector = AvoidanceStrategySelector()

        # Histogram
        self._polar_histogram = [0.0] * self.NUM_SECTORS
        self._binary_histogram = [True] * self.NUM_SECTORS  # True = free

        # State
        self._last_heading = 0.0
        self._avoidance_active = False
        self._avoidance_start_time = 0.0
        self._consecutive_blocks = 0

        # Statistics
        self._replan_count = 0
        self._avoidance_count = 0

    def replan(
        self,
        drone_state: DroneState,
        target: Tuple[float, float, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReplanResult:
        """
        Replan path considering obstacles.

        Args:
            drone_state: Current drone state
            target: Target position (x, y, z)
            constraints: Flight constraints

        Returns:
            ReplanResult with new velocity command
        """
        self._replan_count += 1
        result = ReplanResult(target_position=target)

        # Get obstacles
        obstacles = self._tracker.get_obstacles(confirmed_only=True, max_distance=self.MAX_RANGE)

        # Build polar histogram
        self._build_histogram(drone_state, obstacles)

        # Apply binary threshold
        self._apply_threshold()

        # Calculate goal direction
        goal_direction = self._calculate_goal_direction(drone_state, target)

        # Find best free direction
        selected_direction, is_blocked = self._select_direction(
            goal_direction,
            self._last_heading,
        )

        # Update result
        result.histogram = self._polar_histogram.copy()
        result.selected_sector = int(selected_direction / self.SECTOR_ANGLE) % self.NUM_SECTORS
        result.heading = selected_direction

        if is_blocked:
            result.path_blocked = True
            self._consecutive_blocks += 1

            if self._consecutive_blocks > 10:
                result.detour_required = True
        else:
            self._consecutive_blocks = 0

        # Check for collision threats
        predictions = self._detector.predict_collisions(drone_state, obstacles)
        threat = self._detector.get_highest_threat()

        if threat and threat.avoidance_urgency > 0.5:
            # Use avoidance strategy
            result.avoiding = True
            self._avoidance_active = True
            self._avoidance_count += 1

            strategy, command = self._strategy_selector.select_strategy(
                drone_state, threat, constraints
            )

            result.avoidance_action = command.action
            result.velocity = command.velocity

            # Adjust heading based on avoidance
            if command.action in [AvoidanceAction.DODGE_LEFT]:
                result.heading = (selected_direction - 45) % 360
            elif command.action in [AvoidanceAction.DODGE_RIGHT]:
                result.heading = (selected_direction + 45) % 360
        else:
            self._avoidance_active = False

            # Calculate speed based on obstacle density
            speed = self._calculate_speed(selected_direction, obstacles, drone_state)

            # Calculate velocity toward goal
            heading_rad = math.radians(selected_direction)
            vx = speed * math.cos(heading_rad)
            vy = speed * math.sin(heading_rad)

            # Add vertical component toward target altitude
            target_alt = -target[2]  # NED to altitude
            current_alt = -drone_state.position.z
            alt_error = target_alt - current_alt
            vz = -max(-2.0, min(2.0, alt_error * 0.5))  # Limit vertical speed

            result.velocity = (vx, vy, vz)

        result.success = True
        self._last_heading = result.heading

        return result

    def _build_histogram(
        self,
        drone_state: DroneState,
        obstacles: List[TrackedObstacle],
    ) -> None:
        """Build polar obstacle histogram."""
        # Reset histogram
        self._polar_histogram = [0.0] * self.NUM_SECTORS

        for obstacle in obstacles:
            # Calculate angle to obstacle
            dx = obstacle.position.x - drone_state.position.x
            dy = obstacle.position.y - drone_state.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < 0.1:
                continue

            angle = math.degrees(math.atan2(dy, dx)) % 360

            # Calculate obstacle magnitude
            # Higher for closer obstacles
            if distance < self.SAFETY_DISTANCE:
                magnitude = 1.0
            else:
                magnitude = max(0, 1 - (distance - self.SAFETY_DISTANCE) / (self.MAX_RANGE - self.SAFETY_DISTANCE))

            # Weight by confidence
            magnitude *= obstacle.confidence

            # Expand obstacle by its size
            obstacle_angle_size = math.degrees(
                math.atan2(obstacle.get_bounding_sphere_radius(), distance)
            )

            # Add to histogram sectors
            for delta in range(-int(obstacle_angle_size / self.SECTOR_ANGLE) - 1,
                              int(obstacle_angle_size / self.SECTOR_ANGLE) + 2):
                sector_angle = angle + delta * self.SECTOR_ANGLE
                sector = int(sector_angle / self.SECTOR_ANGLE) % self.NUM_SECTORS
                self._polar_histogram[sector] += magnitude

    def _apply_threshold(self, threshold_low: float = 0.3, threshold_high: float = 0.7) -> None:
        """Apply binary threshold with hysteresis."""
        for i in range(self.NUM_SECTORS):
            if self._polar_histogram[i] > threshold_high:
                self._binary_histogram[i] = False  # Blocked
            elif self._polar_histogram[i] < threshold_low:
                self._binary_histogram[i] = True   # Free
            # Otherwise keep previous state (hysteresis)

    def _calculate_goal_direction(
        self,
        drone_state: DroneState,
        target: Tuple[float, float, float],
    ) -> float:
        """Calculate direction to goal in degrees."""
        dx = target[0] - drone_state.position.x
        dy = target[1] - drone_state.position.y

        return math.degrees(math.atan2(dy, dx)) % 360

    def _select_direction(
        self,
        goal_direction: float,
        previous_direction: float,
    ) -> Tuple[float, bool]:
        """
        Select best direction using VFH+ cost function.

        Returns:
            Tuple of (selected direction in degrees, is_blocked)
        """
        # Find all free sectors
        openings = self._find_openings()

        if not openings:
            # All blocked - return goal direction anyway
            return goal_direction, True

        # Evaluate each opening
        best_score = float('inf')
        best_direction = goal_direction
        is_blocked = False

        for opening in openings:
            # Opening is (start_sector, end_sector, width)
            start_sector, end_sector, width = opening

            # Calculate candidate directions from opening
            candidates = self._get_candidate_directions(
                start_sector, end_sector, goal_direction
            )

            for candidate in candidates:
                score = self._cost_function(
                    candidate, goal_direction, previous_direction, width
                )

                if score < best_score:
                    best_score = score
                    best_direction = candidate

        # Check if goal direction is blocked
        goal_sector = int(goal_direction / self.SECTOR_ANGLE) % self.NUM_SECTORS
        if not self._binary_histogram[goal_sector]:
            is_blocked = True

        return best_direction, is_blocked

    def _find_openings(self) -> List[Tuple[int, int, int]]:
        """Find all free openings in the histogram."""
        openings = []
        in_opening = False
        start = 0

        # Check if sector 0 is free to handle wraparound
        starts_free = self._binary_histogram[0]

        for i in range(self.NUM_SECTORS):
            if self._binary_histogram[i]:
                if not in_opening:
                    in_opening = True
                    start = i
            else:
                if in_opening:
                    in_opening = False
                    width = i - start
                    openings.append((start, i - 1, width))

        # Handle wraparound
        if in_opening:
            if starts_free:
                # Merge with first opening
                width = self.NUM_SECTORS - start
                if openings:
                    first = openings[0]
                    openings[0] = (start, first[1], width + first[2])
                else:
                    openings.append((start, self.NUM_SECTORS - 1, width))
            else:
                openings.append((start, self.NUM_SECTORS - 1, self.NUM_SECTORS - start))

        return openings

    def _get_candidate_directions(
        self,
        start_sector: int,
        end_sector: int,
        goal_direction: float,
    ) -> List[float]:
        """Get candidate directions from an opening."""
        candidates = []

        # Center of opening
        if end_sector >= start_sector:
            center_sector = (start_sector + end_sector) // 2
        else:
            # Wraparound
            total = (end_sector + self.NUM_SECTORS - start_sector)
            center_sector = (start_sector + total // 2) % self.NUM_SECTORS

        center_direction = center_sector * self.SECTOR_ANGLE
        candidates.append(center_direction)

        # Edges of opening
        candidates.append(start_sector * self.SECTOR_ANGLE + self.SECTOR_ANGLE / 2)
        candidates.append(end_sector * self.SECTOR_ANGLE - self.SECTOR_ANGLE / 2)

        # Goal direction if within opening
        goal_sector = int(goal_direction / self.SECTOR_ANGLE) % self.NUM_SECTORS
        if self._is_sector_in_opening(goal_sector, start_sector, end_sector):
            candidates.append(goal_direction)

        return candidates

    def _is_sector_in_opening(self, sector: int, start: int, end: int) -> bool:
        """Check if sector is within opening."""
        if end >= start:
            return start <= sector <= end
        else:
            return sector >= start or sector <= end

    def _cost_function(
        self,
        candidate: float,
        goal: float,
        previous: float,
        opening_width: int,
    ) -> float:
        """
        VFH+ cost function.

        Lower cost = better direction.
        """
        # Angular differences
        goal_diff = abs(self._angle_diff(candidate, goal))
        smooth_diff = abs(self._angle_diff(candidate, previous))

        # Narrower openings are less desirable
        width_penalty = max(0, 10 - opening_width)

        cost = (
            self.GOAL_WEIGHT * goal_diff +
            self.SMOOTH_WEIGHT * smooth_diff +
            self.WIDE_WEIGHT * width_penalty
        )

        return cost

    def _angle_diff(self, a: float, b: float) -> float:
        """Calculate signed angle difference."""
        diff = a - b
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _calculate_speed(
        self,
        direction: float,
        obstacles: List[TrackedObstacle],
        drone_state: DroneState,
    ) -> float:
        """Calculate appropriate speed based on obstacle proximity."""
        # Get minimum obstacle distance in travel direction
        heading_rad = math.radians(direction)
        dx = math.cos(heading_rad)
        dy = math.sin(heading_rad)

        min_distance = self.MAX_RANGE

        for obstacle in obstacles:
            # Calculate if obstacle is in path
            ox = obstacle.position.x - drone_state.position.x
            oy = obstacle.position.y - drone_state.position.y

            # Project obstacle onto heading direction
            projection = ox * dx + oy * dy

            if projection > 0:  # Obstacle is ahead
                # Distance perpendicular to heading
                perp_dist = abs(ox * dy - oy * dx)

                # Consider if obstacle is within collision corridor
                corridor_width = drone_state.safety_radius + obstacle.get_bounding_sphere_radius()
                if perp_dist < corridor_width:
                    dist = math.sqrt(ox*ox + oy*oy)
                    min_distance = min(min_distance, dist)

        # Speed based on distance
        if min_distance < self.SAFETY_DISTANCE:
            return self.MIN_SPEED
        elif min_distance < self.SAFETY_DISTANCE * 3:
            # Linear interpolation
            ratio = (min_distance - self.SAFETY_DISTANCE) / (self.SAFETY_DISTANCE * 2)
            return self.MIN_SPEED + ratio * (self.MAX_SPEED - self.MIN_SPEED)
        else:
            return self.MAX_SPEED

    def get_statistics(self) -> Dict[str, Any]:
        """Get replanner statistics."""
        return {
            'replan_count': self._replan_count,
            'avoidance_count': self._avoidance_count,
            'consecutive_blocks': self._consecutive_blocks,
            'avoidance_active': self._avoidance_active,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get replanner status."""
        free_sectors = sum(1 for s in self._binary_histogram if s)
        blocked_sectors = self.NUM_SECTORS - free_sectors

        return {
            'active': True,
            'free_sectors': free_sectors,
            'blocked_sectors': blocked_sectors,
            'avoidance_active': self._avoidance_active,
            'last_heading': self._last_heading,
            'histogram': self._polar_histogram,
        }


class DynamicReplanner:
    """
    High-level dynamic replanner that integrates obstacle tracking,
    collision detection, and VFH+ path replanning.
    """

    def __init__(self):
        self._tracker = ObstacleTracker()
        self._detector = CollisionDetector()
        self._vfh = VFHPlusReplanner(self._tracker, self._detector)

        self._enabled = True
        self._last_update = time.time()

    @property
    def obstacle_tracker(self) -> ObstacleTracker:
        """Get obstacle tracker."""
        return self._tracker

    @property
    def collision_detector(self) -> CollisionDetector:
        """Get collision detector."""
        return self._detector

    def enable(self) -> None:
        """Enable replanning."""
        self._enabled = True

    def disable(self) -> None:
        """Disable replanning."""
        self._enabled = False

    def update(self, dt: float) -> None:
        """
        Update obstacle tracking.

        Args:
            dt: Time delta since last update
        """
        self._tracker.update(dt)
        self._last_update = time.time()

    def process_lidar(self, detections: List[Dict[str, float]]) -> None:
        """Process LiDAR detections."""
        self._tracker.process_lidar_detections(detections)

    def process_camera(self, detections: List[Dict[str, Any]]) -> None:
        """Process camera detections."""
        self._tracker.process_camera_detections(detections)

    def process_ultrasonic(self, distance: float, direction: str = 'forward') -> None:
        """Process ultrasonic detection."""
        self._tracker.process_ultrasonic_detection(distance, direction)

    def replan(
        self,
        drone_state: DroneState,
        target: Tuple[float, float, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReplanResult:
        """
        Replan path to target avoiding obstacles.

        Args:
            drone_state: Current drone state
            target: Target position
            constraints: Flight constraints

        Returns:
            ReplanResult with velocity command
        """
        if not self._enabled:
            # Return direct path to target
            dx = target[0] - drone_state.position.x
            dy = target[1] - drone_state.position.y
            dz = target[2] - drone_state.position.z

            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            speed = min(5.0, dist * 0.5)

            if dist > 0:
                vx = dx / dist * speed
                vy = dy / dist * speed
                vz = dz / dist * speed
            else:
                vx, vy, vz = 0, 0, 0

            return ReplanResult(
                success=True,
                velocity=(vx, vy, vz),
                heading=math.degrees(math.atan2(dy, dx)),
                target_position=target,
            )

        # Update threat levels
        self._tracker.update_threat_levels(
            drone_velocity=(
                drone_state.velocity.vx,
                drone_state.velocity.vy,
                drone_state.velocity.vz,
            ),
            safety_radius=drone_state.safety_radius,
        )

        # Replan with VFH+
        return self._vfh.replan(drone_state, target, constraints)

    def is_path_clear(
        self,
        drone_state: DroneState,
        target: Tuple[float, float, float],
    ) -> bool:
        """Check if path to target is clear of obstacles."""
        obstacles = self._tracker.get_obstacles(confirmed_only=True)

        dx = target[0] - drone_state.position.x
        dy = target[1] - drone_state.position.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist == 0:
            return True

        # Normalize direction
        dx, dy = dx/dist, dy/dist

        for obstacle in obstacles:
            ox = obstacle.position.x - drone_state.position.x
            oy = obstacle.position.y - drone_state.position.y
            odist = math.sqrt(ox*ox + oy*oy)

            if odist > dist:
                continue

            # Project onto path
            projection = ox * dx + oy * dy
            if projection < 0:
                continue

            # Perpendicular distance
            perp = abs(ox * dy - oy * dx)
            corridor = drone_state.safety_radius + obstacle.get_bounding_sphere_radius()

            if perp < corridor:
                return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get replanner status."""
        return {
            'enabled': self._enabled,
            'tracker': self._tracker.get_status(),
            'detector': self._detector.get_status(),
            'vfh': self._vfh.get_status(),
        }
