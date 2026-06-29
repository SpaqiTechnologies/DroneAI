"""
Collision Detector Module

Predicts potential collisions based on current trajectories
and provides collision warnings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any, Callable
import time
import math

from .obstacle_tracker import TrackedObstacle, ObstaclePosition, ObstacleVelocity


class CollisionSeverity(Enum):
    """Severity level of collision warning."""
    NONE = "none"
    LOW = "low"             # Monitor situation
    MEDIUM = "medium"       # Prepare for avoidance
    HIGH = "high"           # Take action soon
    CRITICAL = "critical"   # Immediate action required


class CollisionType(Enum):
    """Type of potential collision."""
    CONVERGING = "converging"       # Paths converging
    HEAD_ON = "head_on"            # Direct collision course
    CROSSING = "crossing"          # Crossing paths
    OVERTAKING = "overtaking"      # Catching up to obstacle
    STATIC = "static"              # Collision with static obstacle


@dataclass
class CollisionPrediction:
    """Prediction of a potential collision."""
    obstacle: TrackedObstacle
    severity: CollisionSeverity = CollisionSeverity.NONE
    collision_type: CollisionType = CollisionType.CONVERGING

    # Timing
    time_to_collision: float = float('inf')     # seconds
    time_to_closest_approach: float = 0.0       # seconds

    # Geometry
    closest_approach_distance: float = 0.0      # meters
    collision_point: Optional[ObstaclePosition] = None

    # Action needed
    avoidance_urgency: float = 0.0              # 0-1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'obstacle_id': self.obstacle.id,
            'severity': self.severity.value,
            'collision_type': self.collision_type.value,
            'time_to_collision': self.time_to_collision if self.time_to_collision < float('inf') else None,
            'time_to_closest_approach': self.time_to_closest_approach,
            'closest_approach_distance': self.closest_approach_distance,
            'collision_point': self.collision_point.to_tuple() if self.collision_point else None,
            'avoidance_urgency': self.avoidance_urgency,
        }


@dataclass
class DroneState:
    """Current drone state for collision prediction."""
    position: ObstaclePosition = field(default_factory=ObstaclePosition)
    velocity: ObstacleVelocity = field(default_factory=ObstacleVelocity)
    heading: float = 0.0            # degrees
    safety_radius: float = 2.0      # meters


class CollisionDetector:
    """
    Collision detection and prediction system.

    Analyzes tracked obstacles and drone trajectory to predict
    potential collisions and calculate avoidance windows.
    """

    # Prediction parameters
    PREDICTION_HORIZON = 10.0       # seconds to look ahead
    TIME_STEP = 0.1                 # seconds per prediction step

    # Threshold distances
    CRITICAL_DISTANCE = 3.0         # meters
    HIGH_DISTANCE = 5.0             # meters
    MEDIUM_DISTANCE = 10.0          # meters
    LOW_DISTANCE = 20.0             # meters

    # Time thresholds
    CRITICAL_TIME = 2.0             # seconds
    HIGH_TIME = 5.0                 # seconds
    MEDIUM_TIME = 10.0              # seconds

    def __init__(self):
        self._predictions: List[CollisionPrediction] = []
        self._last_update_time = time.time()
        self._callbacks: List[Callable[[CollisionPrediction], None]] = []

        # Statistics
        self._total_predictions = 0
        self._critical_count = 0

    def predict_collisions(
        self,
        drone_state: DroneState,
        obstacles: List[TrackedObstacle],
    ) -> List[CollisionPrediction]:
        """
        Predict potential collisions with obstacles.

        Args:
            drone_state: Current drone state
            obstacles: List of tracked obstacles

        Returns:
            List of collision predictions sorted by urgency
        """
        self._predictions.clear()

        for obstacle in obstacles:
            prediction = self._analyze_collision(drone_state, obstacle)
            if prediction.severity != CollisionSeverity.NONE:
                self._predictions.append(prediction)
                self._total_predictions += 1

                if prediction.severity == CollisionSeverity.CRITICAL:
                    self._critical_count += 1
                    self._notify_collision(prediction)

        # Sort by urgency (most urgent first)
        self._predictions.sort(key=lambda p: -p.avoidance_urgency)

        self._last_update_time = time.time()
        return self._predictions

    def _analyze_collision(
        self,
        drone_state: DroneState,
        obstacle: TrackedObstacle,
    ) -> CollisionPrediction:
        """Analyze potential collision with a single obstacle."""
        prediction = CollisionPrediction(obstacle=obstacle)

        # Get relative state
        dx = obstacle.position.x - drone_state.position.x
        dy = obstacle.position.y - drone_state.position.y
        dz = obstacle.position.z - drone_state.position.z

        dvx = obstacle.velocity.vx - drone_state.velocity.vx
        dvy = obstacle.velocity.vy - drone_state.velocity.vy
        dvz = obstacle.velocity.vz - drone_state.velocity.vz

        # Current distance
        current_distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Combined safety radius
        safety_radius = drone_state.safety_radius + obstacle.get_bounding_sphere_radius()

        # Check if already in collision
        if current_distance < safety_radius:
            prediction.severity = CollisionSeverity.CRITICAL
            prediction.time_to_collision = 0.0
            prediction.closest_approach_distance = current_distance
            prediction.avoidance_urgency = 1.0
            prediction.collision_type = CollisionType.STATIC
            return prediction

        # Calculate closest point of approach (CPA)
        relative_speed_sq = dvx*dvx + dvy*dvy + dvz*dvz

        if relative_speed_sq < 0.01:
            # Static or parallel motion - check current distance only
            prediction.closest_approach_distance = current_distance
            prediction.time_to_closest_approach = 0.0

            if not obstacle.is_moving():
                prediction.collision_type = CollisionType.STATIC
        else:
            # Calculate time to CPA
            dot_product = dx*dvx + dy*dvy + dz*dvz
            t_cpa = -dot_product / relative_speed_sq

            if t_cpa < 0:
                # CPA is in the past
                prediction.time_to_closest_approach = 0.0
                prediction.closest_approach_distance = current_distance
            elif t_cpa > self.PREDICTION_HORIZON:
                # CPA is too far in future
                prediction.time_to_closest_approach = t_cpa
                prediction.closest_approach_distance = float('inf')
            else:
                prediction.time_to_closest_approach = t_cpa

                # Calculate distance at CPA
                cpa_dx = dx + dvx * t_cpa
                cpa_dy = dy + dvy * t_cpa
                cpa_dz = dz + dvz * t_cpa
                prediction.closest_approach_distance = math.sqrt(
                    cpa_dx*cpa_dx + cpa_dy*cpa_dy + cpa_dz*cpa_dz
                )

                # Calculate collision point
                prediction.collision_point = ObstaclePosition(
                    x=drone_state.position.x + drone_state.velocity.vx * t_cpa,
                    y=drone_state.position.y + drone_state.velocity.vy * t_cpa,
                    z=drone_state.position.z + drone_state.velocity.vz * t_cpa,
                )

            # Determine collision type
            prediction.collision_type = self._determine_collision_type(
                dx, dy, dz, dvx, dvy, dvz, obstacle
            )

        # Calculate time to collision if CPA distance is less than safety radius
        if prediction.closest_approach_distance < safety_radius:
            # Solve quadratic for when distance = safety_radius
            a = relative_speed_sq
            b = 2 * (dx*dvx + dy*dvy + dz*dvz)
            c = dx*dx + dy*dy + dz*dz - safety_radius*safety_radius

            discriminant = b*b - 4*a*c
            if discriminant >= 0 and a > 0:
                t1 = (-b - math.sqrt(discriminant)) / (2*a)
                if t1 > 0:
                    prediction.time_to_collision = t1
                else:
                    t2 = (-b + math.sqrt(discriminant)) / (2*a)
                    if t2 > 0:
                        prediction.time_to_collision = t2

        # Determine severity
        prediction.severity = self._determine_severity(prediction)

        # Calculate urgency (0-1)
        prediction.avoidance_urgency = self._calculate_urgency(prediction)

        return prediction

    def _determine_collision_type(
        self,
        dx: float, dy: float, dz: float,
        dvx: float, dvy: float, dvz: float,
        obstacle: TrackedObstacle,
    ) -> CollisionType:
        """Determine the type of potential collision."""
        if not obstacle.is_moving():
            return CollisionType.STATIC

        # Calculate approach angle
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0:
            return CollisionType.CONVERGING

        # Normalize position vector
        nx, ny, nz = dx/dist, dy/dist, dz/dist

        # Relative velocity magnitude
        rel_speed = math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
        if rel_speed == 0:
            return CollisionType.CONVERGING

        # Normalize velocity vector
        vx, vy, vz = dvx/rel_speed, dvy/rel_speed, dvz/rel_speed

        # Dot product gives alignment
        alignment = nx*vx + ny*vy + nz*vz

        if alignment < -0.9:
            return CollisionType.HEAD_ON
        elif alignment > 0.9:
            return CollisionType.OVERTAKING
        elif abs(alignment) < 0.3:
            return CollisionType.CROSSING
        else:
            return CollisionType.CONVERGING

    def _determine_severity(self, prediction: CollisionPrediction) -> CollisionSeverity:
        """Determine collision severity based on distance and time."""
        ttc = prediction.time_to_collision
        cpa_dist = prediction.closest_approach_distance

        # Check time to collision
        if ttc < self.CRITICAL_TIME:
            return CollisionSeverity.CRITICAL
        elif ttc < self.HIGH_TIME:
            return CollisionSeverity.HIGH
        elif ttc < self.MEDIUM_TIME:
            return CollisionSeverity.MEDIUM

        # Check closest approach distance
        if cpa_dist < self.CRITICAL_DISTANCE:
            return CollisionSeverity.CRITICAL
        elif cpa_dist < self.HIGH_DISTANCE:
            return CollisionSeverity.HIGH
        elif cpa_dist < self.MEDIUM_DISTANCE:
            return CollisionSeverity.MEDIUM
        elif cpa_dist < self.LOW_DISTANCE:
            return CollisionSeverity.LOW

        return CollisionSeverity.NONE

    def _calculate_urgency(self, prediction: CollisionPrediction) -> float:
        """Calculate urgency score 0-1."""
        urgency = 0.0

        # Time-based urgency
        ttc = prediction.time_to_collision
        if ttc < float('inf'):
            if ttc < self.CRITICAL_TIME:
                urgency += 0.5
            else:
                urgency += 0.3 * max(0, 1 - ttc / self.PREDICTION_HORIZON)

        # Distance-based urgency
        cpa_dist = prediction.closest_approach_distance
        if cpa_dist < self.CRITICAL_DISTANCE:
            urgency += 0.5
        elif cpa_dist < self.HIGH_DISTANCE:
            urgency += 0.3
        elif cpa_dist < self.MEDIUM_DISTANCE:
            urgency += 0.2
        else:
            urgency += 0.1 * max(0, 1 - cpa_dist / self.LOW_DISTANCE)

        return min(1.0, urgency)

    def get_highest_threat(self) -> Optional[CollisionPrediction]:
        """Get the most urgent collision prediction."""
        if not self._predictions:
            return None
        return self._predictions[0]

    def get_threats_above_severity(
        self,
        min_severity: CollisionSeverity = CollisionSeverity.MEDIUM,
    ) -> List[CollisionPrediction]:
        """Get all threats at or above specified severity."""
        severity_order = [
            CollisionSeverity.NONE,
            CollisionSeverity.LOW,
            CollisionSeverity.MEDIUM,
            CollisionSeverity.HIGH,
            CollisionSeverity.CRITICAL,
        ]

        min_idx = severity_order.index(min_severity)

        return [
            p for p in self._predictions
            if severity_order.index(p.severity) >= min_idx
        ]

    def is_collision_imminent(self) -> bool:
        """Check if there's an imminent collision."""
        for prediction in self._predictions:
            if prediction.severity in [CollisionSeverity.CRITICAL, CollisionSeverity.HIGH]:
                return True
        return False

    def get_safe_directions(
        self,
        drone_state: DroneState,
        obstacles: List[TrackedObstacle],
        num_directions: int = 16,
    ) -> List[Tuple[float, float, float]]:
        """
        Get list of relatively safe directions to move.

        Args:
            drone_state: Current drone state
            obstacles: List of obstacles
            num_directions: Number of directions to evaluate

        Returns:
            List of safe direction vectors sorted by safety
        """
        directions = []

        # Generate directions around the drone
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            dx = math.cos(angle)
            dy = math.sin(angle)

            # Check vertical directions too
            for dz in [-0.5, 0, 0.5]:
                mag = math.sqrt(dx*dx + dy*dy + dz*dz)
                if mag > 0:
                    directions.append((dx/mag, dy/mag, dz/mag))

        # Score each direction
        scored_directions = []
        for direction in directions:
            score = self._score_direction(drone_state, obstacles, direction)
            scored_directions.append((score, direction))

        # Sort by score (higher = safer)
        scored_directions.sort(key=lambda x: -x[0])

        return [d[1] for d in scored_directions]

    def _score_direction(
        self,
        drone_state: DroneState,
        obstacles: List[TrackedObstacle],
        direction: Tuple[float, float, float],
    ) -> float:
        """Score a direction based on obstacle proximity."""
        dx, dy, dz = direction
        score = 1.0

        for obstacle in obstacles:
            # Vector to obstacle
            ox = obstacle.position.x - drone_state.position.x
            oy = obstacle.position.y - drone_state.position.y
            oz = obstacle.position.z - drone_state.position.z

            dist = math.sqrt(ox*ox + oy*oy + oz*oz)
            if dist == 0:
                continue

            # Normalize
            ox, oy, oz = ox/dist, oy/dist, oz/dist

            # Alignment with direction (negative = away from obstacle = good)
            alignment = dx*ox + dy*oy + dz*oz

            # Penalize directions toward obstacles
            if alignment > 0:
                # More penalty for closer obstacles
                proximity_factor = max(0, 1 - dist / 20)
                score -= alignment * proximity_factor * obstacle.confidence

        return max(0, score)

    def add_collision_callback(
        self,
        callback: Callable[[CollisionPrediction], None]
    ) -> None:
        """Add callback for critical collision warnings."""
        self._callbacks.append(callback)

    def _notify_collision(self, prediction: CollisionPrediction) -> None:
        """Notify callbacks of critical collision."""
        for callback in self._callbacks:
            try:
                callback(prediction)
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'total_predictions': self._total_predictions,
            'critical_count': self._critical_count,
            'current_threats': len(self._predictions),
            'highest_urgency': max(
                (p.avoidance_urgency for p in self._predictions),
                default=0.0
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        highest = self.get_highest_threat()

        return {
            'active': True,
            'threat_count': len(self._predictions),
            'imminent_collision': self.is_collision_imminent(),
            'highest_severity': highest.severity.value if highest else 'none',
            'highest_urgency': highest.avoidance_urgency if highest else 0.0,
            'predictions': [p.to_dict() for p in self._predictions[:5]],
        }
