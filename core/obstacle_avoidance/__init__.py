"""
Obstacle Avoidance Package

Provides comprehensive obstacle detection, tracking, and avoidance:
- Multi-sensor fusion obstacle tracking
- Collision prediction and detection
- Avoidance strategy selection
- VFH+ dynamic path replanning
"""

from .obstacle_tracker import (
    ObstacleTracker,
    TrackedObstacle,
    ObstaclePosition,
    ObstacleVelocity,
    ObstacleType,
    ObstacleSource,
)

from .collision_detector import (
    CollisionDetector,
    CollisionPrediction,
    CollisionSeverity,
    CollisionType,
    DroneState,
)

from .avoidance_strategies import (
    AvoidanceStrategy,
    AvoidanceStrategySelector,
    AvoidanceCommand,
    AvoidanceAction,
    StopStrategy,
    SlowDownStrategy,
    DodgeStrategy,
    ClimbStrategy,
    DescendStrategy,
    ReverseStrategy,
)

from .dynamic_replanner import (
    DynamicReplanner,
    VFHPlusReplanner,
    ReplanResult,
)

__all__ = [
    # Obstacle Tracker
    'ObstacleTracker',
    'TrackedObstacle',
    'ObstaclePosition',
    'ObstacleVelocity',
    'ObstacleType',
    'ObstacleSource',
    # Collision Detector
    'CollisionDetector',
    'CollisionPrediction',
    'CollisionSeverity',
    'CollisionType',
    'DroneState',
    # Avoidance Strategies
    'AvoidanceStrategy',
    'AvoidanceStrategySelector',
    'AvoidanceCommand',
    'AvoidanceAction',
    'StopStrategy',
    'SlowDownStrategy',
    'DodgeStrategy',
    'ClimbStrategy',
    'DescendStrategy',
    'ReverseStrategy',
    # Dynamic Replanner
    'DynamicReplanner',
    'VFHPlusReplanner',
    'ReplanResult',
]
