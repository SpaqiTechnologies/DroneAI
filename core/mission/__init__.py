"""
Mission Planning Package

Provides comprehensive mission planning capabilities including:
- Waypoint and mission data structures
- Mission validation
- Survey pattern generation
- Mission storage (JSON persistence)
- Mission execution management
"""

from .waypoint import (
    Waypoint,
    WaypointType,
    WaypointAction,
    WaypointActionItem,
    AltitudeFrame,
)

from .mission import (
    Mission,
    MissionType,
    MissionState,
    MissionStatistics,
)

from .mission_validator import (
    MissionValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationType,
)

from .survey_patterns import SurveyPatternGenerator

from .mission_storage import MissionStorage

from .mission_manager import (
    MissionManager,
    MissionExecutionState,
    MissionProgress,
)

__all__ = [
    # Waypoint
    'Waypoint',
    'WaypointType',
    'WaypointAction',
    'WaypointActionItem',
    'AltitudeFrame',
    # Mission
    'Mission',
    'MissionType',
    'MissionState',
    'MissionStatistics',
    # Validation
    'MissionValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'ValidationType',
    # Survey patterns
    'SurveyPatternGenerator',
    # Storage
    'MissionStorage',
    # Manager
    'MissionManager',
    'MissionExecutionState',
    'MissionProgress',
]
