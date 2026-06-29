"""
Flight Modes Package

Provides different flight modes for autonomous drone control:
- STABILIZE: Manual attitude control with auto-leveling
- LOITER: GPS-based position hold
- AUTO: Autonomous mission execution
- GUIDED: Single waypoint navigation
- RTL: Return-to-launch
- LAND: Automated landing
"""

from .base_mode import (
    FlightMode,
    FlightModeHandler,
    FlightCommand,
    ModeTransitionError,
    ModeState,
)

from .stabilize_mode import StabilizeModeHandler
from .loiter_mode import LoiterModeHandler
from .guided_mode import GuidedModeHandler
from .rtl_mode import RTLModeHandler, RTLPhase
from .auto_mode import AutoModeHandler

# Re-export mission classes from mission package for convenience
from core.mission import (
    Mission,
    MissionType,
    MissionState,
    MissionStatistics,
    Waypoint,
    WaypointType,
    WaypointAction,
    WaypointActionItem,
    AltitudeFrame,
)

__all__ = [
    # Base classes
    'FlightMode',
    'FlightModeHandler',
    'FlightCommand',
    'ModeTransitionError',
    'ModeState',
    # Mode handlers
    'StabilizeModeHandler',
    'LoiterModeHandler',
    'GuidedModeHandler',
    'RTLModeHandler',
    'RTLPhase',
    'AutoModeHandler',
    # Mission classes (re-exported from core.mission)
    'Mission',
    'MissionType',
    'MissionState',
    'MissionStatistics',
    'Waypoint',
    'WaypointType',
    'WaypointAction',
    'WaypointActionItem',
    'AltitudeFrame',
]
