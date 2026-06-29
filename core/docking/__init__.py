"""Drone-in-a-box: docking station, auto-charging, scheduled patrols."""

from core.docking.docking_station import (
    DockingStation,
    DockState,
    DockStatus,
    ChargeProfile,
)
from core.docking.patrol_scheduler import (
    PatrolScheduler,
    PatrolJob,
    PatrolJobStatus,
)
from core.docking.autonomy_adapter import (
    DockAutonomyAdapter,
    PatrolPath,
)

__all__ = [
    "DockingStation",
    "DockState",
    "DockStatus",
    "ChargeProfile",
    "PatrolScheduler",
    "PatrolJob",
    "PatrolJobStatus",
    "DockAutonomyAdapter",
    "PatrolPath",
]
