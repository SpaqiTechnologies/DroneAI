"""
Application modules for specialized drone operations.

Provides ready-to-use applications for common drone use cases
including mapping, inspection, search & rescue, and personnel-recovery
(survival) tooling.
"""

from .base_application import BaseApplication, ApplicationState
from .mapping.aerial_mapper import AerialMapper, MappingConfig
from .inspection.inspector import (
    Inspector,
    InspectionConfig,
    InspectionType,
)
from .search_rescue import (
    SearchPatternGenerator,
    SearchPattern,
    SearchType,
    SearchConfig,
    SARMission,
    SARTarget,
    SARReport,
    SARState,
    SwarmSARMission,
    SwarmSARReport,
)
from .survival import (
    BeaconLocator,
    BeaconReading,
    BeaconFix,
    SupplyDropPlanner,
    SupplyDropPlan,
    DropParameters,
    SafeCorridorPlanner,
    SafeCorridor,
    CorridorSegment,
    ThreatZone,
)

__all__ = [
    # Base
    "BaseApplication",
    "ApplicationState",
    # Mapping
    "AerialMapper",
    "MappingConfig",
    # Inspection
    "Inspector",
    "InspectionConfig",
    "InspectionType",
    # Search & Rescue
    "SearchPatternGenerator",
    "SearchPattern",
    "SearchType",
    "SearchConfig",
    "SARMission",
    "SARTarget",
    "SARReport",
    "SARState",
    "SwarmSARMission",
    "SwarmSARReport",
    # Survival
    "BeaconLocator",
    "BeaconReading",
    "BeaconFix",
    "SupplyDropPlanner",
    "SupplyDropPlan",
    "DropParameters",
    "SafeCorridorPlanner",
    "SafeCorridor",
    "CorridorSegment",
    "ThreatZone",
]
