"""Soldier-survival application layer.

Defensive / personnel-recovery use cases: locate emergency beacons via
RSSI trilateration, plan wind-corrected supply drop release points, and
plan ground safe corridors that avoid known threat polygons + obstacles.

Explicitly excluded: any targeting, strike, weapons fire-control, or
combat-intent logic.
"""

from applications.survival.beacon_locator import (
    BeaconLocator,
    BeaconReading,
    BeaconFix,
    rssi_to_distance,
)
from applications.survival.supply_drop import (
    SupplyDropPlanner,
    SupplyDropPlan,
    DropParameters,
)
from applications.survival.safe_corridor import (
    SafeCorridorPlanner,
    SafeCorridor,
    CorridorSegment,
    ThreatZone,
)

__all__ = [
    "BeaconLocator",
    "BeaconReading",
    "BeaconFix",
    "rssi_to_distance",
    "SupplyDropPlanner",
    "SupplyDropPlan",
    "DropParameters",
    "SafeCorridorPlanner",
    "SafeCorridor",
    "CorridorSegment",
    "ThreatZone",
]
