"""Tests for the safe corridor planner."""

import math

from applications.survival.safe_corridor import (
    SafeCorridorPlanner,
    ThreatZone,
)


def _meters_between(a, b) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def test_no_threats_direct_path() -> None:
    planner = SafeCorridorPlanner()
    corridor = planner.plan(start=(47.50, -122.30), goal=(47.51, -122.31))
    assert corridor.safe
    assert len(corridor.waypoints) == 2
    assert corridor.total_length_m > 0
    assert len(corridor.segments) == 1


def test_threat_in_the_way_routes_around() -> None:
    # threat sits directly between start (47.50, -122.30) and goal (47.51, -122.30)
    threat = ThreatZone(latitude=47.505, longitude=-122.30, radius_m=120.0, name="checkpoint")
    planner = SafeCorridorPlanner(threats=[threat], margin_m=20.0)
    corridor = planner.plan(start=(47.50, -122.30), goal=(47.51, -122.30))
    assert corridor.safe
    assert len(corridor.waypoints) > 2
    direct = _meters_between((47.50, -122.30), (47.51, -122.30))
    assert corridor.total_length_m > direct
    assert corridor.min_threat_distance_m >= 0


def test_start_inside_threat_keepout_rejected() -> None:
    threat = ThreatZone(latitude=47.50001, longitude=-122.30001, radius_m=100.0)
    planner = SafeCorridorPlanner(threats=[threat], margin_m=10.0)
    corridor = planner.plan(start=(47.50, -122.30), goal=(47.52, -122.30))
    assert not corridor.safe
    assert corridor.failure_reason is not None
    assert "start" in corridor.failure_reason


def test_goal_inside_threat_keepout_rejected() -> None:
    threat = ThreatZone(latitude=47.51, longitude=-122.30, radius_m=120.0)
    planner = SafeCorridorPlanner(threats=[threat], margin_m=5.0)
    corridor = planner.plan(start=(47.50, -122.30), goal=(47.51001, -122.30001))
    assert not corridor.safe
    assert "goal" in (corridor.failure_reason or "")


def test_corridor_to_dict_serializable() -> None:
    planner = SafeCorridorPlanner()
    corridor = planner.plan(start=(47.50, -122.30), goal=(47.501, -122.301))
    d = corridor.to_dict()
    assert d["safe"] is True
    assert "waypoints" in d
    assert "segments" in d
