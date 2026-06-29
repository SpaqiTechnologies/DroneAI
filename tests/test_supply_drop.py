"""Tests for the supply drop planner."""

import math

import pytest

from applications.survival.supply_drop import (
    DropParameters,
    SupplyDropPlanner,
)


def _meters_between(a, b) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def test_no_wind_release_near_target() -> None:
    planner = SupplyDropPlanner(DropParameters(
        release_altitude_agl_m=30.0,
        release_speed_mps=0.0,
        drag_coefficient=0.1,
    ))
    plan = planner.plan(target_lat=47.5, target_lon=-122.3,
                        wind_speed_mps=0.0, wind_direction_deg=0.0,
                        release_heading_deg=0.0)
    dist = _meters_between((plan.release_lat, plan.release_lon), (47.5, -122.3))
    assert dist < 5.0


def test_release_point_is_upwind_of_target() -> None:
    # Wind from 270 (out of the west) blows package toward east. Release
    # point should compensate by being west of target. Drag must be > 0 so
    # the package actually couples to the wind during fall.
    planner = SupplyDropPlanner(DropParameters(
        release_altitude_agl_m=40.0,
        release_speed_mps=0.0,
        drag_coefficient=0.3,
    ))
    plan = planner.plan(target_lat=47.5, target_lon=-122.3,
                        wind_speed_mps=10.0, wind_direction_deg=270.0,
                        release_heading_deg=270.0)
    # 270 is "from west" → wind blows east → release upwind = west of target
    assert plan.release_lon < -122.3
    assert plan.horizontal_drift_m > 1.0


def test_higher_altitude_means_longer_fall_time() -> None:
    p_lo = SupplyDropPlanner(DropParameters(release_altitude_agl_m=20.0)).plan(
        47.5, -122.3, wind_speed_mps=0.0
    )
    p_hi = SupplyDropPlanner(DropParameters(release_altitude_agl_m=80.0)).plan(
        47.5, -122.3, wind_speed_mps=0.0
    )
    assert p_hi.fall_time_s > p_lo.fall_time_s


def test_parachute_extends_fall_time() -> None:
    no_chute = SupplyDropPlanner(DropParameters(
        release_altitude_agl_m=60.0,
        parachute_deploy_at_m=None,
    )).plan(47.5, -122.3)
    with_chute = SupplyDropPlanner(DropParameters(
        release_altitude_agl_m=60.0,
        parachute_deploy_at_m=30.0,
        parachute_terminal_speed_mps=4.0,
    )).plan(47.5, -122.3)
    assert with_chute.fall_time_s > no_chute.fall_time_s


def test_negative_wind_rejected() -> None:
    with pytest.raises(ValueError):
        SupplyDropPlanner().plan(47.5, -122.3, wind_speed_mps=-1.0)


def test_to_dict_round_trip() -> None:
    planner = SupplyDropPlanner()
    plan = planner.plan(47.5, -122.3, wind_speed_mps=5.0, wind_direction_deg=180.0)
    d = plan.to_dict()
    assert d["target"]["latitude"] == 47.5
    assert d["release_point"]["altitude_agl_m"] > 0
    assert "wind" in d
