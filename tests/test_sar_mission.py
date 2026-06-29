"""Tests for the SAR mission orchestrator and the swarm variant."""

import tempfile

import pytest

from applications.search_rescue import (
    SARMission,
    SARState,
    SARReport,
    SearchConfig,
    SearchType,
    SwarmSARMission,
    allocate_subareas,
)
from core.drone import Drone


def _make_drone() -> Drone:
    d = Drone()
    d.current_position = (47.5, -122.3)
    d.current_altitude = 0.0
    d.camera_sensor.start()
    return d


def _run_to_completion(sar: SARMission, max_ticks: int = 5000, dt: float = 0.1) -> None:
    sar.start()
    for _ in range(max_ticks):
        sar.tick(dt)
        if sar.is_done:
            return
    raise AssertionError("SAR mission did not complete within tick budget")


def test_sar_mission_completes_pattern() -> None:
    drone = _make_drone()
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=20.0, track_spacing=30.0, speed=20.0, legs=4),
        cruise_speed_mps=50.0,
        photo_interval_s=10.0,
        return_to_start=False,
    )
    _run_to_completion(sar)
    report = sar.report()
    assert report.state == SARState.COMPLETED
    assert report.waypoints_completed >= 4
    assert report.distance_traveled_m > 0


def test_sar_mission_records_targets_from_camera() -> None:
    drone = _make_drone()
    drone.camera_sensor.simulate_obstacle(
        x=400, y=200, width=60, height=60,
        distance=8.0, confidence=0.9,
    )
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=20.0, track_spacing=80.0, speed=20.0, legs=4),
        cruise_speed_mps=80.0,
        photo_interval_s=10.0,
        dedupe_radius_m=30.0,
        return_to_start=False,
    )
    _run_to_completion(sar)
    report = sar.report()
    assert len(report.targets) >= 1
    assert all(t.confidence >= 0.6 for t in report.targets)


def test_sar_mission_returns_to_start_when_enabled() -> None:
    drone = _make_drone()
    start_pos = drone.current_position
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=20.0, track_spacing=40.0, speed=20.0, legs=4),
        cruise_speed_mps=80.0,
        photo_interval_s=10.0,
        return_to_start=True,
    )
    _run_to_completion(sar)
    # After return-to-start, drone should be back near the original spot
    dx = abs(drone.current_position[0] - start_pos[0])
    dy = abs(drone.current_position[1] - start_pos[1])
    assert dx < 1e-4 and dy < 1e-4


def test_sar_mission_abort_terminates() -> None:
    drone = _make_drone()
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=20.0, track_spacing=40.0, speed=20.0, legs=8),
        cruise_speed_mps=1.0,
        return_to_start=False,
    )
    sar.start()
    sar.tick(0.1)
    sar.abort("test abort")
    assert sar.is_done
    assert sar.state == SARState.ABORTED


def test_allocate_subareas_returns_n_slots() -> None:
    slots = allocate_subareas((47.5, -122.3), radius_m=500.0, n=4)
    assert len(slots) == 4
    for (center, radius, heading) in slots:
        assert radius > 0
        assert 0 <= heading < 360.0


def test_allocate_subareas_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        allocate_subareas((0.0, 0.0), radius_m=100.0, n=0)
    with pytest.raises(ValueError):
        allocate_subareas((0.0, 0.0), radius_m=-1.0, n=2)


def test_swarm_sar_completes() -> None:
    drones = {f"drone_{i}": _make_drone() for i in range(3)}
    swarm = SwarmSARMission(
        drones=drones,
        center=(47.5, -122.3),
        radius_m=200.0,
        config=SearchConfig(altitude=20.0, track_spacing=40.0, speed=30.0, legs=4),
        pattern_type=SearchType.SECTOR,
        return_to_start=False,
    )
    swarm.start()
    for _ in range(5000):
        swarm.tick(0.1)
        if swarm.is_done:
            break
    assert swarm.is_done
    rep = swarm.report()
    assert rep.completed_drones == 3
    assert rep.drone_count == 3


def test_swarm_sar_rejects_empty_drones() -> None:
    with pytest.raises(ValueError):
        SwarmSARMission(drones={}, center=(47.5, -122.3))


def test_swarm_sar_aggregates_and_dedupes_targets() -> None:
    drones = {f"drone_{i}": _make_drone() for i in range(2)}
    # both drones see "the same" obstacle in their feed
    for d in drones.values():
        d.camera_sensor.simulate_obstacle(
            x=400, y=200, width=60, height=60,
            distance=8.0, confidence=0.9,
        )
    swarm = SwarmSARMission(
        drones=drones,
        center=(47.5, -122.3),
        radius_m=200.0,
        config=SearchConfig(altitude=20.0, track_spacing=50.0, speed=30.0, legs=4),
        pattern_type=SearchType.SECTOR,
        return_to_start=False,
        dedupe_radius_m=100.0,
    )
    swarm.start()
    for _ in range(5000):
        swarm.tick(0.1)
        if swarm.is_done:
            break
    rep = swarm.report()
    total_individual = sum(len(r.targets) for r in rep.per_drone_reports.values())
    # aggregated should not exceed the sum of all individual reports
    assert len(rep.aggregated_targets) <= total_individual
