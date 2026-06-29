"""Tests for the dock→autonomy adapter and the dashboard route."""

import time

import pytest

from core.docking import (
    DockingStation,
    DockState,
    ChargeProfile,
    DockAutonomyAdapter,
    PatrolPath,
)
from core.docking.autonomy_adapter import _generate_orbit_waypoints
from core.drone import Drone
from core.landing import LandingMode

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


def _drone_at(lat: float, lon: float, battery: float = 95.0) -> Drone:
    d = Drone()
    d.current_position = (lat, lon)
    d.current_altitude = 0.0
    d.battery_level = battery
    return d


# ============================ Adapter ============================


def test_generate_orbit_waypoints_closed_ring() -> None:
    wps = _generate_orbit_waypoints(
        center=(47.5, -122.3), altitude_m=20.0, radius_m=50.0, points=8, laps=2,
    )
    assert len(wps) == 16  # 8 points × 2 laps
    # All waypoints should be at the orbit altitude
    assert all(abs(wp[2] - 20.0) < 0.01 for wp in wps)


def test_adapter_attach_sets_hooks() -> None:
    drone = _drone_at(47.5, -122.3)
    dock = DockingStation(drone, 47.5, -122.3)
    adapter = DockAutonomyAdapter(dock=dock, drone=drone)
    assert adapter._attached is False
    adapter.attach()
    assert adapter._attached is True
    # The dock should have a non-default deploy hook now
    assert dock._on_deploy_hook is not None
    adapter.detach()
    assert adapter._attached is False


def test_adapter_runs_full_patrol_loop() -> None:
    drone = _drone_at(47.5, -122.3)
    dock = DockingStation(
        drone, 47.5, -122.3,
        charge_profile=ChargeProfile(rate_pct_per_s=50.0, launch_min_pct=80.0),
        tick_hz=50.0,
    )
    adapter = DockAutonomyAdapter(
        dock=dock, drone=drone,
        default_path=PatrolPath(
            altitude_m=10.0, radius_m=20.0, lap_count=1,
            points_per_lap=4, cruise_speed_mps=80.0,
        ),
        landing_mode=LandingMode.EMERGENCY,
        tick_hz=50.0,
    )
    adapter.attach()
    dock.start()
    try:
        time.sleep(0.1)  # let dock reach READY
        ok, _ = dock.deploy()
        assert ok
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if adapter.completions + adapter.aborts >= 1:
                break
            time.sleep(0.1)
        assert adapter.deployments == 1
        assert adapter.completions == 1
        assert adapter.aborts == 0
        last = adapter.last_runtime_status()
        assert last is not None
        assert last["phase"] == "completed"
    finally:
        dock.stop()


def test_adapter_abort_on_recall() -> None:
    drone = _drone_at(47.5, -122.3)
    dock = DockingStation(drone, 47.5, -122.3, tick_hz=50.0)
    adapter = DockAutonomyAdapter(
        dock=dock, drone=drone,
        default_path=PatrolPath(
            altitude_m=10.0, radius_m=20.0, lap_count=20,
            points_per_lap=8, cruise_speed_mps=0.5,  # slow → long flight
        ),
        landing_mode=LandingMode.EMERGENCY,
        tick_hz=50.0,
    )
    adapter.attach()
    dock.start()
    try:
        time.sleep(0.1)
        ok, _ = dock.deploy()
        assert ok
        time.sleep(0.2)
        rt = adapter.current_runtime
        assert rt is not None and rt.is_running
        dock.recall(reason="test recall")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not rt.is_running:
                break
            time.sleep(0.05)
        assert not rt.is_running
    finally:
        dock.stop()


def test_adapter_rejects_bad_tick_hz() -> None:
    drone = _drone_at(47.5, -122.3)
    dock = DockingStation(drone, 47.5, -122.3)
    with pytest.raises(ValueError):
        DockAutonomyAdapter(dock=dock, drone=drone, tick_hz=0)


# ============================ Dashboard ============================


pytestmark = pytest.mark.skipif(TestClient is None, reason="FastAPI TestClient unavailable")


@pytest.fixture(scope="module")
def client():
    from simulation.fastapi_server import app, state
    state.initialize_drone()
    with TestClient(app) as c:
        yield c


def test_dashboard_route_returns_html(client) -> None:
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "<title>Drone AI Dashboard</title>" in body
    assert "/api/media/latest.png" in body
    assert "/api/dock/status" in body
    assert "/api/scan3d/" in body


def test_root_route_links_to_dashboard(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "/dashboard" in r.text


def test_dock_setup_endpoint_attaches_adapter(client) -> None:
    body = {
        "latitude": 47.5, "longitude": -122.3,
        "rate_pct_per_s": 50.0, "launch_min_pct": 50.0,
        "patrol_altitude_m": 10.0, "patrol_radius_m": 20.0,
        "patrol_lap_count": 1, "patrol_points_per_lap": 4,
        "patrol_cruise_speed_mps": 80.0,
    }
    r = client.post("/api/dock/setup", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["adapter_attached"] is True


def test_dock_status_includes_adapter_counters(client) -> None:
    r = client.get("/api/dock/status")
    assert r.status_code == 200
    data = r.json()
    assert "adapter" in data
    assert "deployments" in data["adapter"]
