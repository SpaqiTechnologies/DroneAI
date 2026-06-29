"""Smoke tests for the new FastAPI routes using TestClient (in-process)."""

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


pytestmark = pytest.mark.skipif(TestClient is None, reason="FastAPI TestClient unavailable")


@pytest.fixture(scope="module")
def client():
    from simulation.fastapi_server import app, state
    state.initialize_drone()
    with TestClient(app) as c:
        yield c


def test_supply_drop_endpoint(client) -> None:
    body = {
        "target": {"latitude": 47.5, "longitude": -122.3},
        "wind_speed_mps": 4.0,
        "wind_direction_deg": 180.0,
        "release_altitude_agl_m": 40.0,
        "release_speed_mps": 10.0,
    }
    r = client.post("/api/survival/supply-drop/plan", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "release_point" in data
    assert data["fall_time_s"] > 0


def test_corridor_endpoint(client) -> None:
    body = {
        "start": {"latitude": 47.5, "longitude": -122.3},
        "goal": {"latitude": 47.51, "longitude": -122.3},
        "threats": [{
            "latitude": 47.505, "longitude": -122.30,
            "radius_m": 100.0, "name": "test",
        }],
        "margin_m": 20.0,
    }
    r = client.post("/api/survival/corridor/plan", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "safe" in data and "waypoints" in data


def test_beacon_flow(client) -> None:
    client.post("/api/survival/beacon/reset", json={})
    for lat, lon, rssi in [
        (47.5000, -122.3000, -55),
        (47.5005, -122.3005, -65),
        (47.5002, -122.3010, -70),
    ]:
        r = client.post(
            "/api/survival/beacon/sample",
            json={"latitude": lat, "longitude": lon, "altitude": 20, "rssi_dbm": rssi},
        )
        assert r.status_code == 200
    r = client.get("/api/survival/beacon/fix")
    assert r.status_code == 200
    data = r.json()
    assert data["fix"] is not None
    assert data["sample_count"] == 3


def test_sar_plan_endpoint(client) -> None:
    body = {
        "center": {"latitude": 47.5, "longitude": -122.3},
        "pattern_type": "expanding_square",
        "altitude": 30, "track_spacing": 50, "speed": 8, "legs": 6,
    }
    r = client.post("/api/sar/plan", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["waypoint_count"] >= 6
    assert len(data["waypoints"]) == data["waypoint_count"]


def test_media_snapshot_and_listing(client) -> None:
    r = client.post("/api/media/snapshot", json={})
    assert r.status_code == 200
    assert r.json()["success"]
    r = client.get("/api/media")
    assert r.status_code == 200
    data = r.json()
    assert len(data["photos"]) >= 1


def test_media_latest_png_returns_image(client) -> None:
    r = client.get("/api/media/latest.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    body = r.content
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_media_stream_emits_at_least_one_frame(client) -> None:
    # Bounded SSE stream so the generator exits cleanly without us racing
    # the close on the sync TestClient.
    r = client.get("/api/media/stream?fps=20&max_frames=1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "event: frame" in text
    assert "png_base64" in text


def test_autonomy_run_and_status(client) -> None:
    body = {
        "target_altitude": 3.0,
        "landing_mode": "emergency",
        "use_takeoff_manager": False,
        "cruise_seconds": 0.1,
        "tick_hz": 50.0,
    }
    r = client.post("/api/autonomy/run", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    rid = data["runtime_id"]
    assert rid

    # second concurrent start should 409
    r2 = client.post("/api/autonomy/run", json=body)
    assert r2.status_code in (409, 200)

    # poll until finished or timeout
    import time
    deadline = time.time() + 10.0
    while time.time() < deadline:
        s = client.get(f"/api/autonomy/{rid}/status").json()
        if not s["is_running"]:
            break
        time.sleep(0.1)
    final = client.get(f"/api/autonomy/{rid}/status").json()
    assert final["status"]["phase"] in ("completed", "landing", "aborted")


def test_autonomy_abort(client) -> None:
    body = {
        "target_altitude": 3.0,
        "landing_mode": "emergency",
        "use_takeoff_manager": False,
        "cruise_seconds": 60.0,
        "tick_hz": 20.0,
    }
    r = client.post("/api/autonomy/run", json=body)
    assert r.status_code == 200
    rid = r.json()["runtime_id"]
    r2 = client.post(f"/api/autonomy/{rid}/abort")
    assert r2.status_code == 200
    assert r2.json()["status"]["phase"] in ("aborted", "completed", "landing", "failed")


def test_autonomy_list_endpoint(client) -> None:
    r = client.get("/api/autonomy")
    assert r.status_code == 200
    data = r.json()
    assert "runtime_ids" in data and "active" in data
