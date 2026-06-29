"""Tests for the unified autonomy runtime."""

import time

from core.autonomy_runtime import AutonomyRuntime, RuntimePhase
from core.drone import Drone
from core.landing import LandingMode


def _make_drone() -> Drone:
    d = Drone()
    d.current_position = (47.5, -122.3)
    d.current_altitude = 0.0
    d.camera_sensor.start()
    return d


def test_runtime_completes_with_no_managers() -> None:
    drone = _make_drone()
    runtime = AutonomyRuntime(drone=drone, tick_hz=50.0)

    def cruise_tick(dt, rt):
        return True  # immediately complete cruise

    # EMERGENCY landing bypasses wind/abort checks so a sim drone with
    # default wind sensor noise won't abort the descent.
    started = runtime.start_flight(
        target_altitude=3.0,
        cruise_tick=cruise_tick,
        landing_mode=LandingMode.EMERGENCY,
    )
    assert started
    runtime.join(timeout=10.0)
    assert not runtime.is_running
    status = runtime.status()
    assert status.phase in (RuntimePhase.COMPLETED, RuntimePhase.LANDING)


def test_runtime_progresses_phases() -> None:
    drone = _make_drone()
    runtime = AutonomyRuntime(drone=drone, tick_hz=50.0)
    seen = []
    runtime.add_phase_callback(lambda p, s: seen.append(p))

    elapsed = {"t": 0.0}
    def cruise_tick(dt, rt):
        elapsed["t"] += dt
        return elapsed["t"] >= 0.05

    runtime.start_flight(
        target_altitude=3.0,
        cruise_tick=cruise_tick,
        landing_mode=LandingMode.EMERGENCY,
    )
    runtime.join(timeout=10.0)
    assert RuntimePhase.TAKEOFF in seen
    assert RuntimePhase.CRUISE in seen
    assert RuntimePhase.LANDING in seen


def test_runtime_abort_stops_flight() -> None:
    drone = _make_drone()
    runtime = AutonomyRuntime(drone=drone, tick_hz=20.0)

    def cruise_tick(dt, rt):
        time.sleep(0.01)
        return False

    runtime.start_flight(target_altitude=10.0, cruise_tick=cruise_tick)
    time.sleep(0.1)
    runtime.abort("test")
    runtime.join(timeout=3.0)
    status = runtime.status()
    assert status.phase in (RuntimePhase.ABORTED, RuntimePhase.COMPLETED)


def test_runtime_rejects_double_start() -> None:
    drone = _make_drone()
    runtime = AutonomyRuntime(drone=drone, tick_hz=20.0)

    def cruise_tick(dt, rt):
        time.sleep(0.005)
        return False

    assert runtime.start_flight(target_altitude=10.0, cruise_tick=cruise_tick)
    assert not runtime.start_flight(target_altitude=10.0, cruise_tick=cruise_tick)
    runtime.abort()
    runtime.join(timeout=3.0)


def test_runtime_status_to_dict_serializable() -> None:
    drone = _make_drone()
    runtime = AutonomyRuntime(drone=drone, tick_hz=20.0)
    status = runtime.status()
    d = status.to_dict()
    assert "phase" in d and "tick_count" in d
