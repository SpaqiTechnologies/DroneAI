"""Smoke test: docking station + adapter actually fly a patrol via AutonomyRuntime."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.docking import (
    DockingStation,
    DockState,
    ChargeProfile,
    DockAutonomyAdapter,
    PatrolPath,
)
from core.drone import Drone
from core.landing import LandingMode


def main() -> int:
    drone = Drone()
    drone.current_position = (47.5, -122.3)
    drone.current_altitude = 0.0
    drone.battery_level = 95.0

    dock = DockingStation(
        drone=drone,
        latitude=47.5,
        longitude=-122.3,
        altitude_msl_m=0.0,
        charge_profile=ChargeProfile(rate_pct_per_s=50.0, launch_min_pct=80.0, target_pct=95.0),
        tick_hz=50.0,
    )
    adapter = DockAutonomyAdapter(
        dock=dock,
        drone=drone,
        default_path=PatrolPath(
            altitude_m=15.0,
            radius_m=30.0,
            lap_count=1,
            points_per_lap=4,
            cruise_speed_mps=50.0,  # fast for sim
        ),
        landing_mode=LandingMode.EMERGENCY,
        tick_hz=50.0,
    )
    adapter.attach()
    dock.start()

    time.sleep(0.2)
    print(f"pre-deploy dock state: {dock.state.value}, battery: {drone.battery_level:.1f}")

    ok, msg = dock.deploy(altitude_m=15.0, radius_m=30.0)
    print(f"deploy: ok={ok} msg={msg}")

    # Wait for the runtime to finish.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if adapter.completions + adapter.aborts > 0:
            break
        time.sleep(0.1)

    print(f"deployments: {adapter.deployments}")
    print(f"completions: {adapter.completions}")
    print(f"aborts: {adapter.aborts}")
    last = adapter.last_runtime_status()
    if last is not None:
        print(f"final phase: {last.get('phase')}, tick_count: {last.get('tick_count')}")

    time.sleep(0.5)  # let dock tick land/charge
    print(f"final drone alt: {drone.current_altitude:.2f}, dock state: {dock.state.value}")

    dock.stop()
    ok = adapter.completions >= 1 and dock.state in (DockState.CHARGING, DockState.READY, DockState.DOCKED)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
