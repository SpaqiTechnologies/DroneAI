"""Smoke test for the new survival + SAR + autonomy modules."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applications.survival import (
    BeaconLocator,
    SupplyDropPlanner,
    DropParameters,
    SafeCorridorPlanner,
    ThreatZone,
)
from applications.search_rescue import (
    SARMission,
    SARState,
    SearchType,
    SearchConfig,
    SwarmSARMission,
)
from core.drone import Drone


def smoke_beacon() -> None:
    print("\n=== BeaconLocator ===")
    bl = BeaconLocator(tx_power_dbm=-40.0, path_loss_exponent=2.5)
    true_lat, true_lon = 47.5001, -122.3001

    samples = [
        (47.5000, -122.3000, 10.0, -55.0),
        (47.5005, -122.3000, 10.0, -65.0),
        (47.5000, -122.3010, 10.0, -68.0),
        (47.5005, -122.3010, 10.0, -70.0),
    ]
    for lat, lon, alt, rssi in samples:
        bl.add_reading(lat, lon, alt, rssi)
    fix = bl.compute_fix()
    assert fix is not None
    print(f"fix lat={fix.latitude:.5f} lon={fix.longitude:.5f} method={fix.method} "
          f"conf={fix.confidence:.2f} residual={fix.residual_m:.1f}m samples={fix.sample_count}")


def smoke_supply() -> None:
    print("\n=== SupplyDropPlanner ===")
    planner = SupplyDropPlanner(DropParameters(
        release_altitude_agl_m=40.0,
        release_speed_mps=10.0,
        parachute_deploy_at_m=15.0,
    ))
    plan = planner.plan(
        target_lat=47.5,
        target_lon=-122.3,
        wind_speed_mps=6.0,
        wind_direction_deg=270.0,
    )
    import json
    print(json.dumps(plan.to_dict(), indent=2))


def smoke_corridor() -> None:
    print("\n=== SafeCorridorPlanner ===")
    planner = SafeCorridorPlanner(threats=[
        ThreatZone(latitude=47.5005, longitude=-122.3005, radius_m=100.0, name="threat A"),
        ThreatZone(latitude=47.5008, longitude=-122.3015, radius_m=75.0, name="threat B"),
    ], margin_m=25.0)
    corridor = planner.plan(start=(47.5000, -122.3000), goal=(47.5015, -122.3025))
    import json
    print("safe:", corridor.safe, "length:", corridor.total_length_m, "waypoints:", len(corridor.waypoints))
    print(json.dumps([s.to_dict() for s in corridor.segments[:3]], indent=2))


def smoke_sar() -> None:
    print("\n=== SARMission ===")
    drone = Drone()
    drone.current_position = (47.5, -122.3)
    drone.current_altitude = 0.0
    drone.camera_sensor.start()

    drone.camera_sensor.simulate_obstacle(
        x=400, y=200, width=60, height=60,
        distance=8.0, confidence=0.85,
    )

    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=30.0, track_spacing=50.0, speed=20.0, legs=4),
        cruise_speed_mps=50.0,
        photo_interval_s=0.0,
        return_to_start=False,
    )
    sar.start()
    for _ in range(2000):
        sar.tick(0.1)
        if sar.is_done:
            break
    report = sar.report()
    print(f"state={report.state.value} progress={report.progress_percent:.0f}% "
          f"targets={len(report.targets)} distance={report.distance_traveled_m:.0f}m "
          f"photos={report.photos_taken}")
    if report.targets:
        t0 = report.targets[0]
        print(f"  first target: {t0.detection_type} conf={t0.confidence:.2f} "
              f"at ({t0.latitude:.5f},{t0.longitude:.5f})")


def smoke_swarm_sar() -> None:
    print("\n=== SwarmSARMission ===")
    drones = {}
    for i in range(3):
        d = Drone()
        d.current_position = (47.5, -122.3)
        d.current_altitude = 0.0
        d.camera_sensor.start()
        drones[f"drone_{i}"] = d
    swarm = SwarmSARMission(
        drones=drones,
        center=(47.5, -122.3),
        radius_m=200.0,
        config=SearchConfig(altitude=30.0, track_spacing=50.0, speed=20.0, legs=4),
        pattern_type=SearchType.SECTOR,
        return_to_start=False,
    )
    swarm.start()
    for _ in range(2000):
        swarm.tick(0.1)
        if swarm.is_done:
            break
    rep = swarm.report()
    print(f"completed={rep.completed_drones}/{rep.drone_count} "
          f"aborted={rep.aborted_drones} total_targets={len(rep.aggregated_targets)} "
          f"duration={rep.duration_s:.1f}s")


def main() -> int:
    smoke_beacon()
    smoke_supply()
    smoke_corridor()
    smoke_sar()
    smoke_swarm_sar()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
