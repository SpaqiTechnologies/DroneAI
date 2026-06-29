"""Smoke test: SAR mission must descend + multi-photo on a high-conf detection."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applications.search_rescue import SARMission, SARState, SearchConfig, SearchType
from core.drone import Drone
from sensors.media import MediaStorage


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        drone = Drone()
        drone.current_position = (47.5, -122.3)
        drone.current_altitude = 30.0  # cruise altitude
        drone.camera_sensor._media_storage = MediaStorage(root=td)
        drone.camera_sensor._synth_resolution = (64, 36)
        drone.camera_sensor.start()
        # Simulate a high-confidence detection that persists
        drone.camera_sensor.simulate_obstacle(
            x=300, y=200, width=80, height=80,
            distance=8.0, confidence=0.92,
        )
        sar = SARMission(
            drone=drone,
            pattern_type=SearchType.EXPANDING_SQUARE,
            center=(47.5, -122.3),
            config=SearchConfig(altitude=30.0, track_spacing=60.0, speed=20.0, legs=4),
            cruise_speed_mps=80.0,
            photo_interval_s=10.0,  # don't spam photos during cruise
            dedupe_radius_m=200.0,  # one investigation per area
            investigate_targets=True,
            investigation_confidence=0.8,
            investigation_altitude_m=10.0,
            investigation_duration_s=0.5,
            investigation_photos=3,
            investigation_descent_mps=100.0,  # fast for sim
            return_to_start=False,
        )
        states_seen: list[str] = []
        sar.start()
        for _ in range(2000):
            states_seen.append(sar.state.value)
            sar.tick(0.05)
            if sar.is_done:
                break
        report = sar.report()
        print(f"state={report.state.value}")
        print(f"waypoints_completed={report.waypoints_completed}/{report.waypoints_total}")
        print(f"targets={len(report.targets)} investigations_completed={report.investigations_completed}")
        print(f"photos={report.photos_taken} distance={report.distance_traveled_m:.0f}m")
        print(f"distinct states visited: {sorted(set(states_seen))}")
        if report.targets:
            t = report.targets[0]
            inv_photos = t.metadata.get("investigation_photos", [])
            print(f"first target: investigated={t.metadata.get('investigated')} "
                  f"photo_count={t.metadata.get('investigation_photo_count')} "
                  f"photo_paths={len(inv_photos)}")
        ok = (
            "investigating" in states_seen
            and report.investigations_completed >= 1
            and report.targets
            and report.targets[0].metadata.get("investigated") is True
        )
        print("OK" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
