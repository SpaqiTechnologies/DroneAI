"""
Fly a planned mission against ArduPilot SITL.

Builds a small Mission in code (takeoff position + 3 waypoints around it +
RTL), uploads it via MAVLink, arms, switches to AUTO, and watches the
autopilot execute it.

You can swap in a real mission loaded from disk by uncommenting the
load-from-storage path.

THIS WILL FLY A DRONE. Use only against SITL or a known-safe vehicle.

Usage:
    python scripts/sitl_mission_demo.py --confirm
    python scripts/sitl_mission_demo.py --confirm --alt 30 --leg-m 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.communication.mavlink_backend import MAVLinkBackend      # noqa: E402
from core.mission import Mission, Waypoint                          # noqa: E402


def offset_north_east(lat_deg: float, lon_deg: float,
                       north_m: float, east_m: float) -> tuple[float, float]:
    """Approximate small-offset lat/lon shift. Good to ~1m at city scales."""
    import math
    new_lat = lat_deg + (north_m / 111_320.0)
    new_lon = lon_deg + (east_m / (111_320.0 * math.cos(math.radians(lat_deg))))
    return new_lat, new_lon


def build_square_mission(home_lat: float, home_lon: float,
                          alt_m: float, leg_m: float) -> Mission:
    """A simple 4-corner box at given altitude, returning home at the end."""
    mission = Mission(
        name="sitl-demo-square",
        default_altitude=alt_m,
        return_home=True,
    )
    corners = [
        (leg_m, 0),         # north
        (leg_m, leg_m),     # northeast
        (0, leg_m),         # east
        (0, 0),             # back to start
    ]
    for north_m, east_m in corners:
        lat, lon = offset_north_east(home_lat, home_lon, north_m, east_m)
        mission.add_waypoint(Waypoint(
            latitude=lat,
            longitude=lon,
            altitude=alt_m,
            acceptance_radius=3.0,
        ))
    return mission


def step(title: str) -> None:
    print("\n" + "=" * 60 + f"\n  {title}\n" + "=" * 60)


def wait_for(predicate, *, timeout: float, poll: float = 0.5,
             description: str = "condition") -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    print(f"  timed out waiting for {description} ({timeout:.0f}s)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", nargs="?", default="udpout:127.0.0.1:14550")
    parser.add_argument("--alt", type=float, default=20.0,
                        help="cruise altitude in meters (default 20)")
    parser.add_argument("--leg-m", type=float, default=50.0,
                        help="square leg length in meters (default 50)")
    parser.add_argument("--mission-timeout", type=float, default=300.0,
                        help="max seconds to wait for mission completion + disarm")
    parser.add_argument("--confirm", action="store_true",
                        help="required confirmation that you accept the risk")
    args = parser.parse_args()

    if not args.confirm:
        print("This script will arm and fly a mission. Re-run with --confirm.")
        return 1

    backend = MAVLinkBackend()
    try:
        step(f"connect — {args.endpoint}")
        peer = backend.connect(args.endpoint, heartbeat_timeout=10.0)
        print(f"  peer: sys={peer.system} comp={peer.component} "
              f"type={peer.type} autopilot={peer.autopilot}")

        print("  waiting for GPS position...")
        if not wait_for(lambda: backend.position is not None,
                        timeout=30.0, description="GPS position"):
            return 2
        home = backend.position
        print(f"  home: ({home['lat']:.6f}, {home['lon']:.6f})")

        step("build mission")
        mission = build_square_mission(
            home_lat=home["lat"],
            home_lon=home["lon"],
            alt_m=args.alt,
            leg_m=args.leg_m,
        )
        print(f"  {mission.name}: {len(mission.waypoints)} waypoints, "
              f"alt={args.alt}m, leg={args.leg_m}m, RTL=True")

        step("clear any existing mission")
        result = backend.clear_mission()
        print(f"  {'OK' if result.success else 'FAIL'}: {result.ack_name or result.error}")

        step("upload mission")
        result = backend.upload_mission(mission, takeoff_alt_m=args.alt,
                                         timeout=20.0)
        if not result.success:
            print(f"  upload failed: {result.ack_name or result.error}")
            return 3
        print(f"  uploaded {result.item_count} items (takeoff + waypoints + RTL)")

        step("verify by downloading back")
        readback = backend.download_mission(timeout=20.0)
        if readback.success:
            print(f"  autopilot reports {readback.item_count} items stored")
        else:
            print(f"  download readback failed: {readback.error}")

        step("switch to GUIDED, arm")
        # ArduCopter likes GUIDED for the arming step; AUTO starts the mission.
        mr = backend.set_mode("GUIDED")
        print(f"  GUIDED: {mr.result_name}")
        ar = backend.arm()
        print(f"  arm: {ar.result_name}: {ar.message}")
        if not ar.accepted:
            print("  arm refused — check pre-arm checks (in SITL: param set ARMING_CHECK 0)")
            return 4
        wait_for(lambda: backend.armed is True, timeout=5.0, description="armed=True")

        step("start mission (AUTO + MISSION_START)")
        ms = backend.start_mission()
        print(f"  {ms.result_name}: {ms.message}")
        if not ms.accepted:
            print("  start_mission refused; trying just AUTO mode")
            backend.set_mode("AUTO")

        step("watching mission execute")
        deadline = time.monotonic() + args.mission_timeout
        last_print = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_print >= 2.0:
                p = backend.position or {}
                print(f"  mode={backend.mode}  armed={backend.armed}  "
                      f"pos=({p.get('lat', 0):.6f}, {p.get('lon', 0):.6f}) "
                      f"alt={p.get('alt_rel_m', 0):.1f}m")
                last_print = now
            if backend.armed is False:
                print("  drone disarmed — mission complete.")
                return 0
            time.sleep(0.5)

        print("  mission_timeout reached; vehicle may still be flying.")
        return 5

    except TimeoutError as e:
        print(f"connect failed: {e}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted. attempting RTL...")
        try:
            backend.set_mode("RTL")
        except Exception as e:
            print(f"  RTL request failed: {e}")
        return 130
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
