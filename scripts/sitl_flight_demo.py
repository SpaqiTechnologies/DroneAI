"""
End-to-end flight demo against ArduPilot SITL.

Sequence:
    1. Connect to MAVLink endpoint, wait for heartbeat
    2. Switch to GUIDED mode (ArduCopter takes commands from us)
    3. Arm
    4. Takeoff to 20 m
    5. Fly to a point ~50 m north of the takeoff position
    6. Hold for 5 s
    7. Switch to RTL — autopilot flies home and lands
    8. Wait for disarm, disconnect

THIS WILL FLY A DRONE. Only point it at SITL or a real vehicle where you
know what you're doing and have a clear flight area.

Usage:
    python scripts/sitl_flight_demo.py                        # default udpout:127.0.0.1:14550
    python scripts/sitl_flight_demo.py udpout:127.0.0.1:14550
    python scripts/sitl_flight_demo.py --altitude 30 --north-m 100
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.communication.mavlink_backend import MAVLinkBackend     # noqa: E402


def offset_north(lat_deg: float, north_m: float) -> float:
    """Approximate: 1 degree latitude ≈ 111_320 m."""
    return lat_deg + (north_m / 111_320.0)


def step(name: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {name}\n{bar}")


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
    parser.add_argument("--altitude", type=float, default=20.0,
                        help="takeoff altitude in meters (default 20)")
    parser.add_argument("--north-m", type=float, default=50.0,
                        help="meters north of takeoff to fly to (default 50)")
    parser.add_argument("--hold-s", type=float, default=5.0,
                        help="seconds to hold at the target (default 5)")
    parser.add_argument("--rtl-timeout", type=float, default=120.0,
                        help="max seconds to wait for RTL+disarm (default 120)")
    parser.add_argument("--confirm", action="store_true",
                        help="required confirmation that you accept the risk")
    args = parser.parse_args()

    if not args.confirm:
        print("This script will arm and fly a drone. Re-run with --confirm to proceed.")
        print("Only do this against SITL or a vehicle where you have a safe flight area.")
        return 1

    backend = MAVLinkBackend()
    try:
        step(f"connect — {args.endpoint}")
        peer = backend.connect(args.endpoint, heartbeat_timeout=10.0)
        print(f"  peer: sys={peer.system} comp={peer.component} "
              f"type={peer.type} autopilot={peer.autopilot}")

        # Let telemetry stream populate so we have a takeoff position
        print("  waiting for position fix...")
        if not wait_for(lambda: backend.position is not None,
                        timeout=30.0, description="GPS position"):
            return 2
        takeoff_pos = backend.position
        print(f"  takeoff pos: ({takeoff_pos['lat']:.6f}, "
              f"{takeoff_pos['lon']:.6f})  alt_rel={takeoff_pos['alt_rel_m']:.1f}m")

        step("switch to GUIDED")
        result = backend.set_mode("GUIDED")
        print(f"  {result.result_name}: {result.message}")
        if not result.accepted:
            return 3

        step("arm")
        result = backend.arm()
        print(f"  {result.result_name}: {result.message}")
        if not result.accepted:
            print("  ArduPilot refused to arm. Pre-arm checks may be failing.")
            print("  In SITL, try setting ARMING_CHECK=0 first if you're just")
            print("  validating the link.")
            return 4
        wait_for(lambda: backend.armed is True, timeout=5.0, description="armed=True")

        step(f"takeoff to {args.altitude:.0f}m")
        result = backend.takeoff(args.altitude)
        print(f"  {result.result_name}: {result.message}")
        if not result.accepted:
            return 5
        # Climb is autopilot-managed; wait until we're near the target altitude.
        target_alt = args.altitude
        if not wait_for(
            lambda: backend.position is not None
                    and backend.position["alt_rel_m"] >= target_alt - 1.0,
            timeout=60.0, description=f"alt >= {target_alt}m"
        ):
            return 6
        print(f"  reached {backend.position['alt_rel_m']:.1f}m")

        step(f"fly {args.north_m:.0f}m north")
        target_lat = offset_north(takeoff_pos["lat"], args.north_m)
        target_lon = takeoff_pos["lon"]
        print(f"  target: ({target_lat:.6f}, {target_lon:.6f}) alt={target_alt:.0f}m")
        backend.goto_position(target_lat, target_lon, target_alt)
        if not backend.wait_until_reached(target_lat, target_lon, target_alt,
                                          radius_m=3.0, alt_tolerance_m=2.0,
                                          timeout=90.0):
            print("  did not reach target — continuing anyway to RTL")
        else:
            print(f"  arrived at target")

        step(f"hold {args.hold_s:.0f}s")
        end = time.monotonic() + args.hold_s
        while time.monotonic() < end:
            p = backend.position or {}
            print(f"  pos=({p.get('lat', 0):.6f}, {p.get('lon', 0):.6f}) "
                  f"alt={p.get('alt_rel_m', 0):.1f}m  mode={backend.mode}")
            time.sleep(1.0)

        step("RTL — return to launch")
        result = backend.set_mode("RTL")
        print(f"  {result.result_name}: {result.message}")

        step(f"wait for disarm (up to {args.rtl_timeout:.0f}s)")
        disarmed = wait_for(lambda: backend.armed is False,
                             timeout=args.rtl_timeout, poll=1.0,
                             description="disarm after RTL")
        if disarmed:
            print("  drone disarmed. flight complete.")
            return 0
        print("  did not disarm within timeout; check vehicle state.")
        return 7

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
