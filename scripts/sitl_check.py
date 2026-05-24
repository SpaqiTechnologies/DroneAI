"""
Smoke test against a real SITL (or hardware) endpoint.

Usage:
    python scripts/sitl_check.py                          # udpout:127.0.0.1:14550
    python scripts/sitl_check.py udpout:127.0.0.1:14550
    python scripts/sitl_check.py udpin:0.0.0.0:14550
    python scripts/sitl_check.py tcp:127.0.0.1:5760
    python scripts/sitl_check.py COM3 --baud 57600

What it does:
    1. Connect to the endpoint.
    2. Wait up to 10 seconds for a HEARTBEAT.
    3. Print peer identity (system/component, MAV_TYPE, autopilot).
    4. Stream telemetry once per second for 15 seconds.
    5. Disconnect cleanly.

It does NOT arm or move the vehicle. It's read-only by design — a way to
prove the link works before you trust it with arm/takeoff commands.

To spin up ArduPilot SITL (see docs in mavlink_link.py for full setup):
    # In WSL or Linux:
    sim_vehicle.py -v ArduCopter --console --map
    # SITL then listens on udp:127.0.0.1:14550 by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from anywhere: add project root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.communication.mavlink_link import MAVLinkLink   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "endpoint",
        nargs="?",
        default="udpout:127.0.0.1:14550",
        help="MAVLink endpoint (default: udpout:127.0.0.1:14550)",
    )
    parser.add_argument(
        "--baud", type=int, default=57600,
        help="Serial baud rate (default: 57600); ignored for UDP/TCP",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0,
        help="Telemetry stream duration in seconds (default: 15)",
    )
    args = parser.parse_args()

    print(f"connecting to {args.endpoint} ...")
    link = MAVLinkLink()
    try:
        link.connect(args.endpoint, baud=args.baud)
        print("connected. waiting for heartbeat (10s)...")

        peer = link.wait_heartbeat(timeout=10.0)
        if peer is None:
            print("ERROR: no heartbeat received within 10 seconds.")
            print("  - Is SITL running?")
            print("  - For ArduPilot SITL the default is udpout:127.0.0.1:14550")
            print("  - For PX4 SITL try udpout:127.0.0.1:14540")
            return 2

        print(f"peer found: system={peer.system} component={peer.component} "
              f"type={peer.type} autopilot={peer.autopilot}")
        print()
        print(f"streaming telemetry for {args.duration:.0f}s...")
        print("-" * 60)

        deadline = time.monotonic() + args.duration
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(max(0.0, next_tick - time.monotonic()))
            next_tick += 1.0
            tel = link.get_telemetry()
            print(_format_telemetry(tel))

        print("-" * 60)
        print("done.")
        return 0

    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130
    finally:
        link.close()


def _format_telemetry(tel) -> str:
    parts = []
    if tel.mode:
        parts.append(f"mode={tel.mode}")
    if tel.armed is not None:
        parts.append("ARMED" if tel.armed else "disarmed")
    if tel.position:
        p = tel.position
        parts.append(
            f"pos=({p['lat']:.6f},{p['lon']:.6f}) "
            f"alt={p['alt_rel_m']:.1f}m"
        )
    if tel.attitude:
        a = tel.attitude
        parts.append(
            f"rpy=({a['roll']:+.2f},{a['pitch']:+.2f},{a['yaw']:+.2f})"
        )
    if tel.battery_pct is not None and tel.battery_pct >= 0:
        parts.append(f"bat={tel.battery_pct}%")
    if tel.heartbeat_age_s is not None:
        parts.append(f"hb_age={tel.heartbeat_age_s:.1f}s")
    return "  ".join(parts) if parts else "(no telemetry yet)"


if __name__ == "__main__":
    raise SystemExit(main())
