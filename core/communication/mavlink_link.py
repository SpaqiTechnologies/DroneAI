"""
MAVLink link to a flight controller — real hardware or SITL.

Thin wrapper around pymavlink. Keeps a background RX thread that caches the
latest message of each type, and exposes a small command surface (arm, mode,
takeoff). Same code works against:

  - ArduPilot SITL  : udpin:127.0.0.1:14550 (SITL listens, GCS connects out
                      via udpout — see scripts/sitl_check.py)
  - PX4 SITL        : udpin:127.0.0.1:14540
  - Real Pixhawk    : /dev/ttyUSB0 @ 57600 or 921600 (or telem radio)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

# ArduPilot's magic "force arm" value for COMPONENT_ARM_DISARM param2.
# See https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
_FORCE_ARM_MAGIC = 21196


@dataclass
class PeerInfo:
    system: int
    component: int
    type: int       # MAV_TYPE (e.g., MAV_TYPE_QUADROTOR = 2)
    autopilot: int  # MAV_AUTOPILOT (e.g., MAV_AUTOPILOT_ARDUPILOTMEGA = 3)


@dataclass
class Telemetry:
    attitude: Optional[Dict[str, float]] = None       # roll/pitch/yaw radians
    position: Optional[Dict[str, float]] = None       # lat/lon deg, alt meters
    velocity_ned: Optional[Dict[str, float]] = None   # vx/vy/vz m/s
    battery_pct: Optional[int] = None                 # 0..100 (-1 if unknown)
    armed: Optional[bool] = None
    mode: Optional[str] = None
    heartbeat_age_s: Optional[float] = None


class MAVLinkLink:
    """
    Bidirectional MAVLink session.

    Lifecycle:
        link = MAVLinkLink()
        link.connect("udpout:127.0.0.1:14550")
        peer = link.wait_heartbeat(timeout=10)   # blocks until peer seen
        link.set_mode("GUIDED")
        link.arm()
        link.takeoff(altitude_m=10)
        ...
        link.close()
    """

    def __init__(
        self,
        source_system: int = 255,        # 255 = ground control station
        source_component: int = mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
        heartbeat_period_s: float = 1.0,
    ) -> None:
        self.source_system = source_system
        self.source_component = source_component
        self._heartbeat_period_s = heartbeat_period_s

        self._conn: Optional[mavutil.mavfile] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}    # msg_type -> last message
        self._last_peer_hb: float = 0.0      # monotonic timestamp of last peer HB

    # ------------------------------------------------------------------ connect

    def connect(self, endpoint: str, baud: int = 57600) -> None:
        """
        Open the underlying transport. Does NOT block on heartbeat —
        call wait_heartbeat() for that.

        Endpoint examples:
            udpout:127.0.0.1:14550    -> connect to ArduPilot SITL (typical)
            udpin:0.0.0.0:14550       -> bind & wait for autopilot to find us
            tcp:127.0.0.1:5760        -> ArduPilot SITL TCP
            COM3                      -> Windows serial
            /dev/ttyUSB0              -> Linux serial
        """
        if self._conn is not None:
            raise RuntimeError("already connected; call close() first")

        self._conn = mavutil.mavlink_connection(
            endpoint,
            baud=baud,
            source_system=self.source_system,
            source_component=self.source_component,
            autoreconnect=False,
        )
        # Prime the socket: on Windows, a udpout UDP socket throws
        # WSAEINVAL (WinError 10022) from recvfrom if nothing has been
        # sent yet. Sending one heartbeat synchronously here keeps both
        # the main thread (wait_heartbeat) and the RX thread happy on
        # the first recv.
        try:
            self._conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
        except Exception:
            # Serial/TCP endpoints may not be writable yet; tolerate it.
            pass
        # Defensive ConnectionResetError handling lives in wait_heartbeat()
        # and the RX loop — those cover the Windows "ICMP port unreachable
        # turns into WSAECONNRESET on next recvfrom" case without needing
        # the SIO_UDP_CONNRESET ioctl (which CPython's socket.ioctl rejects).
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="mavlink-rx", daemon=True
        )
        self._rx_thread.start()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name="mavlink-hb", daemon=True
        )
        self._hb_thread.start()

    def close(self) -> None:
        self._running = False
        for t in (self._rx_thread, self._hb_thread):
            if t is not None:
                t.join(timeout=2.0)
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "MAVLinkLink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ handshake

    def wait_heartbeat(self, timeout: float = 10.0) -> Optional[PeerInfo]:
        """
        Block until we receive a HEARTBEAT from any peer or timeout.
        Returns peer identity on success, None on timeout.
        """
        if self._conn is None:
            raise RuntimeError("not connected")
        try:
            msg = self._conn.wait_heartbeat(timeout=timeout)
        except (ConnectionResetError, OSError):
            # Belt-and-braces: even with SIO_UDP_CONNRESET cleared, some
            # Windows configurations still surface socket errors here.
            # Treat as no-peer.
            return None
        if msg is None:
            return None
        with self._lock:
            self._latest["HEARTBEAT"] = msg
            self._last_peer_hb = time.monotonic()
        # pymavlink only records target_system from HEARTBEAT (not
        # target_component), so read the component from the message
        # itself rather than from the connection.
        return PeerInfo(
            system=msg.get_srcSystem(),
            component=msg.get_srcComponent(),
            type=msg.type,
            autopilot=msg.autopilot,
        )

    # ------------------------------------------------------------------ commands

    def arm(self, force: bool = False) -> None:
        self._require_peer()
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,                                              # confirmation
            1,                                              # 1=arm
            _FORCE_ARM_MAGIC if force else 0,
            0, 0, 0, 0, 0,
        )

    def disarm(self, force: bool = False) -> None:
        self._require_peer()
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,                                              # 0=disarm
            _FORCE_ARM_MAGIC if force else 0,
            0, 0, 0, 0, 0,
        )

    def set_mode(self, mode_name: str) -> None:
        """Set a named flight mode (e.g. 'GUIDED', 'LOITER', 'RTL')."""
        self._require_peer()
        mapping = self._conn.mode_mapping() or {}
        mode_id = mapping.get(mode_name.upper())
        if mode_id is None:
            raise ValueError(
                f"unknown mode {mode_name!r}; known modes: {sorted(mapping)}"
            )
        self._conn.mav.set_mode_send(
            self._conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

    def takeoff(self, altitude_m: float) -> None:
        """Request takeoff to given altitude (meters above home)."""
        self._require_peer()
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,    # pitch, empty, empty, yaw — unused for multirotor
            0, 0,          # lat, lon = 0 means current
            float(altitude_m),
        )

    def land(self) -> None:
        self._require_peer()
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    def send_heartbeat(self) -> None:
        """Send one HEARTBEAT now (the bg thread does this on a 1s cadence)."""
        if self._conn is None:
            raise RuntimeError("not connected")
        self._conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    # ------------------------------------------------------------------ telemetry

    def get_telemetry(self) -> Telemetry:
        with self._lock:
            att = self._latest.get("ATTITUDE")
            gpos = self._latest.get("GLOBAL_POSITION_INT")
            sys_status = self._latest.get("SYS_STATUS")
            hb = self._latest.get("HEARTBEAT")
            last_hb = self._last_peer_hb

        out = Telemetry()
        if att is not None:
            out.attitude = {"roll": att.roll, "pitch": att.pitch, "yaw": att.yaw}
        if gpos is not None:
            out.position = {
                "lat": gpos.lat / 1e7,
                "lon": gpos.lon / 1e7,
                "alt_msl_m": gpos.alt / 1000.0,
                "alt_rel_m": gpos.relative_alt / 1000.0,
            }
            out.velocity_ned = {
                "vx": gpos.vx / 100.0,
                "vy": gpos.vy / 100.0,
                "vz": gpos.vz / 100.0,
            }
        if sys_status is not None:
            # battery_remaining is -1 when the autopilot can't estimate
            out.battery_pct = sys_status.battery_remaining
        if hb is not None:
            out.armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            out.mode = self._mode_id_to_name(hb.custom_mode)
        if last_hb:
            out.heartbeat_age_s = time.monotonic() - last_hb
        return out

    def latest(self, msg_type: str) -> Optional[Any]:
        """Most recent message of the given type, or None."""
        with self._lock:
            return self._latest.get(msg_type)

    # ------------------------------------------------------------------ internals

    def _require_peer(self) -> None:
        if self._conn is None:
            raise RuntimeError("not connected")
        if self._conn.target_system == 0:
            raise RuntimeError(
                "no peer yet — call wait_heartbeat() before sending commands"
            )

    def _mode_id_to_name(self, mode_id: int) -> Optional[str]:
        if self._conn is None:
            return None
        mapping = self._conn.mode_mapping() or {}
        for name, mid in mapping.items():
            if mid == mode_id:
                return name
        return None

    def _rx_loop(self) -> None:
        # Cache every incoming message by type. The pymavlink connection's
        # blocking recv_match has its own internal mutex.
        while self._running and self._conn is not None:
            try:
                msg = self._conn.recv_match(blocking=True, timeout=0.5)
            except Exception:
                # Transient transport hiccups shouldn't kill the thread.
                time.sleep(0.05)
                continue
            if msg is None:
                continue
            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                continue
            with self._lock:
                self._latest[msg_type] = msg
                if msg_type == "HEARTBEAT":
                    self._last_peer_hb = time.monotonic()

    def _heartbeat_loop(self) -> None:
        while self._running and self._conn is not None:
            try:
                self._conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            except Exception:
                pass
            time.sleep(self._heartbeat_period_s)


# Convenience for ad-hoc use: `python -m core.communication.mavlink_link <endpoint>`
if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    endpoint = _sys.argv[1] if len(_sys.argv) > 1 else "udpout:127.0.0.1:14550"
    with MAVLinkLink() as _link:
        _link.connect(endpoint)
        print(f"connected to {endpoint}, waiting for heartbeat...")
        peer = _link.wait_heartbeat(timeout=10)
        if peer is None:
            print("no heartbeat in 10s — is SITL running?")
            raise SystemExit(1)
        print(f"peer: sys={peer.system} comp={peer.component} "
              f"type={peer.type} autopilot={peer.autopilot}")
        for _ in range(5):
            time.sleep(1)
            print(_link.get_telemetry())
