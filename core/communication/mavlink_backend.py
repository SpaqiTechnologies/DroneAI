"""
MAVLink-backed drone backend.

Sits on top of MAVLinkLink and provides:

  - Command methods that wait for COMMAND_ACK and report success/failure
    instead of fire-and-forget (so the caller knows whether the autopilot
    actually accepted arm/takeoff/etc.)
  - A goto_position() that uses SET_POSITION_TARGET_GLOBAL_INT, the
    standard "fly to GPS coord" message in GUIDED mode
  - A small state surface (position, attitude, battery, mode, armed)
    that downstream code can read without learning MAVLink

This is the building block for a GCS-style dashboard panel: the dashboard
points at a SITL or real flight controller via the backend, and reads/sends
through it.

For an end-to-end usage example, see scripts/sitl_flight_demo.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from pymavlink import mavutil

from core.communication.mavlink_link import MAVLinkLink, PeerInfo, Telemetry
from core.communication.mavlink_mission import (
    MissionItem,
    MissionResult,
    MissionTransfer,
)


# Position-only typemask for SET_POSITION_TARGET_GLOBAL_INT:
# ignore vx/vy/vz, ax/ay/az, yaw, yaw_rate. Bits 0-2 (position) stay
# cleared so the autopilot uses them.
_POS_TARGET_IGNORE_NONPOS = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


@dataclass
class CommandResult:
    accepted: bool
    result_code: int            # MAV_RESULT value
    result_name: str            # human-readable
    message: str = ""

    @classmethod
    def timeout(cls) -> "CommandResult":
        return cls(False, -1, "TIMEOUT", "no COMMAND_ACK within timeout")

    @classmethod
    def from_ack(cls, ack_msg) -> "CommandResult":
        code = ack_msg.result
        name = mavutil.mavlink.enums["MAV_RESULT"][code].name if code in mavutil.mavlink.enums["MAV_RESULT"] else f"UNKNOWN({code})"
        accepted = code == mavutil.mavlink.MAV_RESULT_ACCEPTED
        return cls(accepted, code, name)


class MAVLinkBackend:
    """
    Higher-level wrapper around MAVLinkLink with ack-waiting commands.

    Thread-safety: command methods are serialized by an internal lock so
    that ack-waiting from one command doesn't interleave with another's.
    Telemetry reads are independent and don't take the lock.
    """

    def __init__(self, link: Optional[MAVLinkLink] = None) -> None:
        self.link = link or MAVLinkLink()
        self._cmd_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def connect(self, endpoint: str, baud: int = 57600,
                heartbeat_timeout: float = 10.0) -> PeerInfo:
        """Connect and wait for the autopilot's first heartbeat."""
        self.link.connect(endpoint, baud=baud)
        peer = self.link.wait_heartbeat(timeout=heartbeat_timeout)
        if peer is None:
            self.link.close()
            raise TimeoutError(
                f"no MAVLink heartbeat from {endpoint} within "
                f"{heartbeat_timeout:.0f}s"
            )
        return peer

    def close(self) -> None:
        self.link.close()

    def __enter__(self) -> "MAVLinkBackend":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ state

    @property
    def is_connected(self) -> bool:
        return self.link._conn is not None

    @property
    def telemetry(self) -> Telemetry:
        return self.link.get_telemetry()

    @property
    def armed(self) -> Optional[bool]:
        return self.telemetry.armed

    @property
    def mode(self) -> Optional[str]:
        return self.telemetry.mode

    @property
    def position(self) -> Optional[dict]:
        return self.telemetry.position

    @property
    def heartbeat_age_s(self) -> Optional[float]:
        return self.telemetry.heartbeat_age_s

    # ------------------------------------------------------------------ commands

    def arm(self, force: bool = False, timeout: float = 5.0) -> CommandResult:
        return self._send_and_wait_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            send=lambda: self.link.arm(force=force),
            timeout=timeout,
        )

    def disarm(self, force: bool = False, timeout: float = 5.0) -> CommandResult:
        return self._send_and_wait_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            send=lambda: self.link.disarm(force=force),
            timeout=timeout,
        )

    def takeoff(self, altitude_m: float, timeout: float = 5.0) -> CommandResult:
        return self._send_and_wait_ack(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            send=lambda: self.link.takeoff(altitude_m=altitude_m),
            timeout=timeout,
        )

    def land(self, timeout: float = 5.0) -> CommandResult:
        return self._send_and_wait_ack(
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            send=lambda: self.link.land(),
            timeout=timeout,
        )

    def set_mode(self, mode_name: str) -> CommandResult:
        """
        Set flight mode. Mode changes don't ACK via COMMAND_ACK on
        ArduPilot — they show up as a HEARTBEAT custom_mode change.
        We send the command, then poll telemetry briefly to confirm.
        """
        target = mode_name.upper()
        self.link.set_mode(target)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.mode == target:
                return CommandResult(True, 0, "ACCEPTED",
                                     f"mode={target}")
            time.sleep(0.1)
        actual = self.mode
        return CommandResult(
            False, -1, "MODE_NOT_CONFIRMED",
            f"requested {target}, autopilot reports {actual!r} after 3s",
        )

    def goto_position(self, lat_deg: float, lon_deg: float,
                      alt_rel_m: float) -> None:
        """
        Fly to a GPS coordinate at given altitude above home (GUIDED mode).

        Fire-and-forget: SET_POSITION_TARGET_GLOBAL_INT is a streaming
        setpoint, not a command, so there's no COMMAND_ACK to wait on.
        Call wait_until_reached() to block on arrival.
        """
        conn = self.link._conn
        if conn is None:
            raise RuntimeError("not connected")
        conn.mav.set_position_target_global_int_send(
            0,                                       # time_boot_ms (autopilot ignores)
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            _POS_TARGET_IGNORE_NONPOS,
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            float(alt_rel_m),
            0, 0, 0,                                 # vx, vy, vz (ignored)
            0, 0, 0,                                 # ax, ay, az (ignored)
            0, 0,                                    # yaw, yaw_rate (ignored)
        )

    # ------------------------------------------------------------------ mission

    def upload_mission(self, mission_or_items,
                       takeoff_alt_m: Optional[float] = None,
                       append_rtl: Optional[bool] = None,
                       timeout: float = 30.0) -> MissionResult:
        """
        Upload a mission to the autopilot.

        Accepts either a list of MissionItem (raw MAVLink form) or a
        core.mission.Mission instance, which gets converted using the
        adapter below. ArduCopter expects seq=0 to be a TAKEOFF item;
        if takeoff_alt_m is given (or omitted, in which case we use the
        mission's default_altitude) we prepend one automatically.

        If append_rtl is True, append MAV_CMD_NAV_RETURN_TO_LAUNCH after
        the last waypoint. Defaults to mission.return_home for Mission
        objects, False otherwise.
        """
        if isinstance(mission_or_items, list):
            items = list(mission_or_items)
        else:
            items = mission_to_mavlink_items(
                mission_or_items,
                takeoff_alt_m=takeoff_alt_m,
                append_rtl=append_rtl,
            )
        return MissionTransfer(self.link).upload(items, timeout=timeout)

    def download_mission(self, timeout: float = 30.0) -> MissionResult:
        """Read the mission currently loaded on the autopilot."""
        return MissionTransfer(self.link).download(timeout=timeout)

    def clear_mission(self, timeout: float = 5.0) -> MissionResult:
        """Clear all mission items from the autopilot."""
        return MissionTransfer(self.link).clear(timeout=timeout)

    def start_mission(self) -> CommandResult:
        """
        Tell the autopilot to begin its loaded mission. Switches to AUTO
        and issues MAV_CMD_MISSION_START. Assumes the drone is already
        armed; the typical sequence is upload → arm → start_mission.
        """
        mode_result = self.set_mode("AUTO")
        if not mode_result.accepted:
            return mode_result
        return self._send_and_wait_ack(
            mavutil.mavlink.MAV_CMD_MISSION_START,
            send=lambda: self._send_mission_start(),
            timeout=5.0,
        )

    def _send_mission_start(self) -> None:
        conn = self.link._conn
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    # ------------------------------------------------------------------ navigation

    def wait_until_reached(self, lat_deg: float, lon_deg: float,
                           alt_rel_m: float,
                           radius_m: float = 2.0,
                           alt_tolerance_m: float = 1.0,
                           timeout: float = 60.0) -> bool:
        """
        Poll telemetry until the drone is within radius_m of the target
        (great-circle distance) AND within alt_tolerance_m of the target
        altitude. Returns True on arrival, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self.position
            if p is not None:
                dist = _haversine_m(p["lat"], p["lon"], lat_deg, lon_deg)
                alt_err = abs(p["alt_rel_m"] - alt_rel_m)
                if dist <= radius_m and alt_err <= alt_tolerance_m:
                    return True
            time.sleep(0.2)
        return False

    # ------------------------------------------------------------------ internals

    def _send_and_wait_ack(self, command: int, send, timeout: float) -> CommandResult:
        """
        Send a command and wait for the matching COMMAND_ACK.

        pymavlink delivers COMMAND_ACK through the RX loop into the
        message cache. We snapshot what's cached *before* sending, then
        poll for a new ACK with the right command id.
        """
        with self._cmd_lock:
            prev_ack = self.link.latest("COMMAND_ACK")
            send()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                ack = self.link.latest("COMMAND_ACK")
                if ack is not None and ack is not prev_ack and ack.command == command:
                    result = CommandResult.from_ack(ack)
                    # Some autopilots send an IN_PROGRESS first; keep waiting
                    # for the terminal result.
                    if result.result_code == mavutil.mavlink.MAV_RESULT_IN_PROGRESS:
                        prev_ack = ack
                        time.sleep(0.05)
                        continue
                    return result
                time.sleep(0.05)
            return CommandResult.timeout()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    import math
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def mission_to_mavlink_items(mission,
                              takeoff_alt_m: Optional[float] = None,
                              append_rtl: Optional[bool] = None) -> list:
    """
    Convert a core.mission.Mission into a list of MissionItem ready for
    upload. ArduCopter convention:

      seq=0    MAV_CMD_NAV_TAKEOFF (altitude = takeoff_alt_m or mission default)
      seq=1..N MAV_CMD_NAV_WAYPOINT for each Waypoint
      seq=N+1  MAV_CMD_NAV_RETURN_TO_LAUNCH (optional)
    """
    items: list = []

    # TAKEOFF item — ArduCopter requires seq=0 to be a takeoff
    alt = takeoff_alt_m if takeoff_alt_m is not None else mission.default_altitude
    items.append(MissionItem(
        command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        alt_m=float(alt),
    ))

    # Waypoints
    for wp in mission.waypoints:
        items.append(MissionItem(
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            lat_deg=float(wp.latitude),
            lon_deg=float(wp.longitude),
            alt_m=float(wp.altitude),
            param1=float(getattr(wp, "hold_time", 0.0) or 0.0),
            param2=float(getattr(wp, "acceptance_radius", 2.0) or 2.0),
        ))

    # Optional RTL
    if append_rtl is None:
        append_rtl = bool(getattr(mission, "return_home", False))
    if append_rtl:
        items.append(MissionItem(
            command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        ))

    return items
