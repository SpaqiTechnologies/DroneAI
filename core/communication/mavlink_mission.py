"""
MAVLink mission upload / download / clear.

Runs the standard MAVLink mission protocol:

    GCS                              autopilot
     |--- MISSION_COUNT(n) ---------> |
     |<-- MISSION_REQUEST_INT(0) ---- |
     |--- MISSION_ITEM_INT(0) ------> |
     |<-- MISSION_REQUEST_INT(1) ---- |
     |--- MISSION_ITEM_INT(1) ------> |
     |               ...              |
     |<-- MISSION_ACK -------------- |

This module deals in MAVLink-flavored items (the MissionItem dataclass)
and knows nothing about the app's Mission/Waypoint types. The adapter
that converts core.mission.Mission to MissionItem lives in
mavlink_backend.py so this layer stays test-isolated.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pymavlink import mavutil

from core.communication.mavlink_link import MAVLinkLink


@dataclass
class MissionItem:
    """One MAVLink mission item (one row in MISSION_ITEM_INT)."""
    command: int                # MAV_CMD_NAV_* (16=waypoint, 22=takeoff, ...)
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0
    param1: float = 0.0         # hold time / pitch / etc — varies by command
    param2: float = 0.0         # acceptance radius for waypoint
    param3: float = 0.0         # pass-through for waypoint
    param4: float = float("nan")  # yaw (NaN = current)
    frame: int = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    autocontinue: int = 1


@dataclass
class MissionResult:
    success: bool
    item_count: int = 0
    ack_type: Optional[int] = None       # MAV_MISSION_RESULT enum value
    ack_name: Optional[str] = None
    error: str = ""
    items: List[MissionItem] = field(default_factory=list)


class MissionTransfer:
    """
    Runs upload / download / clear against an autopilot via MAVLinkLink.

    Each call is synchronous and self-contained: it snapshots which
    MISSION_* messages are cached in the link, then watches for new ones
    to drive the state machine. Subsequent calls don't share state.
    """

    DEFAULT_TIMEOUT = 30.0
    REQUEST_WAIT = 5.0           # max seconds between item requests
    ACK_WAIT = 5.0               # max seconds after final item for MISSION_ACK

    def __init__(self, link: MAVLinkLink,
                 mission_type: int = mavutil.mavlink.MAV_MISSION_TYPE_MISSION):
        self.link = link
        self.mission_type = mission_type

    # ------------------------------------------------------------------ upload

    def upload(self, items: List[MissionItem],
               timeout: float = DEFAULT_TIMEOUT) -> MissionResult:
        """Push the given mission to the autopilot, replacing what's there."""
        conn = self._require_peer()

        # Snapshot what was previously in the cache so we can tell a new
        # message from a stale one.
        prev_req = self.link.latest("MISSION_REQUEST_INT")
        prev_req_old = self.link.latest("MISSION_REQUEST")
        prev_ack = self.link.latest("MISSION_ACK")

        # Send MISSION_COUNT
        conn.mav.mission_count_send(
            conn.target_system,
            conn.target_component,
            len(items),
            self.mission_type,
        )

        # Empty mission case: autopilot ACKs immediately, no item requests
        if not items:
            ack = self._wait_for_ack(prev_ack, timeout=self.ACK_WAIT)
            return self._result_from_ack(ack, item_count=0)

        # Drive item requests until we've sent the last one
        deadline = time.monotonic() + timeout
        sent = set()
        last_req_seq = -1
        while time.monotonic() < deadline:
            # Did the autopilot ACK early (probably an error)?
            ack = self.link.latest("MISSION_ACK")
            if ack is not None and ack is not prev_ack:
                return self._result_from_ack(ack, item_count=len(sent))

            # Look for an item request
            req_seq = self._next_request_seq(prev_req, prev_req_old, last_req_seq)
            if req_seq is None:
                time.sleep(0.05)
                continue

            if req_seq < 0 or req_seq >= len(items):
                return MissionResult(
                    success=False, item_count=len(sent),
                    error=f"autopilot requested seq {req_seq} but we only have "
                          f"{len(items)} items",
                )
            self._send_item(items[req_seq], req_seq)
            sent.add(req_seq)
            last_req_seq = req_seq

            # After we've sent the last item, wait for the final ACK
            if req_seq == len(items) - 1:
                ack = self._wait_for_ack(prev_ack, timeout=self.ACK_WAIT)
                return self._result_from_ack(ack, item_count=len(items))

        return MissionResult(
            success=False, item_count=len(sent),
            error=f"upload timed out after {timeout:.0f}s "
                  f"(sent {len(sent)}/{len(items)} items)",
        )

    # ------------------------------------------------------------------ clear

    def clear(self, timeout: float = ACK_WAIT) -> MissionResult:
        """Clear all mission items on the autopilot."""
        conn = self._require_peer()
        prev_ack = self.link.latest("MISSION_ACK")
        conn.mav.mission_clear_all_send(
            conn.target_system,
            conn.target_component,
            self.mission_type,
        )
        ack = self._wait_for_ack(prev_ack, timeout=timeout)
        return self._result_from_ack(ack, item_count=0)

    # ------------------------------------------------------------------ download

    def download(self, timeout: float = DEFAULT_TIMEOUT) -> MissionResult:
        """Request the current mission from the autopilot."""
        conn = self._require_peer()
        prev_count = self.link.latest("MISSION_COUNT")
        prev_item = self.link.latest("MISSION_ITEM_INT")
        prev_ack = self.link.latest("MISSION_ACK")

        # Request the count
        conn.mav.mission_request_list_send(
            conn.target_system,
            conn.target_component,
            self.mission_type,
        )
        count_msg = self._wait_for_new("MISSION_COUNT", prev_count, timeout=5.0)
        if count_msg is None:
            return MissionResult(
                success=False, error="no MISSION_COUNT from autopilot",
            )
        count = count_msg.count

        if count == 0:
            # Some autopilots send an immediate MISSION_ACK; the success path
            # is just an empty mission.
            return MissionResult(success=True, item_count=0, items=[])

        # Request each item in sequence
        items: List[MissionItem] = []
        last_prev_item = prev_item
        for seq in range(count):
            conn.mav.mission_request_int_send(
                conn.target_system,
                conn.target_component,
                seq,
                self.mission_type,
            )
            msg = self._wait_for_new("MISSION_ITEM_INT", last_prev_item, timeout=self.REQUEST_WAIT)
            if msg is None:
                return MissionResult(
                    success=False, item_count=len(items),
                    error=f"no MISSION_ITEM_INT for seq {seq}",
                )
            last_prev_item = msg
            items.append(MissionItem(
                command=msg.command,
                lat_deg=msg.x / 1e7,
                lon_deg=msg.y / 1e7,
                alt_m=msg.z,
                param1=msg.param1,
                param2=msg.param2,
                param3=msg.param3,
                param4=msg.param4,
                frame=msg.frame,
                autocontinue=msg.autocontinue,
            ))

        # Acknowledge the download so the autopilot knows we're done
        conn.mav.mission_ack_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_MISSION_ACCEPTED,
            self.mission_type,
        )
        return MissionResult(success=True, item_count=count, items=items)

    # ------------------------------------------------------------------ set current

    def set_current(self, seq: int) -> None:
        """Tell the autopilot which mission seq to start from."""
        conn = self._require_peer()
        conn.mav.mission_set_current_send(
            conn.target_system,
            conn.target_component,
            seq,
        )

    # ------------------------------------------------------------------ internals

    def _require_peer(self):
        conn = self.link._conn
        if conn is None:
            raise RuntimeError("not connected")
        if conn.target_system == 0:
            raise RuntimeError("no peer — wait_heartbeat() first")
        return conn

    def _send_item(self, item: MissionItem, seq: int) -> None:
        conn = self.link._conn
        conn.mav.mission_item_int_send(
            conn.target_system,
            conn.target_component,
            seq,
            item.frame,
            item.command,
            1 if seq == 0 else 0,           # current
            item.autocontinue,
            float(item.param1),
            float(item.param2),
            float(item.param3),
            float(item.param4) if not math.isnan(item.param4) else float("nan"),
            int(item.lat_deg * 1e7),
            int(item.lon_deg * 1e7),
            float(item.alt_m),
            self.mission_type,
        )

    def _next_request_seq(self, prev_req_int, prev_req_old,
                           last_seen: int) -> Optional[int]:
        """Return seq of a new (post-snapshot) item request, or None."""
        cur = self.link.latest("MISSION_REQUEST_INT")
        if cur is not None and cur is not prev_req_int and cur.seq != last_seen:
            return cur.seq
        # Some autopilots fall back to MISSION_REQUEST (without _INT)
        cur_old = self.link.latest("MISSION_REQUEST")
        if cur_old is not None and cur_old is not prev_req_old and cur_old.seq != last_seen:
            return cur_old.seq
        return None

    def _wait_for_ack(self, prev_ack, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ack = self.link.latest("MISSION_ACK")
            if ack is not None and ack is not prev_ack:
                return ack
            time.sleep(0.05)
        return None

    def _wait_for_new(self, msg_type: str, prev, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cur = self.link.latest(msg_type)
            if cur is not None and cur is not prev:
                return cur
            time.sleep(0.05)
        return None

    def _result_from_ack(self, ack, item_count: int) -> MissionResult:
        if ack is None:
            return MissionResult(
                success=False, item_count=item_count,
                error="no MISSION_ACK from autopilot",
            )
        ack_type = ack.type
        ack_name = self._mission_result_name(ack_type)
        success = (ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED)
        return MissionResult(
            success=success, item_count=item_count,
            ack_type=ack_type, ack_name=ack_name,
            error="" if success else f"autopilot rejected mission: {ack_name}",
        )

    @staticmethod
    def _mission_result_name(code: int) -> str:
        enums = mavutil.mavlink.enums.get("MAV_MISSION_RESULT", {})
        if code in enums:
            return enums[code].name
        return f"UNKNOWN({code})"
