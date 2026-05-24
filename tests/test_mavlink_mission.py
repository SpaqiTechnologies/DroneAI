"""
Loopback tests for the MAVLink mission protocol.

Uses a fake autopilot that drives the request/item exchange and records
what it receives, so we can assert on the full upload state machine
without needing a real SITL.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from pymavlink import mavutil

from core.communication.mavlink_backend import (
    MAVLinkBackend,
    mission_to_mavlink_items,
)
from core.communication.mavlink_link import MAVLinkLink
from core.communication.mavlink_mission import (
    MissionItem,
    MissionResult,
    MissionTransfer,
)


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeMissionAutopilot:
    """
    Fake autopilot that implements the MAVLink mission protocol enough
    to upload/download/clear.

    Knobs:
      ack_result            -- what MAV_MISSION_RESULT to send in MISSION_ACK
      drop_first_request    -- if True, skip the first MISSION_REQUEST_INT
                                (simulates a glitchy link / re-request)
      reply_with_old_request -- if True, send MISSION_REQUEST (no _INT)
      stored_mission        -- list of MissionItem the autopilot "has"
                                (used for download tests)
    """

    def __init__(self, port: int):
        self.port = port
        self.conn = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{port}",
            source_system=1,
            source_component=1,
        )
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Knobs
        self.ack_result = mavutil.mavlink.MAV_MISSION_ACCEPTED
        self.drop_first_request = False
        self.reply_with_old_request = False
        self.silent_after_count = False        # send no requests at all

        # Captured state
        self.received_items: list = []          # MissionItem-shaped
        self.received_clears: int = 0
        self.received_count: int = 0           # MISSION_COUNT n
        self.received_request_lists: int = 0    # MISSION_REQUEST_LIST count
        self.received_acks: list = []           # acks from GCS (download flow)

        # For download tests
        self.stored_mission: list = []

        # Upload-side state machine
        self._expected_count: int = 0
        self._next_request: int = 0
        self._request_dropped_once: bool = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.conn.close()

    def _loop(self):
        last_hb = 0.0
        while self._running:
            now = time.monotonic()
            if now - last_hb >= 0.2:
                self.conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    base_mode=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    custom_mode=0,
                    system_status=mavutil.mavlink.MAV_STATE_STANDBY,
                )
                last_hb = now

            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "MISSION_COUNT":
                self._handle_count(msg)
            elif t == "MISSION_ITEM_INT":
                self._handle_item(msg)
            elif t == "MISSION_CLEAR_ALL":
                self._handle_clear()
            elif t == "MISSION_REQUEST_LIST":
                self._handle_request_list()
            elif t == "MISSION_REQUEST_INT":
                self._handle_download_request(msg.seq)
            elif t == "MISSION_ACK":
                self.received_acks.append(msg)

    # upload side -----------------------------------------------------------

    def _handle_count(self, msg):
        with self._lock:
            self._expected_count = msg.count
            self._next_request = 0
            self._request_dropped_once = False
            self.received_count = msg.count
            self.received_items = []
        if msg.count == 0:
            self._send_ack()
            return
        if self.silent_after_count:
            return
        self._request_next()

    def _request_next(self):
        with self._lock:
            seq = self._next_request
            if self.drop_first_request and not self._request_dropped_once:
                self._request_dropped_once = True
                return        # client must time out and re-send count, or we eventually re-request
        if self.reply_with_old_request:
            self.conn.mav.mission_request_send(
                msg_target_system := 255,    # any GCS
                msg_target_component := 0,
                seq,
            )
        else:
            self.conn.mav.mission_request_int_send(
                255, 0, seq,
            )

    def _handle_item(self, msg):
        with self._lock:
            self.received_items.append(MissionItem(
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
            self._next_request = msg.seq + 1
            done = self._next_request >= self._expected_count
        if done:
            self._send_ack()
        else:
            self._request_next()

    def _send_ack(self):
        self.conn.mav.mission_ack_send(
            255, 0,
            self.ack_result,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

    def _handle_clear(self):
        self.received_clears += 1
        self._send_ack()

    # download side ---------------------------------------------------------

    def _handle_request_list(self):
        self.received_request_lists += 1
        self.conn.mav.mission_count_send(
            255, 0,
            len(self.stored_mission),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

    def _handle_download_request(self, seq: int):
        if seq < 0 or seq >= len(self.stored_mission):
            return
        item = self.stored_mission[seq]
        self.conn.mav.mission_item_int_send(
            255, 0, seq,
            item.frame, item.command,
            1 if seq == 0 else 0,
            item.autocontinue,
            float(item.param1), float(item.param2),
            float(item.param3), float(item.param4),
            int(item.lat_deg * 1e7), int(item.lon_deg * 1e7),
            float(item.alt_m),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transfer_pair():
    port = _free_udp_port()
    ap = _FakeMissionAutopilot(port)
    ap.start()
    link = MAVLinkLink(heartbeat_period_s=0.2)
    link.connect(f"udpout:127.0.0.1:{port}")
    assert link.wait_heartbeat(timeout=5.0) is not None
    transfer = MissionTransfer(link)
    try:
        yield ap, link, transfer
    finally:
        link.close()
        ap.stop()


@pytest.fixture
def backend_pair():
    port = _free_udp_port()
    ap = _FakeMissionAutopilot(port)
    ap.start()
    backend = MAVLinkBackend(MAVLinkLink(heartbeat_period_s=0.2))
    backend.connect(f"udpout:127.0.0.1:{port}", heartbeat_timeout=5.0)
    try:
        yield ap, backend
    finally:
        backend.close()
        ap.stop()


# ---------------------------------------------------------------------------
# upload tests
# ---------------------------------------------------------------------------

def test_upload_empty_mission_acks_immediately(transfer_pair):
    ap, _link, transfer = transfer_pair
    result = transfer.upload([])
    assert result.success
    assert result.item_count == 0
    assert ap.received_count == 0
    assert ap.received_items == []


def test_upload_three_waypoints(transfer_pair):
    ap, _link, transfer = transfer_pair
    items = [
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=10.0),
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    lat_deg=47.5, lon_deg=-122.3, alt_m=10.0,
                    param2=2.0),
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH),
    ]
    result = transfer.upload(items, timeout=10.0)
    assert result.success, f"got {result}"
    assert result.item_count == 3
    assert ap.received_count == 3
    assert len(ap.received_items) == 3

    # Item 0 should be TAKEOFF with alt=10
    got = ap.received_items[0]
    assert got.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert got.alt_m == pytest.approx(10.0)

    # Item 1 should be the waypoint
    got = ap.received_items[1]
    assert got.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
    assert got.lat_deg == pytest.approx(47.5, abs=1e-5)
    assert got.lon_deg == pytest.approx(-122.3, abs=1e-5)
    assert got.alt_m == pytest.approx(10.0)
    assert got.param2 == pytest.approx(2.0)

    # Item 2 should be RTL
    assert ap.received_items[2].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH


def test_upload_rejected_by_autopilot(transfer_pair):
    ap, _link, transfer = transfer_pair
    ap.ack_result = mavutil.mavlink.MAV_MISSION_ERROR
    items = [MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=5.0)]
    result = transfer.upload(items, timeout=5.0)
    assert not result.success
    assert result.ack_type == mavutil.mavlink.MAV_MISSION_ERROR
    assert "MAV_MISSION_ERROR" in (result.ack_name or "")


def test_upload_times_out_when_autopilot_silent(transfer_pair):
    ap, _link, transfer = transfer_pair
    ap.silent_after_count = True
    items = [MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=5.0)]
    result = transfer.upload(items, timeout=0.6)
    assert not result.success
    assert "timed out" in result.error


def test_upload_handles_legacy_mission_request(transfer_pair):
    ap, _link, transfer = transfer_pair
    ap.reply_with_old_request = True
    items = [MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=5.0)]
    result = transfer.upload(items, timeout=5.0)
    assert result.success, f"got {result}"
    assert ap.received_count == 1


# ---------------------------------------------------------------------------
# clear / download
# ---------------------------------------------------------------------------

def test_clear_sends_clear_all(transfer_pair):
    ap, _link, transfer = transfer_pair
    result = transfer.clear(timeout=5.0)
    assert result.success
    assert ap.received_clears == 1


def test_download_three_waypoints(transfer_pair):
    ap, _link, transfer = transfer_pair
    ap.stored_mission = [
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=15.0),
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    lat_deg=47.61, lon_deg=-122.33, alt_m=20.0,
                    param2=3.0),
        MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH),
    ]
    result = transfer.download(timeout=10.0)
    assert result.success, f"got {result}"
    assert result.item_count == 3
    assert len(result.items) == 3
    assert result.items[0].command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert result.items[0].alt_m == pytest.approx(15.0)
    assert result.items[1].lat_deg == pytest.approx(47.61, abs=1e-5)
    assert result.items[2].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH


def test_download_empty(transfer_pair):
    ap, _link, transfer = transfer_pair
    ap.stored_mission = []
    result = transfer.download(timeout=5.0)
    assert result.success
    assert result.item_count == 0
    assert result.items == []


# ---------------------------------------------------------------------------
# Mission -> MissionItem adapter
# ---------------------------------------------------------------------------

def test_mission_adapter_prepends_takeoff_and_appends_rtl():
    """Test the adapter without spinning up MAVLink at all."""
    from core.mission import Mission, Waypoint, WaypointType

    m = Mission(name="x", default_altitude=12.0, return_home=True)
    m.add_waypoint(Waypoint(latitude=47.5, longitude=-122.3,
                             altitude=20.0, acceptance_radius=3.0))
    m.add_waypoint(Waypoint(latitude=47.51, longitude=-122.31, altitude=25.0))

    items = mission_to_mavlink_items(m)

    assert len(items) == 4    # TAKEOFF + 2 waypoints + RTL
    assert items[0].command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert items[0].alt_m == pytest.approx(12.0)
    assert items[1].command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
    assert items[1].lat_deg == pytest.approx(47.5)
    assert items[1].param2 == pytest.approx(3.0)  # acceptance radius
    assert items[2].lat_deg == pytest.approx(47.51)
    assert items[3].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH


def test_mission_adapter_no_rtl_by_default():
    from core.mission import Mission, Waypoint

    m = Mission(name="x", default_altitude=10.0)
    m.add_waypoint(Waypoint(latitude=47.5, longitude=-122.3, altitude=10.0))
    items = mission_to_mavlink_items(m)
    assert len(items) == 2     # TAKEOFF + waypoint
    assert items[-1].command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT


def test_mission_adapter_override_takeoff_alt():
    from core.mission import Mission

    m = Mission(name="x", default_altitude=10.0)
    items = mission_to_mavlink_items(m, takeoff_alt_m=25.0)
    assert items[0].command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert items[0].alt_m == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# backend integration
# ---------------------------------------------------------------------------

def test_backend_upload_mission(backend_pair):
    ap, backend = backend_pair
    from core.mission import Mission, Waypoint

    m = Mission(name="t", default_altitude=15.0, return_home=True)
    m.add_waypoint(Waypoint(latitude=47.5, longitude=-122.3, altitude=15.0))
    m.add_waypoint(Waypoint(latitude=47.51, longitude=-122.31, altitude=20.0))

    result = backend.upload_mission(m, timeout=10.0)
    assert result.success, f"got {result}"
    assert result.item_count == 4    # takeoff + 2 wp + rtl
    # Autopilot should have received them in order
    cmds = [i.command for i in ap.received_items]
    assert cmds == [
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
    ]


def test_backend_clear_mission(backend_pair):
    ap, backend = backend_pair
    result = backend.clear_mission()
    assert result.success
    assert ap.received_clears == 1


def test_backend_upload_raw_items(backend_pair):
    ap, backend = backend_pair
    items = [MissionItem(command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, alt_m=10.0)]
    result = backend.upload_mission(items, timeout=5.0)
    assert result.success
    assert len(ap.received_items) == 1
