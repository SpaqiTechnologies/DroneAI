"""
Loopback tests for MAVLinkLink.

These tests don't need a real SITL — they spin up two MAVLinkLink instances
talking over UDP loopback and verify that command construction, heartbeat
exchange, and message caching all work end-to-end against the real pymavlink
parser.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from pymavlink import mavutil

from core.communication.mavlink_link import MAVLinkLink, PeerInfo, Telemetry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeAutopilot:
    """
    Minimal autopilot-side endpoint. Binds udpin and emits a heartbeat once
    per second so the GCS-side link can discover a peer. Records every
    inbound command_long so tests can assert on what was sent.
    """

    def __init__(self, port: int, sys_id: int = 1, comp_id: int = 1):
        self.port = port
        self.sys_id = sys_id
        self.comp_id = comp_id
        self.conn = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{port}",
            source_system=sys_id,
            source_component=comp_id,
        )
        self.commands: list = []     # captured COMMAND_LONG messages
        self.set_mode_msgs: list = []
        self._running = False
        self._thread = None

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
            if now - last_hb >= 0.2:    # 5 Hz — fast for tests
                # ArduPilot-flavored heartbeat: MAV_TYPE_QUADROTOR,
                # MAV_AUTOPILOT_ARDUPILOTMEGA, with GUIDED custom_mode (4).
                self.conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    base_mode=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    custom_mode=4,    # ArduCopter GUIDED
                    system_status=mavutil.mavlink.MAV_STATE_STANDBY,
                )
                last_hb = now
            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "COMMAND_LONG":
                self.commands.append(msg)
            elif t == "SET_MODE":
                self.set_mode_msgs.append(msg)

    def send_attitude(self, roll: float, pitch: float, yaw: float) -> None:
        self.conn.mav.attitude_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            roll, pitch, yaw, 0.0, 0.0, 0.0,
        )

    def send_global_position(self, lat_deg: float, lon_deg: float,
                              alt_msl_m: float, alt_rel_m: float) -> None:
        self.conn.mav.global_position_int_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            int(alt_msl_m * 1000),
            int(alt_rel_m * 1000),
            0, 0, 0,                            # vx, vy, vz cm/s
            0,                                  # hdg
        )


@pytest.fixture
def autopilot_and_link():
    port = _free_udp_port()
    autopilot = _FakeAutopilot(port)
    autopilot.start()
    link = MAVLinkLink(heartbeat_period_s=0.2)
    link.connect(f"udpout:127.0.0.1:{port}")
    peer = link.wait_heartbeat(timeout=5.0)
    assert peer is not None, "no heartbeat from fake autopilot within 5s"
    try:
        yield autopilot, link, peer
    finally:
        link.close()
        autopilot.stop()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_heartbeat_discovers_peer(autopilot_and_link):
    _autopilot, _link, peer = autopilot_and_link
    assert isinstance(peer, PeerInfo)
    assert peer.system == 1
    assert peer.component == 1
    assert peer.type == mavutil.mavlink.MAV_TYPE_QUADROTOR
    assert peer.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA


def test_arm_emits_correct_command(autopilot_and_link):
    autopilot, link, _peer = autopilot_and_link
    link.arm()
    # Allow the autopilot's RX loop to drain
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not autopilot.commands:
        time.sleep(0.05)
    assert autopilot.commands, "arm() didn't reach the autopilot"
    cmd = autopilot.commands[-1]
    assert cmd.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert cmd.param1 == 1.0     # arm
    assert cmd.param2 == 0.0     # not forced


def test_disarm_force_uses_magic_value(autopilot_and_link):
    autopilot, link, _peer = autopilot_and_link
    link.disarm(force=True)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not autopilot.commands:
        time.sleep(0.05)
    cmd = autopilot.commands[-1]
    assert cmd.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert cmd.param1 == 0.0     # disarm
    assert cmd.param2 == 21196.0  # ArduPilot force magic


def test_takeoff_includes_altitude(autopilot_and_link):
    autopilot, link, _peer = autopilot_and_link
    link.takeoff(altitude_m=12.5)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not autopilot.commands:
        time.sleep(0.05)
    cmd = autopilot.commands[-1]
    assert cmd.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert cmd.param7 == pytest.approx(12.5)


def test_set_mode_sends_setmode(autopilot_and_link):
    autopilot, link, _peer = autopilot_and_link
    link.set_mode("LOITER")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not autopilot.set_mode_msgs:
        time.sleep(0.05)
    assert autopilot.set_mode_msgs, "set_mode() didn't reach the autopilot"


def test_set_mode_unknown_raises(autopilot_and_link):
    _autopilot, link, _peer = autopilot_and_link
    with pytest.raises(ValueError, match="unknown mode"):
        link.set_mode("FANTASY_LAND")


def test_telemetry_aggregates_messages(autopilot_and_link):
    autopilot, link, _peer = autopilot_and_link
    autopilot.send_attitude(0.1, -0.2, 1.5)
    autopilot.send_global_position(47.5, -122.3, 105.0, 5.0)
    # Wait for RX thread to ingest
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        tel = link.get_telemetry()
        if tel.attitude is not None and tel.position is not None:
            break
        time.sleep(0.05)
    tel = link.get_telemetry()
    assert tel.attitude is not None
    assert tel.attitude["roll"] == pytest.approx(0.1, abs=1e-3)
    assert tel.attitude["yaw"] == pytest.approx(1.5, abs=1e-3)
    assert tel.position is not None
    assert tel.position["lat"] == pytest.approx(47.5, abs=1e-5)
    assert tel.position["alt_rel_m"] == pytest.approx(5.0, abs=1e-3)
    assert tel.mode == "GUIDED"     # from the autopilot's heartbeat
    # heartbeat_age_s should be small (we just received one)
    assert tel.heartbeat_age_s is not None
    assert tel.heartbeat_age_s < 1.0


def test_command_before_peer_raises():
    link = MAVLinkLink()
    port = _free_udp_port()
    link.connect(f"udpout:127.0.0.1:{port}")
    try:
        # No peer ever shows up, so target_system is still 0.
        with pytest.raises(RuntimeError, match="no peer"):
            link.arm()
    finally:
        link.close()


def test_command_without_connect_raises():
    link = MAVLinkLink()
    with pytest.raises(RuntimeError, match="not connected"):
        link.wait_heartbeat(timeout=0.1)


def test_double_connect_raises():
    link = MAVLinkLink()
    port = _free_udp_port()
    link.connect(f"udpout:127.0.0.1:{port}")
    try:
        with pytest.raises(RuntimeError, match="already connected"):
            link.connect(f"udpout:127.0.0.1:{port}")
    finally:
        link.close()


def test_context_manager_closes(autopilot_and_link):
    autopilot, _link, _peer = autopilot_and_link
    # Separate link to test the with-statement contract.
    port = _free_udp_port()
    autopilot2 = _FakeAutopilot(port)
    autopilot2.start()
    try:
        with MAVLinkLink(heartbeat_period_s=0.2) as link:
            link.connect(f"udpout:127.0.0.1:{port}")
            assert link.wait_heartbeat(timeout=5.0) is not None
        # After __exit__, the connection should be closed.
        assert link._conn is None
    finally:
        autopilot2.stop()
