"""
Loopback tests for MAVLinkBackend.

Spins up a fake autopilot that responds with COMMAND_ACK to incoming
COMMAND_LONGs (and broadcasts heartbeats with a real mode mapping), so
the backend's ack-waiting and mode-confirmation logic gets exercised
without a real SITL.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from pymavlink import mavutil

from core.communication.mavlink_backend import (
    CommandResult,
    MAVLinkBackend,
    _haversine_m,
)
from core.communication.mavlink_link import MAVLinkLink


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeArduCopter:
    """
    Autopilot that ACKs COMMAND_LONG. Honors SET_MODE by changing its
    own heartbeat custom_mode. Honors COMPONENT_ARM_DISARM by changing
    the SAFETY_ARMED base_mode bit. NAV_TAKEOFF is acknowledged but
    doesn't actually move the simulated position (tests should pass
    targets explicitly).
    """

    # ArduCopter custom mode IDs (subset)
    MODES = {
        "STABILIZE": 0,
        "ALT_HOLD": 2,
        "AUTO": 3,
        "GUIDED": 4,
        "LOITER": 5,
        "RTL": 6,
        "LAND": 9,
    }

    def __init__(self, port: int):
        self.port = port
        self.conn = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{port}",
            source_system=1,
            source_component=1,
        )
        self.commands: list = []
        self.set_mode_msgs: list = []
        self.armed: bool = False
        self.custom_mode: int = self.MODES["STABILIZE"]
        # Knob for tests that need to exercise failure paths
        self.next_ack_result = mavutil.mavlink.MAV_RESULT_ACCEPTED
        self.ack_command: bool = True   # set False to test ack timeout
        # Knob for SET_POSITION_TARGET_GLOBAL_INT capture
        self.position_targets: list = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        # Simulated position
        self.sim_lat = 47.5
        self.sim_lon = -122.3
        self.sim_alt_rel = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.conn.close()

    def set_simulated_position(self, lat: float, lon: float, alt_rel: float) -> None:
        with self._lock:
            self.sim_lat = lat
            self.sim_lon = lon
            self.sim_alt_rel = alt_rel

    def _loop(self):
        last_hb = 0.0
        last_pos = 0.0
        while self._running:
            now = time.monotonic()
            # heartbeat
            if now - last_hb >= 0.2:
                with self._lock:
                    base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                    if self.armed:
                        base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    cm = self.custom_mode
                self.conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    base_mode=base_mode,
                    custom_mode=cm,
                    system_status=mavutil.mavlink.MAV_STATE_STANDBY,
                )
                last_hb = now
            # global position broadcast (10 Hz)
            if now - last_pos >= 0.1:
                with self._lock:
                    lat, lon, alt = self.sim_lat, self.sim_lon, self.sim_alt_rel
                self.conn.mav.global_position_int_send(
                    int(now * 1000) & 0xFFFFFFFF,
                    int(lat * 1e7),
                    int(lon * 1e7),
                    int(alt * 1000) + 50_000,   # MSL offset
                    int(alt * 1000),
                    0, 0, 0, 0,
                )
                last_pos = now
            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "COMMAND_LONG":
                self.commands.append(msg)
                self._handle_command(msg)
            elif t == "SET_MODE":
                self.set_mode_msgs.append(msg)
                # Update our heartbeat custom_mode to match
                with self._lock:
                    self.custom_mode = msg.custom_mode
            elif t == "SET_POSITION_TARGET_GLOBAL_INT":
                self.position_targets.append(msg)

    def _handle_command(self, msg):
        # Update simulated state for arm/disarm so heartbeat reflects it
        if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            with self._lock:
                self.armed = (msg.param1 == 1.0)
        # Send COMMAND_ACK
        if self.ack_command:
            self.conn.mav.command_ack_send(
                msg.command,
                self.next_ack_result,
            )


@pytest.fixture
def backend_pair():
    port = _free_udp_port()
    ap = _FakeArduCopter(port)
    ap.start()
    backend = MAVLinkBackend(MAVLinkLink(heartbeat_period_s=0.2))
    peer = backend.connect(f"udpout:127.0.0.1:{port}", heartbeat_timeout=5.0)
    assert peer is not None
    # Give telemetry a moment to populate
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and backend.mode is None:
        time.sleep(0.05)
    try:
        yield ap, backend
    finally:
        backend.close()
        ap.stop()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_connect_returns_peer(backend_pair):
    _ap, backend = backend_pair
    assert backend.is_connected
    assert backend.mode == "STABILIZE"


def test_connect_timeout_raises():
    port = _free_udp_port()
    backend = MAVLinkBackend()
    with pytest.raises(TimeoutError, match="no MAVLink heartbeat"):
        backend.connect(f"udpout:127.0.0.1:{port}", heartbeat_timeout=0.5)


def test_arm_succeeds(backend_pair):
    _ap, backend = backend_pair
    result = backend.arm()
    assert result.accepted, f"got {result}"
    assert result.result_code == 0
    assert result.result_name == "MAV_RESULT_ACCEPTED"
    # And the autopilot's armed flag flipped
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and backend.armed is not True:
        time.sleep(0.05)
    assert backend.armed is True


def test_arm_denied(backend_pair):
    ap, backend = backend_pair
    ap.next_ack_result = mavutil.mavlink.MAV_RESULT_DENIED
    result = backend.arm()
    assert not result.accepted
    assert result.result_code == mavutil.mavlink.MAV_RESULT_DENIED
    assert result.result_name == "MAV_RESULT_DENIED"


def test_arm_timeout_when_autopilot_silent(backend_pair):
    ap, backend = backend_pair
    ap.ack_command = False
    result = backend.arm(timeout=0.5)
    assert not result.accepted
    assert result.result_name == "TIMEOUT"


def test_takeoff_acks(backend_pair):
    ap, backend = backend_pair
    result = backend.takeoff(altitude_m=10.0)
    assert result.accepted, f"got {result}"
    cmd = ap.commands[-1]
    assert cmd.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert cmd.param7 == pytest.approx(10.0)


def test_set_mode_confirmed_via_heartbeat(backend_pair):
    _ap, backend = backend_pair
    result = backend.set_mode("GUIDED")
    assert result.accepted, f"got {result}"
    assert backend.mode == "GUIDED"


def test_set_mode_unknown_raises(backend_pair):
    _ap, backend = backend_pair
    with pytest.raises(ValueError):
        backend.set_mode("HOVER_LIKE_A_HUMMINGBIRD")


def test_goto_position_sends_target(backend_pair):
    ap, backend = backend_pair
    backend.goto_position(lat_deg=47.51, lon_deg=-122.31, alt_rel_m=15.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not ap.position_targets:
        time.sleep(0.05)
    assert ap.position_targets, "no SET_POSITION_TARGET_GLOBAL_INT received"
    msg = ap.position_targets[-1]
    assert msg.lat_int == int(47.51 * 1e7)
    assert msg.lon_int == int(-122.31 * 1e7)
    assert msg.alt == pytest.approx(15.0)
    assert msg.coordinate_frame == mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT


def test_wait_until_reached_succeeds(backend_pair):
    ap, backend = backend_pair
    ap.set_simulated_position(47.5, -122.3, 0.0)
    # Move the simulated autopilot to the target after a short delay
    def move_after():
        time.sleep(0.3)
        ap.set_simulated_position(47.5001, -122.2999, 10.0)
    threading.Thread(target=move_after, daemon=True).start()
    arrived = backend.wait_until_reached(
        47.5001, -122.2999, 10.0,
        radius_m=5.0, alt_tolerance_m=1.0, timeout=3.0,
    )
    assert arrived


def test_wait_until_reached_times_out(backend_pair):
    ap, backend = backend_pair
    ap.set_simulated_position(47.5, -122.3, 0.0)
    arrived = backend.wait_until_reached(
        48.0, -121.0, 100.0,   # far away
        radius_m=2.0, alt_tolerance_m=1.0, timeout=0.5,
    )
    assert not arrived


def test_haversine():
    # Known reference: 1 degree of latitude is ~111 km
    d = _haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_000 < d < 112_000
    # Same point → zero
    assert _haversine_m(47.5, -122.3, 47.5, -122.3) == pytest.approx(0.0, abs=0.001)


def test_disarm_succeeds(backend_pair):
    _ap, backend = backend_pair
    # Arm first so the disarm transition is observable
    arm_result = backend.arm()
    assert arm_result.accepted
    disarm_result = backend.disarm()
    assert disarm_result.accepted
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and backend.armed is not False:
        time.sleep(0.05)
    assert backend.armed is False
