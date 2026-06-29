"""
Tests for the Drone <-> MAVLink wiring.

Uses an in-process FakeMAVLinkBackend (no sockets, no threads) so we can
exercise the override / freshness / lifecycle logic in Drone without
spinning up a real autopilot. The actual MAVLink socket layer is covered
by tests/test_mavlink_link.py and tests/test_mavlink_backend.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core.drone import Drone
from core.communication.mavlink_link import PeerInfo, Telemetry


class FakeMAVLinkBackend:
    """Drop-in replacement for MAVLinkBackend with no I/O."""

    def __init__(self, telemetry: Telemetry):
        self._telemetry = telemetry
        self._connected = True
        self.closed = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    @property
    def heartbeat_age_s(self):
        return self._telemetry.heartbeat_age_s

    def close(self) -> None:
        self.closed = True
        self._connected = False


def _telemetry(*, lat=52.5, lon=13.4, alt_rel_m=25.0, alt_msl_m=100.0,
               battery_pct=66, mode="GUIDED", armed=True, heartbeat_age_s=0.5,
               roll=0.1, pitch=0.2, yaw=0.3, vx=1.0, vy=2.0, vz=-0.5) -> Telemetry:
    return Telemetry(
        attitude={"roll": roll, "pitch": pitch, "yaw": yaw},
        position={"lat": lat, "lon": lon, "alt_msl_m": alt_msl_m, "alt_rel_m": alt_rel_m},
        velocity_ned={"vx": vx, "vy": vy, "vz": vz},
        battery_pct=battery_pct,
        armed=armed,
        mode=mode,
        heartbeat_age_s=heartbeat_age_s,
    )


class TestMAVLinkFreshness(unittest.TestCase):
    def setUp(self):
        self.drone = Drone(enable_logging=False)

    def test_no_backend_not_connected(self):
        self.assertIsNone(self.drone.mavlink_backend)
        self.assertFalse(self.drone.is_mavlink_connected)
        self.assertFalse(self.drone._mavlink_telemetry_fresh())

    def test_fresh_heartbeat_is_fresh(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=0.5))
        self.assertTrue(self.drone.is_mavlink_connected)
        self.assertTrue(self.drone._mavlink_telemetry_fresh())

    def test_stale_heartbeat_is_not_fresh(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=30.0))
        self.assertTrue(self.drone.is_mavlink_connected)
        self.assertFalse(self.drone._mavlink_telemetry_fresh())

    def test_missing_heartbeat_age_is_not_fresh(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=None))
        self.assertFalse(self.drone._mavlink_telemetry_fresh())

    def test_freshness_threshold_is_configurable(self):
        self.drone._mavlink_heartbeat_max_age_s = 5.0
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=4.0))
        self.assertTrue(self.drone._mavlink_telemetry_fresh())
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=6.0))
        self.assertFalse(self.drone._mavlink_telemetry_fresh())


class TestMAVLinkApplyTelemetry(unittest.TestCase):
    def setUp(self):
        self.drone = Drone(enable_logging=False)

    def test_apply_overrides_position_altitude_battery(self):
        self.drone.current_position = (0.0, 0.0)
        self.drone.current_altitude = 0.0
        self.drone.battery_level = 100.0

        self.drone.mavlink_backend = FakeMAVLinkBackend(
            _telemetry(lat=52.52, lon=13.405, alt_rel_m=42.5, battery_pct=37)
        )
        self.drone._apply_mavlink_telemetry()

        self.assertEqual(self.drone.current_position, (52.52, 13.405))
        self.assertEqual(self.drone.current_altitude, 42.5)
        self.assertEqual(self.drone.battery_level, 37.0)

    def test_apply_uses_alt_rel_not_msl(self):
        """alt_rel_m (above home) is the right field — alt_msl_m would be
        thousands of meters off in many places."""
        self.drone.mavlink_backend = FakeMAVLinkBackend(
            _telemetry(alt_msl_m=540.0, alt_rel_m=10.0)
        )
        self.drone._apply_mavlink_telemetry()
        self.assertEqual(self.drone.current_altitude, 10.0)

    def test_apply_skips_unknown_battery(self):
        """battery_pct=-1 means the autopilot can't estimate; keep sim value."""
        self.drone.battery_level = 88.0
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(battery_pct=-1))
        self.drone._apply_mavlink_telemetry()
        self.assertEqual(self.drone.battery_level, 88.0)

    def test_apply_skips_missing_position(self):
        """No GLOBAL_POSITION_INT yet -> leave position alone."""
        self.drone.current_position = (1.0, 2.0)
        self.drone.current_altitude = 3.0
        tel = _telemetry()
        tel.position = None
        self.drone.mavlink_backend = FakeMAVLinkBackend(tel)
        self.drone._apply_mavlink_telemetry()
        self.assertEqual(self.drone.current_position, (1.0, 2.0))
        self.assertEqual(self.drone.current_altitude, 3.0)


class TestUpdateSensorsIntegration(unittest.TestCase):
    """End-to-end: update_sensors() honors the MAVLink override path."""

    def setUp(self):
        self.drone = Drone(enable_logging=False)

    def test_fresh_backend_drives_public_state(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(
            _telemetry(lat=10.0, lon=20.0, alt_rel_m=15.5, battery_pct=42,
                       heartbeat_age_s=0.5)
        )
        self.drone.update_sensors()
        self.assertEqual(self.drone.current_position, (10.0, 20.0))
        self.assertEqual(self.drone.current_altitude, 15.5)
        self.assertEqual(self.drone.battery_level, 42.0)

    def test_stale_backend_does_not_override(self):
        """Stale heartbeat -> simulated stack stays authoritative.
        We don't care about the exact sim values, only that they're not
        equal to the absurd autopilot values."""
        self.drone.mavlink_backend = FakeMAVLinkBackend(
            _telemetry(lat=999.0, lon=999.0, alt_rel_m=999.0, battery_pct=1,
                       heartbeat_age_s=30.0)
        )
        self.drone.update_sensors()
        self.assertNotEqual(self.drone.current_position, (999.0, 999.0))
        self.assertNotEqual(self.drone.current_altitude, 999.0)
        self.assertNotEqual(self.drone.battery_level, 1.0)


class TestMAVLinkConnectionLifecycle(unittest.TestCase):
    def setUp(self):
        self.drone = Drone(enable_logging=False)

    def test_connect_creates_backend_and_returns_peer(self):
        peer = PeerInfo(system=1, component=1, type=2, autopilot=3)

        with patch("core.drone.MAVLinkBackend") as mock_cls:
            inst = mock_cls.return_value
            inst.connect.return_value = peer
            inst.is_connected = True

            info = self.drone.connect_mavlink("udpout:127.0.0.1:14550")

        self.assertIs(self.drone.mavlink_backend, inst)
        inst.connect.assert_called_once_with(
            "udpout:127.0.0.1:14550", baud=57600, heartbeat_timeout=10.0
        )
        self.assertEqual(info["system"], 1)
        self.assertEqual(info["autopilot"], 3)
        self.assertEqual(info["endpoint"], "udpout:127.0.0.1:14550")

    def test_connect_when_already_connected_raises(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry())
        with self.assertRaises(RuntimeError):
            self.drone.connect_mavlink("udpout:127.0.0.1:14550")

    def test_disconnect_closes_and_clears(self):
        fake = FakeMAVLinkBackend(_telemetry())
        self.drone.mavlink_backend = fake
        self.drone.disconnect_mavlink()
        self.assertTrue(fake.closed)
        self.assertIsNone(self.drone.mavlink_backend)
        self.assertFalse(self.drone.is_mavlink_connected)

    def test_disconnect_when_not_connected_is_noop(self):
        self.drone.disconnect_mavlink()  # should not raise
        self.assertIsNone(self.drone.mavlink_backend)

    def test_disconnect_clears_even_if_close_raises(self):
        class ExplodingBackend(FakeMAVLinkBackend):
            def close(self):
                raise RuntimeError("close failed")

        self.drone.mavlink_backend = ExplodingBackend(_telemetry())
        with self.assertRaises(RuntimeError):
            self.drone.disconnect_mavlink()
        # Even on failure, the field is cleared so we don't get stuck
        self.assertIsNone(self.drone.mavlink_backend)


class TestMAVLinkStatus(unittest.TestCase):
    def setUp(self):
        self.drone = Drone(enable_logging=False)

    def test_status_when_disconnected(self):
        self.assertEqual(self.drone.get_mavlink_status(), {"connected": False})

    def test_status_when_connected(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(
            _telemetry(lat=1.0, lon=2.0, alt_rel_m=3.0, battery_pct=80,
                       armed=True, mode="LOITER", heartbeat_age_s=0.2)
        )
        status = self.drone.get_mavlink_status()
        self.assertTrue(status["connected"])
        self.assertTrue(status["fresh"])
        self.assertEqual(status["mode"], "LOITER")
        self.assertTrue(status["armed"])
        self.assertEqual(status["battery_pct"], 80)
        self.assertEqual(status["position"]["lat"], 1.0)
        self.assertEqual(status["attitude"]["roll"], 0.1)
        self.assertEqual(status["velocity_ned"]["vx"], 1.0)

    def test_status_stale_reports_not_fresh(self):
        self.drone.mavlink_backend = FakeMAVLinkBackend(_telemetry(heartbeat_age_s=30.0))
        status = self.drone.get_mavlink_status()
        self.assertTrue(status["connected"])
        self.assertFalse(status["fresh"])


if __name__ == "__main__":
    unittest.main()
