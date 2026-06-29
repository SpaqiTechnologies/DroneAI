"""Tests for the three "AI in the loop" features:

  1. SAR mission investigates a high-confidence detection (state machine).
  2. Anomaly detector → failsafe manager bridge.
  3. LLM CommandInterpreter → real ``Mission`` materialization.
"""

import math
import tempfile

import pytest

from applications.search_rescue import (
    SARMission,
    SARState,
    SearchConfig,
    SearchType,
)
from core.drone import Drone
from sensors.media import MediaStorage
from ai.anomaly.anomaly_detector import AnomalyDetector, AnomalySeverity, AnomalyType
from core.failsafe import FailsafeManager, FailsafeType
from core.anomaly_failsafe import AnomalyFailsafeBridge
from ai.llm.command_interpreter import (
    CommandInterpreter,
    CommandType,
    DroneCommand,
    WaypointDef,
)


# ============================ SAR investigation ============================


def _drone_with_storage(tmp_path) -> Drone:
    d = Drone()
    d.current_position = (47.5, -122.3)
    d.current_altitude = 30.0
    d.camera_sensor._media_storage = MediaStorage(root=str(tmp_path))
    d.camera_sensor._synth_resolution = (48, 27)
    d.camera_sensor.start()
    return d


def test_sar_enters_investigation_on_high_confidence_target(tmp_path) -> None:
    drone = _drone_with_storage(tmp_path)
    drone.camera_sensor.simulate_obstacle(
        x=300, y=200, width=80, height=80, distance=8.0, confidence=0.92,
    )
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=30.0, track_spacing=60.0, speed=20.0, legs=4),
        cruise_speed_mps=80.0,
        photo_interval_s=10.0,
        dedupe_radius_m=200.0,
        investigate_targets=True,
        investigation_confidence=0.8,
        investigation_altitude_m=10.0,
        investigation_duration_s=0.5,
        investigation_photos=3,
        investigation_descent_mps=100.0,
        return_to_start=False,
    )
    states_seen: set[str] = set()
    sar.start()
    for _ in range(2000):
        states_seen.add(sar.state.value)
        sar.tick(0.05)
        if sar.is_done:
            break
    assert "investigating" in states_seen
    report = sar.report()
    assert report.investigations_completed >= 1
    assert report.targets
    t = report.targets[0]
    assert t.metadata.get("investigated") is True
    assert t.metadata.get("investigation_photo_count") == 3


def test_sar_skips_investigation_when_disabled(tmp_path) -> None:
    drone = _drone_with_storage(tmp_path)
    drone.camera_sensor.simulate_obstacle(
        x=200, y=150, width=60, height=60, distance=8.0, confidence=0.95,
    )
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=30.0, track_spacing=60.0, speed=20.0, legs=3),
        cruise_speed_mps=80.0,
        photo_interval_s=10.0,
        dedupe_radius_m=200.0,
        investigate_targets=False,
        return_to_start=False,
    )
    sar.start()
    for _ in range(2000):
        sar.tick(0.05)
        if sar.is_done:
            break
    rep = sar.report()
    assert rep.investigations_completed == 0
    assert rep.targets  # detection was logged
    assert rep.targets[0].metadata.get("investigated") is not True


def test_sar_skips_investigation_below_confidence_threshold(tmp_path) -> None:
    drone = _drone_with_storage(tmp_path)
    drone.camera_sensor.simulate_obstacle(
        x=200, y=150, width=60, height=60, distance=8.0, confidence=0.7,
    )
    sar = SARMission(
        drone=drone,
        pattern_type=SearchType.EXPANDING_SQUARE,
        center=(47.5, -122.3),
        config=SearchConfig(altitude=30.0, track_spacing=60.0, speed=20.0, legs=3),
        cruise_speed_mps=80.0,
        photo_interval_s=10.0,
        dedupe_radius_m=200.0,
        investigate_targets=True,
        investigation_confidence=0.85,  # 0.7 detection is below
        return_to_start=False,
    )
    sar.start()
    for _ in range(2000):
        sar.tick(0.05)
        if sar.is_done:
            break
    assert sar.report().investigations_completed == 0


# ============================ Anomaly → Failsafe ===========================


def _prime_baseline(detector: AnomalyDetector, sensor: str, value: float, n: int = 40) -> None:
    for _ in range(n):
        detector.update(sensor, value, component="motor_1")


def test_anomaly_detector_invokes_callbacks() -> None:
    detector = AnomalyDetector()
    fired: list = []
    detector.add_callback(lambda a: fired.append(a))
    _prime_baseline(detector, "motor_vibration", 0.5)
    detector.update("motor_vibration", 50.0, component="motor_1")
    assert len(fired) >= 1


def test_anomaly_detector_remove_callback() -> None:
    detector = AnomalyDetector()
    fired: list = []
    cb = lambda a: fired.append(a)
    detector.add_callback(cb)
    detector.remove_callback(cb)
    _prime_baseline(detector, "motor_vibration", 0.5)
    detector.update("motor_vibration", 50.0, component="motor_1")
    assert fired == []


def test_anomaly_failsafe_bridge_forwards_critical_to_failsafe() -> None:
    detector = AnomalyDetector()
    failsafe = FailsafeManager()
    bridge = AnomalyFailsafeBridge(detector, failsafe, min_severity=AnomalySeverity.CRITICAL)
    bridge.attach()
    _prime_baseline(detector, "motor_vibration", 0.5)
    detector.update("motor_vibration", 50.0, component="motor_1")
    assert len(bridge.history) == 1
    trig = bridge.history[0]
    assert trig.failsafe_type == "MOTOR_FAILURE"
    assert trig.failsafe_action == "land_immediately"
    assert FailsafeType.MOTOR_FAILURE in failsafe._active_failsafes


def test_anomaly_failsafe_bridge_ignores_warnings() -> None:
    detector = AnomalyDetector()
    failsafe = FailsafeManager()
    bridge = AnomalyFailsafeBridge(detector, failsafe, min_severity=AnomalySeverity.CRITICAL)
    bridge.attach()
    _prime_baseline(detector, "motor_vibration", 0.5)
    # Small spike → WARNING (2σ < x < 3.5σ ideally, but z-score depends on baseline)
    detector.update("motor_vibration", 1.5, component="motor_1")
    # Warnings should not trigger failsafe
    assert all(t.severity != "warning" for t in bridge.history)


def test_anomaly_failsafe_bridge_detach() -> None:
    detector = AnomalyDetector()
    failsafe = FailsafeManager()
    bridge = AnomalyFailsafeBridge(detector, failsafe)
    bridge.attach()
    bridge.detach()
    _prime_baseline(detector, "motor_vibration", 0.5)
    detector.update("motor_vibration", 50.0, component="motor_1")
    assert bridge.history == []


def test_anomaly_failsafe_bridge_custom_mapping() -> None:
    detector = AnomalyDetector()
    failsafe = FailsafeManager()
    bridge = AnomalyFailsafeBridge(
        detector, failsafe,
        custom_mapping={AnomalyType.MOTOR_VIBRATION: FailsafeType.SENSOR_FAILURE},
    )
    bridge.attach()
    _prime_baseline(detector, "motor_vibration", 0.5)
    detector.update("motor_vibration", 50.0, component="motor_1")
    assert len(bridge.history) == 1
    assert bridge.history[0].failsafe_type == "SENSOR_FAILURE"


def test_drone_has_anomaly_bridge_wired() -> None:
    drone = Drone()
    assert drone.anomaly_failsafe_bridge._attached
    _prime_baseline(drone.anomaly_detector, "motor_vibration", 0.5)
    drone.anomaly_detector.update("motor_vibration", 50.0, component="motor_1")
    assert len(drone.anomaly_failsafe_bridge.history) == 1


# ============================ LLM → Mission ==============================


def test_drone_command_to_mission_for_goto_uses_waypoints() -> None:
    cmd = DroneCommand(
        command_type=CommandType.GOTO,
        waypoints=[WaypointDef(latitude=47.5, longitude=-122.3, altitude=20.0)],
        confidence=0.9,
        raw_text="fly to A",
    )
    m = cmd.to_mission()
    assert m is not None
    assert m.mission_type.value == "waypoint"
    assert len(m.waypoints) == 1
    assert m.waypoints[0].latitude == 47.5
    assert m.metadata.get("source") == "llm"


def test_drone_command_to_mission_for_mission_with_multiple_waypoints() -> None:
    cmd = DroneCommand(
        command_type=CommandType.MISSION,
        waypoints=[
            WaypointDef(latitude=47.50, longitude=-122.30, altitude=25.0, hold_time=2.0),
            WaypointDef(latitude=47.51, longitude=-122.31, altitude=30.0),
            WaypointDef(latitude=47.52, longitude=-122.30, altitude=30.0),
        ],
        confidence=0.85,
    )
    m = cmd.to_mission()
    assert m is not None
    assert len(m.waypoints) == 3
    assert m.waypoints[0].hold_time == 2.0
    assert m.waypoints[0].sequence == 0
    assert m.waypoints[2].sequence == 2


def test_drone_command_to_mission_for_survey_generates_grid() -> None:
    cmd = DroneCommand(
        command_type=CommandType.SURVEY,
        parameters={"pattern": "grid"},
        confidence=0.7,
    )
    m = cmd.to_mission(
        home_position=(47.5, -122.3),
        survey_width_m=100.0,
        survey_height_m=80.0,
        survey_spacing_m=25.0,
    )
    assert m is not None
    assert m.mission_type.value == "survey"
    assert len(m.waypoints) >= 6  # grid needs multiple lines


def test_drone_command_to_mission_for_orbit_generates_circle() -> None:
    cmd = DroneCommand(
        command_type=CommandType.ORBIT,
        parameters={"radius": 25.0},
        confidence=0.8,
    )
    m = cmd.to_mission(
        home_position=(47.5, -122.3),
        orbit_points=8,
    )
    assert m is not None
    assert len(m.waypoints) >= 8


def test_drone_command_to_mission_for_single_shot_returns_none() -> None:
    for ct in (CommandType.LAND, CommandType.ARM, CommandType.HOVER, CommandType.RTL):
        cmd = DroneCommand(command_type=ct)
        assert cmd.to_mission() is None


def test_drone_command_to_mission_goto_falls_back_to_home_when_missing_coords() -> None:
    cmd = DroneCommand(
        command_type=CommandType.GOTO,
        waypoints=[WaypointDef(altitude=20.0)],  # no lat/lon
    )
    m = cmd.to_mission(home_position=(47.5, -122.3))
    assert m is not None
    assert m.waypoints[0].latitude == 47.5
    assert m.waypoints[0].longitude == -122.3


def test_command_interpreter_one_shot_to_mission() -> None:
    interp = CommandInterpreter()
    cmd, mission = interp.interpret_to_mission(
        "survey a grid here",
        home_position=(47.5, -122.3),
    )
    assert isinstance(cmd, DroneCommand)
    # Survey/grid commands should produce a multi-waypoint Mission
    assert mission is not None
    assert len(mission.waypoints) >= 2


def test_command_interpreter_to_mission_returns_none_for_single_shot() -> None:
    interp = CommandInterpreter()
    cmd, mission = interp.interpret_to_mission("land now")
    assert cmd.command_type == CommandType.LAND
    assert mission is None
