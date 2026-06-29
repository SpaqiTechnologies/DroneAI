"""Tests for the enterprise + media features added in this iteration:
docking station, scheduled patrols, adaptive 3D scan, highlight reel,
inspection report, camera presets, multi-camera array.
"""

import json
import os
import time

import pytest

from core.drone import Drone
from core.docking import (
    DockingStation,
    DockState,
    ChargeProfile,
    PatrolScheduler,
)
from applications.mapping.adaptive_scan import AdaptiveScan, ScanConfig, ScanState
from applications.media import HighlightEvent, HighlightReelBuilder
from applications.inspection.inspector import (
    Inspector,
    InspectionConfig,
    InspectionType,
    DetectedDefect,
    DefectSeverity,
)
from applications.inspection.report import InspectionReportGenerator
from sensors.camera_sensor import CameraSensor
from sensors.camera_array import (
    CameraArray,
    CameraSpec,
    default_inspection_array,
)
from sensors.media import MediaStorage, synthesize_rgb_bytes, write_png, VideoRecorder


# =============================== Docking ===================================


def _make_drone_at(lat: float, lon: float) -> Drone:
    d = Drone()
    d.current_position = (lat, lon)
    d.current_altitude = 0.0
    d.battery_level = 60.0
    return d


def test_docking_station_charges_drone_on_pad() -> None:
    drone = _make_drone_at(47.5, -122.3)
    profile = ChargeProfile(rate_pct_per_s=200.0, target_pct=95.0, launch_min_pct=80.0)
    dock = DockingStation(drone, 47.5, -122.3, charge_profile=profile, tick_hz=50.0)
    dock.start()
    time.sleep(0.4)
    dock.stop()
    assert drone.battery_level >= 90.0
    status = dock.status()
    assert status.state in (DockState.CHARGING, DockState.READY)


def test_dock_blocks_deploy_when_battery_low() -> None:
    drone = _make_drone_at(47.5, -122.3)
    drone.battery_level = 30.0
    dock = DockingStation(
        drone, 47.5, -122.3, charge_profile=ChargeProfile(launch_min_pct=80.0),
    )
    ok, msg = dock.deploy()
    assert not ok and "below launch threshold" in msg


def test_dock_deploy_transitions_state_and_fires_hook() -> None:
    drone = _make_drone_at(47.5, -122.3)
    drone.battery_level = 90.0
    dock = DockingStation(drone, 47.5, -122.3)
    fired = []
    dock.set_deploy_hook(lambda payload: fired.append(payload))
    ok, _ = dock.deploy(mission_id="abc")
    assert ok
    assert dock.state == DockState.DEPLOYED
    assert fired and fired[0]["mission_id"] == "abc"


def test_dock_auto_recall_on_low_battery() -> None:
    drone = _make_drone_at(47.5, -122.3)
    drone.battery_level = 90.0
    dock = DockingStation(
        drone, 47.5, -122.3,
        charge_profile=ChargeProfile(recall_pct=50.0),
        tick_hz=50.0,
    )
    dock.deploy()
    # Move drone away from the pad so we stay in DEPLOYED.
    drone.current_position = (47.6, -122.4)
    drone._is_flying = True
    dock.start()
    time.sleep(0.1)
    drone.battery_level = 40.0  # under recall threshold
    time.sleep(0.2)
    dock.stop()
    status = dock.status()
    assert status.state == DockState.RECALLING


# =============================== Patrols ===================================


def test_patrol_scheduler_deploys_when_dock_ready() -> None:
    drone = _make_drone_at(47.5, -122.3)
    drone.battery_level = 95.0  # already ready
    dock = DockingStation(
        drone, 47.5, -122.3,
        charge_profile=ChargeProfile(launch_min_pct=80.0, target_pct=95.0),
        tick_hz=50.0,
    )
    deployed = []
    dock.set_deploy_hook(lambda payload: deployed.append(payload))
    dock.start()
    time.sleep(0.1)  # let dock tick into READY
    sched = PatrolScheduler(dock=dock, tick_hz=20.0)
    sched.add_job(
        name="perimeter", period_s=60.0, payload={"area": "north"},
        flight_duration_s=0.05, start_immediately=True,
    )
    sched.start()
    time.sleep(0.5)
    sched.stop()
    dock.stop()
    assert deployed, f"scheduler never deployed; dock state was {dock.state}"
    assert deployed[0]["patrol_name"] == "perimeter"


def test_patrol_scheduler_rejects_zero_period() -> None:
    drone = _make_drone_at(47.5, -122.3)
    dock = DockingStation(drone, 47.5, -122.3)
    sched = PatrolScheduler(dock=dock)
    with pytest.raises(ValueError):
        sched.add_job(name="bad", period_s=0)


def test_patrol_scheduler_remove_and_toggle() -> None:
    drone = _make_drone_at(47.5, -122.3)
    dock = DockingStation(drone, 47.5, -122.3)
    sched = PatrolScheduler(dock=dock)
    job = sched.add_job(name="x", period_s=10.0)
    assert sched.enable_job(job.job_id, False)
    assert any(j.job_id == job.job_id and not j.enabled for j in sched.jobs())
    assert sched.remove_job(job.job_id)
    assert not sched.enable_job(job.job_id, True)


# ============================ Adaptive 3D Scan =============================


def test_adaptive_scan_plans_initial_waypoints() -> None:
    scan = AdaptiveScan(
        center=(47.5, -122.3),
        target_height_m=20.0,
        target_radius_m=5.0,
        config=ScanConfig(shells=2, bins_per_shell=6, standoff_m=10.0),
    )
    wps = scan.remaining_waypoints()
    assert len(wps) == 12  # 2 shells × 6 bins
    # All waypoints should be at standoff from center
    for wp in wps:
        assert wp.distance_m == pytest.approx(15.0)
        assert wp.shell_index in (0, 1)


def test_adaptive_scan_completes_and_refines_under_covered_bins() -> None:
    scan = AdaptiveScan(
        center=(47.5, -122.3),
        target_height_m=15.0,
        target_radius_m=3.0,
        config=ScanConfig(
            shells=2, bins_per_shell=4,
            coverage_target=0.85,
            coverage_quality_base=0.5,
            max_refinement_passes=1,
        ),
    )
    scan.start()
    # First pass — captures all initial waypoints
    captured = 0
    while True:
        img = scan.capture_next()
        if img is None:
            break
        captured += 1
        if captured > 100:
            break
    report = scan.report()
    assert report.state == ScanState.COMPLETED
    assert report.captured_waypoints >= 8
    assert report.refinement_waypoints >= 1  # at least some refinement happened
    assert report.overall_coverage > 0.5


def test_adaptive_scan_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError):
        AdaptiveScan((0, 0), target_height_m=0, target_radius_m=1)


# ============================ Highlight Reel ==============================


def _make_recording_bundle(tmp_path) -> str:
    storage = MediaStorage(root=str(tmp_path / "media"))
    rec = VideoRecorder(
        storage=storage, width=32, height=18, target_fps=10.0, frame_format="png",
    )
    rec.start()
    for _ in range(20):  # 2s of frames
        rec.add_frame(synthesize_rgb_bytes(32, 18))
    summary = rec.stop()
    return summary.bundle_path


def test_highlight_reel_selects_clips_around_events(tmp_path) -> None:
    bundle = _make_recording_bundle(tmp_path)
    with open(os.path.join(bundle, "manifest.json")) as fh:
        manifest = json.load(fh)
    # The sim recording is sub-second; place events at the bundle's
    # midpoint so they fall inside the [started_at, stopped_at] window.
    t_start = manifest["started_at"]
    t_stop = manifest["stopped_at"]
    midpoint = (t_start + t_stop) / 2
    events = [
        HighlightEvent(timestamp=midpoint, kind="target", label="person", confidence=0.95),
    ]
    builder = HighlightReelBuilder(
        bundle, pre_event_s=0.3, post_event_s=0.3, max_reel_duration_s=5.0,
    )
    out_path = tmp_path / "highlight_manifest.json"
    reel = builder.build(events=events, output_path=str(out_path))
    assert reel.clip_count >= 1
    assert reel.total_duration_s > 0
    assert os.path.isfile(out_path)
    payload = json.loads(out_path.read_text())
    assert payload["clip_count"] == reel.clip_count
    assert all(c["frame_count"] >= 0 for c in payload["clips"])


def test_highlight_reel_skips_events_outside_recording(tmp_path) -> None:
    bundle = _make_recording_bundle(tmp_path)
    with open(os.path.join(bundle, "manifest.json")) as fh:
        manifest = json.load(fh)
    builder = HighlightReelBuilder(bundle, pre_event_s=0.2, post_event_s=0.2)
    far_future = manifest["stopped_at"] + 60.0
    reel = builder.build(events=[
        HighlightEvent(timestamp=far_future, kind="target", label="late"),
    ])
    assert reel.clip_count == 0


def test_highlight_reel_merges_adjacent_clips(tmp_path) -> None:
    bundle = _make_recording_bundle(tmp_path)
    with open(os.path.join(bundle, "manifest.json")) as fh:
        manifest = json.load(fh)
    t_start = manifest["started_at"]
    t_stop = manifest["stopped_at"]
    # Two near-simultaneous events inside the bundle window; with a 2s
    # merge gap they must collapse into a single clip.
    midpoint = (t_start + t_stop) / 2
    events = [
        HighlightEvent(timestamp=midpoint, kind="target", label="a", confidence=0.9),
        HighlightEvent(timestamp=midpoint, kind="target", label="b", confidence=0.9),
    ]
    builder = HighlightReelBuilder(
        bundle, pre_event_s=0.3, post_event_s=0.5, merge_adjacent_gap_s=2.0,
    )
    reel = builder.build(events=events)
    assert reel.clip_count == 1
    assert len(reel.clips[0].events) == 2


# ============================ Inspection Report ===========================


def test_inspection_report_writes_json_and_markdown(tmp_path) -> None:
    inspector = Inspector(InspectionConfig(inspection_type=InspectionType.TOWER))
    # Inject a defect into the inspector by reaching into its internal list
    # rather than relying on simulation.
    inspector._defects.append(DetectedDefect(
        defect_id=1,
        location=(47.5, -122.3, 30.0),
        severity=DefectSeverity.CRITICAL,
        defect_type="crack",
        confidence=0.92,
        description="vertical crack visible 8m above ground",
        image_id=42,
    ))
    out = tmp_path / "reports"
    gen = InspectionReportGenerator(asset_name="Tower-7")
    report = gen.generate(inspector, output_dir=str(out))
    assert report.json_path and os.path.isfile(report.json_path)
    assert report.markdown_path and os.path.isfile(report.markdown_path)
    data = json.loads(open(report.json_path).read())
    assert data["summary"]["highest_severity"] == "critical"
    assert data["summary"]["actionable"] is True


def test_inspection_report_empty_inspector_is_safe() -> None:
    inspector = Inspector(InspectionConfig(inspection_type=InspectionType.BRIDGE))
    gen = InspectionReportGenerator(asset_name="empty")
    report = gen.generate(inspector)
    assert report.summary["defect_count"] == 0
    assert report.summary["highest_severity"] == "none"
    assert report.summary["actionable"] is False


# ============================ Camera presets / array =====================


def test_camera_resolution_presets_exist() -> None:
    assert CameraSensor.RESOLUTION_6K == (6144, 3240)
    assert CameraSensor.RESOLUTION_8K == (7680, 4320)


def test_camera_set_resolution_and_color_profile() -> None:
    cam = CameraSensor()
    ok, _ = cam.set_resolution(CameraSensor.RESOLUTION_8K)
    assert ok
    cam.enable_hdr(True)
    assert cam.is_hdr_enabled()
    ok, _ = cam.set_color_profile(CameraSensor.COLOR_PROFILE_DLOG_M)
    assert ok
    assert cam.get_color_profile() == "dlog-m"


def test_camera_rejects_invalid_color_profile() -> None:
    cam = CameraSensor()
    ok, msg = cam.set_color_profile("rainbows")
    assert not ok and "unknown" in msg


def test_camera_array_default_inspection_trio() -> None:
    array = default_inspection_array()
    assert array.names == ["wide", "tele", "thermal"]
    assert array.get_by_role("thermal") is not None
    assert array.active == "wide"


def test_camera_array_take_synchronized_snapshot(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    array = CameraArray(
        specs=[
            CameraSpec(name="wide", role="wide",
                       resolution=(160, 90), fov_deg=84.0),
            CameraSpec(name="tele", role="tele",
                       resolution=(160, 90), fov_deg=15.0, zoom=5.0),
        ],
        media_storage=storage,
    )
    array.start_all()
    results = array.take_synchronized_snapshot()
    assert all(ok for ok, _ in results.values())
    arts = storage.list_artifacts(kind="photo")
    assert len(arts) >= 2


def test_camera_array_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        CameraArray(
            specs=[
                CameraSpec(name="a", role="wide", resolution=(160, 90), fov_deg=84),
                CameraSpec(name="a", role="tele", resolution=(160, 90), fov_deg=15),
            ],
        )


def test_camera_array_set_active_and_capture_active(tmp_path) -> None:
    array = default_inspection_array()
    array.start_all()
    ok, _ = array.set_active("tele")
    assert ok
    assert array.active == "tele"
    f = array.capture_active_frame()
    assert f is not None
