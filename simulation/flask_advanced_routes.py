"""Flask mirror of the FastAPI advanced-feature routes.

The original Flask dashboard at ``simulation/templates/dashboard.html``
already covers map / waypoints / geofence / telemetry / 3D viz / gimbal
/ swarm / vision modes / Remote ID / failsafe triggers.

This module ADDS routes for the newer features that were only on the
FastAPI server before: docking station + scheduled patrols + autonomy
adapter, SAR + swarm SAR, beacon trilateration, supply-drop and
safe-corridor planners, adaptive 3D scan, highlight reel builder,
inspection report generator, anomaly bridge history. Same JSON shapes
as the FastAPI equivalents so the same JS works against either backend.

Wire up by calling ``register_advanced_routes(app, get_drone)`` once at
Flask boot.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import jsonify, request

from applications.search_rescue import (
    SARMission,
    SARState,
    SearchConfig,
    SearchPatternGenerator,
    SearchType,
    SwarmSARMission,
)
from applications.survival import (
    BeaconLocator,
    SupplyDropPlanner,
    DropParameters,
    SafeCorridorPlanner,
    ThreatZone,
)
from applications.mapping.adaptive_scan import AdaptiveScan, ScanConfig
from applications.media import HighlightEvent, HighlightReelBuilder
from applications.inspection.report import InspectionReportGenerator
from core.docking import (
    DockingStation,
    PatrolScheduler,
    ChargeProfile,
    DockAutonomyAdapter,
    PatrolPath,
)
from ai.llm.command_interpreter import CommandInterpreter
from core.drone import Drone
from sensors.media import MediaStorage, synthesize_rgb_bytes, encode_png


_PATTERN_TYPES = {
    "expanding_square": SearchType.EXPANDING_SQUARE,
    "sector": SearchType.SECTOR,
    "parallel": SearchType.PARALLEL,
    "creeping_line": SearchType.CREEPING_LINE,
    "spiral": SearchType.SPIRAL,
    "grid": SearchType.GRID,
}


class _Store:
    """In-process registry for active missions / scans / dock state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.beacon = BeaconLocator()
        self.sar: Dict[str, SARMission] = {}
        self.swarm: Dict[str, SwarmSARMission] = {}
        self.scans: Dict[str, AdaptiveScan] = {}
        self.dock: Optional[DockingStation] = None
        self.scheduler: Optional[PatrolScheduler] = None
        self.adapter: Optional[DockAutonomyAdapter] = None
        self.interpreter: Optional[CommandInterpreter] = None

    def add_sar(self, mission: SARMission) -> str:
        mid = uuid.uuid4().hex[:12]
        with self._lock:
            self.sar[mid] = mission
        return mid

    def add_swarm(self, mission: SwarmSARMission) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self.swarm[sid] = mission
        return sid

    def add_scan(self, scan: AdaptiveScan) -> None:
        with self._lock:
            self.scans[scan.scan_id] = scan


def _ensure_media_storage(drone: Drone) -> Optional[MediaStorage]:
    cam = getattr(drone, "camera_sensor", None)
    if cam is None:
        return None
    existing = getattr(cam, "_media_storage", None)
    if existing is not None:
        return existing
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flight_logs", "media",
    )
    storage = MediaStorage(root=root)
    cam._media_storage = storage
    if not getattr(cam, "_synth_resolution", None):
        cam._synth_resolution = (160, 90)
    return storage


def _build_search_config(payload: Dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        altitude=float(payload.get("altitude", 30.0)),
        track_spacing=float(payload.get("track_spacing", 40.0)),
        speed=float(payload.get("speed", 8.0)),
        search_radius=float(payload.get("radius", payload.get("radius_m", 500.0))),
        legs=int(payload.get("legs", 8)),
    )


def _to_pattern_type(name: str) -> SearchType:
    p = _PATTERN_TYPES.get((name or "").lower())
    if p is None:
        raise ValueError(f"unknown pattern_type: {name}")
    return p


def register_advanced_routes(
    app,
    get_drone: Callable[[], Optional[Drone]],
) -> _Store:
    """Register all advanced routes on ``app``. Returns the route store."""
    store = _Store()

    # ============================ Dock ============================

    @app.route("/api/dock/setup", methods=["POST"])
    def dock_setup():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        req = request.get_json(silent=True) or {}
        if store.dock is not None and store.dock.is_running():
            store.dock.stop()
        if store.scheduler is not None and store.scheduler.is_running:
            store.scheduler.stop()
        profile = ChargeProfile(
            rate_pct_per_s=float(req.get("rate_pct_per_s", 5.0)),
            target_pct=float(req.get("target_pct", 95.0)),
            launch_min_pct=float(req.get("launch_min_pct", 80.0)),
            recall_pct=float(req.get("recall_pct", 25.0)),
        )
        dock = DockingStation(
            drone=drone,
            latitude=float(req.get("latitude", 47.5)),
            longitude=float(req.get("longitude", -122.3)),
            altitude_msl_m=float(req.get("altitude_msl_m", 0.0)),
            charge_profile=profile,
            tick_hz=float(req.get("tick_hz", 10.0)),
        )
        adapter = DockAutonomyAdapter(
            dock=dock,
            drone=drone,
            default_path=PatrolPath(
                altitude_m=float(req.get("patrol_altitude_m", 20.0)),
                radius_m=float(req.get("patrol_radius_m", 40.0)),
                lap_count=int(req.get("patrol_lap_count", 1)),
                points_per_lap=int(req.get("patrol_points_per_lap", 8)),
                cruise_speed_mps=float(req.get("patrol_cruise_speed_mps", 12.0)),
            ),
        )
        adapter.attach()
        dock.start()
        sched = PatrolScheduler(dock=dock)
        sched.start()
        store.dock, store.scheduler, store.adapter = dock, sched, adapter
        return jsonify({"ok": True, "dock": dock.status().to_dict(), "adapter_attached": True})

    @app.route("/api/dock/status", methods=["GET"])
    def dock_status():
        # Polled every 2s by the dashboard — return 200 with a stable
        # "not_configured" payload so the browser console doesn't fill
        # with 503s before the operator clicks Setup Dock.
        if store.dock is None:
            return jsonify({
                "state": "not_configured",
                "battery_pct": 0.0,
                "is_charging": False,
                "drone_at_dock": False,
                "deployments": 0,
                "recalls": 0,
                "auto_recall_armed": False,
                "elapsed_in_state_s": 0.0,
                "last_event_at": 0.0,
                "configured": False,
            })
        out = store.dock.status().to_dict()
        out["configured"] = True
        if store.adapter is not None:
            out["adapter"] = {
                "deployments": store.adapter.deployments,
                "completions": store.adapter.completions,
                "aborts": store.adapter.aborts,
                "last_runtime_status": store.adapter.last_runtime_status(),
            }
        return jsonify(out)

    @app.route("/api/dock/deploy", methods=["POST"])
    def dock_deploy():
        if store.dock is None:
            return jsonify({"detail": "no dock configured"}), 503
        payload = request.get_json(silent=True) or {}
        ok, msg = store.dock.deploy(**payload)
        if not ok:
            return jsonify({"detail": msg}), 400
        return jsonify({"ok": True, "message": msg, "dock": store.dock.status().to_dict()})

    @app.route("/api/dock/recall", methods=["POST"])
    def dock_recall():
        if store.dock is None:
            return jsonify({"detail": "no dock configured"}), 503
        reason = request.args.get("reason", "manual")
        store.dock.recall(reason=reason)
        return jsonify({"ok": True, "dock": store.dock.status().to_dict()})

    # ============================ Patrols ============================

    @app.route("/api/patrols/add", methods=["POST"])
    def patrols_add():
        if store.scheduler is None:
            return jsonify({"detail": "no patrol scheduler"}), 503
        req = request.get_json(silent=True) or {}
        job = store.scheduler.add_job(
            name=str(req.get("name", "patrol")),
            period_s=float(req.get("period_s", 30.0)),
            payload=req.get("payload") or {},
            flight_duration_s=float(req.get("flight_duration_s", 30.0)),
            start_immediately=bool(req.get("start_immediately", False)),
        )
        return jsonify(job.to_dict())

    @app.route("/api/patrols", methods=["GET"])
    def patrols_list():
        if store.scheduler is None:
            return jsonify({"jobs": []})
        return jsonify({"jobs": [j.to_dict() for j in store.scheduler.jobs()]})

    @app.route("/api/patrols/<job_id>", methods=["DELETE"])
    def patrols_remove(job_id):
        if store.scheduler is None or not store.scheduler.remove_job(job_id):
            return jsonify({"detail": "not found"}), 404
        return jsonify({"ok": True})

    # ============================ SAR ============================

    @app.route("/api/sar/plan", methods=["POST"])
    def sar_plan():
        req = request.get_json(silent=True) or {}
        try:
            pt = _to_pattern_type(req.get("pattern_type", "expanding_square"))
        except ValueError as e:
            return jsonify({"detail": str(e)}), 400
        gen = SearchPatternGenerator(config=_build_search_config(req))
        center = req.get("center") or {"latitude": 47.5, "longitude": -122.3}
        pattern = gen.generate(pt, (float(center["latitude"]), float(center["longitude"])))
        return jsonify({
            "pattern_type": pattern.pattern_type.value,
            "waypoint_count": len(pattern.waypoints),
            "waypoints": [list(wp) for wp in pattern.waypoints],
            "total_distance_m": pattern.total_distance,
            "estimated_time_s": pattern.estimated_time,
            "coverage_probability": pattern.coverage_probability,
            "area_sqm": pattern.area_sqm,
        })

    @app.route("/api/sar/run", methods=["POST"])
    def sar_run():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        req = request.get_json(silent=True) or {}
        _ensure_media_storage(drone)
        try:
            drone.camera_sensor.start()
        except Exception:
            pass
        try:
            pt = _to_pattern_type(req.get("pattern_type", "expanding_square"))
        except ValueError as e:
            return jsonify({"detail": str(e)}), 400
        center = req.get("center") or {"latitude": 47.5, "longitude": -122.3}
        sar = SARMission(
            drone=drone,
            pattern_type=pt,
            center=(float(center["latitude"]), float(center["longitude"])),
            config=_build_search_config(req),
            cruise_speed_mps=float(req.get("speed", 8)) * 2.0,
            photo_interval_s=float(req.get("photo_interval_s", 1.0)),
            return_to_start=bool(req.get("return_to_start", False)),
        )
        sar.run_in_background(tick_hz=20.0)
        return jsonify({"mission_id": store.add_sar(sar), "state": sar.state.value})

    @app.route("/api/sar/<mission_id>/report", methods=["GET"])
    def sar_report(mission_id):
        sar = store.sar.get(mission_id)
        if sar is None:
            return jsonify({"detail": "mission not found"}), 404
        return jsonify(sar.report().to_dict())

    @app.route("/api/sar/<mission_id>/abort", methods=["POST"])
    def sar_abort(mission_id):
        sar = store.sar.get(mission_id)
        if sar is None:
            return jsonify({"detail": "mission not found"}), 404
        sar.abort("api abort")
        return jsonify({"ok": True, "state": sar.state.value})

    # ============================ Swarm SAR ============================

    @app.route("/api/swarm-sar/run", methods=["POST"])
    def swarm_sar_run():
        req = request.get_json(silent=True) or {}
        try:
            pt = _to_pattern_type(req.get("pattern_type", "sector"))
        except ValueError as e:
            return jsonify({"detail": str(e)}), 400
        center = req.get("center") or {"latitude": 47.5, "longitude": -122.3}
        center_t = (float(center["latitude"]), float(center["longitude"]))
        drones: Dict[str, Drone] = {}
        for i in range(int(req.get("drone_count", 3))):
            d = Drone()
            d.current_position = center_t
            d.current_altitude = 0.0
            try:
                d.camera_sensor.start()
            except Exception:
                pass
            drones[f"drone_{i}"] = d
        swarm = SwarmSARMission(
            drones=drones,
            center=center_t,
            radius_m=float(req.get("radius_m", 500.0)),
            config=_build_search_config(req),
            pattern_type=pt,
            return_to_start=bool(req.get("return_to_start", False)),
        )
        swarm.run_in_background(tick_hz=20.0)
        return jsonify({
            "swarm_id": store.add_swarm(swarm),
            "drone_ids": swarm.drone_ids,
            "drone_count": len(drones),
        })

    @app.route("/api/swarm-sar/<sid>/report", methods=["GET"])
    def swarm_sar_report(sid):
        sw = store.swarm.get(sid)
        if sw is None:
            return jsonify({"detail": "swarm mission not found"}), 404
        return jsonify(sw.report().to_dict())

    # ============================ Beacon ============================

    @app.route("/api/survival/beacon/sample", methods=["POST"])
    def beacon_sample():
        req = request.get_json(silent=True) or {}
        reading = store.beacon.add_reading(
            latitude=float(req["latitude"]),
            longitude=float(req["longitude"]),
            altitude=float(req["altitude"]),
            rssi_dbm=float(req["rssi_dbm"]),
        )
        return jsonify({"sample_count": store.beacon.sample_count, "reading": reading.to_dict()})

    @app.route("/api/survival/beacon/fix", methods=["GET"])
    def beacon_fix():
        fix = store.beacon.compute_fix()
        return jsonify({
            "fix": fix.to_dict() if fix else None,
            "sample_count": store.beacon.sample_count,
        })

    @app.route("/api/survival/beacon/reset", methods=["POST"])
    def beacon_reset():
        store.beacon.reset()
        return jsonify({"sample_count": 0})

    # ============================ Supply drop ============================

    @app.route("/api/survival/supply-drop/plan", methods=["POST"])
    def supply_drop_plan():
        req = request.get_json(silent=True) or {}
        params = DropParameters(
            release_altitude_agl_m=float(req.get("release_altitude_agl_m", 30.0)),
            release_speed_mps=float(req.get("release_speed_mps", 8.0)),
            parachute_deploy_at_m=req.get("parachute_deploy_at_m"),
        )
        plan = SupplyDropPlanner(params).plan(
            target_lat=float(req["target"]["latitude"]),
            target_lon=float(req["target"]["longitude"]),
            wind_speed_mps=float(req.get("wind_speed_mps", 0.0)),
            wind_direction_deg=float(req.get("wind_direction_deg", 0.0)),
        )
        return jsonify(plan.to_dict())

    # ============================ Safe corridor ============================

    @app.route("/api/survival/corridor/plan", methods=["POST"])
    def corridor_plan():
        req = request.get_json(silent=True) or {}
        threats = [
            ThreatZone(
                latitude=float(t["latitude"]),
                longitude=float(t["longitude"]),
                radius_m=float(t["radius_m"]),
                name=str(t.get("name", "threat")),
                severity=float(t.get("severity", 1.0)),
            )
            for t in (req.get("threats") or [])
        ]
        planner = SafeCorridorPlanner(threats=threats, margin_m=float(req.get("margin_m", 25.0)))
        corridor = planner.plan(
            start=(float(req["start"]["latitude"]), float(req["start"]["longitude"])),
            goal=(float(req["goal"]["latitude"]), float(req["goal"]["longitude"])),
        )
        return jsonify(corridor.to_dict())

    # ============================ Adaptive 3D scan ============================

    @app.route("/api/scan3d/plan", methods=["POST"])
    def scan_plan():
        req = request.get_json(silent=True) or {}
        cfg = ScanConfig(
            standoff_m=float(req.get("standoff_m", 15.0)),
            shells=int(req.get("shells", 3)),
            bins_per_shell=int(req.get("bins_per_shell", 12)),
            coverage_target=float(req.get("coverage_target", 0.85)),
        )
        scan = AdaptiveScan(
            center=(float(req["center_lat"]), float(req["center_lon"])),
            target_height_m=float(req["target_height_m"]),
            target_radius_m=float(req["target_radius_m"]),
            ground_altitude_m=float(req.get("ground_altitude_m", 0.0)),
            config=cfg,
        )
        store.add_scan(scan)
        scan.start()
        return jsonify({
            "scan_id": scan.scan_id,
            "planned_waypoints": [wp.to_dict() for wp in scan.remaining_waypoints()],
            "report": scan.report().to_dict(),
        })

    @app.route("/api/scan3d/<scan_id>/capture", methods=["POST"])
    def scan_capture(scan_id):
        scan = store.scans.get(scan_id)
        if scan is None:
            return jsonify({"detail": "scan not found"}), 404
        img = scan.capture_next()
        return jsonify({
            "captured": img.to_dict() if img else None,
            "report": scan.report().to_dict(),
        })

    @app.route("/api/scan3d/<scan_id>/report", methods=["GET"])
    def scan_report(scan_id):
        scan = store.scans.get(scan_id)
        if scan is None:
            return jsonify({"detail": "scan not found"}), 404
        return jsonify(scan.report().to_dict())

    # ============================ Highlight reel ============================

    @app.route("/api/media/highlight-reel", methods=["POST"])
    def media_highlight_reel():
        req = request.get_json(silent=True) or {}
        try:
            builder = HighlightReelBuilder(
                recording_bundle_path=req["bundle_path"],
                pre_event_s=float(req.get("pre_event_s", 1.5)),
                post_event_s=float(req.get("post_event_s", 2.5)),
                max_reel_duration_s=float(req.get("max_reel_duration_s", 30.0)),
            )
        except FileNotFoundError as exc:
            return jsonify({"detail": str(exc)}), 404
        events = [
            HighlightEvent(
                timestamp=float(e["timestamp"]),
                kind=str(e.get("kind", "event")),
                label=str(e.get("label", "")),
                importance=float(e.get("importance", 1.0)),
                confidence=float(e.get("confidence", 1.0)),
                metadata=e.get("metadata", {}) or {},
            )
            for e in req.get("events") or []
        ]
        reel = builder.build(events=events, output_path=req.get("output_path"))
        return jsonify(reel.to_dict())

    # ============================ Inspection report ============================

    @app.route("/api/inspection/report", methods=["POST"])
    def inspection_report():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        asset = request.args.get("asset_name", "asset")
        operator = request.args.get("operator_id")
        output_dir = request.args.get("output_dir")
        # The Flask server's existing global state has an Inspector instance.
        # Look it up via the same name the FastAPI variant uses.
        inspector = None
        try:
            import simulation.web_server as ws  # type: ignore
            inspector = getattr(ws, "inspector_app", None) or getattr(ws, "inspector", None)
        except Exception:
            pass
        if inspector is None:
            return jsonify({"detail": "no Inspector instance available"}), 400
        storage = _ensure_media_storage(drone)
        media = [a.to_dict() for a in storage.list_artifacts(kind="photo")] if storage else []
        gen = InspectionReportGenerator(asset_name=asset, operator_id=operator)
        report = gen.generate(inspector, media_artifacts=media, output_dir=output_dir)
        return jsonify(report.to_dict())

    # ============================ LLM mission ============================

    @app.route("/api/llm/mission/plan", methods=["POST"])
    def llm_mission_plan():
        if store.interpreter is None:
            store.interpreter = CommandInterpreter()
        req = request.get_json(silent=True) or {}
        home = None
        if isinstance(req.get("home_position"), dict):
            hp = req["home_position"]
            home = (float(hp["latitude"]), float(hp["longitude"]))
        else:
            d = get_drone()
            if d is not None and getattr(d, "current_position", None):
                home = tuple(d.current_position[:2])
        cmd, mission = store.interpreter.interpret_to_mission(
            req["command"],
            home_position=home,
            default_altitude=float(req.get("default_altitude", 30.0)),
            default_speed=float(req.get("default_speed", 5.0)),
            survey_width_m=float(req.get("survey_width_m", 200.0)),
            survey_height_m=float(req.get("survey_height_m", 200.0)),
            survey_spacing_m=float(req.get("survey_spacing_m", 25.0)),
            orbit_radius_m=float(req.get("orbit_radius_m", 30.0)),
        )
        return jsonify({
            "command": cmd.to_dict(),
            "mission": (
                {
                    "id": mission.id, "name": mission.name,
                    "mission_type": mission.mission_type.value,
                    "default_speed": mission.default_speed,
                    "default_altitude": mission.default_altitude,
                    "waypoint_count": len(mission.waypoints),
                    "waypoints": [
                        {"sequence": w.sequence, "lat": w.latitude, "lon": w.longitude,
                         "alt": w.altitude, "speed": w.speed, "hold_time": w.hold_time}
                        for w in mission.waypoints
                    ],
                    "metadata": mission.metadata,
                } if mission is not None else None
            ),
            "convertible": mission is not None,
            "reason": None if mission else
                f"command_type={cmd.command_type.value} is single-shot",
        })

    # ============================ Anomaly history ============================

    @app.route("/api/anomalies/history", methods=["GET"])
    def anomalies_history():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        bridge = getattr(drone, "anomaly_failsafe_bridge", None)
        if bridge is None:
            return jsonify({"history": [], "note": "no anomaly bridge attached"})
        return jsonify({"history": [t.to_dict() for t in bridge.history]})

    @app.route("/api/anomalies/inject", methods=["POST"])
    def anomalies_inject():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        detector = getattr(drone, "anomaly_detector", None)
        if detector is None:
            return jsonify({"detail": "no anomaly detector"}), 503
        sensor = request.args.get("sensor", "motor_vibration")
        component = request.args.get("component", "motor_1")
        baseline_value = float(request.args.get("baseline_value", 0.5))
        spike_value = float(request.args.get("spike_value", 50.0))
        baseline_samples = int(request.args.get("baseline_samples", 40))
        for _ in range(baseline_samples):
            detector.update(sensor, baseline_value, component=component)
        anomaly = detector.update(sensor, spike_value, component=component)
        bridge = getattr(drone, "anomaly_failsafe_bridge", None)
        return jsonify({
            "ok": True,
            "anomaly_fired": anomaly is not None,
            "severity": getattr(getattr(anomaly, "severity", None), "value", None),
            "history_count": len(bridge.history) if bridge else 0,
        })

    # ============================ Media snapshot / recording ================

    @app.route("/api/media/snapshot", methods=["POST"])
    def media_snapshot():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        storage = _ensure_media_storage(drone)
        try:
            drone.camera_sensor.start()
        except Exception:
            pass
        ok, msg = drone.take_snapshot()
        arts = storage.list_artifacts(kind="photo") if storage else []
        latest = arts[-1].to_dict() if arts else None
        return jsonify({"success": ok, "message": msg, "artifact": latest})

    @app.route("/api/media/recording/start", methods=["POST"])
    def media_recording_start():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        _ensure_media_storage(drone)
        try:
            drone.camera_sensor.start()
            drone.camera_sensor.start_streaming()
        except Exception:
            pass
        ok, msg = drone.start_recording()
        return jsonify({"success": ok, "message": msg})

    @app.route("/api/media/recording/stop", methods=["POST"])
    def media_recording_stop():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        ok, msg = drone.stop_recording()
        summary = drone.camera_sensor.get_last_recording_summary()
        try:
            drone.camera_sensor.stop_streaming()
        except Exception:
            pass
        return jsonify({"success": ok, "message": msg, "summary": summary})

    @app.route("/api/media", methods=["GET"])
    def media_list():
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        storage = _ensure_media_storage(drone)
        if storage is None:
            return jsonify({"photos": [], "videos": []})
        return jsonify({
            "photos": [a.to_dict() for a in storage.list_artifacts(kind="photo")],
            "videos": [a.to_dict() for a in storage.list_artifacts(kind="video")],
            "total_bytes": storage.total_bytes(),
            "root": storage.root,
        })

    # ============================ Live PNG frame ============================

    @app.route("/api/media/latest.png", methods=["GET"])
    def media_latest_png():
        from flask import Response
        drone = get_drone()
        if drone is None:
            return jsonify({"detail": "drone not initialized"}), 503
        cam = getattr(drone, "camera_sensor", None)
        if cam is None:
            return jsonify({"detail": "no camera"}), 503
        try:
            cam.start()
        except Exception:
            pass
        frame = cam.capture_frame()
        if frame is None:
            return jsonify({"detail": "no frame"}), 503
        sw, sh = getattr(cam, "_synth_resolution", (160, 90))
        boxes = cam._scaled_detection_boxes(frame, sw, sh)
        rgb = synthesize_rgb_bytes(
            sw, sh, vision_mode=frame.mode.value,
            detections=boxes, timestamp=frame.timestamp,
        )
        png = encode_png(rgb, sw, sh)
        return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})

    return store
