"""FastAPI routes for the enterprise + media features:
docking station, scheduled patrols, adaptive 3D scan, highlight reel,
inspection report.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from applications.mapping.adaptive_scan import AdaptiveScan, ScanConfig
from applications.media import (
    HighlightEvent,
    HighlightReelBuilder,
)
from applications.inspection.report import InspectionReportGenerator
from core.docking import (
    DockingStation,
    PatrolScheduler,
    ChargeProfile,
    DockAutonomyAdapter,
    PatrolPath,
)


class DockSetupRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude_msl_m: float = 0.0
    rate_pct_per_s: float = 0.5
    launch_min_pct: float = 80.0
    recall_pct: float = 25.0
    target_pct: float = 95.0
    tick_hz: float = 5.0
    patrol_altitude_m: float = 20.0
    patrol_radius_m: float = 50.0
    patrol_lap_count: int = 1
    patrol_points_per_lap: int = 12
    patrol_cruise_speed_mps: float = 8.0


class PatrolAddRequest(BaseModel):
    name: str
    period_s: float = Field(..., gt=0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    flight_duration_s: float = 60.0
    start_immediately: bool = False


class AdaptiveScanRequest(BaseModel):
    center_lat: float = Field(..., ge=-90, le=90)
    center_lon: float = Field(..., ge=-180, le=180)
    target_height_m: float = Field(..., gt=0)
    target_radius_m: float = Field(..., gt=0)
    ground_altitude_m: float = 0.0
    standoff_m: float = 15.0
    shells: int = Field(3, ge=1, le=10)
    bins_per_shell: int = Field(12, ge=4, le=72)
    coverage_target: float = Field(0.85, gt=0, le=1.0)


class HighlightReelRequest(BaseModel):
    bundle_path: str
    events: List[Dict[str, Any]] = Field(default_factory=list)
    pre_event_s: float = 1.5
    post_event_s: float = 2.5
    max_reel_duration_s: float = 30.0
    output_path: Optional[str] = None


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dock: Optional[DockingStation] = None
        self._scheduler: Optional[PatrolScheduler] = None
        self._adapter: Optional[DockAutonomyAdapter] = None
        self._scans: Dict[str, AdaptiveScan] = {}

    def set_dock(
        self,
        dock: DockingStation,
        scheduler: PatrolScheduler,
        adapter: Optional[DockAutonomyAdapter] = None,
    ) -> None:
        with self._lock:
            self._dock = dock
            self._scheduler = scheduler
            self._adapter = adapter

    @property
    def adapter(self) -> Optional[DockAutonomyAdapter]:
        with self._lock:
            return self._adapter

    @property
    def dock(self) -> Optional[DockingStation]:
        with self._lock:
            return self._dock

    @property
    def scheduler(self) -> Optional[PatrolScheduler]:
        with self._lock:
            return self._scheduler

    def add_scan(self, scan: AdaptiveScan) -> None:
        with self._lock:
            self._scans[scan.scan_id] = scan

    def get_scan(self, scan_id: str) -> Optional[AdaptiveScan]:
        with self._lock:
            return self._scans.get(scan_id)

    def list_scans(self) -> List[str]:
        with self._lock:
            return list(self._scans.keys())


def build_router(get_state) -> APIRouter:
    router = APIRouter()
    store = _Store()

    def _require_drone():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        return state, state.drone

    # ============================ Docking ============================

    @router.post("/api/dock/setup")
    def dock_setup(req: DockSetupRequest):
        state, drone = _require_drone()
        if store.dock is not None and store.dock.is_running():
            store.dock.stop()
        if store.scheduler is not None and store.scheduler.is_running:
            store.scheduler.stop()
        profile = ChargeProfile(
            rate_pct_per_s=req.rate_pct_per_s,
            target_pct=req.target_pct,
            launch_min_pct=req.launch_min_pct,
            recall_pct=req.recall_pct,
        )
        dock = DockingStation(
            drone=drone,
            latitude=req.latitude,
            longitude=req.longitude,
            altitude_msl_m=req.altitude_msl_m,
            charge_profile=profile,
            tick_hz=req.tick_hz,
        )
        # Adapter must attach BEFORE the dock thread starts so it owns the hook.
        adapter = DockAutonomyAdapter(
            dock=dock,
            drone=drone,
            default_path=PatrolPath(
                altitude_m=req.patrol_altitude_m,
                radius_m=req.patrol_radius_m,
                lap_count=req.patrol_lap_count,
                points_per_lap=req.patrol_points_per_lap,
                cruise_speed_mps=req.patrol_cruise_speed_mps,
            ),
        )
        adapter.attach()
        dock.start()
        sched = PatrolScheduler(dock=dock)
        sched.start()
        store.set_dock(dock, sched, adapter)
        # Expose on simulation state for other routers if needed.
        state.docking_station = dock
        state.patrol_scheduler = sched
        state.dock_adapter = adapter
        return {
            "ok": True,
            "dock": dock.status().to_dict(),
            "adapter_attached": True,
        }

    @router.get("/api/dock/status")
    def dock_status():
        dock = store.dock
        if dock is None:
            raise HTTPException(status_code=503, detail="no dock configured")
        out = dock.status().to_dict()
        adapter = store.adapter
        if adapter is not None:
            out["adapter"] = {
                "deployments": adapter.deployments,
                "completions": adapter.completions,
                "aborts": adapter.aborts,
                "last_runtime_status": adapter.last_runtime_status(),
            }
        return out

    @router.post("/api/dock/deploy")
    def dock_deploy(payload: Dict[str, Any] = None):
        dock = store.dock
        if dock is None:
            raise HTTPException(status_code=503, detail="no dock configured")
        ok, msg = dock.deploy(**(payload or {}))
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"ok": True, "message": msg, "dock": dock.status().to_dict()}

    @router.post("/api/dock/recall")
    def dock_recall(reason: str = "manual"):
        dock = store.dock
        if dock is None:
            raise HTTPException(status_code=503, detail="no dock configured")
        dock.recall(reason=reason)
        return {"ok": True, "dock": dock.status().to_dict()}

    # =========================== Patrols ============================

    @router.post("/api/patrols/add")
    def patrol_add(req: PatrolAddRequest):
        sched = store.scheduler
        if sched is None:
            raise HTTPException(status_code=503, detail="no patrol scheduler")
        job = sched.add_job(
            name=req.name,
            period_s=req.period_s,
            payload=req.payload,
            flight_duration_s=req.flight_duration_s,
            start_immediately=req.start_immediately,
        )
        return job.to_dict()

    @router.get("/api/patrols")
    def patrols_list():
        sched = store.scheduler
        if sched is None:
            return {"jobs": []}
        return {"jobs": [j.to_dict() for j in sched.jobs()]}

    @router.delete("/api/patrols/{job_id}")
    def patrol_remove(job_id: str):
        sched = store.scheduler
        if sched is None or not sched.remove_job(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return {"ok": True}

    @router.post("/api/patrols/{job_id}/enabled/{enabled}")
    def patrol_enable(job_id: str, enabled: bool):
        sched = store.scheduler
        if sched is None or not sched.enable_job(job_id, enabled):
            raise HTTPException(status_code=404, detail="job not found")
        return {"ok": True}

    # =========================== Adaptive scan =====================

    @router.post("/api/scan3d/plan")
    def scan_plan(req: AdaptiveScanRequest):
        cfg = ScanConfig(
            standoff_m=req.standoff_m,
            shells=req.shells,
            bins_per_shell=req.bins_per_shell,
            coverage_target=req.coverage_target,
        )
        scan = AdaptiveScan(
            center=(req.center_lat, req.center_lon),
            target_height_m=req.target_height_m,
            target_radius_m=req.target_radius_m,
            ground_altitude_m=req.ground_altitude_m,
            config=cfg,
        )
        store.add_scan(scan)
        scan.start()
        return {
            "scan_id": scan.scan_id,
            "planned_waypoints": [wp.to_dict() for wp in scan.remaining_waypoints()],
            "report": scan.report().to_dict(),
        }

    @router.post("/api/scan3d/{scan_id}/capture")
    def scan_capture(scan_id: str):
        scan = store.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        img = scan.capture_next()
        return {
            "captured": img.to_dict() if img is not None else None,
            "report": scan.report().to_dict(),
        }

    @router.get("/api/scan3d/{scan_id}/report")
    def scan_report(scan_id: str):
        scan = store.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return scan.report().to_dict()

    @router.get("/api/scan3d")
    def scan_list():
        return {"scan_ids": store.list_scans()}

    # =========================== Highlight reel =====================

    @router.post("/api/media/highlight-reel")
    def highlight_reel(req: HighlightReelRequest):
        try:
            builder = HighlightReelBuilder(
                recording_bundle_path=req.bundle_path,
                pre_event_s=req.pre_event_s,
                post_event_s=req.post_event_s,
                max_reel_duration_s=req.max_reel_duration_s,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        events = [
            HighlightEvent(
                timestamp=float(e["timestamp"]),
                kind=str(e.get("kind", "event")),
                label=str(e.get("label", "")),
                importance=float(e.get("importance", 1.0)),
                confidence=float(e.get("confidence", 1.0)),
                metadata=e.get("metadata", {}) or {},
            )
            for e in req.events
        ]
        reel = builder.build(events=events, output_path=req.output_path)
        return reel.to_dict()

    # =========================== Anomaly history ====================

    @router.get("/api/anomalies/history")
    def anomalies_history():
        state = get_state()
        drone = getattr(state, "drone", None)
        if drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        bridge = getattr(drone, "anomaly_failsafe_bridge", None)
        if bridge is None:
            return {"history": [], "note": "no anomaly bridge attached"}
        return {"history": [t.to_dict() for t in bridge.history]}

    @router.post("/api/anomalies/inject")
    def anomalies_inject(
        sensor: str = "motor_vibration",
        component: str = "motor_1",
        baseline_value: float = 0.5,
        spike_value: float = 50.0,
        baseline_samples: int = 40,
    ):
        """Test hook: warm up the baseline and inject a spike to fire the bridge."""
        state = get_state()
        drone = getattr(state, "drone", None)
        if drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        detector = getattr(drone, "anomaly_detector", None)
        if detector is None:
            raise HTTPException(status_code=503, detail="no anomaly detector")
        for _ in range(baseline_samples):
            detector.update(sensor, baseline_value, component=component)
        anomaly = detector.update(sensor, spike_value, component=component)
        bridge = getattr(drone, "anomaly_failsafe_bridge", None)
        return {
            "ok": True,
            "anomaly_fired": anomaly is not None,
            "severity": getattr(getattr(anomaly, "severity", None), "value", None),
            "history_count": len(bridge.history) if bridge else 0,
        }

    # =========================== Inspection report ==================

    @router.post("/api/inspection/report")
    def inspection_report(
        output_dir: Optional[str] = None,
        asset_name: str = "asset",
        operator_id: Optional[str] = None,
    ):
        state, drone = _require_drone()
        inspector = getattr(state, "inspector", None)
        if inspector is None:
            raise HTTPException(
                status_code=400,
                detail="no Inspector instance attached to simulation state",
            )
        media: List[Dict[str, Any]] = []
        cam = getattr(drone, "camera_sensor", None)
        storage = getattr(cam, "media_storage", None) if cam else None
        if storage is not None:
            media = [a.to_dict() for a in storage.list_artifacts(kind="photo")]
        gen = InspectionReportGenerator(asset_name=asset_name, operator_id=operator_id)
        report = gen.generate(
            inspector=inspector, media_artifacts=media, output_dir=output_dir,
        )
        return report.to_dict()

    return router
