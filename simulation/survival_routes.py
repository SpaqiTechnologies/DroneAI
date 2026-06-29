"""FastAPI routes for the SAR, swarm SAR, and survival applications.

Exposed via ``build_router(state)`` so the routes can read/write the
existing ``SimulationState`` singleton without circular imports.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import asyncio
import base64
import json as _json

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from applications.search_rescue import (
    SARMission,
    SARState,
    SearchConfig,
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
from core.drone import Drone
from sensors.media import MediaStorage, encode_png, synthesize_rgb_bytes


# --------------------------- Pydantic models -------------------------------


class _LatLon(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class SARPlanRequest(BaseModel):
    center: _LatLon
    pattern_type: str = "expanding_square"
    altitude: float = 30.0
    track_spacing: float = 40.0
    speed: float = 8.0
    radius: float = 500.0
    legs: int = 8


class SARRunRequest(SARPlanRequest):
    return_to_start: bool = True
    photo_interval_s: float = 1.0


class SwarmSARRunRequest(BaseModel):
    drone_count: int = Field(..., ge=2, le=20)
    center: _LatLon
    radius_m: float = 500.0
    pattern_type: str = "sector"
    altitude: float = 30.0
    track_spacing: float = 40.0
    speed: float = 10.0
    return_to_start: bool = False


class BeaconSampleRequest(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    rssi_dbm: float


class SupplyDropRequest(BaseModel):
    target: _LatLon
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0
    release_altitude_agl_m: float = 30.0
    release_speed_mps: float = 8.0
    parachute_deploy_at_m: Optional[float] = None


class ThreatZoneModel(BaseModel):
    latitude: float
    longitude: float
    radius_m: float
    name: str = "threat"
    severity: float = 1.0


class CorridorRequest(BaseModel):
    start: _LatLon
    goal: _LatLon
    threats: List[ThreatZoneModel] = []
    margin_m: float = 25.0


# --------------------------- Background-task store ------------------------


class _MissionStore:
    """Hold running SAR / swarm-SAR missions keyed by id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sar: Dict[str, SARMission] = {}
        self._swarm: Dict[str, SwarmSARMission] = {}
        self._beacon = BeaconLocator()

    @property
    def beacon(self) -> BeaconLocator:
        return self._beacon

    def add_sar(self, mission: SARMission) -> str:
        mid = uuid.uuid4().hex[:12]
        with self._lock:
            self._sar[mid] = mission
        return mid

    def add_swarm(self, mission: SwarmSARMission) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._swarm[sid] = mission
        return sid

    def get_sar(self, mid: str) -> Optional[SARMission]:
        with self._lock:
            return self._sar.get(mid)

    def get_swarm(self, sid: str) -> Optional[SwarmSARMission]:
        with self._lock:
            return self._swarm.get(sid)

    def list_sar(self) -> List[str]:
        with self._lock:
            return list(self._sar.keys())

    def list_swarm(self) -> List[str]:
        with self._lock:
            return list(self._swarm.keys())


_PATTERN_TYPES = {
    "expanding_square": SearchType.EXPANDING_SQUARE,
    "sector": SearchType.SECTOR,
    "parallel": SearchType.PARALLEL,
    "creeping_line": SearchType.CREEPING_LINE,
    "spiral": SearchType.SPIRAL,
    "grid": SearchType.GRID,
}


def _to_pattern_type(name: str) -> SearchType:
    p = _PATTERN_TYPES.get(name.lower())
    if p is None:
        raise HTTPException(status_code=400, detail=f"unknown pattern_type: {name}")
    return p


def _build_search_config(req: SARPlanRequest | SwarmSARRunRequest) -> SearchConfig:
    return SearchConfig(
        altitude=req.altitude,
        track_spacing=req.track_spacing,
        speed=req.speed,
        search_radius=getattr(req, "radius", getattr(req, "radius_m", 500.0)),
        legs=getattr(req, "legs", 8),
    )


# --------------------------- Router builder -------------------------------


def build_router(get_state) -> APIRouter:
    """``get_state`` is a 0-arg callable returning the global SimulationState."""

    router = APIRouter()
    store = _MissionStore()

    def _ensure_media_storage(drone: Drone) -> Optional[MediaStorage]:
        cam = getattr(drone, "camera_sensor", None)
        if cam is None:
            return None
        existing = getattr(cam, "media_storage", None)
        if existing is not None:
            return existing
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "flight_logs", "media",
        )
        storage = MediaStorage(root=root)
        cam._media_storage = storage
        if not cam._synth_resolution:
            cam._synth_resolution = (160, 90)
        return storage

    # ============================ SAR ============================

    @router.post("/api/sar/plan")
    def sar_plan(req: SARPlanRequest):
        from applications.search_rescue import SearchPatternGenerator
        gen = SearchPatternGenerator(config=_build_search_config(req))
        pattern = gen.generate(
            _to_pattern_type(req.pattern_type),
            (req.center.latitude, req.center.longitude),
        )
        return {
            "pattern_type": pattern.pattern_type.value,
            "waypoint_count": len(pattern.waypoints),
            "waypoints": [list(wp) for wp in pattern.waypoints],
            "total_distance_m": pattern.total_distance,
            "estimated_time_s": pattern.estimated_time,
            "coverage_probability": pattern.coverage_probability,
            "area_sqm": pattern.area_sqm,
        }

    @router.post("/api/sar/run")
    def sar_run(req: SARRunRequest):
        state = get_state()
        drone = state.drone
        if drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        _ensure_media_storage(drone)
        try:
            drone.camera_sensor.start()
        except Exception:
            pass
        sar = SARMission(
            drone=drone,
            pattern_type=_to_pattern_type(req.pattern_type),
            center=(req.center.latitude, req.center.longitude),
            config=_build_search_config(req),
            cruise_speed_mps=req.speed * 2.0,
            photo_interval_s=req.photo_interval_s,
            return_to_start=req.return_to_start,
        )
        sar.run_in_background(tick_hz=20.0)
        mid = store.add_sar(sar)
        return {"mission_id": mid, "state": sar.state.value}

    @router.get("/api/sar/{mission_id}/report")
    def sar_report(mission_id: str):
        sar = store.get_sar(mission_id)
        if sar is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return sar.report().to_dict()

    @router.post("/api/sar/{mission_id}/abort")
    def sar_abort(mission_id: str):
        sar = store.get_sar(mission_id)
        if sar is None:
            raise HTTPException(status_code=404, detail="mission not found")
        sar.abort("api abort")
        return {"mission_id": mission_id, "state": sar.state.value}

    @router.get("/api/sar")
    def sar_list():
        return {"mission_ids": store.list_sar()}

    # ========================== Swarm SAR ==========================

    @router.post("/api/swarm-sar/run")
    def swarm_sar_run(req: SwarmSARRunRequest):
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        drones: Dict[str, Drone] = {}
        for i in range(req.drone_count):
            d = Drone()
            d.current_position = (req.center.latitude, req.center.longitude)
            d.current_altitude = 0.0
            try:
                d.camera_sensor.start()
            except Exception:
                pass
            drones[f"drone_{i}"] = d

        swarm = SwarmSARMission(
            drones=drones,
            center=(req.center.latitude, req.center.longitude),
            radius_m=req.radius_m,
            config=_build_search_config(req),
            pattern_type=_to_pattern_type(req.pattern_type),
            return_to_start=req.return_to_start,
        )
        swarm.run_in_background(tick_hz=20.0)
        sid = store.add_swarm(swarm)
        return {
            "swarm_id": sid,
            "drone_ids": swarm.drone_ids,
            "drone_count": len(drones),
        }

    @router.get("/api/swarm-sar/{swarm_id}/report")
    def swarm_sar_report(swarm_id: str):
        swarm = store.get_swarm(swarm_id)
        if swarm is None:
            raise HTTPException(status_code=404, detail="swarm mission not found")
        return swarm.report().to_dict()

    @router.post("/api/swarm-sar/{swarm_id}/abort")
    def swarm_sar_abort(swarm_id: str):
        swarm = store.get_swarm(swarm_id)
        if swarm is None:
            raise HTTPException(status_code=404, detail="swarm mission not found")
        swarm.abort("api abort")
        return {"swarm_id": swarm_id}

    @router.get("/api/swarm-sar")
    def swarm_sar_list():
        return {"swarm_ids": store.list_swarm()}

    # ========================= Beacon =============================

    @router.post("/api/survival/beacon/sample")
    def beacon_sample(req: BeaconSampleRequest):
        reading = store.beacon.add_reading(
            latitude=req.latitude,
            longitude=req.longitude,
            altitude=req.altitude,
            rssi_dbm=req.rssi_dbm,
        )
        return {
            "sample_count": store.beacon.sample_count,
            "reading": reading.to_dict(),
        }

    @router.get("/api/survival/beacon/fix")
    def beacon_fix():
        fix = store.beacon.compute_fix()
        if fix is None:
            return {"fix": None, "sample_count": 0}
        return {"fix": fix.to_dict(), "sample_count": store.beacon.sample_count}

    @router.post("/api/survival/beacon/reset")
    def beacon_reset():
        store.beacon.reset()
        return {"sample_count": 0}

    # ========================= Supply drop ========================

    @router.post("/api/survival/supply-drop/plan")
    def supply_drop_plan(req: SupplyDropRequest):
        params = DropParameters(
            release_altitude_agl_m=req.release_altitude_agl_m,
            release_speed_mps=req.release_speed_mps,
            parachute_deploy_at_m=req.parachute_deploy_at_m,
        )
        plan = SupplyDropPlanner(params).plan(
            target_lat=req.target.latitude,
            target_lon=req.target.longitude,
            wind_speed_mps=req.wind_speed_mps,
            wind_direction_deg=req.wind_direction_deg,
        )
        return plan.to_dict()

    # ========================= Safe corridor ======================

    @router.post("/api/survival/corridor/plan")
    def corridor_plan(req: CorridorRequest):
        threats = [
            ThreatZone(
                latitude=t.latitude,
                longitude=t.longitude,
                radius_m=t.radius_m,
                name=t.name,
                severity=t.severity,
            )
            for t in req.threats
        ]
        planner = SafeCorridorPlanner(threats=threats, margin_m=req.margin_m)
        corridor = planner.plan(
            start=(req.start.latitude, req.start.longitude),
            goal=(req.goal.latitude, req.goal.longitude),
        )
        return corridor.to_dict()

    # =========================== Media ============================

    @router.post("/api/media/snapshot")
    def media_snapshot():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        storage = _ensure_media_storage(state.drone)
        try:
            state.drone.camera_sensor.start()
        except Exception:
            pass
        ok, msg = state.drone.take_snapshot()
        arts = storage.list_artifacts(kind="photo") if storage else []
        latest = arts[-1].to_dict() if arts else None
        return {"success": ok, "message": msg, "artifact": latest}

    @router.post("/api/media/recording/start")
    def media_recording_start():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        _ensure_media_storage(state.drone)
        try:
            state.drone.camera_sensor.start()
            state.drone.camera_sensor.start_streaming()
        except Exception:
            pass
        ok, msg = state.drone.start_recording()
        return {"success": ok, "message": msg}

    @router.post("/api/media/recording/stop")
    def media_recording_stop():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        ok, msg = state.drone.stop_recording()
        summary = state.drone.camera_sensor.get_last_recording_summary()
        try:
            state.drone.camera_sensor.stop_streaming()
        except Exception:
            pass
        return {"success": ok, "message": msg, "summary": summary}

    def _capture_live_png(drone: Drone) -> Optional[Tuple[bytes, int, int]]:
        cam = getattr(drone, "camera_sensor", None)
        if cam is None:
            return None
        try:
            cam.start()
        except Exception:
            pass
        frame = cam.capture_frame()
        if frame is None:
            return None
        sw, sh = getattr(cam, "_synth_resolution", (160, 90))
        det_boxes = cam._scaled_detection_boxes(frame, sw, sh)
        rgb = synthesize_rgb_bytes(
            sw, sh,
            vision_mode=frame.mode.value,
            detections=det_boxes,
            timestamp=frame.timestamp,
        )
        return encode_png(rgb, sw, sh), sw, sh

    @router.get("/api/media/latest.png")
    def media_latest_png():
        """Synthesize the camera's current frame and return as a PNG image."""
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        result = _capture_live_png(state.drone)
        if result is None:
            raise HTTPException(status_code=503, detail="no frame available")
        png_bytes, _, _ = result
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/api/media/stream")
    def media_stream(fps: float = 5.0, max_frames: int = 0):
        """Server-sent events: emit base64 PNG frames at ``fps``.

        Streams until the client disconnects, or until ``max_frames`` have
        been emitted if ``max_frames > 0``. Browsers will set ``max_frames=0``
        for a live view; tests use a small cap to ensure a clean exit.
        """
        if fps <= 0 or fps > 30:
            raise HTTPException(status_code=400, detail="fps must be in (0, 30]")
        if max_frames < 0:
            raise HTTPException(status_code=400, detail="max_frames must be >= 0")
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")

        drone = state.drone
        interval = 1.0 / float(fps)
        cap = int(max_frames)

        async def event_source():
            seq = 0
            try:
                while True:
                    result = _capture_live_png(drone)
                    if result is None:
                        yield "event: error\ndata: no frame\n\n"
                    else:
                        png_bytes, w, h = result
                        seq += 1
                        payload = {
                            "seq": seq,
                            "width": w,
                            "height": h,
                            "png_base64": base64.b64encode(png_bytes).decode("ascii"),
                            "ts": time.time(),
                        }
                        yield f"event: frame\ndata: {_json.dumps(payload)}\n\n"
                        if cap and seq >= cap:
                            return
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @router.get("/api/media")
    def media_list():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        storage = _ensure_media_storage(state.drone)
        if storage is None:
            return {"photos": [], "videos": []}
        return {
            "photos": [a.to_dict() for a in storage.list_artifacts(kind="photo")],
            "videos": [a.to_dict() for a in storage.list_artifacts(kind="video")],
            "total_bytes": storage.total_bytes(),
            "root": storage.root,
        }

    return router
