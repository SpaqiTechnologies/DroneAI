"""FastAPI routes for the unified ``AutonomyRuntime``.

Exposed via ``build_router(get_state)`` so the routes can read/write the
existing ``SimulationState`` singleton without circular imports.

Only one runtime is active per drone at a time (since the singleton drone
state is shared); attempting to start a second concurrent runtime returns
HTTP 409.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.autonomy_runtime import AutonomyRuntime, RuntimePhase, RuntimeStatus
from core.landing import LandingMode

try:
    from core.takeoff.takeoff_manager import TakeoffManager, TakeoffMode
except Exception:  # pragma: no cover
    TakeoffManager = None  # type: ignore
    TakeoffMode = None  # type: ignore


class AutonomyRunRequest(BaseModel):
    """Start a takeoff → cruise → land sequence on the active drone."""

    target_altitude: float = Field(10.0, ge=2.0, le=400.0)
    landing_mode: str = "normal"
    takeoff_mode: str = "normal"
    use_takeoff_manager: bool = True
    cruise_seconds: float = Field(
        5.0, ge=0.0, le=600.0,
        description="When no mission is supplied, hover for this many seconds before landing.",
    )
    mission_id: Optional[str] = None
    tick_hz: float = Field(10.0, gt=0.0, le=100.0)


_LANDING_MODES = {
    "normal":         LandingMode.NORMAL,
    "precision":      LandingMode.PRECISION,
    "emergency":      LandingMode.EMERGENCY,
    "terrain_follow": LandingMode.TERRAIN_FOLLOW,
    "safe_zone":      LandingMode.SAFE_ZONE,
}


def _resolve_takeoff_mode(name: str):
    if TakeoffMode is None:
        return None
    try:
        return TakeoffMode[name.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown takeoff_mode: {name}")


class _RuntimeStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: Dict[str, AutonomyRuntime] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def add(self, runtime: AutonomyRuntime, meta: Dict[str, Any]) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self._runtimes[rid] = runtime
            self._meta[rid] = meta
        return rid

    def get(self, rid: str) -> Optional[AutonomyRuntime]:
        with self._lock:
            return self._runtimes.get(rid)

    def get_meta(self, rid: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._meta.get(rid, {}))

    def active_for_drone(self, drone) -> Optional[Tuple[str, AutonomyRuntime]]:
        with self._lock:
            for rid, rt in self._runtimes.items():
                if rt.is_running and self._meta.get(rid, {}).get("drone_id") == id(drone):
                    return rid, rt
        return None

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._runtimes.keys())


def build_router(get_state) -> APIRouter:
    router = APIRouter()
    store = _RuntimeStore()

    def _get_drone():
        state = get_state()
        if state.drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")
        return state, state.drone

    @router.post("/api/autonomy/run")
    def autonomy_run(req: AutonomyRunRequest):
        state, drone = _get_drone()

        existing = store.active_for_drone(drone)
        if existing is not None:
            rid_existing, rt_existing = existing
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "another autonomy runtime is already active for this drone",
                    "runtime_id": rid_existing,
                    "phase": rt_existing.phase.value,
                },
            )

        landing = _LANDING_MODES.get(req.landing_mode.lower())
        if landing is None:
            raise HTTPException(
                status_code=400, detail=f"unknown landing_mode: {req.landing_mode}"
            )
        takeoff_mode = _resolve_takeoff_mode(req.takeoff_mode) if req.use_takeoff_manager else None

        takeoff_mgr = None
        if req.use_takeoff_manager and TakeoffManager is not None:
            takeoff_mgr = getattr(state, "takeoff_manager", None) or TakeoffManager(drone)
            state.takeoff_manager = takeoff_mgr

        mission_mgr = None
        if req.mission_id:
            mm = getattr(state, "mission_manager", None)
            if mm is None:
                raise HTTPException(
                    status_code=400,
                    detail="state has no mission_manager attached for mission_id",
                )
            mission_mgr = mm

        runtime = AutonomyRuntime(
            drone=drone,
            takeoff_manager=takeoff_mgr,
            mission_manager=mission_mgr,
            tick_hz=req.tick_hz,
        )

        cruise_tick = None
        if mission_mgr is None:
            elapsed = {"t": 0.0}
            duration = float(req.cruise_seconds)

            def cruise_tick_fn(dt, rt):
                elapsed["t"] += dt
                return elapsed["t"] >= duration

            cruise_tick = cruise_tick_fn

        ok = runtime.start_flight(
            target_altitude=req.target_altitude,
            cruise_tick=cruise_tick,
            landing_mode=landing,
            takeoff_mode=takeoff_mode,
        )
        if not ok:
            raise HTTPException(status_code=500, detail="failed to start runtime")
        meta = {
            "drone_id": id(drone),
            "request": req.model_dump(),
        }
        rid = store.add(runtime, meta)
        return {
            "runtime_id": rid,
            "phase": runtime.phase.value,
            "status": runtime.status().to_dict(),
        }

    @router.get("/api/autonomy/{runtime_id}/status")
    def autonomy_status(runtime_id: str):
        rt = store.get(runtime_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="runtime not found")
        return {
            "runtime_id": runtime_id,
            "is_running": rt.is_running,
            "status": rt.status().to_dict(),
            "meta": store.get_meta(runtime_id),
        }

    @router.post("/api/autonomy/{runtime_id}/abort")
    def autonomy_abort(runtime_id: str):
        rt = store.get(runtime_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="runtime not found")
        rt.abort("api abort")
        rt.join(timeout=5.0)
        return {
            "runtime_id": runtime_id,
            "phase": rt.phase.value,
            "status": rt.status().to_dict(),
        }

    @router.get("/api/autonomy")
    def autonomy_list():
        ids = store.list_ids()
        return {
            "runtime_ids": ids,
            "active": [
                rid for rid in ids
                if (rt := store.get(rid)) is not None and rt.is_running
            ],
        }

    return router
