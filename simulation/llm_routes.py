"""FastAPI routes for LLM-driven mission planning.

These endpoints turn a natural-language string into a real
``core.mission.Mission`` (a sequence of waypoints + actions) and can
either return the plan, store it, or load it into ``MissionManager``
for execution.

The LLM call uses whatever ``CommandInterpreter`` backend is selected
(Ollama → Transformers → OpenAI → mock). When all real backends fail
the interpreter falls back to its rule-based regex parser.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai.llm.command_interpreter import CommandInterpreter, DroneCommand
from core.mission.mission_manager import MissionManager
from core.mission.mission_storage import MissionStorage


class _LatLon(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class LLMMissionRequest(BaseModel):
    """Natural language command, with optional planning defaults."""
    command: str = Field(..., min_length=1)
    home_position: Optional[_LatLon] = None
    default_altitude: float = 30.0
    default_speed: float = 5.0
    survey_width_m: float = 200.0
    survey_height_m: float = 200.0
    survey_spacing_m: float = 25.0
    orbit_radius_m: float = 30.0


class LLMMissionLoadRequest(LLMMissionRequest):
    """Plan + load the resulting mission into ``MissionManager``."""
    persist: bool = True
    start: bool = False


def _serialize_mission(m) -> Dict[str, Any]:
    return {
        "id": m.id,
        "name": m.name,
        "description": m.description,
        "mission_type": m.mission_type.value,
        "default_speed": m.default_speed,
        "default_altitude": m.default_altitude,
        "waypoint_count": len(m.waypoints),
        "waypoints": [
            {
                "sequence": wp.sequence,
                "lat": wp.latitude,
                "lon": wp.longitude,
                "alt": wp.altitude,
                "speed": wp.speed,
                "hold_time": wp.hold_time,
                "name": wp.name,
            }
            for wp in m.waypoints
        ],
        "metadata": m.metadata,
    }


def build_router(get_state) -> APIRouter:
    router = APIRouter()
    interpreter_lock = threading.Lock()
    interpreter: Optional[CommandInterpreter] = None

    def _get_interpreter() -> CommandInterpreter:
        nonlocal interpreter
        if interpreter is None:
            with interpreter_lock:
                if interpreter is None:
                    interpreter = CommandInterpreter()
        return interpreter

    def _resolve_home(req: LLMMissionRequest, state) -> Optional[Tuple[float, float]]:
        if req.home_position is not None:
            return req.home_position.latitude, req.home_position.longitude
        drone = getattr(state, "drone", None)
        if drone is not None and getattr(drone, "current_position", None):
            return tuple(drone.current_position[:2])  # type: ignore[return-value]
        return None

    @router.post("/api/llm/mission/plan")
    def llm_mission_plan(req: LLMMissionRequest):
        state = get_state()
        home = _resolve_home(req, state)
        interp = _get_interpreter()
        cmd, mission = interp.interpret_to_mission(
            req.command,
            home_position=home,
            default_altitude=req.default_altitude,
            default_speed=req.default_speed,
            survey_width_m=req.survey_width_m,
            survey_height_m=req.survey_height_m,
            survey_spacing_m=req.survey_spacing_m,
            orbit_radius_m=req.orbit_radius_m,
        )
        response: Dict[str, Any] = {
            "command": cmd.to_dict(),
            "mission": _serialize_mission(mission) if mission is not None else None,
            "convertible": mission is not None,
        }
        if mission is None:
            response["reason"] = (
                f"command_type={cmd.command_type.value} is single-shot; use the "
                f"direct command endpoint instead."
            )
        return response

    @router.post("/api/llm/mission/load")
    def llm_mission_load(req: LLMMissionLoadRequest):
        state = get_state()
        drone = getattr(state, "drone", None)
        if drone is None:
            raise HTTPException(status_code=503, detail="drone not initialized")

        home = _resolve_home(req, state)
        interp = _get_interpreter()
        cmd, mission = interp.interpret_to_mission(
            req.command,
            home_position=home,
            default_altitude=req.default_altitude,
            default_speed=req.default_speed,
            survey_width_m=req.survey_width_m,
            survey_height_m=req.survey_height_m,
            survey_spacing_m=req.survey_spacing_m,
            orbit_radius_m=req.orbit_radius_m,
        )
        if mission is None:
            raise HTTPException(
                status_code=400,
                detail=f"command '{cmd.command_type.value}' has no waypoint sequence",
            )

        mm: Optional[MissionManager] = getattr(state, "mission_manager", None)
        if mm is None:
            mm = MissionManager(drone)
            state.mission_manager = mm

        ok, msg = mm.load_mission(mission)
        if not ok:
            raise HTTPException(status_code=400, detail=f"load_mission failed: {msg}")

        saved_path: Optional[str] = None
        if req.persist:
            storage = getattr(state, "mission_storage", None) or MissionStorage()
            try:
                saved_path = storage.save(mission)
            except Exception as exc:  # pragma: no cover
                saved_path = f"save_failed: {exc}"

        started = False
        start_msg: Optional[str] = None
        if req.start:
            try:
                started, start_msg = mm.start()
            except Exception as exc:  # pragma: no cover
                started, start_msg = False, str(exc)

        return {
            "command": cmd.to_dict(),
            "mission": _serialize_mission(mission),
            "loaded": True,
            "saved_path": saved_path,
            "started": started,
            "start_message": start_msg,
        }

    return router
