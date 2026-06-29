"""Wire a ``DockingStation``'s deploy_hook into ``AutonomyRuntime``.

When the dock fires ``deploy(payload)`` (either manually or via the
``PatrolScheduler``), this adapter:

  1. Spins up a fresh ``AutonomyRuntime`` for the drone.
  2. Hands it a ``cruise_tick`` that flies a configurable patrol pattern
     (radius, lap count, optional waypoints from the payload).
  3. On cruise complete, snaps the drone back to the dock coordinates
     so the dock's own tick sees it land on the pad → CHARGING.

The dock keeps owning state ("DEPLOYED" while in flight, "RECALLING"
when battery low). The adapter exits cleanly when the runtime finishes;
each deployment gets its own runtime instance.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from core.autonomy_runtime import AutonomyRuntime, RuntimePhase
from core.landing import LandingMode

if TYPE_CHECKING:
    from core.docking.docking_station import DockingStation
    from core.drone import Drone


@dataclass
class PatrolPath:
    """Where the drone should fly while deployed.

    If ``waypoints`` is provided (lat, lon, alt triples), the drone
    visits them in order. Otherwise it flies a single circular lap of
    radius ``radius_m`` around the dock at ``altitude_m``.
    """
    altitude_m: float = 20.0
    radius_m: float = 50.0
    lap_count: int = 1
    points_per_lap: int = 12
    cruise_speed_mps: float = 8.0
    waypoints: Optional[List[Tuple[float, float, float]]] = None


def _generate_orbit_waypoints(
    center: Tuple[float, float],
    altitude_m: float,
    radius_m: float,
    points: int,
    laps: int,
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    lat0 = center[0]
    cos_lat = max(0.01, math.cos(math.radians(lat0)))
    for _ in range(max(1, laps)):
        for i in range(max(3, points)):
            theta = 2 * math.pi * i / points
            d_north = radius_m * math.cos(theta)
            d_east = radius_m * math.sin(theta)
            dlat = (d_north / 6371000.0) * (180.0 / math.pi)
            dlon = (d_east / (6371000.0 * cos_lat)) * (180.0 / math.pi)
            out.append((lat0 + dlat, center[1] + dlon, altitude_m))
    return out


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


class DockAutonomyAdapter:
    """Bridges a DockingStation to the AutonomyRuntime for unattended flight."""

    DEFAULT_TICK_HZ = 20.0

    def __init__(
        self,
        dock: "DockingStation",
        drone: "Drone",
        default_path: Optional[PatrolPath] = None,
        landing_mode: LandingMode = LandingMode.EMERGENCY,
        tick_hz: float = DEFAULT_TICK_HZ,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        self._dock = dock
        self._drone = drone
        self._default_path = default_path or PatrolPath()
        self._landing_mode = landing_mode
        self._tick_hz = float(tick_hz)
        self._on_complete = on_complete
        self._lock = threading.Lock()
        self._current_runtime: Optional[AutonomyRuntime] = None
        self._current_payload: Optional[Dict[str, Any]] = None
        self._last_runtime_status: Optional[Dict[str, Any]] = None
        self._deployments: int = 0
        self._completions: int = 0
        self._aborts: int = 0
        self._attached = False

    @property
    def current_runtime(self) -> Optional[AutonomyRuntime]:
        with self._lock:
            return self._current_runtime

    @property
    def deployments(self) -> int:
        with self._lock:
            return self._deployments

    @property
    def completions(self) -> int:
        with self._lock:
            return self._completions

    @property
    def aborts(self) -> int:
        with self._lock:
            return self._aborts

    def last_runtime_status(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_runtime_status) if self._last_runtime_status else None

    def attach(self) -> None:
        if self._attached:
            return
        self._dock.set_deploy_hook(self._on_deploy)
        self._dock.set_recall_hook(self._on_recall)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self._dock.set_deploy_hook(lambda payload: None)
        self._dock.set_recall_hook(lambda: None)
        self._attached = False

    # ------------------------------ internals -----------------------------

    def _resolve_path(self, payload: Dict[str, Any]) -> PatrolPath:
        return PatrolPath(
            altitude_m=float(payload.get("altitude_m", self._default_path.altitude_m)),
            radius_m=float(payload.get("radius_m", self._default_path.radius_m)),
            lap_count=int(payload.get("lap_count", self._default_path.lap_count)),
            points_per_lap=int(payload.get("points_per_lap", self._default_path.points_per_lap)),
            cruise_speed_mps=float(payload.get("cruise_speed_mps", self._default_path.cruise_speed_mps)),
            waypoints=payload.get("waypoints") or self._default_path.waypoints,
        )

    def _on_deploy(self, payload: Dict[str, Any]) -> None:
        path = self._resolve_path(payload)
        center = (self._dock.position[0], self._dock.position[1])
        if path.waypoints:
            waypoints = list(path.waypoints)
        else:
            waypoints = _generate_orbit_waypoints(
                center=center,
                altitude_m=path.altitude_m,
                radius_m=path.radius_m,
                points=path.points_per_lap,
                laps=path.lap_count,
            )
        cruise_state = {"index": 0, "wps": waypoints, "speed": path.cruise_speed_mps}

        def cruise_tick(dt: float, rt: AutonomyRuntime) -> bool:
            idx = cruise_state["index"]
            wps = cruise_state["wps"]
            if idx >= len(wps):
                # Snap drone home so the dock-tick lands it on the pad.
                self._move_drone((center[0], center[1], path.altitude_m))
                return True
            target = wps[idx]
            pos = (self._drone.current_position[0], self._drone.current_position[1])
            dist = _haversine_m(pos, (target[0], target[1]))
            step = max(0.0, cruise_state["speed"] * dt)
            if step >= dist or dist < 1.0:
                self._move_drone(target)
                cruise_state["index"] = idx + 1
            else:
                frac = step / max(1.0, dist)
                new_lat = pos[0] + (target[0] - pos[0]) * frac
                new_lon = pos[1] + (target[1] - pos[1]) * frac
                cur_alt = float(getattr(self._drone, "current_altitude", target[2]))
                new_alt = cur_alt + (target[2] - cur_alt) * frac
                self._move_drone((new_lat, new_lon, new_alt))
            return False

        runtime = AutonomyRuntime(drone=self._drone, tick_hz=self._tick_hz)

        def on_phase(phase: RuntimePhase, status) -> None:
            if phase in (RuntimePhase.COMPLETED, RuntimePhase.ABORTED, RuntimePhase.FAILED):
                self._move_drone((center[0], center[1], self._dock.position[2]))
                self._drone._is_flying = False  # type: ignore[attr-defined]
                with self._lock:
                    if phase == RuntimePhase.COMPLETED:
                        self._completions += 1
                    else:
                        self._aborts += 1
                    self._last_runtime_status = status.to_dict()
                if self._on_complete is not None:
                    try:
                        self._on_complete(status.to_dict())
                    except Exception:
                        pass

        runtime.add_phase_callback(on_phase)
        with self._lock:
            self._current_runtime = runtime
            self._current_payload = dict(payload)
            self._deployments += 1
        runtime.start_flight(
            target_altitude=path.altitude_m,
            cruise_tick=cruise_tick,
            landing_mode=self._landing_mode,
        )

    def _on_recall(self) -> None:
        with self._lock:
            rt = self._current_runtime
        if rt is not None and rt.is_running:
            rt.abort("dock recall")

    def _move_drone(self, position: Tuple[float, float, float]) -> None:
        try:
            self._drone.current_position = (position[0], position[1])
            self._drone.current_altitude = float(position[2])
        except Exception:
            pass
