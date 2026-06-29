"""Docking station: drone-in-a-box state machine + auto-charging.

A ``DockingStation`` owns a single ``Drone`` and models the box that
holds it between missions: precision-land on the dock, auto-charge,
hold ready, deploy on command (or by schedule), recall when low
battery or signal lost.

Pure-sim by default; on a real rig you'd wire the charge tick to the
charger telemetry and the deploy hook to the autonomy runtime.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.drone import Drone


class DockState(Enum):
    EMPTY = "empty"                  # No drone at the dock
    DOCKED = "docked"                # Drone parked, idle
    CHARGING = "charging"            # Drone parked + charging
    READY = "ready"                  # Charged above launch threshold, ready to fly
    DEPLOYED = "deployed"            # Drone airborne, dock empty
    RECALLING = "recalling"          # Recall requested, drone returning
    ERROR = "error"


@dataclass
class ChargeProfile:
    """Linear charge approximation. Real chargers are CC/CV; this is fine for sim."""
    rate_pct_per_s: float = 0.5           # 0.5%/s → ~3.3 min to 100%
    target_pct: float = 95.0              # stop charging at this level
    launch_min_pct: float = 80.0          # need at least this much before launch
    recall_pct: float = 25.0              # auto-recall at this battery


@dataclass
class DockStatus:
    state: DockState
    battery_pct: float
    is_charging: bool
    elapsed_in_state_s: float
    last_event_at: float
    drone_at_dock: bool
    deployments: int
    recalls: int
    auto_recall_armed: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "battery_pct": self.battery_pct,
            "is_charging": self.is_charging,
            "elapsed_in_state_s": self.elapsed_in_state_s,
            "last_event_at": self.last_event_at,
            "drone_at_dock": self.drone_at_dock,
            "deployments": self.deployments,
            "recalls": self.recalls,
            "auto_recall_armed": self.auto_recall_armed,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class DockingStation:
    """Background-driven docking station that auto-charges a single drone.

    The dock owns the drone, monitors its position, charges it when on the pad,
    and exposes ``deploy(...)`` / ``recall()`` so an external scheduler
    (e.g. :class:`PatrolScheduler`) can run unattended missions.
    """

    def __init__(
        self,
        drone: "Drone",
        latitude: float,
        longitude: float,
        altitude_msl_m: float = 0.0,
        charge_profile: Optional[ChargeProfile] = None,
        tick_hz: float = 5.0,
        pad_radius_m: float = 1.0,
        auto_recall: bool = True,
    ) -> None:
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        self._drone = drone
        self._lat = float(latitude)
        self._lon = float(longitude)
        self._alt = float(altitude_msl_m)
        self._profile = charge_profile or ChargeProfile()
        self._tick_interval = 1.0 / float(tick_hz)
        self._pad_radius = float(pad_radius_m)
        self._auto_recall = bool(auto_recall)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Snap drone to the pad on startup if it's not flying.
        if not getattr(drone, "_is_flying", False):
            try:
                drone.current_position = (self._lat, self._lon)
                drone.current_altitude = self._alt
            except Exception:
                pass

        self._state = DockState.DOCKED if self._drone_at_pad() else DockState.EMPTY
        self._state_entered_at = time.time()
        self._last_event_at = time.time()
        self._deployments = 0
        self._recalls = 0
        self._error: Optional[str] = None
        self._listeners: List[Callable[[DockStatus], None]] = []
        self._on_deploy_hook: Optional[Callable[[Dict[str, Any]], None]] = None
        self._on_recall_hook: Optional[Callable[[], None]] = None

    # ------------------------------ properties ----------------------------

    @property
    def position(self) -> Tuple[float, float, float]:
        return self._lat, self._lon, self._alt

    @property
    def state(self) -> DockState:
        with self._lock:
            return self._state

    def status(self) -> DockStatus:
        with self._lock:
            return DockStatus(
                state=self._state,
                battery_pct=float(getattr(self._drone, "battery_level", 0.0)),
                is_charging=(self._state == DockState.CHARGING),
                elapsed_in_state_s=time.time() - self._state_entered_at,
                last_event_at=self._last_event_at,
                drone_at_dock=self._drone_at_pad(),
                deployments=self._deployments,
                recalls=self._recalls,
                auto_recall_armed=self._auto_recall,
                error=self._error,
            )

    def add_listener(self, cb: Callable[[DockStatus], None]) -> None:
        self._listeners.append(cb)

    def set_deploy_hook(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Called when ``deploy(payload)`` fires. Payload is the deploy kwargs.

        Use this to plug in the autonomy runtime, mission manager, or any
        other launcher. Hook runs in the background tick thread.
        """
        self._on_deploy_hook = hook

    def set_recall_hook(self, hook: Callable[[], None]) -> None:
        """Called when a recall is initiated (battery, signal, manual)."""
        self._on_recall_hook = hook

    # ----------------------------- lifecycle ------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="dock-tick",
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----------------------------- commands -------------------------------

    def deploy(self, **payload) -> Tuple[bool, str]:
        """Authorise a deployment if state allows. Returns (ok, message)."""
        with self._lock:
            if self._state not in (DockState.READY, DockState.DOCKED, DockState.CHARGING):
                return False, f"cannot deploy in state: {self._state.value}"
            battery = float(getattr(self._drone, "battery_level", 0.0))
            if battery < self._profile.launch_min_pct:
                return False, (
                    f"battery {battery:.1f}% below launch threshold "
                    f"{self._profile.launch_min_pct:.0f}%"
                )
            self._transition_locked(DockState.DEPLOYED)
            self._deployments += 1
            payload = dict(payload)

        if self._on_deploy_hook is not None:
            try:
                self._on_deploy_hook(payload)
            except Exception as exc:
                with self._lock:
                    self._error = f"deploy hook failed: {exc}"
                return False, f"deploy hook failed: {exc}"
        return True, "deployed"

    def recall(self, reason: str = "manual") -> None:
        with self._lock:
            if self._state in (DockState.DEPLOYED, DockState.RECALLING):
                self._transition_locked(DockState.RECALLING)
                self._recalls += 1
                self._error = None
        if self._on_recall_hook is not None:
            try:
                self._on_recall_hook()
            except Exception as exc:
                with self._lock:
                    self._error = f"recall hook failed: {exc}"

    # ----------------------------- internals ------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                with self._lock:
                    self._error = f"tick error: {exc}"
                    self._transition_locked(DockState.ERROR)
            time.sleep(self._tick_interval)

    def _tick(self) -> None:
        battery = float(getattr(self._drone, "battery_level", 0.0))
        at_pad = self._drone_at_pad()
        flying = bool(getattr(self._drone, "_is_flying", False))

        with self._lock:
            state = self._state

            if state == DockState.DEPLOYED:
                if self._auto_recall and battery <= self._profile.recall_pct:
                    self._transition_locked(DockState.RECALLING)
                elif at_pad and not flying:
                    self._transition_locked(
                        DockState.CHARGING
                        if battery < self._profile.target_pct
                        else DockState.READY
                    )

            elif state == DockState.RECALLING:
                if at_pad and not flying:
                    self._transition_locked(
                        DockState.CHARGING
                        if battery < self._profile.target_pct
                        else DockState.READY
                    )

            elif state in (DockState.DOCKED, DockState.CHARGING, DockState.READY):
                if not at_pad or flying:
                    self._transition_locked(DockState.DEPLOYED)
                    return
                if battery < self._profile.target_pct:
                    self._charge_step_locked(self._tick_interval)
                    self._transition_locked(DockState.CHARGING)
                else:
                    self._transition_locked(DockState.READY)

            elif state == DockState.EMPTY:
                if at_pad and not flying:
                    self._transition_locked(DockState.DOCKED)

        for cb in list(self._listeners):
            try:
                cb(self.status())
            except Exception:
                pass

    def _charge_step_locked(self, dt: float) -> None:
        battery = float(getattr(self._drone, "battery_level", 0.0))
        new = min(self._profile.target_pct, battery + self._profile.rate_pct_per_s * dt)
        try:
            self._drone.battery_level = new
        except Exception:
            pass

    def _transition_locked(self, new_state: DockState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self._state_entered_at = time.time()
        self._last_event_at = self._state_entered_at

    def _drone_at_pad(self) -> bool:
        pos = getattr(self._drone, "current_position", None)
        alt = float(getattr(self._drone, "current_altitude", 0.0))
        if pos is None:
            return False
        from math import asin, cos, radians, sin, sqrt
        lat1, lon1 = radians(pos[0]), radians(pos[1])
        lat2, lon2 = radians(self._lat), radians(self._lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        dist = 2 * 6371000.0 * asin(sqrt(h))
        return dist <= self._pad_radius and abs(alt - self._alt) <= 1.0
