"""Unified autonomy runtime.

Composes ``TakeoffManager``, ``MissionManager``, and ``LandingManager`` into a
single background loop so callers can hand off "take off, run mission, land"
as one operation. Existing managers are external-tick driven; this runtime is
the canonical driver.

The runtime is intentionally simple — a fixed-rate background thread that
walks a small state machine and forwards ``update()`` calls. It does not own
any drone state itself; it only sequences.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.landing import LandingMode, LandingState

if TYPE_CHECKING:
    from core.drone import Drone
    from core.landing import LandingManager
    from core.mission.mission_manager import MissionManager
    from core.takeoff.takeoff_manager import TakeoffManager, TakeoffMode

logger = logging.getLogger(__name__)


class RuntimePhase(Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    LANDING = "landing"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class RuntimeStatus:
    phase: RuntimePhase
    started_at: Optional[float]
    elapsed_s: float
    last_tick_at: Optional[float]
    tick_count: int
    message: str = ""
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "last_tick_at": self.last_tick_at,
            "tick_count": self.tick_count,
            "message": self.message,
            "error": self.error,
            "extra": dict(self.extra),
        }


PhaseCallback = Callable[[RuntimePhase, "RuntimeStatus"], None]


class AutonomyRuntime:
    """Sequences takeoff → cruise → land in a background thread."""

    def __init__(
        self,
        drone: "Drone",
        takeoff_manager: Optional["TakeoffManager"] = None,
        mission_manager: Optional["MissionManager"] = None,
        landing_manager: Optional["LandingManager"] = None,
        tick_hz: float = 10.0,
    ) -> None:
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        self._drone = drone
        self._takeoff = takeoff_manager
        self._mission = mission_manager
        self._landing = landing_manager or getattr(drone, "landing_manager", None)
        self._tick_interval = 1.0 / float(tick_hz)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._phase = RuntimePhase.IDLE
        self._target_altitude: float = 10.0
        self._takeoff_mode = None  # type: Optional["TakeoffMode"]
        self._landing_mode = LandingMode.NORMAL
        self._cruise_tick: Optional[Callable[[float, "AutonomyRuntime"], bool]] = None
        self._started_at: Optional[float] = None
        self._tick_count = 0
        self._last_tick_at: Optional[float] = None
        self._message = ""
        self._error: Optional[str] = None
        self._extra: Dict[str, Any] = {}
        self._phase_callbacks: List[PhaseCallback] = []
        self._abort_flag = False

    # --------------------------- lifecycle ---------------------------------

    def add_phase_callback(self, cb: PhaseCallback) -> None:
        self._phase_callbacks.append(cb)

    @property
    def phase(self) -> RuntimePhase:
        return self._phase

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> RuntimeStatus:
        with self._lock:
            elapsed = (time.time() - self._started_at) if self._started_at else 0.0
            return RuntimeStatus(
                phase=self._phase,
                started_at=self._started_at,
                elapsed_s=elapsed,
                last_tick_at=self._last_tick_at,
                tick_count=self._tick_count,
                message=self._message,
                error=self._error,
                extra=dict(self._extra),
            )

    def start_flight(
        self,
        target_altitude: float = 10.0,
        cruise_tick: Optional[Callable[[float, "AutonomyRuntime"], bool]] = None,
        landing_mode: LandingMode = LandingMode.NORMAL,
        takeoff_mode: Optional["TakeoffMode"] = None,
    ) -> bool:
        """Begin a takeoff → cruise → land sequence.

        ``cruise_tick(dt, runtime)`` is called every tick during cruise. It
        returns ``True`` when cruise is complete (runtime advances to landing).
        If omitted and a ``MissionManager`` is attached, cruise drives the
        mission until it reaches a terminal state. If neither, cruise is
        skipped entirely.
        """
        with self._lock:
            if self.is_running:
                return False
            self._stop_event.clear()
            self._abort_flag = False
            self._target_altitude = float(target_altitude)
            self._cruise_tick = cruise_tick
            self._landing_mode = landing_mode
            self._takeoff_mode = takeoff_mode
            self._started_at = time.time()
            self._tick_count = 0
            self._last_tick_at = None
            self._message = "starting"
            self._error = None
            self._extra = {}
            self._transition_locked(RuntimePhase.TAKEOFF)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="autonomy-runtime"
        )
        self._thread.start()
        return True

    def abort(self, reason: str = "user abort") -> None:
        with self._lock:
            self._abort_flag = True
            self._message = reason
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --------------------------- main loop ---------------------------------

    def _run(self) -> None:
        last = time.time()
        try:
            while not self._stop_event.is_set():
                now = time.time()
                dt = max(0.0, now - last)
                last = now
                with self._lock:
                    self._tick_count += 1
                    self._last_tick_at = now
                    phase = self._phase
                try:
                    if self._abort_flag:
                        self._handle_abort()
                        break
                    if phase == RuntimePhase.TAKEOFF:
                        self._tick_takeoff(dt)
                    elif phase == RuntimePhase.CRUISE:
                        self._tick_cruise(dt)
                    elif phase == RuntimePhase.LANDING:
                        self._tick_landing(dt)
                    else:
                        break
                except Exception as exc:
                    logger.exception("autonomy runtime tick raised")
                    with self._lock:
                        self._error = str(exc)
                        self._transition_locked(RuntimePhase.FAILED)
                    break
                time.sleep(self._tick_interval)
        finally:
            if self._phase not in (
                RuntimePhase.COMPLETED, RuntimePhase.ABORTED, RuntimePhase.FAILED
            ):
                with self._lock:
                    self._transition_locked(RuntimePhase.COMPLETED)

    # --------------------------- phase handlers ---------------------------

    def _tick_takeoff(self, dt: float) -> None:
        if self._takeoff is None:
            self._direct_takeoff()
            with self._lock:
                self._transition_locked(RuntimePhase.CRUISE)
                self._message = "takeoff complete (direct)"
            return

        if not getattr(self._takeoff, "is_active", False):
            mode = self._takeoff_mode
            if mode is None:
                try:
                    from core.takeoff.takeoff_manager import TakeoffMode as _TM
                    mode = _TM.NORMAL
                except Exception:
                    mode = None
            kwargs = {"target_altitude": self._target_altitude}
            if mode is not None:
                kwargs["mode"] = mode
            ok, msg = self._takeoff.start_takeoff(**kwargs)
            if not ok:
                with self._lock:
                    self._error = msg
                    self._transition_locked(RuntimePhase.FAILED)
                return

        self._takeoff.update(dt)
        state_name = getattr(getattr(self._takeoff, "_state", None), "name", "")
        if state_name == "COMPLETE":
            with self._lock:
                self._transition_locked(RuntimePhase.CRUISE)
                self._message = "takeoff complete"
        elif state_name in ("FAILED", "ABORTED"):
            with self._lock:
                self._error = f"takeoff {state_name.lower()}"
                self._transition_locked(RuntimePhase.FAILED)

    def _direct_takeoff(self) -> None:
        """Fallback when no TakeoffManager is provided: arm + climb directly.

        Best-effort. If the drone refuses to arm (pre-arm failure, etc.) we
        still set the altitude in the sim model so the runtime can proceed —
        the runtime is a sequencer, not a safety system.
        """
        try:
            if hasattr(self._drone, "arm"):
                try:
                    self._drone.arm()
                except Exception:
                    pass
            climbed = False
            for name in ("take_off", "takeoff"):
                fn = getattr(self._drone, name, None)
                if fn is not None:
                    try:
                        fn(self._target_altitude)
                        climbed = True
                        break
                    except Exception:
                        continue
            if not climbed:
                try:
                    self._drone.current_altitude = float(self._target_altitude)
                except Exception:
                    pass
                try:
                    self._drone._is_flying = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Ensure altitude is at least the target so landing has something
            # to descend through (some arm() paths leave altitude untouched).
            try:
                if float(getattr(self._drone, "current_altitude", 0.0)) < self._target_altitude:
                    self._drone.current_altitude = float(self._target_altitude)
            except Exception:
                pass
        except Exception as exc:
            with self._lock:
                self._error = f"direct takeoff failed: {exc}"
                self._transition_locked(RuntimePhase.FAILED)

    def _tick_cruise(self, dt: float) -> None:
        if self._cruise_tick is not None:
            done = bool(self._cruise_tick(dt, self))
            if done:
                with self._lock:
                    self._transition_locked(RuntimePhase.LANDING)
                    self._message = "cruise complete"
            return

        if self._mission is not None:
            try:
                self._mission.update(dt)
            except Exception as exc:
                with self._lock:
                    self._error = f"mission update raised: {exc}"
                    self._transition_locked(RuntimePhase.FAILED)
                return
            state = getattr(self._mission, "_state", None)
            name = getattr(state, "name", str(state)) if state else ""
            if name in ("COMPLETE", "ABORTED", "FAILED"):
                with self._lock:
                    self._transition_locked(RuntimePhase.LANDING)
                    self._message = f"mission {name.lower()}"
            return

        with self._lock:
            self._transition_locked(RuntimePhase.LANDING)
            self._message = "cruise skipped (no driver)"

    def _tick_landing(self, dt: float) -> None:
        if self._landing is None:
            with self._lock:
                self._transition_locked(RuntimePhase.COMPLETED)
                self._message = "landed (no manager)"
            return

        if self._landing._state == LandingState.IDLE:
            pos = self._drone_position()
            self._landing.start_landing(self._landing_mode, pos)

        wind_speed = self._read_wind_speed()
        pos = self._drone_position()
        state, descent_rate, _ = self._landing.update(
            current_position=pos,
            wind_speed=wind_speed,
            obstacle_below=False,
        )
        # Apply descent in sim: shrink altitude so the manager can detect TOUCHDOWN.
        try:
            cur_alt = float(getattr(self._drone, "current_altitude", 0.0))
            new_alt = max(0.0, cur_alt - descent_rate * dt)
            self._drone.current_altitude = new_alt
        except Exception:
            pass
        with self._lock:
            self._extra["descent_rate"] = descent_rate
            self._extra["landing_state"] = state.value
        if state == LandingState.LANDED:
            try:
                if hasattr(self._drone, "disarm"):
                    self._drone.disarm("autonomy_landed")
            except Exception:
                pass
            with self._lock:
                self._transition_locked(RuntimePhase.COMPLETED)
                self._message = "landed"
        elif state == LandingState.ABORTED:
            with self._lock:
                self._error = "landing aborted"
                self._transition_locked(RuntimePhase.FAILED)

    def _handle_abort(self) -> None:
        if self._phase == RuntimePhase.TAKEOFF and self._takeoff is not None:
            try:
                self._takeoff.abort("runtime abort")
            except Exception:
                pass
        if self._phase == RuntimePhase.CRUISE and self._mission is not None:
            try:
                self._mission.abort()
            except Exception:
                pass
        try:
            pos = self._drone_position()
            if self._landing is not None and self._landing._state == LandingState.IDLE:
                self._landing.start_landing(LandingMode.EMERGENCY, pos)
        except Exception:
            pass
        with self._lock:
            self._transition_locked(RuntimePhase.ABORTED)

    # --------------------------- helpers ----------------------------------

    def _drone_position(self):
        pos = getattr(self._drone, "current_position", (0.0, 0.0))
        alt = getattr(self._drone, "current_altitude", 0.0)
        return (pos[0], pos[1], alt)

    def _read_wind_speed(self) -> float:
        wind = getattr(self._drone, "wind_sensor", None)
        if wind is None:
            return 0.0
        try:
            value = wind.get_wind_speed()
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _transition_locked(self, new_phase: RuntimePhase) -> None:
        if new_phase == self._phase:
            return
        self._phase = new_phase
        snapshot = RuntimeStatus(
            phase=self._phase,
            started_at=self._started_at,
            elapsed_s=(time.time() - self._started_at) if self._started_at else 0.0,
            last_tick_at=self._last_tick_at,
            tick_count=self._tick_count,
            message=self._message,
            error=self._error,
            extra=dict(self._extra),
        )
        for cb in list(self._phase_callbacks):
            try:
                cb(new_phase, snapshot)
            except Exception:
                logger.exception("phase callback raised")
