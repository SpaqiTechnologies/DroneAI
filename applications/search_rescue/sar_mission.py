"""End-to-end Search-and-Rescue mission orchestrator.

Sequences a search pattern across waypoints, captures imagery at each
waypoint, applies a detector hook (defaults to the camera's existing
detection list), and accumulates a target log. Drives the drone in
simulation by interpolating its position along the pattern at a chosen
cruise speed.

This is the SAR "mission application" — it is independent from the
``AutonomyRuntime`` so a swarm coordinator can run N missions in parallel
without each spawning its own runtime thread.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from applications.search_rescue.search_pattern import (
    SearchPattern,
    SearchPatternGenerator,
    SearchType,
    SearchConfig,
)

if TYPE_CHECKING:
    from core.drone import Drone


class SARState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    INVESTIGATING = "investigating"
    PAUSED = "paused"
    RETURNING = "returning"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class SARTarget:
    """A potential survivor / target identified during the search."""
    latitude: float
    longitude: float
    altitude: float
    confidence: float
    detection_type: str
    waypoint_index: int
    discovered_at: float
    photo_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "confidence": self.confidence,
            "detection_type": self.detection_type,
            "waypoint_index": self.waypoint_index,
            "discovered_at": self.discovered_at,
            "photo_path": self.photo_path,
            "metadata": dict(self.metadata),
        }


@dataclass
class SARReport:
    state: SARState
    waypoints_total: int
    waypoints_completed: int
    targets: List[SARTarget]
    distance_traveled_m: float
    duration_s: float
    started_at: Optional[float]
    finished_at: Optional[float]
    photos_taken: int
    pattern_type: str
    center: Tuple[float, float]
    error: Optional[str] = None
    investigations_completed: int = 0
    current_investigation: Optional[SARTarget] = None

    @property
    def progress_percent(self) -> float:
        if self.waypoints_total == 0:
            return 0.0
        return 100.0 * self.waypoints_completed / self.waypoints_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "waypoints_total": self.waypoints_total,
            "waypoints_completed": self.waypoints_completed,
            "progress_percent": self.progress_percent,
            "targets": [t.to_dict() for t in self.targets],
            "target_count": len(self.targets),
            "distance_traveled_m": self.distance_traveled_m,
            "duration_s": self.duration_s,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "photos_taken": self.photos_taken,
            "pattern_type": self.pattern_type,
            "center": list(self.center),
            "error": self.error,
            "investigations_completed": self.investigations_completed,
            "current_investigation": (
                self.current_investigation.to_dict()
                if self.current_investigation is not None else None
            ),
        }


# Detector hook: given (drone, waypoint_index) returns list of (type, confidence, metadata)
DetectorHook = Callable[
    ["Drone", int],
    List[Tuple[str, float, Dict[str, Any]]],
]


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def default_detector_hook(drone: "Drone", _wp_index: int) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Pull detections from the drone's camera sensor, if available."""
    cam = getattr(drone, "camera_sensor", None)
    if cam is None:
        return []
    try:
        dets = cam.get_detections()
    except Exception:
        return []
    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for d in dets:
        try:
            out.append((d.detection_type.value, float(d.confidence), {
                "bounding_box": d.bounding_box.to_dict(),
                "distance": d.distance,
            }))
        except Exception:
            continue
    return out


class SARMission:
    """A single-drone search-and-rescue mission.

    Typical use:

        sar = SARMission(drone, pattern=pattern)
        sar.start()
        while not sar.is_done:
            sar.tick(dt=0.1)

    Or, for fire-and-forget:

        sar.run_in_background()
        sar.wait_done()
    """

    DEFAULT_PHOTO_COOLDOWN_S = 0.5
    DEFAULT_TARGET_CONFIDENCE = 0.6
    DEFAULT_DEDUPE_RADIUS_M = 25.0
    DEFAULT_INVESTIGATION_CONF = 0.8
    DEFAULT_INVESTIGATION_ALT_M = 10.0
    DEFAULT_INVESTIGATION_DURATION_S = 5.0
    DEFAULT_INVESTIGATION_PHOTOS = 4
    DEFAULT_INVESTIGATION_DESCENT_MPS = 4.0

    def __init__(
        self,
        drone: "Drone",
        pattern: Optional[SearchPattern] = None,
        pattern_type: SearchType = SearchType.EXPANDING_SQUARE,
        center: Optional[Tuple[float, float]] = None,
        config: Optional[SearchConfig] = None,
        cruise_speed_mps: Optional[float] = None,
        photo_interval_s: float = DEFAULT_PHOTO_COOLDOWN_S,
        min_confidence: float = DEFAULT_TARGET_CONFIDENCE,
        detector_hook: Optional[DetectorHook] = None,
        return_to_start: bool = True,
        on_target: Optional[Callable[[SARTarget], None]] = None,
        on_waypoint: Optional[Callable[[int, Tuple[float, float, float]], None]] = None,
        dedupe_radius_m: float = DEFAULT_DEDUPE_RADIUS_M,
        investigate_targets: bool = True,
        investigation_confidence: float = DEFAULT_INVESTIGATION_CONF,
        investigation_altitude_m: float = DEFAULT_INVESTIGATION_ALT_M,
        investigation_duration_s: float = DEFAULT_INVESTIGATION_DURATION_S,
        investigation_photos: int = DEFAULT_INVESTIGATION_PHOTOS,
        investigation_descent_mps: float = DEFAULT_INVESTIGATION_DESCENT_MPS,
        on_investigation_start: Optional[Callable[[SARTarget], None]] = None,
        on_investigation_end: Optional[Callable[[SARTarget, int], None]] = None,
    ) -> None:
        self._drone = drone
        self._lock = threading.Lock()
        self._state = SARState.IDLE
        self._error: Optional[str] = None

        if pattern is None:
            if center is None:
                pos = getattr(drone, "current_position", (0.0, 0.0))
                center = (pos[0], pos[1])
            gen = SearchPatternGenerator(config=config)
            pattern = gen.generate(pattern_type, center)
        self._pattern = pattern
        self._waypoints: List[Tuple[float, float, float]] = list(pattern.waypoints)
        self._cruise_speed = float(
            cruise_speed_mps
            if cruise_speed_mps is not None
            else (config.speed if config is not None else 8.0)
        )
        self._photo_interval = float(photo_interval_s)
        self._min_confidence = float(min_confidence)
        self._detector_hook = detector_hook or default_detector_hook
        self._return_to_start = return_to_start
        self._on_target = on_target
        self._on_waypoint = on_waypoint
        self._dedupe_radius_m = float(dedupe_radius_m)

        # Investigation behavior
        self._investigate_targets = bool(investigate_targets)
        self._investigation_confidence = float(investigation_confidence)
        self._investigation_altitude_m = float(investigation_altitude_m)
        self._investigation_duration_s = float(investigation_duration_s)
        self._investigation_photos = int(investigation_photos)
        self._investigation_descent_mps = float(investigation_descent_mps)
        self._on_investigation_start = on_investigation_start
        self._on_investigation_end = on_investigation_end
        self._investigation_target: Optional[SARTarget] = None
        self._investigation_started_at: Optional[float] = None
        self._investigation_elapsed_s: float = 0.0
        self._investigation_cruise_altitude: Optional[float] = None
        self._investigation_photos_taken: int = 0
        self._investigations_completed: int = 0
        self._investigated_target_ids: set[int] = set()

        self._current_wp_index = 0
        self._waypoints_completed = 0
        self._distance_traveled = 0.0
        self._photos_taken = 0
        self._targets: List[SARTarget] = []
        self._last_photo_at = 0.0
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._start_position: Optional[Tuple[float, float]] = None
        self._returning = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ----------------------------- API ------------------------------------

    @property
    def state(self) -> SARState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state in (SARState.COMPLETED, SARState.ABORTED)

    @property
    def targets(self) -> List[SARTarget]:
        with self._lock:
            return list(self._targets)

    @property
    def pattern(self) -> SearchPattern:
        return self._pattern

    def start(self) -> bool:
        with self._lock:
            if self._state not in (SARState.IDLE, SARState.PAUSED):
                return False
            if not self._waypoints:
                self._state = SARState.COMPLETED
                self._error = "empty pattern"
                return False
            self._state = SARState.SEARCHING
            self._started_at = self._started_at or time.time()
            pos = getattr(self._drone, "current_position", (0.0, 0.0))
            if self._start_position is None:
                self._start_position = (pos[0], pos[1])
        return True

    def pause(self) -> None:
        with self._lock:
            if self._state == SARState.SEARCHING:
                self._state = SARState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state == SARState.PAUSED:
                self._state = SARState.SEARCHING

    def abort(self, reason: str = "user abort") -> None:
        with self._lock:
            self._state = SARState.ABORTED
            self._error = reason
            self._finished_at = time.time()
        self._stop_event.set()

    def tick(self, dt: float) -> None:
        if dt < 0:
            dt = 0.0
        with self._lock:
            current_state = self._state
            if current_state not in (
                SARState.SEARCHING, SARState.INVESTIGATING, SARState.RETURNING,
            ):
                return

        if current_state == SARState.INVESTIGATING:
            self._tick_investigation(dt)
            return

        target_wp = self._next_waypoint()
        if target_wp is None:
            self._finish_locked()
            return

        pos = getattr(self._drone, "current_position", (0.0, 0.0))
        cur_alt = float(getattr(self._drone, "current_altitude", target_wp[2]))
        dist_m = _haversine_m(pos, (target_wp[0], target_wp[1]))
        step_m = max(0.0, self._cruise_speed * dt)

        if step_m >= dist_m or dist_m < 1.0:
            self._move_drone_to(target_wp)
            self._distance_traveled += dist_m
            self._on_waypoint_arrival(target_wp)
        else:
            frac = step_m / dist_m
            new_lat = pos[0] + (target_wp[0] - pos[0]) * frac
            new_lon = pos[1] + (target_wp[1] - pos[1]) * frac
            new_alt = cur_alt + (target_wp[2] - cur_alt) * frac
            self._move_drone_to((new_lat, new_lon, new_alt))
            self._distance_traveled += step_m

        self._maybe_take_photo()
        self._scan_for_targets()

    def run_in_background(self, tick_hz: float = 10.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.start()

        def loop() -> None:
            interval = 1.0 / max(0.1, tick_hz)
            last = time.time()
            while not self._stop_event.is_set() and not self.is_done:
                now = time.time()
                self.tick(now - last)
                last = now
                time.sleep(interval)

        self._thread = threading.Thread(target=loop, daemon=True, name="sar-mission")
        self._thread.start()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        if self._thread is None:
            return self.is_done
        self._thread.join(timeout=timeout)
        return self.is_done

    def report(self) -> SARReport:
        with self._lock:
            duration = 0.0
            if self._started_at is not None:
                end = self._finished_at or time.time()
                duration = end - self._started_at
            return SARReport(
                state=self._state,
                waypoints_total=len(self._waypoints),
                waypoints_completed=self._waypoints_completed,
                targets=list(self._targets),
                distance_traveled_m=self._distance_traveled,
                duration_s=duration,
                started_at=self._started_at,
                finished_at=self._finished_at,
                photos_taken=self._photos_taken,
                pattern_type=self._pattern.pattern_type.value,
                center=self._pattern.center,
                error=self._error,
                investigations_completed=self._investigations_completed,
                current_investigation=self._investigation_target,
            )

    # --------------------------- internals --------------------------------

    def _next_waypoint(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            if self._state == SARState.RETURNING:
                if self._start_position is None:
                    return None
                return (self._start_position[0], self._start_position[1], self._pattern.waypoints[0][2])
            if self._current_wp_index >= len(self._waypoints):
                return None
            return self._waypoints[self._current_wp_index]

    def _move_drone_to(self, position: Tuple[float, float, float]) -> None:
        lat, lon, alt = position
        try:
            self._drone.current_position = (lat, lon)
            self._drone.current_altitude = float(alt)
        except Exception:
            pass

    def _on_waypoint_arrival(self, wp: Tuple[float, float, float]) -> None:
        with self._lock:
            if self._state == SARState.RETURNING:
                self._state = SARState.COMPLETED
                self._finished_at = time.time()
                return
            self._current_wp_index += 1
            self._waypoints_completed += 1
            idx = self._current_wp_index
        if self._on_waypoint is not None:
            try:
                self._on_waypoint(idx, wp)
            except Exception:
                pass

        if self._current_wp_index >= len(self._waypoints):
            self._finish_locked()

    def _finish_locked(self) -> None:
        with self._lock:
            if self._return_to_start and self._start_position is not None and self._state != SARState.RETURNING:
                self._state = SARState.RETURNING
                return
            if self._state != SARState.COMPLETED:
                self._state = SARState.COMPLETED
                self._finished_at = time.time()

    def _maybe_take_photo(self) -> None:
        now = time.time()
        if now - self._last_photo_at < self._photo_interval:
            return
        cam = getattr(self._drone, "camera_sensor", None)
        if cam is None:
            return
        try:
            ok, msg = cam.take_snapshot()
        except Exception:
            return
        if ok:
            with self._lock:
                self._photos_taken += 1
                self._last_photo_at = now

    def _scan_for_targets(self) -> None:
        try:
            results = self._detector_hook(self._drone, self._current_wp_index)
        except Exception:
            return
        if not results:
            return
        pos = getattr(self._drone, "current_position", (0.0, 0.0))
        alt = float(getattr(self._drone, "current_altitude", 0.0))
        cam = getattr(self._drone, "camera_sensor", None)
        photo_path: Optional[str] = None
        storage = getattr(cam, "media_storage", None) if cam else None
        if storage is not None:
            arts = storage.list_artifacts(kind="photo")
            if arts:
                photo_path = arts[-1].path
        new_targets: List[SARTarget] = []
        for det_type, conf, meta in results:
            if conf < self._min_confidence:
                continue
            if self._is_duplicate(pos, det_type):
                continue
            target = SARTarget(
                latitude=pos[0],
                longitude=pos[1],
                altitude=alt,
                confidence=conf,
                detection_type=det_type,
                waypoint_index=self._current_wp_index,
                discovered_at=time.time(),
                photo_path=photo_path,
                metadata=meta,
            )
            new_targets.append(target)
        if not new_targets:
            return
        with self._lock:
            self._targets.extend(new_targets)
        if self._on_target is not None:
            for t in new_targets:
                try:
                    self._on_target(t)
                except Exception:
                    pass

        if not self._investigate_targets:
            return
        with self._lock:
            if self._state != SARState.SEARCHING:
                return
            best: Optional[SARTarget] = None
            for t in new_targets:
                if t.confidence < self._investigation_confidence:
                    continue
                if id(t) in self._investigated_target_ids:
                    continue
                if best is None or t.confidence > best.confidence:
                    best = t
            if best is None:
                return
            self._investigation_target = best
            self._investigation_started_at = time.time()
            self._investigation_elapsed_s = 0.0
            self._investigation_cruise_altitude = float(
                getattr(self._drone, "current_altitude", best.altitude)
            )
            self._investigation_photos_taken = 0
            self._investigated_target_ids.add(id(best))
            self._state = SARState.INVESTIGATING
        if self._on_investigation_start is not None:
            try:
                self._on_investigation_start(self._investigation_target)
            except Exception:
                pass

    def _tick_investigation(self, dt: float) -> None:
        target = self._investigation_target
        if target is None or self._investigation_started_at is None:
            with self._lock:
                self._state = SARState.SEARCHING
            return
        with self._lock:
            self._investigation_elapsed_s += dt
            elapsed = self._investigation_elapsed_s
        cur_alt = float(getattr(self._drone, "current_altitude", target.altitude))
        target_alt = self._investigation_altitude_m
        descent_step = max(0.0, self._investigation_descent_mps * dt)
        new_alt = max(target_alt, cur_alt - descent_step)
        try:
            self._drone.current_position = (target.latitude, target.longitude)
            self._drone.current_altitude = new_alt
        except Exception:
            pass

        photos_target = self._investigation_photos
        photo_cadence = (
            self._investigation_duration_s / max(1, photos_target)
            if photos_target > 0 else self._investigation_duration_s
        )
        if (
            photos_target > 0
            and self._investigation_photos_taken < photos_target
            and elapsed >= photo_cadence * self._investigation_photos_taken
        ):
            cam = getattr(self._drone, "camera_sensor", None)
            if cam is not None:
                try:
                    ok, _ = cam.take_snapshot()
                except Exception:
                    ok = False
                if ok:
                    storage = getattr(cam, "media_storage", None)
                    photo_path: Optional[str] = None
                    if storage is not None:
                        arts = storage.list_artifacts(kind="photo")
                        if arts:
                            photo_path = arts[-1].path
                    with self._lock:
                        self._investigation_photos_taken += 1
                        self._photos_taken += 1
                        target.metadata.setdefault("investigation_photos", []).append(
                            photo_path or ""
                        )

        if elapsed >= self._investigation_duration_s:
            cruise_alt = self._investigation_cruise_altitude or target.altitude
            try:
                self._drone.current_altitude = float(cruise_alt)
            except Exception:
                pass
            taken = self._investigation_photos_taken
            with self._lock:
                self._investigations_completed += 1
                target.metadata["investigated"] = True
                target.metadata["investigation_photo_count"] = taken
                self._investigation_target = None
                self._investigation_started_at = None
                self._investigation_elapsed_s = 0.0
                self._investigation_cruise_altitude = None
                self._investigation_photos_taken = 0
                self._state = SARState.SEARCHING
            if self._on_investigation_end is not None:
                try:
                    self._on_investigation_end(target, taken)
                except Exception:
                    pass

    def _is_duplicate(self, pos: Tuple[float, float], det_type: str) -> bool:
        for existing in self._targets:
            if existing.detection_type != det_type:
                continue
            if _haversine_m(pos, (existing.latitude, existing.longitude)) < self._dedupe_radius_m:
                return True
        return False
