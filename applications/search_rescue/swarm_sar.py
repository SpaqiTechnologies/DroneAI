"""Multi-drone Search-and-Rescue mission.

Slices a circular AOI into N non-overlapping sectors (or non-overlapping
parallel-track lanes) and assigns one ``SARMission`` per drone. Aggregates
target reports across the swarm and deduplicates by (type, location).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from applications.search_rescue.sar_mission import (
    SARMission,
    SARReport,
    SARState,
    SARTarget,
)
from applications.search_rescue.search_pattern import (
    SearchPattern,
    SearchPatternGenerator,
    SearchType,
    SearchConfig,
)

if TYPE_CHECKING:
    from core.drone import Drone


def allocate_subareas(
    center: Tuple[float, float],
    radius_m: float,
    n: int,
    sector_overlap_deg: float = 0.0,
) -> List[Tuple[Tuple[float, float], float, float]]:
    """Return ``n`` sub-area assignments around ``center``.

    Each assignment is ``(sub_center, sub_radius, heading_deg)`` so a drone
    can run a SECTOR or PARALLEL pattern inside its slice without colliding
    with neighbours.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if n == 1:
        return [(center, radius_m, 0.0)]
    slice_deg = 360.0 / n
    sub_radius = radius_m * 0.95
    # place each sub-center half-way along the radial of its slice so
    # the sub-areas tile concentrically without overlapping the next slice
    sub_distance = radius_m * 0.5
    out: List[Tuple[Tuple[float, float], float, float]] = []
    lat = center[0]
    cos_lat = math.cos(math.radians(lat))
    for i in range(n):
        heading_deg = i * slice_deg
        heading_rad = math.radians(heading_deg)
        dlat = (sub_distance * math.cos(heading_rad)) / 111000.0
        dlon = (sub_distance * math.sin(heading_rad)) / (111000.0 * max(0.01, cos_lat))
        sub_center = (lat + dlat, center[1] + dlon)
        sub_r = sub_radius * (slice_deg / 360.0) * 2.0
        out.append((sub_center, sub_r, heading_deg))
    _ = sector_overlap_deg  # accepted for API symmetry; not used yet
    return out


@dataclass
class SwarmSARReport:
    drone_count: int
    started_at: Optional[float]
    finished_at: Optional[float]
    completed_drones: int
    aborted_drones: int
    aggregated_targets: List[SARTarget]
    per_drone_reports: Dict[str, SARReport] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_count": self.drone_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "completed_drones": self.completed_drones,
            "aborted_drones": self.aborted_drones,
            "aggregated_targets": [t.to_dict() for t in self.aggregated_targets],
            "target_count": len(self.aggregated_targets),
            "per_drone": {k: v.to_dict() for k, v in self.per_drone_reports.items()},
        }


class SwarmSARMission:
    """Coordinator for an N-drone SAR sweep."""

    DEFAULT_DEDUPE_RADIUS_M = 10.0

    def __init__(
        self,
        drones: Dict[str, "Drone"],
        center: Tuple[float, float],
        radius_m: float = 500.0,
        config: Optional[SearchConfig] = None,
        pattern_type: SearchType = SearchType.SECTOR,
        return_to_start: bool = True,
        dedupe_radius_m: float = DEFAULT_DEDUPE_RADIUS_M,
    ) -> None:
        if not drones:
            raise ValueError("at least one drone is required")
        if radius_m <= 0:
            raise ValueError("radius_m must be positive")
        self._drones = dict(drones)
        self._center = center
        self._radius_m = float(radius_m)
        self._config = config or SearchConfig()
        self._pattern_type = pattern_type
        self._return_to_start = return_to_start
        self._dedupe_radius = float(dedupe_radius_m)
        self._missions: Dict[str, SARMission] = {}
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._lock = threading.Lock()
        self._build_missions()

    @property
    def missions(self) -> Dict[str, SARMission]:
        return dict(self._missions)

    @property
    def drone_ids(self) -> List[str]:
        return list(self._drones.keys())

    def _build_missions(self) -> None:
        assignments = allocate_subareas(
            self._center, self._radius_m, len(self._drones)
        )
        gen = SearchPatternGenerator(config=self._config)
        for (drone_id, drone), (sub_center, sub_radius, heading) in zip(
            self._drones.items(), assignments
        ):
            kwargs: Dict[str, Any] = {}
            if self._pattern_type == SearchType.SECTOR:
                kwargs["radius"] = sub_radius
                kwargs["angle"] = max(30.0, 360.0 / len(self._drones))
            elif self._pattern_type == SearchType.PARALLEL:
                kwargs["width"] = sub_radius
                kwargs["height"] = sub_radius
                kwargs["heading"] = heading
            elif self._pattern_type == SearchType.EXPANDING_SQUARE:
                kwargs["legs"] = 6
            pattern: SearchPattern = gen.generate(self._pattern_type, sub_center, **kwargs)
            mission = SARMission(
                drone=drone,
                pattern=pattern,
                cruise_speed_mps=self._config.speed,
                return_to_start=self._return_to_start,
            )
            self._missions[drone_id] = mission

    # ----------------------------- control --------------------------------

    def start(self) -> None:
        with self._lock:
            self._started_at = self._started_at or time.time()
        for m in self._missions.values():
            m.start()

    def tick(self, dt: float) -> None:
        for m in self._missions.values():
            m.tick(dt)
        if all(m.is_done for m in self._missions.values()):
            with self._lock:
                if self._finished_at is None:
                    self._finished_at = time.time()

    def run_in_background(self, tick_hz: float = 10.0) -> None:
        for m in self._missions.values():
            m.run_in_background(tick_hz=tick_hz)
        with self._lock:
            self._started_at = self._started_at or time.time()

    def abort(self, reason: str = "user abort") -> None:
        for m in self._missions.values():
            m.abort(reason)
        with self._lock:
            self._finished_at = self._finished_at or time.time()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.time() + timeout
        for m in self._missions.values():
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            m.wait_done(timeout=remaining)
        return all(m.is_done for m in self._missions.values())

    @property
    def is_done(self) -> bool:
        return all(m.is_done for m in self._missions.values())

    # ----------------------------- reporting ------------------------------

    def report(self) -> SwarmSARReport:
        per_drone: Dict[str, SARReport] = {}
        aggregated: List[SARTarget] = []
        completed = 0
        aborted = 0
        for did, m in self._missions.items():
            r = m.report()
            per_drone[did] = r
            if r.state == SARState.COMPLETED:
                completed += 1
            elif r.state == SARState.ABORTED:
                aborted += 1
            for t in r.targets:
                if not self._is_dup(aggregated, t):
                    aggregated.append(t)
        with self._lock:
            return SwarmSARReport(
                drone_count=len(self._missions),
                started_at=self._started_at,
                finished_at=self._finished_at,
                completed_drones=completed,
                aborted_drones=aborted,
                aggregated_targets=aggregated,
                per_drone_reports=per_drone,
            )

    def _is_dup(self, existing: List[SARTarget], cand: SARTarget) -> bool:
        for e in existing:
            if e.detection_type != cand.detection_type:
                continue
            d = self._haversine(e.latitude, e.longitude, cand.latitude, cand.longitude)
            if d < self._dedupe_radius:
                return True
        return False

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        a = math.radians(lat1)
        b = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        h = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
        return 2 * 6371000.0 * math.asin(math.sqrt(h))
