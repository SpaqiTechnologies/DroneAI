"""Coverage-aware multi-shell orbit photogrammetry.

A target structure is approximated as a vertical cylinder (center, radius,
height). We plan a sequence of concentric orbit "shells" at multiple
altitudes, capturing imagery at each waypoint. After each shell we tally
which azimuth bins have been imaged from a "useful" viewpoint and add
additional capture waypoints into bins below the per-bin target.

The planner runs in software — no actual photogrammetry pipeline — but
produces a deterministic, inspectable capture plan + a coverage report
that downstream image processing can consume to drive real reconstruction.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ScanState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    CAPTURING = "capturing"
    REFINING = "refining"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ScanWaypoint:
    """A single capture pose: orbit position + view direction."""
    latitude: float
    longitude: float
    altitude: float
    azimuth_deg: float            # viewing direction (toward the center)
    pitch_deg: float              # downward tilt
    distance_m: float             # standoff from target
    shell_index: int
    bin_index: int                # azimuth bin
    is_refinement: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "azimuth_deg": self.azimuth_deg,
            "pitch_deg": self.pitch_deg,
            "distance_m": self.distance_m,
            "shell_index": self.shell_index,
            "bin_index": self.bin_index,
            "is_refinement": self.is_refinement,
        }


@dataclass
class CapturedImage:
    waypoint: ScanWaypoint
    photo_path: Optional[str]
    captured_at: float
    coverage_quality: float       # 0..1; placeholder; real systems would
                                  # estimate from sharpness + view angle.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waypoint": self.waypoint.to_dict(),
            "photo_path": self.photo_path,
            "captured_at": self.captured_at,
            "coverage_quality": self.coverage_quality,
        }


@dataclass
class ScanReport:
    scan_id: str
    state: ScanState
    target_center: Tuple[float, float]
    target_height_m: float
    target_radius_m: float
    shells: int
    bins_per_shell: int
    planned_waypoints: int
    captured_waypoints: int
    refinement_waypoints: int
    images: List[CapturedImage]
    coverage_per_shell: List[List[float]]  # 0..1 per bin, per shell
    overall_coverage: float
    started_at: Optional[float]
    finished_at: Optional[float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "state": self.state.value,
            "target_center": list(self.target_center),
            "target_height_m": self.target_height_m,
            "target_radius_m": self.target_radius_m,
            "shells": self.shells,
            "bins_per_shell": self.bins_per_shell,
            "planned_waypoints": self.planned_waypoints,
            "captured_waypoints": self.captured_waypoints,
            "refinement_waypoints": self.refinement_waypoints,
            "image_count": len(self.images),
            "coverage_per_shell": [list(row) for row in self.coverage_per_shell],
            "overall_coverage": self.overall_coverage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "images": [img.to_dict() for img in self.images],
        }


@dataclass
class ScanConfig:
    standoff_m: float = 15.0          # horizontal distance from cylinder surface
    shells: int = 3                   # vertical slices (top/mid/bottom etc.)
    bins_per_shell: int = 12          # azimuth bins per shell
    coverage_target: float = 0.85     # per-bin target quality after refinement
    max_refinement_passes: int = 2
    coverage_quality_base: float = 0.6   # base quality assigned per capture
    bin_overlap_bonus: float = 0.15      # bonus when a bin is captured twice


def _project_offset(
    origin: Tuple[float, float], east_m: float, north_m: float,
) -> Tuple[float, float]:
    lat0 = origin[0]
    dlat = (north_m / 6371000.0) * (180.0 / math.pi)
    dlon = (east_m / (6371000.0 * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return origin[0] + dlat, origin[1] + dlon


class AdaptiveScan:
    """Orchestrates a coverage-aware orbit scan around a target structure."""

    def __init__(
        self,
        center: Tuple[float, float],
        target_height_m: float,
        target_radius_m: float,
        ground_altitude_m: float = 0.0,
        config: Optional[ScanConfig] = None,
        on_capture: Optional[Callable[[CapturedImage], None]] = None,
    ) -> None:
        if target_height_m <= 0 or target_radius_m <= 0:
            raise ValueError("target_height_m and target_radius_m must be positive")
        self._scan_id = uuid.uuid4().hex[:12]
        self._center = center
        self._height = float(target_height_m)
        self._radius = float(target_radius_m)
        self._ground_alt = float(ground_altitude_m)
        self._config = config or ScanConfig()
        self._on_capture = on_capture

        self._planned: List[ScanWaypoint] = []
        self._captured: List[CapturedImage] = []
        self._coverage: List[List[float]] = [
            [0.0] * self._config.bins_per_shell
            for _ in range(self._config.shells)
        ]
        self._refinement_count = 0
        self._refinement_passes_done = 0
        self._state = ScanState.IDLE
        self._lock = threading.Lock()
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._error: Optional[str] = None

        self._build_initial_plan()

    # ------------------------------- planning -----------------------------

    def _build_initial_plan(self) -> None:
        cfg = self._config
        # Shells distributed evenly along the structure's height
        for s in range(cfg.shells):
            t = (s + 0.5) / cfg.shells if cfg.shells > 0 else 0.5
            shell_alt = self._ground_alt + t * self._height
            for b in range(cfg.bins_per_shell):
                wp = self._make_waypoint(
                    shell_idx=s, bin_idx=b, altitude_m=shell_alt,
                    is_refinement=False,
                )
                self._planned.append(wp)

    def _make_waypoint(
        self, shell_idx: int, bin_idx: int, altitude_m: float, is_refinement: bool,
    ) -> ScanWaypoint:
        cfg = self._config
        azimuth = (360.0 * bin_idx / cfg.bins_per_shell) % 360.0
        # Place the drone at standoff distance opposite the target face that
        # corresponds to this azimuth bin.
        r_total = self._radius + cfg.standoff_m
        radial_east = r_total * math.sin(math.radians(azimuth))
        radial_north = r_total * math.cos(math.radians(azimuth))
        lat, lon = _project_offset(self._center, radial_east, radial_north)
        # Camera points back toward the structure center.
        view_az = (azimuth + 180.0) % 360.0
        view_pitch = math.degrees(
            math.atan2(altitude_m - (self._ground_alt + self._height / 2), r_total)
        )
        return ScanWaypoint(
            latitude=lat,
            longitude=lon,
            altitude=altitude_m,
            azimuth_deg=view_az,
            pitch_deg=-abs(view_pitch),
            distance_m=r_total,
            shell_index=shell_idx,
            bin_index=bin_idx,
            is_refinement=is_refinement,
        )

    # ------------------------------- API ----------------------------------

    @property
    def scan_id(self) -> str:
        return self._scan_id

    @property
    def state(self) -> ScanState:
        with self._lock:
            return self._state

    def remaining_waypoints(self) -> List[ScanWaypoint]:
        with self._lock:
            return list(self._planned)

    def captured_images(self) -> List[CapturedImage]:
        with self._lock:
            return list(self._captured)

    def coverage_matrix(self) -> List[List[float]]:
        with self._lock:
            return [list(row) for row in self._coverage]

    def coverage_overall(self) -> float:
        with self._lock:
            return self._coverage_overall_locked()

    def _coverage_overall_locked(self) -> float:
        if not self._coverage or not self._coverage[0]:
            return 0.0
        total = sum(sum(row) for row in self._coverage)
        cells = self._config.shells * self._config.bins_per_shell
        return min(1.0, total / cells) if cells > 0 else 0.0

    def start(self) -> None:
        with self._lock:
            if self._state in (ScanState.CAPTURING, ScanState.REFINING):
                return
            self._state = ScanState.CAPTURING
            if self._started_at is None:
                self._started_at = time.time()

    def abort(self, reason: str = "user abort") -> None:
        with self._lock:
            self._state = ScanState.ABORTED
            self._error = reason
            self._finished_at = time.time()

    def capture_next(
        self,
        photo_path: Optional[str] = None,
        coverage_quality: Optional[float] = None,
    ) -> Optional[CapturedImage]:
        """Mark the next planned waypoint as captured. Returns the captured image."""
        with self._lock:
            if self._state not in (ScanState.CAPTURING, ScanState.REFINING):
                return None
            if not self._planned:
                self._maybe_finish_or_refine_locked()
                return None
            wp = self._planned.pop(0)
            cfg = self._config
            quality = (
                cfg.coverage_quality_base
                if coverage_quality is None else max(0.0, min(1.0, coverage_quality))
            )
            existing = self._coverage[wp.shell_index][wp.bin_index]
            new_quality = min(
                1.0,
                existing + (1.0 - existing) * (quality + (cfg.bin_overlap_bonus if existing > 0 else 0.0)),
            )
            self._coverage[wp.shell_index][wp.bin_index] = new_quality
            if wp.is_refinement:
                self._refinement_count += 1
            captured = CapturedImage(
                waypoint=wp,
                photo_path=photo_path,
                captured_at=time.time(),
                coverage_quality=new_quality,
            )
            self._captured.append(captured)
            self._maybe_finish_or_refine_locked()
        if self._on_capture is not None:
            try:
                self._on_capture(captured)
            except Exception:
                pass
        return captured

    def _maybe_finish_or_refine_locked(self) -> None:
        if self._planned:
            return
        cfg = self._config
        # Identify under-covered bins
        under = [
            (s, b)
            for s, row in enumerate(self._coverage)
            for b, q in enumerate(row)
            if q < cfg.coverage_target
        ]
        if not under or self._refinement_passes_done >= cfg.max_refinement_passes:
            self._state = ScanState.COMPLETED
            self._finished_at = time.time()
            return
        self._refinement_passes_done += 1
        self._state = ScanState.REFINING
        for (s, b) in under:
            t = (s + 0.5) / cfg.shells if cfg.shells > 0 else 0.5
            shell_alt = self._ground_alt + t * self._height
            wp = self._make_waypoint(
                shell_idx=s, bin_idx=b, altitude_m=shell_alt,
                is_refinement=True,
            )
            self._planned.append(wp)

    def report(self) -> ScanReport:
        with self._lock:
            return ScanReport(
                scan_id=self._scan_id,
                state=self._state,
                target_center=self._center,
                target_height_m=self._height,
                target_radius_m=self._radius,
                shells=self._config.shells,
                bins_per_shell=self._config.bins_per_shell,
                planned_waypoints=len(self._planned),
                captured_waypoints=len(self._captured),
                refinement_waypoints=self._refinement_count,
                images=list(self._captured),
                coverage_per_shell=[list(row) for row in self._coverage],
                overall_coverage=self._coverage_overall_locked(),
                started_at=self._started_at,
                finished_at=self._finished_at,
                error=self._error,
            )
