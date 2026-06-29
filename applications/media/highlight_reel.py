"""Auto-edit highlight reel builder.

Takes a recording bundle (manifest.json + frames/) plus a list of
"events" (target detections, anomalies, mission milestones), and emits
a ``highlight_manifest.json`` that selects the best clip windows.

The selection is rank-then-merge:
  1. Each event becomes a candidate clip ``[t - pre, t + post]``.
  2. Rank clips by score (event importance × confidence).
  3. Greedily merge overlapping/adjacent clips while honoring the
     reel's total target duration.

This is *editorial metadata only* — no transcoding. Downstream players
(or a separate ffmpeg job) consume the manifest to render the actual
output. Pure stdlib, deterministic, testable.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class HighlightEvent:
    """A timestamped point of interest from anywhere in the system."""
    timestamp: float
    kind: str                            # "target", "anomaly", "milestone", ...
    label: str
    importance: float = 1.0              # caller-defined weight
    confidence: float = 1.0              # 0..1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.importance * max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "label": self.label,
            "importance": self.importance,
            "confidence": self.confidence,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass
class HighlightClip:
    """A contiguous time window selected for the reel."""
    clip_id: str
    start_timestamp: float
    end_timestamp: float
    start_frame_index: int
    end_frame_index: int
    frame_paths: List[str]
    events: List[HighlightEvent]
    score: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_s": self.duration_s,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "frame_count": len(self.frame_paths),
            "frame_paths": list(self.frame_paths),
            "events": [e.to_dict() for e in self.events],
            "score": self.score,
        }


@dataclass
class HighlightReelManifest:
    reel_id: str
    source_bundle: str
    total_duration_s: float
    clip_count: int
    clips: List[HighlightClip]
    created_at: float
    settings: Dict[str, Any]
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reel_id": self.reel_id,
            "source_bundle": self.source_bundle,
            "total_duration_s": self.total_duration_s,
            "clip_count": self.clip_count,
            "clips": [c.to_dict() for c in self.clips],
            "created_at": self.created_at,
            "settings": dict(self.settings),
            "manifest_path": self.manifest_path,
        }


class HighlightReelBuilder:
    """Builds a highlight reel manifest from a recording + events."""

    DEFAULT_PRE_S = 1.5
    DEFAULT_POST_S = 2.5
    DEFAULT_MAX_DURATION_S = 30.0
    DEFAULT_MERGE_GAP_S = 1.0

    def __init__(
        self,
        recording_bundle_path: str,
        pre_event_s: float = DEFAULT_PRE_S,
        post_event_s: float = DEFAULT_POST_S,
        max_reel_duration_s: float = DEFAULT_MAX_DURATION_S,
        merge_adjacent_gap_s: float = DEFAULT_MERGE_GAP_S,
    ) -> None:
        if pre_event_s < 0 or post_event_s < 0:
            raise ValueError("pre_event_s and post_event_s must be >= 0")
        if max_reel_duration_s <= 0:
            raise ValueError("max_reel_duration_s must be positive")
        self._bundle = os.path.abspath(recording_bundle_path)
        self._pre = float(pre_event_s)
        self._post = float(post_event_s)
        self._max_total = float(max_reel_duration_s)
        self._merge_gap = float(merge_adjacent_gap_s)
        self._manifest = self._load_manifest()

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = os.path.join(self._bundle, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"recording manifest not found at {manifest_path}"
            )
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _frame_paths(self) -> List[str]:
        frames_dir = os.path.join(self._bundle, self._manifest.get("frame_dir", "frames"))
        if not os.path.isdir(frames_dir):
            return []
        return sorted(
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if not name.startswith(".")
        )

    def build(
        self,
        events: Iterable[HighlightEvent],
        reel_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> HighlightReelManifest:
        events_sorted = sorted(events, key=lambda e: e.timestamp)
        manifest = self._manifest
        started_at = float(manifest.get("started_at") or 0.0)
        fps = float(manifest.get("effective_fps") or manifest.get("target_fps") or 1.0)
        if fps <= 0:
            fps = 1.0
        frame_paths = self._frame_paths()
        frame_count = int(manifest.get("frame_count") or len(frame_paths))
        end_at = float(manifest.get("stopped_at") or (started_at + frame_count / fps))

        # Build candidate clip windows clamped to the recording's bounds.
        candidates: List[Tuple[float, float, HighlightEvent]] = []
        for ev in events_sorted:
            if ev.timestamp < started_at - self._pre or ev.timestamp > end_at + self._post:
                continue
            start = max(started_at, ev.timestamp - self._pre)
            stop = min(end_at, ev.timestamp + self._post)
            if stop <= start:
                continue
            candidates.append((start, stop, ev))

        # Greedy selection by event score until we hit the duration budget.
        candidates_by_score = sorted(
            candidates, key=lambda c: c[2].score, reverse=True,
        )
        chosen: List[Tuple[float, float, List[HighlightEvent]]] = []
        total_duration = 0.0
        for (start, stop, ev) in candidates_by_score:
            if total_duration >= self._max_total:
                break
            chosen.append((start, stop, [ev]))
            total_duration += (stop - start)

        # Sort by time, then merge overlapping / nearly-adjacent clips.
        chosen.sort(key=lambda c: c[0])
        merged: List[Tuple[float, float, List[HighlightEvent]]] = []
        for (start, stop, evs) in chosen:
            if not merged:
                merged.append((start, stop, list(evs)))
                continue
            prev_start, prev_stop, prev_evs = merged[-1]
            if start <= prev_stop + self._merge_gap:
                merged[-1] = (
                    prev_start,
                    max(prev_stop, stop),
                    prev_evs + list(evs),
                )
            else:
                merged.append((start, stop, list(evs)))

        clips = [
            self._build_clip(start, stop, evs, frame_paths, started_at, fps)
            for (start, stop, evs) in merged
        ]

        reel = HighlightReelManifest(
            reel_id=reel_id or uuid.uuid4().hex[:12],
            source_bundle=self._bundle,
            total_duration_s=sum(c.duration_s for c in clips),
            clip_count=len(clips),
            clips=clips,
            created_at=time.time(),
            settings={
                "pre_event_s": self._pre,
                "post_event_s": self._post,
                "max_reel_duration_s": self._max_total,
                "merge_adjacent_gap_s": self._merge_gap,
            },
        )

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(reel.to_dict(), fh, indent=2)
            reel.manifest_path = output_path
        return reel

    def _build_clip(
        self,
        start: float,
        stop: float,
        events: List[HighlightEvent],
        frame_paths: List[str],
        started_at: float,
        fps: float,
    ) -> HighlightClip:
        start_idx = max(0, int(math.floor((start - started_at) * fps)))
        end_idx = min(max(start_idx, len(frame_paths) - 1), int(math.ceil((stop - started_at) * fps)))
        # Frame paths are sorted; slice gives the actual frame files for this clip.
        clip_frames = frame_paths[start_idx:end_idx + 1] if frame_paths else []
        score = max((e.score for e in events), default=0.0)
        return HighlightClip(
            clip_id=uuid.uuid4().hex[:8],
            start_timestamp=start,
            end_timestamp=stop,
            start_frame_index=start_idx,
            end_frame_index=end_idx,
            frame_paths=clip_frames,
            events=events,
            score=score,
        )
