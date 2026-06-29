"""Video recorder: writes frames into a `.dvr` bundle directory."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from sensors.media.encoder import write_png, write_ppm
from sensors.media.storage import MediaArtifact, MediaStorage


class RecorderState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"
    ERROR = "error"


_FRAME_WRITERS = {
    "png": (write_png, "png", "png"),
    "ppm": (write_ppm, "ppm", "ppm-p6"),
}


@dataclass
class RecordingSummary:
    bundle_path: str
    frame_count: int
    duration_s: float
    width: int
    height: int
    bytes_written: int
    manifest_path: str
    fps_effective: float
    frame_paths: List[str] = field(default_factory=list)


class VideoRecorder:
    """Accumulates frames into a directory bundle on disk.

    Layout for a single recording:
        recording_<ts>.dvr/
            manifest.json
            frames/000001.ppm
            frames/000002.ppm
            ...

    Thread-safe. `add_frame` may be called from a streaming thread; `stop`
    writes the manifest atomically and returns a summary.
    """

    def __init__(
        self,
        storage: MediaStorage,
        width: int,
        height: int,
        target_fps: float = 30.0,
        max_frames: Optional[int] = None,
        bundle_path: Optional[str] = None,
        frame_format: str = "png",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if frame_format not in _FRAME_WRITERS:
            raise ValueError(
                f"frame_format must be one of {sorted(_FRAME_WRITERS)}, got {frame_format!r}"
            )
        self._storage = storage
        self._width = width
        self._height = height
        self._target_fps = float(target_fps)
        self._max_frames = max_frames
        self._bundle = bundle_path or storage.allocate_video_dir()
        self._frames_dir = os.path.join(self._bundle, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._state = RecorderState.IDLE
        self._frame_count = 0
        self._bytes = 0
        self._frame_paths: List[str] = []
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._last_error: Optional[str] = None
        writer, ext, manifest_tag = _FRAME_WRITERS[frame_format]
        self._frame_writer = writer
        self._frame_ext = ext
        self._frame_format_tag = manifest_tag

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def bundle_path(self) -> str:
        return self._bundle

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> None:
        with self._lock:
            if self._state == RecorderState.RECORDING:
                return
            self._state = RecorderState.RECORDING
            self._start_time = time.time()
            self._stop_time = None
            self._last_error = None

    def add_frame(self, rgb: bytes, timestamp: Optional[float] = None) -> bool:
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return False
            if self._max_frames is not None and self._frame_count >= self._max_frames:
                return False
            seq = self._frame_count + 1
            fname = os.path.join(self._frames_dir, f"{seq:06d}.{self._frame_ext}")
            try:
                written = self._frame_writer(fname, rgb, self._width, self._height)
            except (OSError, ValueError) as exc:
                self._state = RecorderState.ERROR
                self._last_error = str(exc)
                return False
            self._frame_count = seq
            self._bytes += written
            self._frame_paths.append(fname)
            return True

    def stop(self) -> RecordingSummary:
        with self._lock:
            if self._state not in (RecorderState.RECORDING, RecorderState.ERROR):
                self._stop_time = self._stop_time or time.time()
            else:
                self._stop_time = time.time()
            duration = (self._stop_time - (self._start_time or self._stop_time))
            duration = max(0.0, duration)
            fps_eff = (self._frame_count / duration) if duration > 0 else 0.0
            manifest = {
                "version": 1,
                "started_at": self._start_time,
                "stopped_at": self._stop_time,
                "duration_s": duration,
                "width": self._width,
                "height": self._height,
                "target_fps": self._target_fps,
                "effective_fps": fps_eff,
                "frame_count": self._frame_count,
                "bytes_written": self._bytes,
                "frame_format": self._frame_format_tag,
                "frame_dir": "frames",
                "error": self._last_error,
            }
            manifest_path = os.path.join(self._bundle, "manifest.json")
            tmp = manifest_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            os.replace(tmp, manifest_path)
            self._state = RecorderState.STOPPED
            summary = RecordingSummary(
                bundle_path=self._bundle,
                frame_count=self._frame_count,
                duration_s=duration,
                width=self._width,
                height=self._height,
                bytes_written=self._bytes,
                manifest_path=manifest_path,
                fps_effective=fps_eff,
                frame_paths=list(self._frame_paths),
            )

        artifact = MediaArtifact(
            kind="video",
            path=self._bundle,
            width=self._width,
            height=self._height,
            timestamp=self._start_time or time.time(),
            bytes_written=self._bytes,
            extra={
                "frame_count": self._frame_count,
                "duration_s": duration,
                "effective_fps": fps_eff,
                "manifest": manifest_path,
            },
        )
        self._storage.register(artifact)
        return summary
