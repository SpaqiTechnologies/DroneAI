"""Disk storage manager for camera media."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class MediaArtifact:
    """A single on-disk media artifact (photo or recording bundle)."""
    kind: str                       # "photo" or "video"
    path: str                       # absolute path
    width: int
    height: int
    timestamp: float
    bytes_written: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filename"] = os.path.basename(self.path)
        return d


class MediaStorage:
    """Owns the root directory and assigns unique paths for photos/videos.

    Layout:
        <root>/
            photos/photo_<ts>_<seq>.ppm   (also .jpg if Pillow present)
            videos/recording_<ts>_<seq>.dvr/
                manifest.json
                frames/000001.ppm ...
            index.json
    """

    def __init__(self, root: str, session_id: Optional[str] = None) -> None:
        self._root = os.path.abspath(root)
        self._session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self._photos_dir = os.path.join(self._root, "photos")
        self._videos_dir = os.path.join(self._root, "videos")
        self._index_path = os.path.join(self._root, "index.json")
        self._lock = threading.Lock()
        self._artifacts: List[MediaArtifact] = []
        self._photo_seq = 0
        self._video_seq = 0
        os.makedirs(self._photos_dir, exist_ok=True)
        os.makedirs(self._videos_dir, exist_ok=True)
        self._load_index()

    @property
    def root(self) -> str:
        return self._root

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def photos_dir(self) -> str:
        return self._photos_dir

    @property
    def videos_dir(self) -> str:
        return self._videos_dir

    def allocate_photo_path(self, ext: str = "ppm") -> str:
        with self._lock:
            self._photo_seq += 1
            seq = self._photo_seq
        ts = int(time.time() * 1000)
        ext = ext.lstrip(".")
        return os.path.join(self._photos_dir, f"photo_{ts}_{seq:05d}.{ext}")

    def allocate_video_dir(self) -> str:
        with self._lock:
            self._video_seq += 1
            seq = self._video_seq
        ts = int(time.time() * 1000)
        bundle = os.path.join(self._videos_dir, f"recording_{ts}_{seq:05d}.dvr")
        os.makedirs(os.path.join(bundle, "frames"), exist_ok=True)
        return bundle

    def register(self, artifact: MediaArtifact) -> None:
        with self._lock:
            self._artifacts.append(artifact)
            self._save_index_locked()

    def list_artifacts(self, kind: Optional[str] = None) -> List[MediaArtifact]:
        with self._lock:
            if kind is None:
                return list(self._artifacts)
            return [a for a in self._artifacts if a.kind == kind]

    def total_bytes(self) -> int:
        with self._lock:
            return sum(a.bytes_written for a in self._artifacts)

    def clear(self) -> int:
        """Forget all registered artifacts (does NOT delete files on disk)."""
        with self._lock:
            n = len(self._artifacts)
            self._artifacts.clear()
            self._save_index_locked()
            return n

    def _load_index(self) -> None:
        if not os.path.isfile(self._index_path):
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        for raw in payload.get("artifacts", []):
            self._artifacts.append(MediaArtifact(
                kind=raw.get("kind", "photo"),
                path=raw.get("path", ""),
                width=int(raw.get("width", 0)),
                height=int(raw.get("height", 0)),
                timestamp=float(raw.get("timestamp", 0.0)),
                bytes_written=int(raw.get("bytes_written", 0)),
                extra=raw.get("extra", {}) or {},
            ))
        self._photo_seq = max(
            (i for i, a in enumerate(self._artifacts, start=1) if a.kind == "photo"),
            default=0,
        )
        self._video_seq = max(
            (i for i, a in enumerate(self._artifacts, start=1) if a.kind == "video"),
            default=0,
        )

    def _save_index_locked(self) -> None:
        payload = {
            "session_id": self._session_id,
            "saved_at": time.time(),
            "artifacts": [a.to_dict() for a in self._artifacts],
        }
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self._index_path)
