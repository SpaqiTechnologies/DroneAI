"""Multi-camera array: wide / tele / thermal / etc. on one drone.

Real high-end inspection drones carry multiple sensors that share a
single gimbal: a wide RGB for context, a tele for stand-off detail,
and often a thermal for heat/moisture. This module bundles N
``CameraSensor`` instances under a single coordinator so the system
can take a "shot" from all of them at once and pick the right feed
for a given task.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sensors.camera_sensor import CameraSensor, Frame, VisionMode
from sensors.media import MediaStorage


@dataclass
class CameraSpec:
    """Static description of one camera in the array."""
    name: str
    role: str                              # "wide" | "tele" | "thermal" | ...
    resolution: Tuple[int, int]
    fov_deg: float
    fps: int = 30
    vision_mode: VisionMode = VisionMode.NORMAL
    zoom: float = 1.0
    is_thermal: bool = False
    is_low_light: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "resolution": list(self.resolution),
            "fov_deg": self.fov_deg,
            "fps": self.fps,
            "vision_mode": self.vision_mode.value,
            "zoom": self.zoom,
            "is_thermal": self.is_thermal,
            "is_low_light": self.is_low_light,
            "metadata": dict(self.metadata),
        }


class CameraArray:
    """Coordinates a small fleet of ``CameraSensor`` objects.

    Each underlying sensor is independent: it has its own storage,
    streaming thread, and detection pipeline. The array gives you
    fan-out (``take_synchronized_snapshot``), single-feed selection
    (``set_active``), and a unified status view.
    """

    def __init__(
        self,
        specs: List[CameraSpec],
        media_storage: Optional[MediaStorage] = None,
    ) -> None:
        if not specs:
            raise ValueError("at least one camera spec is required")
        seen = set()
        for spec in specs:
            if spec.name in seen:
                raise ValueError(f"duplicate camera name: {spec.name}")
            seen.add(spec.name)
        self._specs: Dict[str, CameraSpec] = {s.name: s for s in specs}
        self._cameras: Dict[str, CameraSensor] = {}
        for spec in specs:
            cam = CameraSensor(
                resolution=spec.resolution,
                fps=spec.fps,
                fov=spec.fov_deg,
                media_storage=media_storage,
            )
            cam.set_vision_mode(spec.vision_mode)
            self._cameras[spec.name] = cam
        self._active: str = specs[0].name
        self._lock = threading.Lock()

    # ----------------------------- introspection --------------------------

    @property
    def names(self) -> List[str]:
        return list(self._cameras.keys())

    def specs(self) -> List[CameraSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> CameraSensor:
        cam = self._cameras.get(name)
        if cam is None:
            raise KeyError(f"no such camera: {name}")
        return cam

    def get_by_role(self, role: str) -> Optional[CameraSensor]:
        for name, spec in self._specs.items():
            if spec.role == role:
                return self._cameras[name]
        return None

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    def set_active(self, name: str) -> Tuple[bool, str]:
        with self._lock:
            if name not in self._cameras:
                return False, f"no such camera: {name}"
            self._active = name
        return True, f"active camera is now {name}"

    def active_camera(self) -> CameraSensor:
        with self._lock:
            return self._cameras[self._active]

    # ----------------------------- lifecycle ------------------------------

    def start_all(self) -> Dict[str, Tuple[bool, str]]:
        return {name: cam.start() for name, cam in self._cameras.items()}

    def stop_all(self) -> Dict[str, Tuple[bool, str]]:
        return {name: cam.stop() for name, cam in self._cameras.items()}

    # ----------------------------- shooting -------------------------------

    def take_synchronized_snapshot(self) -> Dict[str, Tuple[bool, str]]:
        """Fire all cameras at once; returns per-camera (success, message)."""
        return {name: cam.take_snapshot() for name, cam in self._cameras.items()}

    def start_synchronized_recording(self) -> Dict[str, Tuple[bool, str]]:
        return {
            name: cam.start_recording() for name, cam in self._cameras.items()
        }

    def stop_synchronized_recording(self) -> Dict[str, Tuple[bool, str]]:
        return {
            name: cam.stop_recording() for name, cam in self._cameras.items()
        }

    def capture_active_frame(self) -> Optional[Frame]:
        return self.active_camera().capture_frame()

    # ----------------------------- status ---------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "cameras": [
                {
                    "name": name,
                    "spec": self._specs[name].to_dict(),
                    "status": cam.get_status(),
                }
                for name, cam in self._cameras.items()
            ],
        }


def default_inspection_array(
    media_storage: Optional[MediaStorage] = None,
) -> CameraArray:
    """Canonical wide + tele + thermal trio for inspection drones."""
    return CameraArray(
        specs=[
            CameraSpec(
                name="wide",
                role="wide",
                resolution=CameraSensor.RESOLUTION_4K,
                fov_deg=84.0,
                fps=30,
                vision_mode=VisionMode.NORMAL,
                zoom=1.0,
            ),
            CameraSpec(
                name="tele",
                role="tele",
                resolution=CameraSensor.RESOLUTION_4K,
                fov_deg=15.0,
                fps=30,
                vision_mode=VisionMode.NORMAL,
                zoom=7.0,
            ),
            CameraSpec(
                name="thermal",
                role="thermal",
                resolution=(640, 512),
                fov_deg=45.0,
                fps=30,
                vision_mode=VisionMode.THERMAL,
                is_thermal=True,
            ),
        ],
        media_storage=media_storage,
    )
