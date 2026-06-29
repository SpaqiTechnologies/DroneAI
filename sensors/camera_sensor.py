"""
Camera sensor for drone vision processing.
Provides frame capture, object detection, and marker detection for precision landing.
"""

import os
import time
import threading
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any
from sensors.sensor import Sensor
from sensors.media import (
    MediaStorage,
    VideoRecorder,
    RecorderState,
    synthesize_rgb_bytes,
    write_ppm,
    write_png,
    write_jpeg_if_possible,
    PIL_AVAILABLE,
)


class VisionMode(Enum):
    """Camera vision modes."""
    NORMAL = "normal"
    THERMAL = "thermal"
    NIGHT_VISION = "night_vision"
    OBSTACLE_DETECTION = "obstacle_detection"
    MARKER_DETECTION = "marker_detection"


class CameraState(Enum):
    """Camera operational state."""
    OFF = auto()
    INITIALIZING = auto()
    READY = auto()
    STREAMING = auto()
    RECORDING = auto()
    ERROR = auto()


class DetectionType(Enum):
    """Types of detected objects."""
    OBSTACLE = "obstacle"
    LANDING_MARKER = "landing_marker"
    PERSON = "person"
    VEHICLE = "vehicle"
    BUILDING = "building"
    TREE = "tree"
    POWER_LINE = "power_line"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for detected object."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'center': self.center,
        }


@dataclass
class Detection:
    """Detected object information."""
    detection_type: DetectionType
    confidence: float  # 0.0 to 1.0
    bounding_box: BoundingBox
    distance: Optional[float] = None  # Estimated distance in meters
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'type': self.detection_type.value,
            'confidence': self.confidence,
            'bounding_box': self.bounding_box.to_dict(),
            'distance': self.distance,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }


@dataclass
class Frame:
    """Captured camera frame."""
    width: int
    height: int
    timestamp: float
    data: Optional[bytes] = None  # Raw frame data (None for simulation)
    detections: List[Detection] = field(default_factory=list)
    mode: VisionMode = VisionMode.NORMAL

    def to_dict(self) -> dict:
        """Serialize to dictionary (without raw data)."""
        return {
            'width': self.width,
            'height': self.height,
            'timestamp': self.timestamp,
            'mode': self.mode.value,
            'detections': [d.to_dict() for d in self.detections],
            'has_data': self.data is not None,
        }


@dataclass
class GimbalPosition:
    """Camera gimbal position."""
    pitch: float = 0.0  # -90 (down) to +30 (up)
    roll: float = 0.0   # -45 to +45
    yaw: float = 0.0    # -180 to +180

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
        }


class CameraSensor(Sensor):
    """
    Camera sensor with vision processing capabilities.

    Features:
    - Multiple vision modes (normal, thermal, night vision)
    - Object detection (obstacles, markers, people, vehicles)
    - Landing marker detection for precision landing
    - Gimbal control (pitch, roll, yaw)
    - Recording and snapshot capabilities
    """

    # Resolution presets
    RESOLUTION_480P = (640, 480)
    RESOLUTION_720P = (1280, 720)
    RESOLUTION_1080P = (1920, 1080)
    RESOLUTION_2_7K = (2704, 1520)
    RESOLUTION_4K = (3840, 2160)
    RESOLUTION_6K = (6144, 3240)
    RESOLUTION_8K = (7680, 4320)

    # Common cinema color profiles, exposed as metadata for downstream
    # tooling. We don't actually grade in sim — these are tags.
    COLOR_PROFILE_STANDARD = "standard"
    COLOR_PROFILE_HDR_10 = "hdr10"
    COLOR_PROFILE_HLG = "hlg"
    COLOR_PROFILE_DLOG_M = "dlog-m"
    COLOR_PROFILE_REC_709 = "rec709"

    def __init__(
        self,
        resolution: Tuple[int, int] = RESOLUTION_720P,
        fps: int = 30,
        fov: float = 120.0,  # Field of view in degrees
        media_storage: Optional[MediaStorage] = None,
        synth_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize camera sensor.

        Args:
            resolution: Camera resolution (width, height)
            fps: Frames per second
            fov: Field of view in degrees
            media_storage: Optional storage for on-disk photos/videos. If
                provided, ``take_snapshot`` and ``start_recording`` write
                actual files; otherwise they return path-only stubs.
            synth_resolution: Lower internal resolution for synthesized frame
                pixels. Defaults to a 160x90-clamped downscale to keep
                file sizes sane in simulation. Set to ``resolution`` to
                synthesize at full resolution.
        """
        self._resolution = resolution
        self._fps = fps
        self._fov = fov
        self._media_storage = media_storage
        self._hdr_enabled = False
        self._color_profile = self.COLOR_PROFILE_STANDARD
        self._bitrate_mbps: Optional[float] = None
        if synth_resolution is None:
            sw = min(resolution[0], 160)
            sh = max(1, int(round(sw * resolution[1] / max(1, resolution[0]))))
            self._synth_resolution = (sw, sh)
        else:
            self._synth_resolution = synth_resolution
        self._recorder: Optional[VideoRecorder] = None
        self._last_recording_summary: Optional[Dict[str, Any]] = None

        # State
        self._state = CameraState.OFF
        self._vision_mode = VisionMode.NORMAL
        self._gimbal = GimbalPosition()

        # Detection settings
        self._detection_enabled = True
        self._obstacle_detection_enabled = True
        self._marker_detection_enabled = True
        self._min_confidence = 0.5

        # Recording
        self._is_recording = False
        self._recording_start_time: Optional[float] = None
        self._recording_path: Optional[str] = None

        # Threading
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._frame_callbacks: List[Callable[[Frame], None]] = []
        self._detection_callbacks: List[Callable[[Detection], None]] = []

        # Latest data
        self._last_frame: Optional[Frame] = None
        self._detections: List[Detection] = []
        self._marker_detected = False
        self._marker_position: Optional[Tuple[int, int]] = None

        # Simulation data
        self._simulated_obstacles: List[Dict] = []
        self._simulated_marker: Optional[Dict] = None

        # Health status
        self._healthy = True
        self._error_message: Optional[str] = None
        self._last_update = time.time()

    def start(self) -> Tuple[bool, str]:
        """Start the camera."""
        if self._state != CameraState.OFF:
            return False, "Camera already started"

        self._state = CameraState.INITIALIZING

        # Simulate initialization delay
        time.sleep(0.1)

        self._state = CameraState.READY
        self._last_update = time.time()

        return True, "Camera started successfully"

    def stop(self) -> Tuple[bool, str]:
        """Stop the camera."""
        if self._state == CameraState.OFF:
            return False, "Camera already stopped"

        # Stop streaming if active
        if self._streaming:
            self.stop_streaming()

        # Stop recording if active
        if self._is_recording:
            self.stop_recording()

        self._state = CameraState.OFF
        return True, "Camera stopped"

    def start_streaming(self) -> Tuple[bool, str]:
        """Start frame streaming."""
        if self._state not in (CameraState.READY, CameraState.RECORDING):
            return False, f"Cannot start streaming in state: {self._state.name}"

        if self._streaming:
            return False, "Already streaming"

        self._streaming = True
        if self._state == CameraState.READY:
            self._state = CameraState.STREAMING

        self._stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._stream_thread.start()

        return True, "Streaming started"

    def stop_streaming(self) -> Tuple[bool, str]:
        """Stop frame streaming."""
        if not self._streaming:
            return False, "Not streaming"

        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None

        if self._state == CameraState.STREAMING:
            self._state = CameraState.READY

        return True, "Streaming stopped"

    def _streaming_loop(self):
        """Internal streaming loop."""
        frame_interval = 1.0 / self._fps

        while self._streaming:
            frame = self.capture_frame()
            if frame:
                # Push to active recorder, if any
                if self._recorder is not None and self._recorder.state == RecorderState.RECORDING:
                    try:
                        sw, sh = self._synth_resolution
                        det_boxes = self._scaled_detection_boxes(frame, sw, sh)
                        rgb = synthesize_rgb_bytes(
                            sw, sh,
                            vision_mode=frame.mode.value,
                            detections=det_boxes,
                            timestamp=frame.timestamp,
                        )
                        self._recorder.add_frame(rgb, timestamp=frame.timestamp)
                    except Exception:
                        pass

                # Notify callbacks
                for callback in self._frame_callbacks:
                    try:
                        callback(frame)
                    except Exception:
                        pass

            time.sleep(frame_interval)

    def _scaled_detection_boxes(
        self, frame: 'Frame', target_w: int, target_h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Scale detection bounding boxes into the synth-resolution frame."""
        if not frame.detections:
            return []
        sx = target_w / max(1, self._resolution[0])
        sy = target_h / max(1, self._resolution[1])
        boxes: List[Tuple[int, int, int, int]] = []
        for det in frame.detections:
            bb = det.bounding_box
            boxes.append((
                int(bb.x * sx),
                int(bb.y * sy),
                max(1, int(bb.width * sx)),
                max(1, int(bb.height * sy)),
            ))
        return boxes

    def capture_frame(self) -> Optional[Frame]:
        """
        Capture a single frame.

        Returns:
            Captured frame with detections, or None if capture failed
        """
        if self._state in (CameraState.OFF, CameraState.ERROR):
            return None

        timestamp = time.time()

        # Create frame
        frame = Frame(
            width=self._resolution[0],
            height=self._resolution[1],
            timestamp=timestamp,
            mode=self._vision_mode,
        )

        # Run detection if enabled
        if self._detection_enabled:
            detections = self._run_detection()
            frame.detections = detections
            self._detections = detections

            # Notify detection callbacks
            for detection in detections:
                for callback in self._detection_callbacks:
                    try:
                        callback(detection)
                    except Exception:
                        pass

        self._last_frame = frame
        self._last_update = timestamp

        return frame

    def _run_detection(self) -> List[Detection]:
        """
        Run object detection on current view.

        In simulation mode, this generates synthetic detections.
        In real implementation, this would use computer vision/ML.
        """
        detections = []

        # Check for simulated landing marker
        if self._marker_detection_enabled and self._simulated_marker:
            marker = self._simulated_marker
            detection = Detection(
                detection_type=DetectionType.LANDING_MARKER,
                confidence=marker.get('confidence', 0.95),
                bounding_box=BoundingBox(
                    x=marker.get('x', self._resolution[0] // 2 - 50),
                    y=marker.get('y', self._resolution[1] // 2 - 50),
                    width=marker.get('width', 100),
                    height=marker.get('height', 100),
                ),
                distance=marker.get('distance', 5.0),
                metadata={'marker_id': marker.get('id', 'default')},
            )
            detections.append(detection)
            self._marker_detected = True
            self._marker_position = detection.bounding_box.center
        else:
            self._marker_detected = False
            self._marker_position = None

        # Check for simulated obstacles
        if self._obstacle_detection_enabled:
            for obstacle in self._simulated_obstacles:
                if obstacle.get('visible', True):
                    detection = Detection(
                        detection_type=DetectionType(obstacle.get('type', 'obstacle')),
                        confidence=obstacle.get('confidence', 0.8),
                        bounding_box=BoundingBox(
                            x=obstacle.get('x', 100),
                            y=obstacle.get('y', 100),
                            width=obstacle.get('width', 50),
                            height=obstacle.get('height', 50),
                        ),
                        distance=obstacle.get('distance', 10.0),
                        metadata=obstacle.get('metadata', {}),
                    )
                    detections.append(detection)

        return detections

    def start_recording(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """Start video recording.

        If a ``MediaStorage`` was supplied at construction time, a real
        ``.dvr`` bundle directory is created and streaming frames are written
        into it. Otherwise this falls back to a path-only stub.
        """
        if self._state not in (CameraState.READY, CameraState.STREAMING):
            return False, f"Cannot record in state: {self._state.name}"

        if self._is_recording:
            return False, "Already recording"

        self._is_recording = True
        self._recording_start_time = time.time()

        if self._media_storage is not None:
            sw, sh = self._synth_resolution
            bundle = path if path else None
            self._recorder = VideoRecorder(
                storage=self._media_storage,
                width=sw,
                height=sh,
                target_fps=float(self._fps),
                bundle_path=bundle,
            )
            self._recorder.start()
            self._recording_path = self._recorder.bundle_path
        else:
            self._recorder = None
            self._recording_path = path or f"recording_{int(time.time())}.mp4"

        if self._state == CameraState.READY:
            self._state = CameraState.RECORDING

        return True, f"Recording started: {self._recording_path}"

    def stop_recording(self) -> Tuple[bool, str]:
        """Stop video recording."""
        if not self._is_recording:
            return False, "Not recording"

        duration = time.time() - (self._recording_start_time or 0)
        path = self._recording_path
        summary_dict: Optional[Dict[str, Any]] = None

        if self._recorder is not None:
            summary = self._recorder.stop()
            duration = summary.duration_s
            path = summary.bundle_path
            summary_dict = {
                "bundle_path": summary.bundle_path,
                "manifest_path": summary.manifest_path,
                "frame_count": summary.frame_count,
                "duration_s": summary.duration_s,
                "bytes_written": summary.bytes_written,
                "width": summary.width,
                "height": summary.height,
                "effective_fps": summary.fps_effective,
            }
            self._recorder = None

        self._last_recording_summary = summary_dict
        self._is_recording = False
        self._recording_start_time = None
        self._recording_path = None

        if self._state == CameraState.RECORDING:
            self._state = CameraState.READY

        return True, f"Recording saved: {path} ({duration:.1f}s)"

    def take_snapshot(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """Take a snapshot image. Writes real PPM (+JPEG if Pillow) to disk."""
        frame = self.capture_frame()
        if not frame:
            return False, "Failed to capture frame"

        if self._media_storage is None:
            snapshot_path = path or f"snapshot_{int(time.time())}.jpg"
            return True, f"Snapshot saved: {snapshot_path}"

        sw, sh = self._synth_resolution
        det_boxes = self._scaled_detection_boxes(frame, sw, sh)
        rgb = synthesize_rgb_bytes(
            sw, sh,
            vision_mode=frame.mode.value,
            detections=det_boxes,
            timestamp=frame.timestamp,
        )

        if path is None:
            base = self._media_storage.allocate_photo_path(ext="png")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            base = path
        stem = os.path.splitext(base)[0]
        png_path = stem + ".png"
        ppm_path = stem + ".ppm"

        png_bytes = write_png(png_path, rgb, sw, sh)
        ppm_bytes = write_ppm(ppm_path, rgb, sw, sh)
        jpeg_path = stem + ".jpg"
        jpeg_bytes = write_jpeg_if_possible(jpeg_path, rgb, sw, sh)

        # PNG is the user-facing artifact; PPM stays alongside for raw-tooling.
        artifact_path = png_path
        total_bytes = png_bytes + ppm_bytes + (jpeg_bytes or 0)
        from sensors.media.storage import MediaArtifact
        self._media_storage.register(MediaArtifact(
            kind="photo",
            path=artifact_path,
            width=sw,
            height=sh,
            timestamp=frame.timestamp,
            bytes_written=total_bytes,
            extra={
                "png_path": png_path,
                "ppm_path": ppm_path,
                "jpeg_path": jpeg_path if jpeg_bytes is not None else None,
                "vision_mode": frame.mode.value,
                "detection_count": len(frame.detections),
            },
        ))
        return True, f"Snapshot saved: {artifact_path}"

    def get_last_recording_summary(self) -> Optional[Dict[str, Any]]:
        """Return the summary dict for the most recently stopped recording."""
        return self._last_recording_summary

    @property
    def media_storage(self) -> Optional[MediaStorage]:
        return self._media_storage

    # ==================== Resolution / HDR / Color profile ====================

    def set_resolution(self, resolution: Tuple[int, int]) -> Tuple[bool, str]:
        """Switch capture resolution. Updates synth resolution proportionally."""
        if not (isinstance(resolution, tuple) and len(resolution) == 2):
            return False, "resolution must be a (width, height) tuple"
        w, h = resolution
        if w <= 0 or h <= 0:
            return False, "resolution must be positive"
        self._resolution = (int(w), int(h))
        sw = min(w, 160)
        sh = max(1, int(round(sw * h / max(1, w))))
        self._synth_resolution = (sw, sh)
        return True, f"resolution set to {w}x{h}"

    def enable_hdr(self, enabled: bool = True) -> None:
        """Enable HDR capture (metadata flag in sim)."""
        self._hdr_enabled = bool(enabled)

    def is_hdr_enabled(self) -> bool:
        return self._hdr_enabled

    def set_color_profile(self, profile: str) -> Tuple[bool, str]:
        """Tag the color profile (e.g. ``"hlg"``, ``"dlog-m"``)."""
        valid = {
            self.COLOR_PROFILE_STANDARD,
            self.COLOR_PROFILE_HDR_10,
            self.COLOR_PROFILE_HLG,
            self.COLOR_PROFILE_DLOG_M,
            self.COLOR_PROFILE_REC_709,
        }
        if profile not in valid:
            return False, f"unknown profile: {profile}; valid: {sorted(valid)}"
        self._color_profile = profile
        return True, f"color profile set to {profile}"

    def get_color_profile(self) -> str:
        return self._color_profile

    def set_bitrate_mbps(self, mbps: Optional[float]) -> None:
        """Tag the target encoding bitrate (metadata in sim)."""
        self._bitrate_mbps = None if mbps is None else float(max(0.1, mbps))

    def get_bitrate_mbps(self) -> Optional[float]:
        return self._bitrate_mbps

    # ==================== Vision Mode ====================

    def set_vision_mode(self, mode: VisionMode) -> Tuple[bool, str]:
        """Set the vision processing mode."""
        if self._state == CameraState.OFF:
            return False, "Camera is off"

        self._vision_mode = mode
        return True, f"Vision mode set to: {mode.value}"

    def get_vision_mode(self) -> VisionMode:
        """Get current vision mode."""
        return self._vision_mode

    # ==================== Gimbal Control ====================

    def set_gimbal_position(
        self,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Set gimbal position.

        Args:
            pitch: -90 (down) to +30 (up)
            roll: -45 to +45
            yaw: -180 to +180
        """
        if pitch is not None:
            self._gimbal.pitch = max(-90.0, min(30.0, pitch))

        if roll is not None:
            self._gimbal.roll = max(-45.0, min(45.0, roll))

        if yaw is not None:
            self._gimbal.yaw = max(-180.0, min(180.0, yaw))

        return True, f"Gimbal position: pitch={self._gimbal.pitch}, roll={self._gimbal.roll}, yaw={self._gimbal.yaw}"

    def get_gimbal_position(self) -> GimbalPosition:
        """Get current gimbal position."""
        return self._gimbal

    def point_down(self) -> Tuple[bool, str]:
        """Point camera straight down (for landing)."""
        return self.set_gimbal_position(pitch=-90.0, roll=0.0, yaw=0.0)

    def point_forward(self) -> Tuple[bool, str]:
        """Point camera forward."""
        return self.set_gimbal_position(pitch=0.0, roll=0.0, yaw=0.0)

    # ==================== Detection Settings ====================

    def enable_detection(self, enabled: bool = True):
        """Enable/disable object detection."""
        self._detection_enabled = enabled

    def enable_obstacle_detection(self, enabled: bool = True):
        """Enable/disable obstacle detection."""
        self._obstacle_detection_enabled = enabled

    def enable_marker_detection(self, enabled: bool = True):
        """Enable/disable landing marker detection."""
        self._marker_detection_enabled = enabled

    def set_min_confidence(self, confidence: float):
        """Set minimum detection confidence threshold (0.0-1.0)."""
        self._min_confidence = max(0.0, min(1.0, confidence))

    # ==================== Marker Detection ====================

    def is_marker_detected(self) -> bool:
        """Check if landing marker is currently detected."""
        return self._marker_detected

    def get_marker_position(self) -> Optional[Tuple[int, int]]:
        """Get marker position in frame coordinates (x, y)."""
        return self._marker_position

    def get_marker_offset(self) -> Optional[Tuple[float, float]]:
        """
        Get marker offset from frame center as normalized coordinates.

        Returns:
            (x_offset, y_offset) where each is -1.0 to +1.0
            Negative x = marker is left of center
            Negative y = marker is above center
        """
        if not self._marker_position:
            return None

        center_x = self._resolution[0] / 2
        center_y = self._resolution[1] / 2

        x_offset = (self._marker_position[0] - center_x) / center_x
        y_offset = (self._marker_position[1] - center_y) / center_y

        return (x_offset, y_offset)

    # ==================== Simulation Methods ====================

    def simulate_marker(
        self,
        visible: bool = True,
        x: Optional[int] = None,
        y: Optional[int] = None,
        distance: float = 5.0,
        marker_id: str = "landing_pad",
    ):
        """Simulate a landing marker for testing."""
        if visible:
            self._simulated_marker = {
                'id': marker_id,
                'x': x or (self._resolution[0] // 2 - 50),
                'y': y or (self._resolution[1] // 2 - 50),
                'width': 100,
                'height': 100,
                'distance': distance,
                'confidence': 0.95,
            }
        else:
            self._simulated_marker = None

    def simulate_obstacle(
        self,
        obstacle_type: DetectionType = DetectionType.OBSTACLE,
        x: int = 100,
        y: int = 100,
        width: int = 50,
        height: int = 50,
        distance: float = 10.0,
        confidence: float = 0.8,
        visible: bool = True,
    ) -> int:
        """
        Add a simulated obstacle for testing.

        Returns:
            Index of the added obstacle
        """
        obstacle = {
            'type': obstacle_type.value,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'distance': distance,
            'confidence': confidence,
            'visible': visible,
        }
        self._simulated_obstacles.append(obstacle)
        return len(self._simulated_obstacles) - 1

    def clear_simulated_obstacles(self):
        """Clear all simulated obstacles."""
        self._simulated_obstacles = []

    # ==================== Callbacks ====================

    def add_frame_callback(self, callback: Callable[[Frame], None]):
        """Add callback for frame capture events."""
        self._frame_callbacks.append(callback)

    def add_detection_callback(self, callback: Callable[[Detection], None]):
        """Add callback for detection events."""
        self._detection_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[Frame], None]):
        """Remove frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def remove_detection_callback(self, callback: Callable[[Detection], None]):
        """Remove detection callback."""
        if callback in self._detection_callbacks:
            self._detection_callbacks.remove(callback)

    # ==================== Health & Status ====================

    def update(self):
        """Update sensor (required by base class)."""
        self._last_update = time.time()

    def is_valid(self) -> bool:
        """Check if camera is operational."""
        return self._state not in (CameraState.OFF, CameraState.ERROR)

    def is_healthy(self) -> bool:
        """Check if camera is healthy."""
        return self._healthy and self._state != CameraState.ERROR

    def get_state(self) -> CameraState:
        """Get current camera state."""
        return self._state

    def get_last_frame(self) -> Optional[Frame]:
        """Get the last captured frame."""
        return self._last_frame

    def get_detections(self) -> List[Detection]:
        """Get current detections."""
        return self._detections.copy()

    def get_obstacles(self) -> List[Detection]:
        """Get only obstacle detections."""
        return [
            d for d in self._detections
            if d.detection_type in (
                DetectionType.OBSTACLE,
                DetectionType.TREE,
                DetectionType.BUILDING,
                DetectionType.POWER_LINE,
            )
        ]

    def get_closest_obstacle(self) -> Optional[Detection]:
        """Get the closest detected obstacle."""
        obstacles = self.get_obstacles()
        if not obstacles:
            return None

        return min(obstacles, key=lambda d: d.distance or float('inf'))

    def get_status(self) -> dict:
        """Get camera status."""
        return {
            'state': self._state.name,
            'vision_mode': self._vision_mode.value,
            'resolution': self._resolution,
            'fps': self._fps,
            'fov': self._fov,
            'gimbal': self._gimbal.to_dict(),
            'is_streaming': self._streaming,
            'is_recording': self._is_recording,
            'recording_duration': (
                time.time() - self._recording_start_time
                if self._is_recording and self._recording_start_time
                else None
            ),
            'detection_enabled': self._detection_enabled,
            'obstacle_detection_enabled': self._obstacle_detection_enabled,
            'marker_detection_enabled': self._marker_detection_enabled,
            'marker_detected': self._marker_detected,
            'marker_position': self._marker_position,
            'detection_count': len(self._detections),
            'healthy': self._healthy,
            'last_update': self._last_update,
        }

    def to_dict(self) -> dict:
        """Serialize camera state."""
        status = self.get_status()
        status['last_frame'] = self._last_frame.to_dict() if self._last_frame else None
        status['detections'] = [d.to_dict() for d in self._detections]
        return status
