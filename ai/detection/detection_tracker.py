"""
Multi-object tracking for detected objects.

Uses IOU-based association and Kalman filtering
to track objects across frames.
"""

import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

from .yolo_detector import Detection, DetectionClass

logger = logging.getLogger(__name__)

# Optional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TrackState(Enum):
    """State of a tracked object."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class TrackedObject:
    """A tracked object across multiple frames."""
    track_id: int
    detection_class: DetectionClass
    state: TrackState = TrackState.TENTATIVE
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 0
    time_since_update: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: List[Tuple[float, float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    @property
    def lifetime(self) -> float:
        """Time since first seen in seconds."""
        return self.last_seen - self.first_seen

    def predict_position(
        self,
        dt: float = 1.0,
    ) -> Tuple[float, float]:
        """Predict future position based on velocity."""
        px = self.center[0] + self.velocity[0] * dt
        py = self.center[1] + self.velocity[1] * dt
        return (px, py)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'track_id': self.track_id,
            'class': self.detection_class.value,
            'state': self.state.value,
            'bbox': self.bbox,
            'center': self.center,
            'velocity': self.velocity,
            'confidence': self.confidence,
            'hits': self.hits,
            'age': self.age,
            'lifetime': self.lifetime,
        }


@dataclass
class TrackerConfig:
    """Configuration for object tracker."""
    max_age: int = 30  # Frames before deleting lost track
    min_hits: int = 3  # Hits to confirm track
    iou_threshold: float = 0.3
    max_tracks: int = 100
    velocity_smoothing: float = 0.7
    history_length: int = 30


class DetectionTracker:
    """
    Multi-object tracker using IOU association.

    Associates detections across frames and
    maintains track history.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize tracker.

        Args:
            config: Tracker configuration
        """
        self._config = config or TrackerConfig()
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1
        self._frame_count = 0

    def update(
        self,
        detections: List[Detection],
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detections from current frame

        Returns:
            List of active tracked objects
        """
        self._frame_count += 1
        current_time = time.time()

        # Predict new locations for existing tracks
        for track in self._tracks.values():
            self._predict_track(track)

        # Associate detections with tracks
        matched, unmatched_dets, unmatched_tracks = (
            self._associate(detections)
        )

        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            self._update_track(
                self._tracks[track_id],
                det,
                current_time
            )

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(self._tracks) < self._config.max_tracks:
                self._create_track(detections[det_idx], current_time)

        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.misses += 1
            track.time_since_update += 1

            if track.time_since_update > self._config.max_age:
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

        # Remove deleted tracks
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.state != TrackState.DELETED
        }

        # Return active tracks
        return self.get_active_tracks()

    def _predict_track(self, track: TrackedObject) -> None:
        """Predict track position for next frame."""
        track.age += 1

        # Simple velocity-based prediction
        if track.velocity != (0.0, 0.0):
            new_cx = track.center[0] + track.velocity[0]
            new_cy = track.center[1] + track.velocity[1]

            # Update bbox based on predicted center
            half_w = track.width / 2
            half_h = track.height / 2
            track.bbox = (
                int(new_cx - half_w),
                int(new_cy - half_h),
                int(new_cx + half_w),
                int(new_cy + half_h),
            )
            track.center = (new_cx, new_cy)

    def _associate(
        self,
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks.

        Returns:
            matched: List of (track_id, detection_index) pairs
            unmatched_detections: Detection indices without match
            unmatched_tracks: Track IDs without match
        """
        if not self._tracks or not detections:
            return (
                [],
                list(range(len(detections))),
                list(self._tracks.keys())
            )

        # Compute IOU matrix
        track_ids = list(self._tracks.keys())
        iou_matrix = []

        for tid in track_ids:
            track = self._tracks[tid]
            row = []
            for det in detections:
                iou = self._compute_iou(track.bbox, det.bbox)
                row.append(iou)
            iou_matrix.append(row)

        # Greedy matching
        matched = []
        used_dets = set()
        used_tracks = set()

        # Find best matches
        while True:
            best_iou = self._config.iou_threshold
            best_match = None

            for i, tid in enumerate(track_ids):
                if tid in used_tracks:
                    continue
                for j in range(len(detections)):
                    if j in used_dets:
                        continue
                    if iou_matrix[i][j] > best_iou:
                        best_iou = iou_matrix[i][j]
                        best_match = (tid, j)

            if best_match is None:
                break

            matched.append(best_match)
            used_tracks.add(best_match[0])
            used_dets.add(best_match[1])

        unmatched_dets = [
            i for i in range(len(detections))
            if i not in used_dets
        ]
        unmatched_tracks = [
            tid for tid in track_ids
            if tid not in used_tracks
        ]

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _update_track(
        self,
        track: TrackedObject,
        detection: Detection,
        current_time: float,
    ) -> None:
        """Update track with new detection."""
        # Calculate velocity
        old_center = track.center
        new_center = detection.center

        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]

        # Smooth velocity
        alpha = self._config.velocity_smoothing
        track.velocity = (
            alpha * track.velocity[0] + (1 - alpha) * dx,
            alpha * track.velocity[1] + (1 - alpha) * dy,
        )

        # Update position
        track.bbox = detection.bbox
        track.center = detection.center
        track.confidence = detection.confidence
        track.detection_class = detection.detection_class
        track.last_seen = current_time
        track.hits += 1
        track.time_since_update = 0

        # Update history
        track.history.append(track.center)
        if len(track.history) > self._config.history_length:
            track.history.pop(0)

        # Update state
        if (
            track.state == TrackState.TENTATIVE and
            track.hits >= self._config.min_hits
        ):
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _create_track(
        self,
        detection: Detection,
        current_time: float,
    ) -> TrackedObject:
        """Create new track from detection."""
        track = TrackedObject(
            track_id=self._next_id,
            detection_class=detection.detection_class,
            state=TrackState.TENTATIVE,
            bbox=detection.bbox,
            center=detection.center,
            confidence=detection.confidence,
            first_seen=current_time,
            last_seen=current_time,
            history=[detection.center],
        )

        self._tracks[self._next_id] = track
        self._next_id += 1

        return track

    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active (confirmed or lost) tracks."""
        return [
            t for t in self._tracks.values()
            if t.state in [TrackState.CONFIRMED, TrackState.LOST]
        ]

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get only confirmed tracks."""
        return [
            t for t in self._tracks.values()
            if t.state == TrackState.CONFIRMED
        ]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get track by ID."""
        return self._tracks.get(track_id)

    def delete_track(self, track_id: int) -> bool:
        """Delete a specific track."""
        if track_id in self._tracks:
            del self._tracks[track_id]
            return True
        return False

    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    @property
    def track_count(self) -> int:
        """Number of active tracks."""
        return len(self.get_active_tracks())

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        active = self.get_active_tracks()
        return {
            'frame_count': self._frame_count,
            'total_tracks': len(self._tracks),
            'active_tracks': len(active),
            'confirmed_tracks': len(self.get_confirmed_tracks()),
            'next_id': self._next_id,
        }


class TrackAnalyzer:
    """Analyzes tracked objects for patterns."""

    @staticmethod
    def estimate_direction(
        track: TrackedObject,
    ) -> Optional[float]:
        """
        Estimate movement direction in degrees.

        Returns:
            Angle in degrees (0=right, 90=down) or None
        """
        if track.velocity == (0.0, 0.0):
            return None

        vx, vy = track.velocity
        angle = math.atan2(vy, vx)
        return math.degrees(angle)

    @staticmethod
    def estimate_speed(track: TrackedObject) -> float:
        """Estimate speed in pixels per frame."""
        vx, vy = track.velocity
        return math.sqrt(vx * vx + vy * vy)

    @staticmethod
    def is_approaching(
        track: TrackedObject,
        point: Tuple[float, float],
    ) -> bool:
        """Check if track is moving toward a point."""
        if track.velocity == (0.0, 0.0):
            return False

        # Vector from track to point
        to_point = (
            point[0] - track.center[0],
            point[1] - track.center[1],
        )

        # Dot product with velocity
        dot = (
            to_point[0] * track.velocity[0] +
            to_point[1] * track.velocity[1]
        )

        return dot > 0

    @staticmethod
    def predict_collision(
        track: TrackedObject,
        bbox: Tuple[int, int, int, int],
        max_frames: int = 30,
    ) -> Optional[int]:
        """
        Predict if track will collide with bbox.

        Returns:
            Frame count until collision or None
        """
        if track.velocity == (0.0, 0.0):
            return None

        for frame in range(1, max_frames + 1):
            px, py = track.predict_position(frame)

            # Check if predicted center is in bbox
            if (
                bbox[0] <= px <= bbox[2] and
                bbox[1] <= py <= bbox[3]
            ):
                return frame

        return None
