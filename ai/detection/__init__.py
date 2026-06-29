"""Detection module for object detection and tracking."""

from .yolo_detector import YOLODetector, Detection, DetectionClass
from .detection_tracker import DetectionTracker, TrackedObject, TrackState

__all__ = [
    'YOLODetector',
    'Detection',
    'DetectionClass',
    'DetectionTracker',
    'TrackedObject',
    'TrackState',
]
