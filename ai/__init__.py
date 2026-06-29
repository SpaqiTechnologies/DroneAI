"""
AI/ML integration module for drone intelligence.

Provides object detection, tracking, terrain classification,
and predictive maintenance capabilities.
"""

from .detection.yolo_detector import (
    YOLODetector,
    Detection,
    DetectionClass,
)
from .detection.detection_tracker import (
    DetectionTracker,
    TrackedObject,
    TrackState,
)
from .terrain.terrain_classifier import (
    TerrainClassifier,
    TerrainType,
    LandingZone,
)
from .anomaly.anomaly_detector import (
    AnomalyDetector,
    Anomaly,
    AnomalyType,
)

__all__ = [
    # Detection
    'YOLODetector',
    'Detection',
    'DetectionClass',
    # Tracking
    'DetectionTracker',
    'TrackedObject',
    'TrackState',
    # Terrain
    'TerrainClassifier',
    'TerrainType',
    'LandingZone',
    # Anomaly
    'AnomalyDetector',
    'Anomaly',
    'AnomalyType',
]
