"""
YOLO-based object detection for drone vision.

Supports YOLOv8 for real-time detection with
fallback simulation mode when ultralytics unavailable.
"""

import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.info("ultralytics not installed, using simulation mode")

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class DetectionClass(Enum):
    """Common detection classes for drone operations."""
    PERSON = "person"
    VEHICLE = "vehicle"
    CAR = "car"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    BOAT = "boat"
    AIRCRAFT = "aircraft"
    ANIMAL = "animal"
    BUILDING = "building"
    TREE = "tree"
    OBSTACLE = "obstacle"
    LANDING_PAD = "landing_pad"
    MARKER = "marker"
    UNKNOWN = "unknown"


# COCO class mapping to our classes
COCO_TO_DETECTION = {
    0: DetectionClass.PERSON,
    1: DetectionClass.BICYCLE,
    2: DetectionClass.CAR,
    3: DetectionClass.MOTORCYCLE,
    5: DetectionClass.VEHICLE,  # bus
    7: DetectionClass.TRUCK,
    8: DetectionClass.BOAT,
    4: DetectionClass.AIRCRAFT,  # airplane
    14: DetectionClass.ANIMAL,  # bird
    15: DetectionClass.ANIMAL,  # cat
    16: DetectionClass.ANIMAL,  # dog
    17: DetectionClass.ANIMAL,  # horse
    18: DetectionClass.ANIMAL,  # sheep
    19: DetectionClass.ANIMAL,  # cow
}


@dataclass
class Detection:
    """A single object detection result."""
    class_id: int
    class_name: str
    detection_class: DetectionClass
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float
    timestamp: float = field(default_factory=time.time)
    track_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'detection_class': self.detection_class.value,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'timestamp': self.timestamp,
            'track_id': self.track_id,
        }


@dataclass
class DetectorConfig:
    """Configuration for YOLO detector."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[int]] = None  # Filter classes
    device: str = "auto"  # auto, cpu, cuda:0
    img_size: int = 640
    half_precision: bool = False
    verbose: bool = False


class YOLODetector:
    """
    YOLO-based object detector.

    Uses ultralytics YOLOv8 when available,
    falls back to simulation for testing.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize YOLO detector.

        Args:
            config: Detector configuration
        """
        self._config = config or DetectorConfig()
        self._model = None
        self._is_simulation = not YOLO_AVAILABLE
        self._frame_count = 0
        self._last_inference_time = 0.0
        self._class_names: Dict[int, str] = {}

        if not self._is_simulation:
            self._load_model()

    def _load_model(self) -> bool:
        """Load YOLO model."""
        try:
            self._model = YOLO(self._config.model_path)

            # Get class names
            if hasattr(self._model, 'names'):
                self._class_names = self._model.names

            # Set device
            device = self._config.device
            if device == "auto":
                device = "cuda:0" if self._check_cuda() else "cpu"

            logger.info(
                f"YOLO model loaded: {self._config.model_path} "
                f"on {device}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._is_simulation = True
            return False

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_simulation(self) -> bool:
        """Check if running in simulation mode."""
        return self._is_simulation

    @property
    def last_inference_time(self) -> float:
        """Get last inference time in milliseconds."""
        return self._last_inference_time

    def detect(
        self,
        image: Any,
        return_annotated: bool = False,
    ) -> Tuple[List[Detection], Optional[Any]]:
        """
        Run object detection on an image.

        Args:
            image: Input image (numpy array or path)
            return_annotated: Return annotated image

        Returns:
            Tuple of (detections, annotated_image or None)
        """
        self._frame_count += 1

        if self._is_simulation:
            return self._simulate_detection(image)

        return self._run_inference(image, return_annotated)

    def _run_inference(
        self,
        image: Any,
        return_annotated: bool,
    ) -> Tuple[List[Detection], Optional[Any]]:
        """Run actual YOLO inference."""
        start_time = time.time()

        try:
            results = self._model(
                image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                max_det=self._config.max_detections,
                classes=self._config.classes,
                imgsz=self._config.img_size,
                half=self._config.half_precision,
                verbose=self._config.verbose,
            )

            self._last_inference_time = (
                (time.time() - start_time) * 1000
            )

            detections = self._parse_results(results[0])
            annotated = None

            if return_annotated:
                annotated = results[0].plot()

            return detections, annotated

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return [], None

    def _parse_results(self, result: Any) -> List[Detection]:
        """Parse YOLO results to Detection objects."""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            # Get box data
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            # Get class name
            class_name = self._class_names.get(
                class_id,
                f"class_{class_id}"
            )

            # Map to detection class
            det_class = COCO_TO_DETECTION.get(
                class_id,
                DetectionClass.UNKNOWN
            )

            # Calculate center and area
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            # Get track ID if available
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                detection_class=det_class,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area=area,
                track_id=track_id,
            )
            detections.append(detection)

        return detections

    def _simulate_detection(
        self,
        image: Any,
    ) -> Tuple[List[Detection], None]:
        """Generate simulated detections for testing."""
        detections = []

        # Determine image size
        if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = 640, 480

        # Randomly generate 0-3 detections
        num_detections = random.randint(0, 3)

        for i in range(num_detections):
            # Random class
            det_class = random.choice([
                DetectionClass.PERSON,
                DetectionClass.VEHICLE,
                DetectionClass.ANIMAL,
                DetectionClass.OBSTACLE,
            ])

            # Random bbox
            box_w = random.randint(30, 150)
            box_h = random.randint(40, 200)
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            # Random confidence
            conf = random.uniform(0.5, 0.95)

            detection = Detection(
                class_id=0,
                class_name=det_class.value,
                detection_class=det_class,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) / 2, (y1 + y2) / 2),
                area=(x2 - x1) * (y2 - y1),
                track_id=i + self._frame_count * 10,
            )
            detections.append(detection)

        self._last_inference_time = random.uniform(15, 30)
        return detections, None

    def detect_track(
        self,
        image: Any,
        persist: bool = True,
    ) -> List[Detection]:
        """
        Run detection with tracking.

        Args:
            image: Input image
            persist: Persist tracks across frames

        Returns:
            List of detections with track IDs
        """
        if self._is_simulation:
            detections, _ = self._simulate_detection(image)
            return detections

        try:
            results = self._model.track(
                image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                persist=persist,
                verbose=self._config.verbose,
            )
            return self._parse_results(results[0])

        except Exception as e:
            logger.error(f"Track error: {e}")
            return []

    def set_classes(self, class_ids: List[int]) -> None:
        """Set which classes to detect."""
        self._config.classes = class_ids

    def reset_tracker(self) -> None:
        """Reset the internal tracker state."""
        if self._model and hasattr(self._model, 'predictor'):
            if hasattr(self._model.predictor, 'trackers'):
                self._model.predictor.trackers = []

    def warmup(self) -> None:
        """Warmup the model with a dummy image."""
        if self._is_simulation:
            return

        if NUMPY_AVAILABLE:
            dummy = np.zeros(
                (self._config.img_size, self._config.img_size, 3),
                dtype=np.uint8
            )
            self.detect(dummy)
            logger.info("YOLO detector warmed up")

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'is_simulation': self._is_simulation,
            'frame_count': self._frame_count,
            'last_inference_ms': self._last_inference_time,
            'model_path': self._config.model_path,
            'confidence_threshold': self._config.confidence_threshold,
        }
