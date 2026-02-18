"""YOLO-World open-vocabulary object detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Default vocabulary covering common scene objects
DEFAULT_CLASS_NAMES: List[str] = [
    "car", "truck", "bus", "van", "motorcycle", "bicycle",
    "person", "pedestrian", "child",
    "traffic light", "traffic sign", "stop sign",
    "fire hydrant", "bench", "trash can",
    "dog", "cat", "bird",
    "tree", "pole", "building", "fence", "wall",
    "road", "sidewalk", "crosswalk",
    "cone", "barrier", "bollard",
]


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for YOLO-World detector."""
    model_name: str = "yolov8x-worldv2"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda"
    max_detections: int = 50
    class_names: Tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)


@dataclass(frozen=True)
class Detection:
    """Single object detection result."""
    bbox: np.ndarray        # (4,) [x1, y1, x2, y2] in pixel coordinates
    label: str              # detected class name, e.g. "car"
    confidence: float       # detection confidence [0, 1]
    class_id: int           # index into class_names


@dataclass(frozen=True)
class FrameDetections:
    """All detections for a single frame."""
    detections: List[Detection]
    image_size: Tuple[int, int]  # (width, height) of input image


class YOLOWorldDetector:
    """YOLO-World open-vocabulary object detector.

    Usage:
        detector = YOLOWorldDetector(DetectionConfig())
        detections = detector.detect(image_bgr)
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self._config = config or DetectionConfig()
        self._model = None

    def _lazy_load(self) -> None:
        """Load model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ImportError(
                "YOLO-World requires ultralytics. Install with: pip install ultralytics"
            ) from exc

        self._model = YOLOWorld(self._config.model_name)
        self._model.set_classes(list(self._config.class_names))

    def detect(self, image: np.ndarray) -> FrameDetections:
        """Run detection on a single image.

        Args:
            image: (H, W, 3) BGR or RGB uint8 image.

        Returns:
            FrameDetections with all detections for this image.
        """
        self._lazy_load()

        h, w = image.shape[:2]
        results = self._model.predict(
            image,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            device=self._config.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
                confs = boxes.conf.cpu().numpy()       # (N,)
                cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

                # Sort by confidence descending, take top max_detections
                order = np.argsort(-confs)[:self._config.max_detections]

                class_names = list(self._config.class_names)
                for idx in order:
                    cid = int(cls_ids[idx])
                    label = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                    detections.append(Detection(
                        bbox=xyxy[idx].astype(np.float64),
                        label=label,
                        confidence=float(confs[idx]),
                        class_id=cid,
                    ))

        return FrameDetections(detections=detections, image_size=(w, h))

    def detect_batch(self, images: List[np.ndarray]) -> List[FrameDetections]:
        """Run detection on multiple images.

        Args:
            images: List of (H, W, 3) images.

        Returns:
            List of FrameDetections, one per image.
        """
        return [self.detect(img) for img in images]

    def set_classes(self, class_names: List[str]) -> None:
        """Update the detection vocabulary at runtime."""
        self._lazy_load()
        self._model.set_classes(class_names)
        # Update config class_names reference (note: config is frozen, so we just update model)
