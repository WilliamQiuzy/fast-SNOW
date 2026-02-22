"""FastSAM wrapper for Fast-SNOW pipeline.

Provides class-agnostic instance segmentation via FastSAM.
Each detection includes a binary mask and a normalized bounding box
suitable for SAM3 bbox prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from fast_snow.engine.config.fast_snow_config import FastSAMConfig

logger = logging.getLogger(__name__)


@dataclass
class FastSAMDetection:
    """Single FastSAM detection with mask and normalized bbox."""

    mask: np.ndarray  # (H, W) bool, full resolution
    bbox_xywh_norm: Tuple[float, float, float, float]  # [xmin, ymin, w, h] in [0,1]
    score: float
    instance_idx: int


class FastSAMWrapper:
    """Wrapper around FastSAM for class-agnostic instance segmentation."""

    def __init__(self, config: Optional[FastSAMConfig] = None):
        self.config = config or FastSAMConfig()
        self._model = None

    def load(self) -> None:
        """Load the FastSAM model. Called lazily on first inference."""
        if self._model is not None:
            return

        from ultralytics import FastSAM

        model_path = str(Path(self.config.model_path).resolve())
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"FastSAM weights not found: {model_path}. "
                f"Place FastSAM-s.pt in {Path(self.config.model_path).parent}."
            )
        self._model = FastSAM(model_path)
        logger.info("FastSAM loaded from %s", model_path)

    def unload(self) -> None:
        """Release the FastSAM model from GPU memory."""
        if self._model is None:
            return
        import torch
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("FastSAM model unloaded from GPU")

    def detect(self, rgb: np.ndarray) -> List[FastSAMDetection]:
        """Run FastSAM on a single RGB image.

        Args:
            rgb: RGB image as (H, W, 3) uint8 ndarray.

        Returns:
            List of FastSAMDetection sorted by mask area descending.
        """
        self.load()
        import cv2

        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        results = self._model(
            bgr,
            device=self.config.device,
            retina_masks=True,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return []

        masks_data = results[0].masks.data.cpu().numpy()  # (N, H', W')
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else None

        detections: List[FastSAMDetection] = []
        for i, mask_raw in enumerate(masks_data):
            # Resize mask to full frame resolution
            mask_full = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_full > 0.5

            area = int(mask_bool.sum())
            if area == 0:
                continue

            # Extract normalized bbox from mask
            bbox_xywh_norm = _mask_to_bbox_xywh_norm(mask_bool, w, h)
            score = float(confs[i]) if confs is not None and i < len(confs) else 1.0

            detections.append(
                FastSAMDetection(
                    mask=mask_bool,
                    bbox_xywh_norm=bbox_xywh_norm,
                    score=score,
                    instance_idx=i,
                )
            )

        # Sort by mask area descending
        detections.sort(key=lambda d: d.mask.sum(), reverse=True)

        # Limit by max_det
        if self.config.max_det > 0:
            detections = detections[: self.config.max_det]

        return detections


def _mask_to_bbox_xywh_norm(
    mask: np.ndarray, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert a boolean mask to normalized [xmin, ymin, w, h] in [0, 1]."""
    ys, xs = np.where(mask)
    xmin = float(xs.min()) / img_w
    ymin = float(ys.min()) / img_h
    bw = float(xs.max() - xs.min() + 1) / img_w
    bh = float(ys.max() - ys.min() + 1) / img_h
    return (xmin, ymin, bw, bh)
