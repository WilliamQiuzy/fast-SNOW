"""NuScenes LiDAR Segmentation benchmark evaluation.

This module implements evaluation on NuScenes LiDAR Segmentation as described in
SNOW paper Section 4.4.

The task is point-wise semantic segmentation of LiDAR point clouds.
SNOW achieves 38.1 mIoU in a training-free manner.

NuScenes LiDAR Segmentation classes (16 classes):
- barrier, bicycle, bus, car, construction_vehicle
- motorcycle, pedestrian, traffic_cone, trailer, truck
- driveable_surface, other_flat, sidewalk, terrain
- manmade, vegetation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# NuScenes LiDAR Segmentation class names
NUSCENES_LIDAR_CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

# Class ID to name mapping
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(NUSCENES_LIDAR_CLASSES)}
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(NUSCENES_LIDAR_CLASSES)}


@dataclass
class LiDARSegSample:
    """A single LiDAR segmentation sample."""

    sample_token: str
    lidar_path: str
    labels_path: Optional[str] = None
    scene_token: Optional[str] = None


@dataclass
class LiDARSegMetrics:
    """Metrics for LiDAR segmentation evaluation."""

    # Overall mIoU
    miou: float = 0.0

    # Per-class IoU
    class_iou: Dict[str, float] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Per-class counts
    class_tp: Dict[str, int] = field(default_factory=dict)  # True positives
    class_fp: Dict[str, int] = field(default_factory=dict)  # False positives
    class_fn: Dict[str, int] = field(default_factory=dict)  # False negatives

    # Statistics
    total_points: int = 0
    correct_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miou": self.miou,
            "class_iou": self.class_iou,
            "total_points": self.total_points,
            "correct_points": self.correct_points,
            "accuracy": self.correct_points / max(self.total_points, 1),
        }

    def __str__(self) -> str:
        lines = [
            f"NuScenes LiDAR Segmentation Results",
            f"=" * 40,
            f"mIoU: {self.miou:.2%}",
            f"Overall Accuracy: {self.correct_points / max(self.total_points, 1):.2%}",
            f"Total Points: {self.total_points}",
            f"",
            f"Per-Class IoU:",
        ]
        for class_name in NUSCENES_LIDAR_CLASSES:
            if class_name in self.class_iou:
                iou = self.class_iou[class_name]
                lines.append(f"  {class_name}: {iou:.2%}")
        return "\n".join(lines)


def compute_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    class_id: int,
) -> Tuple[float, int, int, int]:
    """Compute IoU for a single class.

    Args:
        pred: Predicted labels (N,)
        gt: Ground truth labels (N,)
        class_id: Class ID to compute IoU for

    Returns:
        Tuple of (iou, tp, fp, fn)
    """
    pred_mask = pred == class_id
    gt_mask = gt == class_id

    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())

    if tp + fp + fn == 0:
        iou = 0.0
    else:
        iou = tp / (tp + fp + fn)

    return iou, tp, fp, fn


def compute_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: Predicted labels (N,)
        gt: Ground truth labels (N,)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        confusion[i, j] = number of points with gt=i and pred=j
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            confusion[i, j] = np.logical_and(gt == i, pred == j).sum()

    return confusion


def compute_metrics_from_confusion(
    confusion: np.ndarray,
    class_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute mIoU and per-class IoU from confusion matrix.

    Args:
        confusion: Confusion matrix (num_classes, num_classes)
        class_names: List of class names

    Returns:
        Tuple of (mIoU, per_class_iou dict)
    """
    num_classes = confusion.shape[0]
    class_iou = {}
    valid_classes = 0

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
            class_iou[class_names[i]] = float(iou)
            valid_classes += 1
        else:
            class_iou[class_names[i]] = 0.0

    if valid_classes > 0:
        miou = sum(class_iou.values()) / valid_classes
    else:
        miou = 0.0

    return miou, class_iou


# Type alias for segmentation function
# Takes: point_cloud (N, 3) or path, scene_data
# Returns: predicted labels (N,)
SegmentationFunc = Callable[[np.ndarray, Any], np.ndarray]


def load_lidar_points(lidar_path: str) -> np.ndarray:
    """Load LiDAR points from file.

    Supports .bin (NuScenes format) and .npy formats.
    """
    path = Path(lidar_path)

    if path.suffix == ".bin":
        # NuScenes format: (N, 5) with [x, y, z, intensity, ring]
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
        return points[:, :3]  # Return only xyz

    elif path.suffix == ".npy":
        points = np.load(str(path))
        if points.shape[1] >= 3:
            return points[:, :3]
        return points

    else:
        raise ValueError(f"Unsupported LiDAR format: {path.suffix}")


def load_labels(labels_path: str) -> np.ndarray:
    """Load segmentation labels from file."""
    path = Path(labels_path)

    if path.suffix == ".bin":
        labels = np.fromfile(str(path), dtype=np.uint8)
        return labels

    elif path.suffix == ".npy":
        return np.load(str(path))

    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return np.array(data["labels"], dtype=np.int32)

    else:
        raise ValueError(f"Unsupported label format: {path.suffix}")


def evaluate_lidar_seg(
    samples: List[LiDARSegSample],
    segment_fn: SegmentationFunc,
    data_root: Optional[Path] = None,
    max_samples: Optional[int] = None,
    save_predictions: Optional[Path] = None,
) -> LiDARSegMetrics:
    """Run LiDAR segmentation evaluation.

    Args:
        samples: List of samples to evaluate
        segment_fn: Function that takes (points, scene_data) and returns labels
        data_root: Optional path to data root
        max_samples: Optional limit on number of samples
        save_predictions: Optional path to save predictions

    Returns:
        LiDARSegMetrics with evaluation results
    """
    metrics = LiDARSegMetrics()

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} LiDAR samples")

    # Accumulate confusion matrix
    num_classes = len(NUSCENES_LIDAR_CLASSES)
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    all_predictions = []

    for i, sample in enumerate(samples):
        try:
            # Load point cloud
            lidar_path = sample.lidar_path
            if data_root is not None:
                lidar_path = str(data_root / lidar_path)

            points = load_lidar_points(lidar_path)

            # Load ground truth labels
            if sample.labels_path is None:
                logger.warning(f"No labels for sample {sample.sample_token}")
                continue

            labels_path = sample.labels_path
            if data_root is not None:
                labels_path = str(data_root / labels_path)

            gt_labels = load_labels(labels_path)

            # Run segmentation
            pred_labels = segment_fn(points, {"sample_token": sample.sample_token})

            # Ensure same length
            min_len = min(len(gt_labels), len(pred_labels))
            gt_labels = gt_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            # Update confusion matrix
            confusion = compute_confusion_matrix(pred_labels, gt_labels, num_classes)
            total_confusion += confusion

            # Update point counts
            metrics.total_points += len(gt_labels)
            metrics.correct_points += (pred_labels == gt_labels).sum()

            # Store predictions if requested
            if save_predictions:
                all_predictions.append({
                    "sample_token": sample.sample_token,
                    "predictions": pred_labels.tolist(),
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")

        except Exception as e:
            logger.warning(f"Error processing sample {sample.sample_token}: {e}")

    # Compute final metrics
    metrics.confusion_matrix = total_confusion
    metrics.miou, metrics.class_iou = compute_metrics_from_confusion(
        total_confusion, NUSCENES_LIDAR_CLASSES
    )

    # Save predictions if requested
    if save_predictions:
        with open(save_predictions, "w") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "predictions": all_predictions,
            }, f)
        logger.info(f"Saved predictions to {save_predictions}")

    return metrics


def create_snow_segmentation_fn(
    snow_pipeline: Any,
    class_mapping: Optional[Dict[str, int]] = None,
) -> SegmentationFunc:
    """Create a segmentation function using SNOW pipeline.

    SNOW performs class-agnostic segmentation, so we need to
    map the segmented instances to semantic classes.

    Args:
        snow_pipeline: Configured SNOW pipeline
        class_mapping: Optional mapping from object descriptions to class IDs

    Returns:
        Segmentation function
    """
    def segment_fn(points: np.ndarray, scene_data: Any) -> np.ndarray:
        # Run SNOW pipeline to get instance segmentation
        # This returns STEP tokens with mask-point associations
        step_tokens = snow_pipeline.process_points(points)

        # Initialize all points as "unknown" (could use a default class)
        labels = np.zeros(len(points), dtype=np.int32)

        # Map instances to semantic classes
        # This requires VLM to classify each instance
        for node_id, step in step_tokens.items():
            # Get points associated with this instance
            # (requires maintaining point-to-token mapping)
            pass

        return labels

    return segment_fn


def get_lidar_seg_baseline_results() -> Dict[str, float]:
    """Get baseline results from SNOW paper Section 4.4."""
    return {
        "SNOW (ours)": 38.1,
        "Seal": 45.0,  # First place (uses training)
        "SAL": 33.0,   # Training-free baseline
    }


def load_nuscenes_lidar_samples(
    nuscenes_root: Path,
    split: str = "val",
) -> List[LiDARSegSample]:
    """Load NuScenes LiDAR segmentation samples.

    This is a placeholder - actual implementation requires
    nuscenes-devkit and lidarseg annotations.
    """
    samples = []

    # In practice:
    # from nuscenes import NuScenes
    # from nuscenes.utils.splits import create_splits_scenes
    #
    # nusc = NuScenes(version='v1.0-trainval', dataroot=str(nuscenes_root))
    # scenes = create_splits_scenes()[split]
    #
    # for scene in scenes:
    #     for sample in scene_samples:
    #         samples.append(LiDARSegSample(
    #             sample_token=sample['token'],
    #             lidar_path=sample['data']['LIDAR_TOP'],
    #             labels_path=sample['lidarseg']['labels'],
    #         ))

    return samples
