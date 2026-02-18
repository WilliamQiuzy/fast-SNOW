"""RoboSpatial-Home benchmark evaluation.

This module implements evaluation on RoboSpatial-Home benchmark as described in
SNOW paper Section 4.2.

RoboSpatial-Home categories:
- Spatial Configuration: Relative spatial relations between objects
- Spatial Context: Spatial localization (point prediction)
- Spatial Compatibility: Spatial feasibility judgment

SNOW achieves 72.29% average accuracy, with +23.82% improvement on Context category.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# RoboSpatial question type mapping
ROBOSPATIAL_CATEGORIES = {
    "configuration": "Spatial Configuration",
    "context": "Spatial Context",
    "compatibility": "Spatial Compatibility",
}


@dataclass
class RoboSpatialSample:
    """A single RoboSpatial-Home sample."""

    sample_id: str
    scene_id: str
    question: str
    answer: Union[str, List[float]]  # String or [x, y, z] for context
    question_type: str
    image_path: Optional[str] = None
    point_cloud_path: Optional[str] = None


@dataclass
class RoboSpatialMetrics:
    """Metrics for RoboSpatial evaluation."""

    # Overall accuracy
    accuracy: float = 0.0
    total_samples: int = 0
    correct_samples: int = 0

    # Per-category metrics
    category_accuracy: Dict[str, float] = field(default_factory=dict)
    category_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Spatial Context specific metrics (point prediction)
    context_mean_error: float = 0.0
    context_errors: List[float] = field(default_factory=list)

    # Detailed results
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def add_prediction(
        self,
        sample_id: str,
        category: str,
        prediction: Any,
        ground_truth: Any,
        is_correct: bool,
        error: Optional[float] = None,
    ) -> None:
        """Add a single prediction result."""
        self.total_samples += 1
        if is_correct:
            self.correct_samples += 1

        if category not in self.category_counts:
            self.category_counts[category] = (0, 0)
        correct, total = self.category_counts[category]
        self.category_counts[category] = (correct + int(is_correct), total + 1)

        if error is not None:
            self.context_errors.append(error)

        self.predictions.append({
            "sample_id": sample_id,
            "category": category,
            "prediction": str(prediction),
            "ground_truth": str(ground_truth),
            "correct": is_correct,
            "error": error,
        })

    def compute_final_metrics(self) -> None:
        """Compute final accuracy metrics."""
        if self.total_samples > 0:
            self.accuracy = self.correct_samples / self.total_samples

        for category, (correct, total) in self.category_counts.items():
            if total > 0:
                self.category_accuracy[category] = correct / total
            else:
                self.category_accuracy[category] = 0.0

        if self.context_errors:
            self.context_mean_error = sum(self.context_errors) / len(self.context_errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "category_accuracy": self.category_accuracy,
            "category_counts": {
                k: {"correct": v[0], "total": v[1]}
                for k, v in self.category_counts.items()
            },
            "context_mean_error": self.context_mean_error,
        }

    def __str__(self) -> str:
        lines = [
            f"RoboSpatial-Home Evaluation Results",
            f"=" * 40,
            f"Overall Accuracy: {self.accuracy:.2%} ({self.correct_samples}/{self.total_samples})",
            f"",
            f"Per-Category Accuracy:",
        ]
        for cat in ["Spatial Configuration", "Spatial Context", "Spatial Compatibility"]:
            if cat in self.category_accuracy:
                acc = self.category_accuracy[cat]
                correct, total = self.category_counts[cat]
                lines.append(f"  {cat}: {acc:.2%} ({correct}/{total})")

        if self.context_mean_error > 0:
            lines.append(f"")
            lines.append(f"Spatial Context Mean Error: {self.context_mean_error:.3f}m")

        return "\n".join(lines)


def load_robospatial_json(json_path: Path) -> List[RoboSpatialSample]:
    """Load RoboSpatial-Home JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []

    # Handle different possible formats
    if isinstance(data, dict) and "samples" in data:
        items = data["samples"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for item in items:
        q_type = item.get("question_type", item.get("type", "unknown"))
        category = ROBOSPATIAL_CATEGORIES.get(q_type.lower(), q_type)

        # Parse answer - could be string or point coordinates
        answer = item.get("answer", "")
        if isinstance(answer, list) and len(answer) == 3:
            # Point coordinates [x, y, z]
            pass
        elif isinstance(answer, dict) and "position" in answer:
            answer = answer["position"]
        else:
            answer = str(answer)

        samples.append(RoboSpatialSample(
            sample_id=str(item.get("id", item.get("sample_id", ""))),
            scene_id=item.get("scene_id", item.get("scene", "")),
            question=item.get("question", ""),
            answer=answer,
            question_type=category,
            image_path=item.get("image_path"),
            point_cloud_path=item.get("point_cloud_path"),
        ))

    return samples


def parse_point_from_response(response: str) -> Optional[List[float]]:
    """Parse 3D point coordinates from model response.

    Handles formats like:
    - [1.5, 2.0, 0.5]
    - (1.5, 2.0, 0.5)
    - x=1.5, y=2.0, z=0.5
    - 1.5 2.0 0.5
    """
    # Try JSON-like format
    match = re.search(r'\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?', response)
    if match:
        return [float(match.group(i)) for i in range(1, 4)]

    # Try parentheses format
    match = re.search(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', response)
    if match:
        return [float(match.group(i)) for i in range(1, 4)]

    # Try x=, y=, z= format
    x_match = re.search(r'x\s*[=:]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
    y_match = re.search(r'y\s*[=:]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
    z_match = re.search(r'z\s*[=:]\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
    if x_match and y_match and z_match:
        return [float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1))]

    # Try space-separated format
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if len(numbers) >= 3:
        return [float(numbers[i]) for i in range(3)]

    return None


def compute_point_error(pred: List[float], gt: List[float]) -> float:
    """Compute Euclidean distance between predicted and ground truth points."""
    if len(pred) != 3 or len(gt) != 3:
        return float('inf')

    return math.sqrt(sum((p - g) ** 2 for p, g in zip(pred, gt)))


def check_answer_robospatial(
    prediction: str,
    ground_truth: Union[str, List[float]],
    question_type: str,
    distance_threshold: float = 0.5,
) -> Tuple[bool, Optional[float]]:
    """Check if prediction matches ground truth for RoboSpatial.

    Args:
        prediction: Model's response
        ground_truth: Ground truth answer
        question_type: Category of question
        distance_threshold: Threshold for point prediction (meters)

    Returns:
        Tuple of (is_correct, error)
    """
    if question_type == "Spatial Context":
        # Point prediction task
        pred_point = parse_point_from_response(prediction)

        if isinstance(ground_truth, list):
            gt_point = ground_truth
        else:
            gt_point = parse_point_from_response(str(ground_truth))

        if pred_point is None or gt_point is None:
            return False, None

        error = compute_point_error(pred_point, gt_point)
        is_correct = error <= distance_threshold

        return is_correct, error

    else:
        # Text-based answer (Configuration, Compatibility)
        pred_norm = prediction.lower().strip()
        gt_norm = str(ground_truth).lower().strip()

        # Remove punctuation for comparison
        pred_clean = re.sub(r'[^\w\s]', '', pred_norm)
        gt_clean = re.sub(r'[^\w\s]', '', gt_norm)

        # Exact match
        if pred_clean == gt_clean:
            return True, None

        # Yes/No normalization for compatibility
        if question_type == "Spatial Compatibility":
            pred_bool = any(word in pred_norm for word in ["yes", "true", "possible", "can", "feasible"])
            gt_bool = any(word in gt_norm for word in ["yes", "true", "possible", "can", "feasible"])
            return pred_bool == gt_bool, None

        # Check if ground truth is contained in prediction
        if gt_clean in pred_clean:
            return True, None

        return False, None


# Type alias for inference function
InferenceFunc = Callable[[str, str, Any], str]


def evaluate_robospatial(
    json_path: Path,
    inference_fn: InferenceFunc,
    data_root: Optional[Path] = None,
    question_types: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    distance_threshold: float = 0.5,
    save_predictions: Optional[Path] = None,
) -> RoboSpatialMetrics:
    """Run RoboSpatial-Home evaluation.

    Args:
        json_path: Path to RoboSpatial JSON file
        inference_fn: Function that takes (question, scene_id, scene_data) and returns response
        data_root: Optional path to RoboSpatial data
        question_types: Optional filter for specific question types
        max_samples: Optional limit on number of samples
        distance_threshold: Threshold for point prediction accuracy (meters)
        save_predictions: Optional path to save detailed predictions

    Returns:
        RoboSpatialMetrics with evaluation results
    """
    metrics = RoboSpatialMetrics()

    # Load samples
    samples = load_robospatial_json(json_path)

    # Filter by question type if specified
    if question_types:
        samples = [s for s in samples if s.question_type in question_types]

    # Limit samples if specified
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} RoboSpatial samples")

    for i, sample in enumerate(samples):
        try:
            # Load scene data if available
            scene_data = None
            if data_root is not None:
                scene_data = load_robospatial_scene(data_root, sample.scene_id)

            # Run inference
            response = inference_fn(sample.question, sample.scene_id, scene_data)

            # Check correctness
            is_correct, error = check_answer_robospatial(
                response,
                sample.answer,
                sample.question_type,
                distance_threshold,
            )

            # Record result
            metrics.add_prediction(
                sample_id=sample.sample_id,
                category=sample.question_type,
                prediction=response,
                ground_truth=sample.answer,
                is_correct=is_correct,
                error=error,
            )

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")

        except Exception as e:
            logger.warning(f"Error processing sample {sample.sample_id}: {e}")
            metrics.add_prediction(
                sample_id=sample.sample_id,
                category=sample.question_type,
                prediction="",
                ground_truth=sample.answer,
                is_correct=False,
            )

    # Compute final metrics
    metrics.compute_final_metrics()

    # Save predictions if requested
    if save_predictions:
        with open(save_predictions, "w") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "predictions": metrics.predictions,
            }, f, indent=2)
        logger.info(f"Saved predictions to {save_predictions}")

    return metrics


def load_robospatial_scene(data_root: Path, scene_id: str) -> Optional[Dict[str, Any]]:
    """Load RoboSpatial scene data.

    This is a placeholder - actual implementation depends on
    how the RoboSpatial data is organized.
    """
    scene_dir = data_root / scene_id
    if not scene_dir.exists():
        return None

    scene_data = {"scene_id": scene_id}

    # Load point cloud if available
    pcd_path = scene_dir / "point_cloud.ply"
    if pcd_path.exists():
        scene_data["point_cloud_path"] = str(pcd_path)

    # Load images if available
    images = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
    if images:
        scene_data["images"] = [str(p) for p in images]

    return scene_data


def get_robospatial_baseline_results() -> Dict[str, Dict[str, float]]:
    """Get baseline results from SNOW paper."""
    return {
        "SNOW (ours)": {
            "Spatial Configuration": 0.7556,
            "Spatial Context": 0.6432,
            "Spatial Compatibility": 0.7700,
            "Overall": 0.7229,
        },
        "SpatialRGPT": {
            "Spatial Configuration": 0.5667,
            "Spatial Context": 0.4050,
            "Spatial Compatibility": 0.5700,
            "Overall": 0.5139,
        },
    }
