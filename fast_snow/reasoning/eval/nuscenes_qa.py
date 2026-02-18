"""NuScenes-QA benchmark evaluation.

This module implements evaluation on NuScenes-QA benchmark as described in
SNOW paper Section 4.1, Table 2.

NuScenes-QA categories:
- Existence (Ext): Object existence judgment
- Count (Cnt): Object counting
- Object (Obj): Object identification
- Status (Sts): Object status (moving/stationary)
- Comparison (Cmp): Comparative relations

SNOW achieves 60.1% overall accuracy, with +23.5% improvement on Status category.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# NuScenes-QA question type mapping
NUSCENES_QA_CATEGORIES = {
    "exist": "Existence",
    "count": "Count",
    "object": "Object",
    "status": "Status",
    "comparison": "Comparison",
}


@dataclass
class NuScenesQASample:
    """A single NuScenes-QA sample."""

    sample_id: str
    scene_token: str
    question: str
    answer: str
    question_type: str

    # Optional metadata
    sample_token: Optional[str] = None
    timestamp: Optional[int] = None


@dataclass
class NuScenesQAMetrics:
    """Metrics for NuScenes-QA evaluation."""

    # Overall accuracy
    accuracy: float = 0.0
    total_samples: int = 0
    correct_samples: int = 0

    # Per-category accuracy
    category_accuracy: Dict[str, float] = field(default_factory=dict)
    category_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Detailed results
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def add_prediction(
        self,
        sample_id: str,
        category: str,
        prediction: str,
        ground_truth: str,
        is_correct: bool,
    ) -> None:
        """Add a single prediction result."""
        self.total_samples += 1
        if is_correct:
            self.correct_samples += 1

        if category not in self.category_counts:
            self.category_counts[category] = (0, 0)
        correct, total = self.category_counts[category]
        self.category_counts[category] = (correct + int(is_correct), total + 1)

        self.predictions.append({
            "sample_id": sample_id,
            "category": category,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct,
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
        }

    def __str__(self) -> str:
        lines = [
            f"NuScenes-QA Evaluation Results",
            f"=" * 40,
            f"Overall Accuracy: {self.accuracy:.2%} ({self.correct_samples}/{self.total_samples})",
            f"",
            f"Per-Category Accuracy:",
        ]
        for cat in ["Existence", "Count", "Object", "Status", "Comparison"]:
            if cat in self.category_accuracy:
                acc = self.category_accuracy[cat]
                correct, total = self.category_counts[cat]
                lines.append(f"  {cat}: {acc:.2%} ({correct}/{total})")
        return "\n".join(lines)


def load_nuscenes_qa_json(json_path: Path) -> List[NuScenesQASample]:
    """Load NuScenes-QA JSON file.

    Expected format:
    {
        "questions": [
            {
                "question_id": "...",
                "scene_token": "...",
                "question": "...",
                "answer": "...",
                "question_type": "exist|count|object|status|comparison"
            },
            ...
        ]
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []

    # Handle different possible formats
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")

    for item in questions:
        q_type = item.get("question_type", item.get("type", "unknown"))
        category = NUSCENES_QA_CATEGORIES.get(q_type, q_type)

        samples.append(NuScenesQASample(
            sample_id=str(item.get("question_id", item.get("id", ""))),
            scene_token=item.get("scene_token", ""),
            question=item.get("question", ""),
            answer=str(item.get("answer", "")),
            question_type=category,
            sample_token=item.get("sample_token"),
            timestamp=item.get("timestamp"),
        ))

    return samples


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    - Convert to lowercase
    - Remove punctuation
    - Normalize numbers
    - Handle common variations
    """
    answer = answer.lower().strip()

    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)

    # Normalize whitespace
    answer = ' '.join(answer.split())

    # Normalize common variations
    answer = answer.replace("metres", "meters")
    answer = answer.replace("metre", "meter")

    # Normalize yes/no
    if answer in ["yes", "yeah", "yep", "true", "correct"]:
        answer = "yes"
    elif answer in ["no", "nope", "false", "incorrect"]:
        answer = "no"

    return answer


def extract_number(text: str) -> Optional[int]:
    """Extract a number from text."""
    # Try to find digit
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())

    # Try word numbers
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "none": 0, "no": 0,
    }
    text_lower = text.lower()
    for word, num in word_to_num.items():
        if word in text_lower:
            return num

    return None


def check_answer_nuscenes(
    prediction: str,
    ground_truth: str,
    question_type: str,
) -> bool:
    """Check if prediction matches ground truth for NuScenes-QA.

    Uses category-specific matching logic.
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Category-specific matching
    if question_type == "Count":
        # Compare numbers
        pred_num = extract_number(prediction)
        gt_num = extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            return pred_num == gt_num

    elif question_type == "Existence":
        # Yes/No matching
        pred_bool = "yes" in pred_norm or "true" in pred_norm
        gt_bool = "yes" in gt_norm or "true" in gt_norm
        return pred_bool == gt_bool

    elif question_type == "Status":
        # Status keywords
        status_keywords = ["moving", "stationary", "stopped", "parked", "driving"]
        for kw in status_keywords:
            if kw in pred_norm and kw in gt_norm:
                return True

    # Check if ground truth is contained in prediction
    if gt_norm in pred_norm:
        return True

    return False


# Type alias for inference function
InferenceFunc = Callable[[str, str, Any], str]


def evaluate_nuscenes_qa(
    qa_json_path: Path,
    inference_fn: InferenceFunc,
    nuscenes_root: Optional[Path] = None,
    question_types: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    save_predictions: Optional[Path] = None,
) -> NuScenesQAMetrics:
    """Run NuScenes-QA evaluation.

    Args:
        qa_json_path: Path to NuScenes-QA JSON file
        inference_fn: Function that takes (question, scene_token, scene_data) and returns response
        nuscenes_root: Optional path to NuScenes dataset
        question_types: Optional filter for specific question types
        max_samples: Optional limit on number of samples
        save_predictions: Optional path to save detailed predictions

    Returns:
        NuScenesQAMetrics with evaluation results
    """
    metrics = NuScenesQAMetrics()

    # Load samples
    samples = load_nuscenes_qa_json(qa_json_path)

    # Filter by question type if specified
    if question_types:
        samples = [s for s in samples if s.question_type in question_types]

    # Limit samples if specified
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} NuScenes-QA samples")

    # Load NuScenes data if available
    scene_data_cache: Dict[str, Any] = {}

    for i, sample in enumerate(samples):
        try:
            # Get scene data (cached)
            scene_data = scene_data_cache.get(sample.scene_token)
            if scene_data is None and nuscenes_root is not None:
                # Load scene data from NuScenes
                scene_data = load_nuscenes_scene(nuscenes_root, sample.scene_token)
                scene_data_cache[sample.scene_token] = scene_data

            # Run inference
            response = inference_fn(sample.question, sample.scene_token, scene_data)

            # Check correctness
            is_correct = check_answer_nuscenes(
                response,
                sample.answer,
                sample.question_type,
            )

            # Record result
            metrics.add_prediction(
                sample_id=sample.sample_id,
                category=sample.question_type,
                prediction=response,
                ground_truth=sample.answer,
                is_correct=is_correct,
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


def load_nuscenes_scene(nuscenes_root: Path, scene_token: str) -> Optional[Dict[str, Any]]:
    """Load NuScenes scene data.

    This is a placeholder - actual implementation depends on
    how the NuScenes data is organized.
    """
    # In practice, this would use the nuscenes-devkit
    # from nuscenes import NuScenes
    # nusc = NuScenes(version='v1.0-trainval', dataroot=str(nuscenes_root))
    # scene = nusc.get('scene', scene_token)
    # return scene
    return None


def get_nuscenes_qa_baseline_results() -> Dict[str, Dict[str, float]]:
    """Get baseline results from SNOW paper Table 2."""
    return {
        "SNOW (ours)": {
            "Existence": 0.802,
            "Count": 0.437,
            "Object": 0.408,
            "Status": 0.871,
            "Comparison": 0.507,
            "Overall": 0.601,
        },
        "LidarLLM": {
            "Existence": 0.744,
            "Count": 0.420,
            "Object": 0.270,
            "Status": 0.636,
            "Comparison": 0.449,
            "Overall": 0.503,
        },
        "DriveMLLM": {
            "Existence": 0.774,
            "Count": 0.452,
            "Object": 0.373,
            "Status": 0.623,
            "Comparison": 0.494,
            "Overall": 0.543,
        },
    }
