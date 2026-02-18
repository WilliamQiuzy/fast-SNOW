"""VLM4D benchmark evaluation.

This module implements evaluation on the VLM4D benchmark as described in
SNOW paper Section 4.2, Table 3.

VLM4D categories:
- Ego-centric: Spatial reasoning from ego perspective
- Exo-centric: Spatial reasoning from external perspective
- Directional: Direction-based reasoning
- False Positive: Detection of false positives

SNOW achieves 73.75% overall accuracy on VLM4D.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fast_snow.data.loaders.vlm4d import VLM4DSample, load_vlm4d_json, iter_vlm4d_samples

logger = logging.getLogger(__name__)


@dataclass
class VLM4DMetrics:
    """Metrics for VLM4D evaluation."""

    # Overall accuracy
    accuracy: float = 0.0
    total_samples: int = 0
    correct_samples: int = 0

    # Per-category accuracy
    category_accuracy: Dict[str, float] = field(default_factory=dict)
    category_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # (correct, total)

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

        # Update category counts
        if category not in self.category_counts:
            self.category_counts[category] = (0, 0)
        correct, total = self.category_counts[category]
        self.category_counts[category] = (correct + int(is_correct), total + 1)

        # Store detailed result
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
            f"VLM4D Evaluation Results",
            f"=" * 40,
            f"Overall Accuracy: {self.accuracy:.2%} ({self.correct_samples}/{self.total_samples})",
            f"",
            f"Per-Category Accuracy:",
        ]
        for cat, acc in sorted(self.category_accuracy.items()):
            correct, total = self.category_counts[cat]
            lines.append(f"  {cat}: {acc:.2%} ({correct}/{total})")
        return "\n".join(lines)


def extract_answer_from_response(response: str, choices: Dict[str, str]) -> str:
    """Extract the answer choice from model response.

    Tries multiple strategies:
    1. Direct match with choice key (A, B, C, D)
    2. Match with choice text
    3. First capital letter in response

    Args:
        response: Model's text response
        choices: Dict mapping choice keys to choice text

    Returns:
        Extracted choice key (A, B, C, D) or empty string if not found
    """
    response_clean = response.strip().upper()

    # Strategy 1: Check if response starts with a choice key
    for key in choices.keys():
        if response_clean.startswith(key.upper()):
            return key.upper()

    # Strategy 2: Check if response contains choice text
    response_lower = response.lower()
    for key, text in choices.items():
        if text.lower() in response_lower:
            return key.upper()

    # Strategy 3: Find first valid choice letter
    for char in response_clean:
        if char in [k.upper() for k in choices.keys()]:
            return char

    # Strategy 4: Look for patterns like "Answer: A" or "(A)"
    import re
    patterns = [
        r"answer[:\s]+([A-D])",
        r"\(([A-D])\)",
        r"option[:\s]+([A-D])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth.

    Args:
        prediction: Predicted answer (A, B, C, D)
        ground_truth: Ground truth answer

    Returns:
        True if correct
    """
    return prediction.upper() == ground_truth.upper()


# Type alias for inference function
# Takes: question (str), choices (dict), video_path (str)
# Returns: response (str)
InferenceFunc = Callable[[str, Dict[str, str], str], str]


def evaluate_vlm4d(
    json_path: Path,
    inference_fn: InferenceFunc,
    local_video_root: Optional[Path] = None,
    question_types: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    save_predictions: Optional[Path] = None,
) -> VLM4DMetrics:
    """Run VLM4D evaluation.

    Args:
        json_path: Path to VLM4D JSON file (real_mc.json or synthetic_mc.json)
        inference_fn: Function that takes (question, choices, video_path) and returns response
        local_video_root: Optional local directory containing videos
        question_types: Optional filter for specific question types
        max_samples: Optional limit on number of samples to evaluate
        save_predictions: Optional path to save detailed predictions

    Returns:
        VLM4DMetrics with evaluation results
    """
    metrics = VLM4DMetrics()

    # Load samples
    samples = list(iter_vlm4d_samples(
        json_path=json_path,
        local_video_root=local_video_root,
        question_type=None,  # We filter manually to support multiple types
    ))

    # Filter by question type if specified
    if question_types:
        samples = [s for s in samples if s.question_type in question_types]

    # Limit samples if specified
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} VLM4D samples")

    for i, sample in enumerate(samples):
        try:
            # Build question with choices
            question_with_choices = f"{sample.question}\n"
            for key, text in sample.choices.items():
                question_with_choices += f"{key}. {text}\n"

            # Run inference
            response = inference_fn(
                question_with_choices,
                sample.choices,
                sample.video,
            )

            # Extract answer
            prediction = extract_answer_from_response(response, sample.choices)

            # Check correctness
            is_correct = check_answer(prediction, sample.answer)

            # Record result
            metrics.add_prediction(
                sample_id=sample.sample_id,
                category=sample.question_type,
                prediction=prediction,
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


def evaluate_vlm4d_with_snow(
    json_path: Path,
    snow_pipeline: Any,  # SNOWPipeline
    vlm: Any,  # VLMInterface
    local_video_root: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> VLM4DMetrics:
    """Run VLM4D evaluation using SNOW pipeline.

    This is a convenience function that integrates the SNOW pipeline
    with VLM4D evaluation.

    Args:
        json_path: Path to VLM4D JSON file
        snow_pipeline: Configured SNOW pipeline
        vlm: VLM interface for inference
        local_video_root: Optional local video directory
        max_samples: Optional sample limit

    Returns:
        VLM4DMetrics
    """
    from fast_snow.reasoning.vlm.prompt_builder import build_messages, PromptConfig

    def inference_fn(question: str, choices: Dict[str, str], video_path: str) -> str:
        # Process video with SNOW pipeline
        # This should extract frames, run MapAnything, build 4DSG
        four_dsg = snow_pipeline.process_video(video_path)

        # Build prompt
        messages = build_messages(question, four_dsg, PromptConfig())

        # Run VLM inference
        response = vlm.generate(messages)

        return response

    return evaluate_vlm4d(
        json_path=json_path,
        inference_fn=inference_fn,
        local_video_root=local_video_root,
        max_samples=max_samples,
    )


# VLM4D question type constants
VLM4D_QUESTION_TYPES = [
    "ego-centric",
    "exo-centric",
    "directional",
    "false_positive",
]


def get_vlm4d_baseline_results() -> Dict[str, float]:
    """Get baseline results from SNOW paper Table 3."""
    return {
        "SNOW (ours)": {
            "ego-centric": 0.7500,
            "exo-centric": 0.7273,
            "directional": 0.7727,
            "false_positive": 0.7000,
            "overall": 0.7375,
        },
        "VideoLLaMA3": {
            "ego-centric": 0.5833,
            "exo-centric": 0.6364,
            "directional": 0.5909,
            "false_positive": 0.6500,
            "overall": 0.6152,
        },
    }
