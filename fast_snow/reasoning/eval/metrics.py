"""Unified metrics computation for VLM4D evaluation.

Phase 8 requirement: Standardized metric calculation aligned with paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvaluationResult:
    """Single evaluation result for one sample."""

    sample_id: str
    category: str
    question: str
    prediction: str
    ground_truth: str
    is_correct: bool
    raw_response: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "question": self.question,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "raw_response": self.raw_response,
            "confidence": self.confidence,
        }


@dataclass
class CategoryMetrics:
    """Metrics for a single category."""

    category: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0

    def compute_accuracy(self) -> None:
        """Compute accuracy from counts."""
        if self.total > 0:
            self.accuracy = self.correct / self.total
        else:
            self.accuracy = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
        }

    def __str__(self) -> str:
        """Format as string."""
        return f"{self.category}: {self.accuracy:.2%} ({self.correct}/{self.total})"


@dataclass
class Phase8Metrics:
    """Complete metrics for Phase 8 VLM4D evaluation.

    Aligned with SNOW paper Table 3 format.
    """

    # Overall metrics
    total_samples: int = 0
    correct_samples: int = 0
    overall_accuracy: float = 0.0

    # Per-category metrics
    categories: Dict[str, CategoryMetrics] = field(default_factory=dict)

    # Detailed results
    results: List[EvaluationResult] = field(default_factory=list)

    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def add_result(self, result: EvaluationResult) -> None:
        """Add a single evaluation result.

        Args:
            result: EvaluationResult to add
        """
        # Update overall counts
        self.total_samples += 1
        if result.is_correct:
            self.correct_samples += 1

        # Update category counts
        if result.category not in self.categories:
            self.categories[result.category] = CategoryMetrics(category=result.category)

        category_metrics = self.categories[result.category]
        category_metrics.total += 1
        if result.is_correct:
            category_metrics.correct += 1

        # Store detailed result
        self.results.append(result)

    def compute_metrics(self) -> None:
        """Compute all final metrics."""
        # Overall accuracy
        if self.total_samples > 0:
            self.overall_accuracy = self.correct_samples / self.total_samples
        else:
            self.overall_accuracy = 0.0

        # Category accuracies
        for category_metrics in self.categories.values():
            category_metrics.compute_accuracy()

    def get_category_accuracy(self, category: str) -> float:
        """Get accuracy for a specific category.

        Args:
            category: Category name

        Returns:
            Accuracy (0.0 if category not found)
        """
        if category in self.categories:
            return self.categories[category].accuracy
        return 0.0

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary metrics as dictionary (without detailed results).

        Returns:
            Dict with overall and per-category metrics
        """
        return {
            "overall_accuracy": self.overall_accuracy,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "category_metrics": {
                cat: metrics.to_dict()
                for cat, metrics in sorted(self.categories.items())
            },
        }

    def to_dict(self, include_results: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_results: Whether to include detailed results

        Returns:
            Dict representation
        """
        data = {
            "config": self.config,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "metrics": self.get_summary_dict(),
        }

        if include_results:
            data["results"] = [r.to_dict() for r in self.results]

        return data

    def save_json(self, path: Path, include_results: bool = True) -> None:
        """Save metrics to JSON file.

        Args:
            path: Output path
            include_results: Whether to include detailed results
        """
        with open(path, "w") as f:
            json.dump(
                self.to_dict(include_results=include_results),
                f,
                indent=2,
            )

    @classmethod
    def load_json(cls, path: Path) -> Phase8Metrics:
        """Load metrics from JSON file.

        Args:
            path: JSON file path

        Returns:
            Phase8Metrics instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        metrics = cls()
        metrics.config = data.get("config", {})
        metrics.environment = data.get("environment", {})
        metrics.timestamp = data.get("timestamp")

        # Load summary metrics
        summary = data.get("metrics", {})
        metrics.total_samples = summary.get("total_samples", 0)
        metrics.correct_samples = summary.get("correct_samples", 0)
        metrics.overall_accuracy = summary.get("overall_accuracy", 0.0)

        # Load category metrics
        for cat_name, cat_data in summary.get("category_metrics", {}).items():
            metrics.categories[cat_name] = CategoryMetrics(
                category=cat_data["category"],
                total=cat_data["total"],
                correct=cat_data["correct"],
                accuracy=cat_data["accuracy"],
            )

        # Load detailed results if present
        if "results" in data:
            for r_data in data["results"]:
                metrics.results.append(EvaluationResult(**r_data))

        return metrics

    def __str__(self) -> str:
        """Format as human-readable string.

        Format matches SNOW paper Table 3:
        ```
        VLM4D Evaluation Results (Phase 8)
        ====================================
        Overall: 73.75% (88/120)

        Per-Category Results:
          Ego-centric:    75.00% (30/40)
          Exo-centric:    72.73% (24/33)
          Directional:    77.27% (17/22)
          False Positive: 70.00% (17/25)
        ```
        """
        lines = [
            "VLM4D Evaluation Results (Phase 8)",
            "=" * 60,
            f"Overall: {self.overall_accuracy:.2%} ({self.correct_samples}/{self.total_samples})",
            "",
            "Per-Category Results:",
        ]

        for cat_name in sorted(self.categories.keys()):
            cat = self.categories[cat_name]
            lines.append(f"  {cat_name:15s} {cat.accuracy:.2%} ({cat.correct}/{cat.total})")

        return "\n".join(lines)

    def format_paper_table(self) -> str:
        """Format results as paper Table 3 style.

        Returns:
            Markdown table string
        """
        lines = [
            "| Category | Accuracy | Correct/Total |",
            "|----------|----------|---------------|",
        ]

        for cat_name in sorted(self.categories.keys()):
            cat = self.categories[cat_name]
            lines.append(
                f"| {cat_name:15s} | {cat.accuracy:.2%} | {cat.correct}/{cat.total} |"
            )

        lines.append(
            f"| **Overall** | **{self.overall_accuracy:.2%}** | {self.correct_samples}/{self.total_samples} |"
        )

        return "\n".join(lines)


def compare_with_baseline(
    metrics: Phase8Metrics,
    baseline_name: str = "SNOW (Paper)",
) -> str:
    """Compare metrics with paper baseline.

    Args:
        metrics: Computed metrics
        baseline_name: Name for baseline

    Returns:
        Comparison table string
    """
    # Paper baselines from Table 3
    paper_baselines = {
        "ego-centric": 0.7500,
        "exo-centric": 0.7273,
        "directional": 0.7727,
        "false_positive": 0.7000,
        "overall": 0.7375,
    }

    lines = [
        "Comparison with Paper Baseline",
        "=" * 60,
        "| Category        | Ours    | Paper   | Diff    |",
        "|-----------------|---------|---------|---------|",
    ]

    for cat_name in sorted(metrics.categories.keys()):
        ours = metrics.get_category_accuracy(cat_name)
        paper = paper_baselines.get(cat_name, 0.0)
        diff = ours - paper

        lines.append(
            f"| {cat_name:15s} | {ours:.2%} | {paper:.2%} | "
            f"{diff:+.2%} |"
        )

    # Overall
    paper_overall = paper_baselines["overall"]
    diff_overall = metrics.overall_accuracy - paper_overall

    lines.append(
        f"| **Overall**     | **{metrics.overall_accuracy:.2%}** | "
        f"**{paper_overall:.2%}** | **{diff_overall:+.2%}** |"
    )

    return "\n".join(lines)


def analyze_errors(metrics: Phase8Metrics, top_k: int = 10) -> str:
    """Analyze most common errors.

    Args:
        metrics: Computed metrics
        top_k: Number of top errors to show

    Returns:
        Error analysis string
    """
    # Get incorrect predictions
    incorrect = [r for r in metrics.results if not r.is_correct]

    if not incorrect:
        return "No errors found!"

    lines = [
        f"Error Analysis ({len(incorrect)} incorrect predictions)",
        "=" * 60,
    ]

    # Group by category
    errors_by_category = {}
    for result in incorrect:
        if result.category not in errors_by_category:
            errors_by_category[result.category] = []
        errors_by_category[result.category].append(result)

    for category, errors in sorted(errors_by_category.items()):
        lines.append(f"\n{category}: {len(errors)} errors")

        for i, error in enumerate(errors[:top_k]):
            lines.append(f"  {i+1}. Sample {error.sample_id}")
            lines.append(f"     Q: {error.question[:60]}...")
            lines.append(f"     Predicted: {error.prediction}, Truth: {error.ground_truth}")

    return "\n".join(lines)
