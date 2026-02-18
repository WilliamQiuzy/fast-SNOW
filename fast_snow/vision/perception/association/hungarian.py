"""Hungarian matching for cross-view mask association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover - optional dependency
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class MatchResult:
    pairs: List[Tuple[int, int]]
    costs: List[float]


def _check_scipy_available() -> None:
    if linear_sum_assignment is None:
        raise ImportError(
            "scipy is required for Hungarian matching. Install scipy to use this module."
        ) from _SCIPY_IMPORT_ERROR


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask shapes must match")
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def build_cost_matrix(
    masks_a: Sequence[np.ndarray],
    masks_b: Sequence[np.ndarray],
) -> np.ndarray:
    """Build cost matrix from IoU (cost = 1 - IoU)."""
    cost = np.ones((len(masks_a), len(masks_b)), dtype=float)
    for i, ma in enumerate(masks_a):
        for j, mb in enumerate(masks_b):
            cost[i, j] = 1.0 - mask_iou(ma, mb)
    return cost


def match_masks_hungarian(
    masks_a: Sequence[np.ndarray],
    masks_b: Sequence[np.ndarray],
    iou_threshold: float = 0.0,
) -> MatchResult:
    """Match masks across two views using Hungarian assignment."""
    _check_scipy_available()
    if len(masks_a) == 0 or len(masks_b) == 0:
        return MatchResult(pairs=[], costs=[])

    cost = build_cost_matrix(masks_a, masks_b)
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: List[Tuple[int, int]] = []
    costs: List[float] = []
    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, c]
        if iou >= iou_threshold:
            pairs.append((int(r), int(c)))
            costs.append(float(cost[r, c]))
    return MatchResult(pairs=pairs, costs=costs)
