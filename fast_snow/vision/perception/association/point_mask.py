"""Associate projected 3D points with 2D masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class PointMaskAssociation:
    mask_to_points: Dict[int, np.ndarray]
    point_to_mask: np.ndarray  # (N,) int, -1 for unassigned


def assign_points_to_masks(
    xy: np.ndarray,
    masks: List[np.ndarray],
    valid: np.ndarray,
) -> PointMaskAssociation:
    """Assign each point to the first mask that contains its projected pixel.

    Args:
        xy: (N, 2) projected pixel coordinates.
        masks: list of (H, W) boolean masks.
        valid: (N,) bool indicating valid projections.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    if valid.shape[0] != xy.shape[0]:
        raise ValueError("valid must have shape (N,)")

    point_to_mask = np.full((xy.shape[0],), -1, dtype=int)
    mask_to_points: Dict[int, List[int]] = {i: [] for i in range(len(masks))}

    for idx, (pt, is_valid) in enumerate(zip(xy, valid)):
        if not is_valid:
            continue
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        for mask_idx, mask in enumerate(masks):
            if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                continue
            if mask[y, x]:
                point_to_mask[idx] = mask_idx
                mask_to_points[mask_idx].append(idx)
                break

    mask_to_points_arr = {
        mask_idx: np.array(indices, dtype=int)
        for mask_idx, indices in mask_to_points.items()
    }
    return PointMaskAssociation(mask_to_points=mask_to_points_arr, point_to_mask=point_to_mask)
