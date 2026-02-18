"""Connect SAM2 output, multi-view matching, and point association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fast_snow.data.transforms.projection import ProjectionResult
from fast_snow.vision.perception.association.multiview_match import match_masks_across_views
from fast_snow.vision.perception.association.point_mask import assign_points_to_masks
from fast_snow.vision.perception.segmentation.sam2_wrapper import SAM2Mask


@dataclass(frozen=True)
class MultiViewAssociation:
    matched: Dict[Tuple[int, int], List[Tuple[int, int]]]
    view_point_assignments: Dict[int, Dict[int, np.ndarray]]


def associate_multiview(
    masks_per_view: Sequence[Sequence[SAM2Mask]],
    projections_per_view: Sequence[ProjectionResult],
    iou_threshold: float = 0.0,
) -> MultiViewAssociation:
    """Associate SAM2 masks across views and map points to matched masks."""
    raw_masks = [[m.mask for m in masks] for masks in masks_per_view]
    matched = match_masks_across_views(raw_masks, iou_threshold=iou_threshold)

    view_point_assignments: Dict[int, Dict[int, np.ndarray]] = {}
    for view_idx, masks in enumerate(raw_masks):
        proj = projections_per_view[view_idx]
        assoc = assign_points_to_masks(proj.xy, masks, proj.valid)
        view_point_assignments[view_idx] = assoc.mask_to_points

    return MultiViewAssociation(
        matched=matched.matches,
        view_point_assignments=view_point_assignments,
    )
