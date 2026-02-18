"""Association module for multi-view and temporal matching.

Includes Hungarian matching, point-mask association, and multi-view matching.
"""

from fast_snow.vision.perception.association.hungarian import (
    match_masks_hungarian,
    build_cost_matrix,
)
from fast_snow.vision.perception.association.point_mask import (
    assign_points_to_masks,
    PointMaskAssociation,
)
from fast_snow.vision.perception.association.multiview_match import (
    MatchedMasks,
    match_masks_across_views,
)

__all__ = [
    # Hungarian matching
    "match_masks_hungarian",
    "build_cost_matrix",
    # Point-mask association
    "assign_points_to_masks",
    "PointMaskAssociation",
    # Multi-view matching
    "MatchedMasks",
    "match_masks_across_views",
]
