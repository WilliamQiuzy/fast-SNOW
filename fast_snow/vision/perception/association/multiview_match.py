"""Cross-view mask matching using Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fast_snow.vision.perception.association.hungarian import match_masks_hungarian


@dataclass(frozen=True)
class MatchedMasks:
    view_pairs: List[Tuple[int, int]]
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]]


def match_masks_across_views(
    masks_per_view: Sequence[Sequence[np.ndarray]],
    iou_threshold: float = 0.0,
) -> MatchedMasks:
    """Match masks across all view pairs.

    Returns a mapping from (view_a, view_b) -> list of (mask_idx_a, mask_idx_b).
    """
    view_pairs: List[Tuple[int, int]] = []
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for i in range(len(masks_per_view)):
        for j in range(i + 1, len(masks_per_view)):
            view_pairs.append((i, j))
            res = match_masks_hungarian(
                masks_per_view[i],
                masks_per_view[j],
                iou_threshold=iou_threshold,
            )
            matches[(i, j)] = res.pairs

    return MatchedMasks(view_pairs=view_pairs, matches=matches)
