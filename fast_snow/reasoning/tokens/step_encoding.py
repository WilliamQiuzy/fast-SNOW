"""STEP token encoding.

Paper reference (Section 3.2, Eq. 4):
    S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t, c_t^k, s_t^k, θ_t^k}

    Where:
    - τ: Image patch tokens (16×16 grid, IoU > 0.5)
    - c: Centroid token (3D center)
    - s: Shape token (Gaussian statistics + extents)
    - θ: Temporal token (t_start, t_end)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

import numpy as np

from fast_snow.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken, build_centroid_token, build_shape_token
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken, mask_to_patch_tokens
from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken


@dataclass(frozen=True)
class STEPToken:
    """Spatio-Temporal Tokenized Patch Encoding for a single object instance.

    This compact multimodal representation captures:
    - Semantic: image patch tokens (localized visual features)
    - Geometric: centroid + shape tokens (3D structure)
    - Temporal: appearance/disappearance timestamps (track-level)

    Attributes:
        patch_tokens: List of (row, col, iou) for 16×16 grid cells with IoU ≥ 0.5
        centroid: 3D center (x̄, ȳ, z̄) from Phase 3 mask-assigned points
        shape: Gaussian statistics (μ, σ, min, max) × 3 axes
        temporal: (t_start, t_end) track-level timestamps
    """
    patch_tokens: List[PatchToken]
    centroid: CentroidToken
    shape: ShapeToken
    temporal: TemporalToken


def build_step_token(
    mask: np.ndarray,
    points_xyz: np.ndarray,
    t_start: int,
    t_end: int,
    grid_size: int = 16,
    iou_threshold: float = 0.5,
) -> STEPToken:
    """Build STEP tokens for a single object instance.

    Paper workflow (Section 3.2):
    1. Partition mask into 16×16 grid
    2. Retain cells with IoU > 0.5 as patch tokens
    3. Compute centroid from 3D points
    4. Compute shape token (Gaussian + extents)
    5. Attach temporal token (t_start, t_end)

    Args:
        mask: (H, W) boolean mask from SAM2 segmentation.
        points_xyz: (N, 3) 3D points from Phase 3 (mask-assigned points).
        t_start: First appearance frame (Phase 4: current frame; Phase 6: track start).
        t_end: Last appearance frame (Phase 4: current frame; Phase 6: track end).
        grid_size: Grid dimension (default 16 for 256 patches).
        iou_threshold: Minimum IoU to retain patch (default 0.5).

    Returns:
        STEPToken with all components initialized.

    Note:
        In Phase 4, t_start = t_end = frame_idx (single-frame placeholder).
        In Phase 6, tracker updates temporal token to reflect true track span.
    """
    patch_tokens = mask_to_patch_tokens(mask, grid_size=grid_size, iou_threshold=iou_threshold)
    centroid = build_centroid_token(points_xyz)
    shape = build_shape_token(points_xyz)
    temporal = TemporalToken(t_start=t_start, t_end=t_end)
    return STEPToken(
        patch_tokens=patch_tokens,
        centroid=centroid,
        shape=shape,
        temporal=temporal,
    )


def update_temporal_token(
    step: STEPToken,
    t_start: int,
    t_end: int,
) -> STEPToken:
    """Update temporal token with track-level timestamps.

    This is called in Phase 6 after cross-frame association to replace
    the single-frame placeholder temporal token with true track-level timestamps.

    Args:
        step: Original STEPToken with placeholder temporal token.
        t_start: First frame of the track.
        t_end: Last frame of the track.

    Returns:
        New STEPToken with updated temporal token (all other fields unchanged).

    Example:
        # Phase 4: Create with placeholder
        step = build_step_token(mask, points, t_start=5, t_end=5, ...)

        # Phase 6: Update with track-level timestamps
        updated_step = update_temporal_token(step, t_start=3, t_end=8)
        assert updated_step.temporal.t_start == 3
        assert updated_step.temporal.t_end == 8
    """
    return replace(step, temporal=TemporalToken(t_start=t_start, t_end=t_end))
