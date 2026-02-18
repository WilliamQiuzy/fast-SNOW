"""Patch tokenizer for STEP encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class PatchToken:
    row: int
    col: int
    iou: float


def mask_to_patch_tokens(
    mask: np.ndarray,
    grid_size: int = 16,
    iou_threshold: float = 0.5,
) -> List[PatchToken]:
    """Convert a binary mask into patch tokens based on IoU with grid cells.

    Paper reference (Section 3.2, Page 3, Lines 259-263):
        "The masked image is partitioned into a fixed 16×16 grid, yielding 256
        patches. Each grid cell is evaluated by its Intersection-over-Union (IoU)
        with the mask. Cells with IoU>0.5 are retained as image patch tokens."

    Implementation note:
        The "IoU" here refers to LOCAL COVERAGE RATIO within each cell, i.e.,
        what fraction of the cell is covered by the mask:
            IoU = (mask pixels in cell) / (total pixels in cell)

        This is NOT the global IoU (intersection / (mask_area + cell_area - intersection)).
        The paper's phrasing "IoU containment" (Figure 3) confirms this interpretation.

    Args:
        mask: (H, W) boolean mask. H and W should be >= grid_size for meaningful results.
        grid_size: Grid dimension (default 16 for 16×16 = 256 patches).
        iou_threshold: Minimum coverage ratio to retain a patch (default 0.5 = 50%).

    Returns:
        List of PatchToken, each containing (row, col, iou) for cells with IoU > threshold.

    Raises:
        ValueError: If mask is not 2D or smaller than grid_size.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    if h < grid_size or w < grid_size:
        raise ValueError(
            f"mask dimensions ({h}×{w}) must be >= grid_size ({grid_size}×{grid_size})"
        )

    # Compute cell dimensions
    # Paper: 16×16 grid, last row/col may be larger to cover remaining pixels
    cell_h = h // grid_size
    cell_w = w // grid_size

    tokens: List[PatchToken] = []
    for r in range(grid_size):
        for c in range(grid_size):
            # Cell boundaries
            y0 = r * cell_h
            x0 = c * cell_w
            # Last row/col extends to image boundary
            y1 = h if r == grid_size - 1 else (r + 1) * cell_h
            x1 = w if c == grid_size - 1 else (c + 1) * cell_w

            # Extract cell mask
            cell = mask[y0:y1, x0:x1]

            # Compute local IoU (coverage ratio)
            # Paper (Section 3.2): "Cells with IoU > 0.5 are retained"
            # Note: Strict inequality (>), not (>=)
            intersection = cell.sum()  # Number of True pixels in cell
            cell_area = cell.size      # Total pixels in cell
            iou = float(intersection / cell_area) if cell_area > 0 else 0.0

            # Retain patch if coverage > threshold (strict inequality per paper)
            if iou > iou_threshold:
                tokens.append(PatchToken(row=r, col=c, iou=iou))

    return tokens
