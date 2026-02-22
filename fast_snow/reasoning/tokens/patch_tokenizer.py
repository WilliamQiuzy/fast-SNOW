"""Patch tokenizer for STEP encoding.

Paper reference (Section 3.2, Page 3):
    "The object mask is isolated by coloring all in-mask pixels. The masked
    image is partitioned into a fixed 16x16 grid, yielding 256 patches. Each
    grid cell is evaluated by its IoU with the mask. Cells with IoU>0.5 are
    retained as image patch tokens."

In the paper, tau (patch tokens) are actual image regions -- visual tokens fed
to the VLM's vision encoder.  Each PatchToken therefore carries both the grid
coordinates (row, col, iou) **and** the pixel crop of that cell region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class PatchToken:
    """A single image patch token from the 16x16 grid.

    Attributes:
        row: Grid row index (0-based).
        col: Grid column index (0-based).
        iou: Mask coverage ratio within this cell (intersection / cell_area).
        image_crop: (cell_h, cell_w, 3) uint8 RGB crop of the masked image
            region.  ``None`` when the tokenizer is called without an image
            (backward-compatible fallback).
    """
    row: int
    col: int
    iou: float
    image_crop: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


def mask_to_patch_tokens(
    mask: np.ndarray,
    grid_size: int = 16,
    iou_threshold: float = 0.5,
    image: Optional[np.ndarray] = None,
    mask_outside: bool = True,
    crop_size: Optional[int] = None,
) -> List[PatchToken]:
    """Convert a binary mask into patch tokens based on IoU with grid cells.

    When *image* is provided, each retained cell also carries the actual pixel
    crop (the "image patch token" in the paper).  Non-mask pixels within the
    cell are zeroed out when *mask_outside* is True (paper default).

    Args:
        mask: (H, W) boolean mask.
        grid_size: Grid dimension (default 16 for 16x16 = 256 patches).
        iou_threshold: Minimum coverage ratio to retain a patch (default 0.5).
        image: (H, W, 3) uint8 RGB image.  When provided, each PatchToken
            will include the cropped image region.  When ``None``, tokens
            contain only grid coordinates and iou (backward-compatible).
        mask_outside: If True and *image* is given, zero out pixels outside
            the mask within each cell crop.  This is the paper's behaviour
            ("The object mask is isolated by coloring all in-mask pixels").
        crop_size: If not None, resize each cell crop to (crop_size, crop_size)
            pixels.  ``None`` keeps the native cell resolution.

    Returns:
        List of PatchToken for cells with IoU > threshold.

    Raises:
        ValueError: If mask is not 2D, smaller than grid_size, or if image
            dimensions don't match mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    if h < grid_size or w < grid_size:
        raise ValueError(
            f"mask dimensions ({h}x{w}) must be >= grid_size ({grid_size}x{grid_size})"
        )

    if image is not None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
        if image.shape[0] != h or image.shape[1] != w:
            raise ValueError(
                f"image/mask shape mismatch: image={image.shape[:2]}, mask={mask.shape}"
            )

    # Precompute masked image once (avoid per-cell masking)
    masked_image: Optional[np.ndarray] = None
    if image is not None and mask_outside:
        masked_image = image.copy()
        masked_image[~mask] = 0
    elif image is not None:
        masked_image = image  # no masking, use original

    cell_h = h // grid_size
    cell_w = w // grid_size

    tokens: List[PatchToken] = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = h if r == grid_size - 1 else (r + 1) * cell_h
            x1 = w if c == grid_size - 1 else (c + 1) * cell_w

            cell_mask = mask[y0:y1, x0:x1]
            intersection = cell_mask.sum()
            cell_area = cell_mask.size
            iou = float(intersection / cell_area) if cell_area > 0 else 0.0

            if iou > iou_threshold:
                crop: Optional[np.ndarray] = None
                if masked_image is not None:
                    crop = masked_image[y0:y1, x0:x1].copy()
                    if crop_size is not None:
                        crop = cv2.resize(
                            crop, (crop_size, crop_size),
                            interpolation=cv2.INTER_AREA,
                        )
                tokens.append(PatchToken(row=r, col=c, iou=iou, image_crop=crop))

    return tokens
