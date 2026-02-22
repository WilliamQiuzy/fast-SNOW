"""Shared fixtures and synthetic data factories for Fast-SNOW tests."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pytest

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastFrameInput,
    FastLocalDetection,
    FastSNOWPipeline,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------

def make_mask(
    h: int = 64,
    w: int = 64,
    y0: int = 10,
    y1: int = 50,
    x0: int = 10,
    x1: int = 50,
) -> np.ndarray:
    """Rectangular boolean mask."""
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def make_depth(h: int = 64, w: int = 64, base: float = 5.0) -> np.ndarray:
    """Depth map with slight vertical gradient (base + row/h)."""
    d = np.full((h, w), base, dtype=np.float32)
    for r in range(h):
        d[r, :] += (r / h) * 0.5
    return d


def make_K(
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 32.0,
    cy: float = 32.0,
) -> np.ndarray:
    """3x3 camera intrinsics matrix."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def make_T_wc(
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """4x4 world→camera transform with optional yaw rotation (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def make_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Random uint8 RGB image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_detection(
    run_id: object = "run0",
    obj_id: object = 0,
    mask: Optional[np.ndarray] = None,
    score: float = 0.9,
    h: int = 64,
    w: int = 64,
) -> FastLocalDetection:
    """Create a FastLocalDetection with a default rectangular mask."""
    if mask is None:
        mask = make_mask(h, w)
    return FastLocalDetection(run_id=run_id, local_obj_id=obj_id, mask=mask, score=score)


def make_frame_input(
    frame_idx: int = 0,
    detections: Optional[Sequence[FastLocalDetection]] = None,
    h: int = 64,
    w: int = 64,
    base_depth: float = 5.0,
    T_wc: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    depth_conf: Optional[np.ndarray] = None,
    depth_is_metric: bool = True,
    timestamp_s: Optional[float] = None,
) -> FastFrameInput:
    """Shorthand for building FastFrameInput with sensible defaults."""
    if detections is None:
        detections = [make_detection(h=h, w=w)]
    if T_wc is None:
        T_wc = np.eye(4, dtype=np.float64)
    if K is None:
        K = make_K(cx=w / 2.0, cy=h / 2.0)
    if timestamp_s is None:
        timestamp_s = float(frame_idx)
    return FastFrameInput(
        frame_idx=frame_idx,
        depth_t=make_depth(h, w, base_depth),
        K_t=K,
        T_wc_t=T_wc,
        detections=detections,
        depth_conf_t=depth_conf,
        depth_is_metric=depth_is_metric,
        timestamp_s=timestamp_s,
    )


# ---------------------------------------------------------------------------
# Config shortcut
# ---------------------------------------------------------------------------

def relaxed_config() -> FastSNOWConfig:
    """Config with relaxed filters so small synthetic data passes."""
    cfg = FastSNOWConfig()
    cfg.depth_filter.min_points = 1
    cfg.depth_filter.max_extent = 1e6
    cfg.depth_filter.conf_thresh = 0.0
    cfg.step.grid_size = 4
    cfg.step.iou_threshold = 0.1
    cfg.step.patch_crop_size = 16
    cfg.step.max_tau_per_step = 0  # unlimited
    cfg.fusion.cross_run_iou_thresh = 0.3
    cfg.fusion.merge_centroid_dist_m = 100.0
    cfg.fusion.merge_temporal_gap = 100
    cfg.fusion.lost_patience = 3
    cfg.fusion.archive_patience = 3
    cfg.edge.motion_window = 3
    return cfg


# ---------------------------------------------------------------------------
# Pipeline shortcut fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    """A pipeline with relaxed config for unit testing."""
    return FastSNOWPipeline(config=relaxed_config())
