"""Tests for STEPToken, build_step_token, update_temporal_token."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken
from fast_snow.reasoning.tokens.step_encoding import (
    STEPToken,
    build_step_token,
    update_temporal_token,
)
from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken


def _make_test_data(h=64, w=64):
    mask = np.zeros((h, w), dtype=bool)
    mask[10:50, 10:50] = True
    pts = np.random.RandomState(0).randn(100, 3).astype(np.float32) + 5.0
    return mask, pts


class TestBuildStepToken:

    def test_basic_all_components(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, t_start=0, t_end=5, grid_size=4, iou_threshold=0.1)
        assert isinstance(step.patch_tokens, list)
        assert isinstance(step.centroid, CentroidToken)
        assert isinstance(step.shape, ShapeToken)
        assert isinstance(step.temporal, TemporalToken)
        assert step.temporal.t_start == 0
        assert step.temporal.t_end == 5

    def test_with_image_crops_populated(self):
        mask, pts = _make_test_data()
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        step = build_step_token(
            mask, pts, 0, 0, grid_size=4, iou_threshold=0.1,
            image=img, crop_size=16,
        )
        has_crop = any(p.image_crop is not None for p in step.patch_tokens)
        assert has_crop, "At least some patches should have image crops"

    def test_without_image_crops_none(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, 0, 0, grid_size=4, iou_threshold=0.1)
        assert all(p.image_crop is None for p in step.patch_tokens)

    def test_grid_params_forwarded(self):
        mask, pts = _make_test_data()
        step_small = build_step_token(mask, pts, 0, 0, grid_size=2, iou_threshold=0.0)
        step_large = build_step_token(mask, pts, 0, 0, grid_size=4, iou_threshold=0.0)
        # Smaller grid → fewer max cells
        assert len(step_small.patch_tokens) <= 4
        assert len(step_large.patch_tokens) <= 16

    def test_high_threshold_fewer_patches(self):
        mask, pts = _make_test_data()
        step_low = build_step_token(mask, pts, 0, 0, grid_size=4, iou_threshold=0.1)
        step_high = build_step_token(mask, pts, 0, 0, grid_size=4, iou_threshold=0.9)
        assert len(step_high.patch_tokens) <= len(step_low.patch_tokens)

    def test_mask_outside_forwarded(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:32, 0:32] = True
        pts = np.ones((10, 3))
        img = np.full((64, 64, 3), 100, dtype=np.uint8)

        step_masked = build_step_token(
            mask, pts, 0, 0, grid_size=2, iou_threshold=0.0,
            image=img, mask_outside=True,
        )
        step_unmasked = build_step_token(
            mask, pts, 0, 0, grid_size=2, iou_threshold=0.0,
            image=img, mask_outside=False,
        )
        # Cell (1,1) is fully outside mask: with mask_outside=True → all zeros
        c11_masked = [p for p in step_masked.patch_tokens if p.row == 1 and p.col == 1]
        c11_unmasked = [p for p in step_unmasked.patch_tokens if p.row == 1 and p.col == 1]
        if c11_masked and c11_masked[0].image_crop is not None:
            assert np.all(c11_masked[0].image_crop == 0)
        if c11_unmasked and c11_unmasked[0].image_crop is not None:
            assert np.all(c11_unmasked[0].image_crop == 100)


class TestUpdateTemporalToken:

    def test_changes_theta(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, t_start=5, t_end=5, grid_size=4, iou_threshold=0.1)
        updated = update_temporal_token(step, t_start=0, t_end=20)
        assert updated.temporal.t_start == 0
        assert updated.temporal.t_end == 20

    def test_preserves_centroid(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, 5, 5, grid_size=4, iou_threshold=0.1)
        updated = update_temporal_token(step, 0, 20)
        assert updated.centroid == step.centroid

    def test_preserves_shape(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, 5, 5, grid_size=4, iou_threshold=0.1)
        updated = update_temporal_token(step, 0, 20)
        assert updated.shape == step.shape

    def test_preserves_patch_tokens(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, 5, 5, grid_size=4, iou_threshold=0.1)
        updated = update_temporal_token(step, 0, 20)
        assert len(updated.patch_tokens) == len(step.patch_tokens)
        for orig, upd in zip(step.patch_tokens, updated.patch_tokens):
            assert orig.row == upd.row
            assert orig.col == upd.col
            assert orig.iou == upd.iou

    def test_returns_new_object(self):
        mask, pts = _make_test_data()
        step = build_step_token(mask, pts, 5, 5, grid_size=4, iou_threshold=0.1)
        updated = update_temporal_token(step, 0, 20)
        assert step is not updated
        assert step.temporal.t_start == 5  # original unchanged
