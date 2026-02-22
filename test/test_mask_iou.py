"""Tests for _mask_iou static utility."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline


class TestMaskIoU:

    def test_identical_masks(self):
        m = np.ones((10, 10), dtype=bool)
        assert FastSNOWPipeline._mask_iou(m, m) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:5, :] = True
        b[5:10, :] = True
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:6, :] = True   # 60 pixels
        b[4:10, :] = True  # 60 pixels
        # intersection: rows 4-5 → 20 pixels
        # union: rows 0-9 → 100 pixels
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(20.0 / 100.0)

    def test_subset_mask(self):
        """Small mask fully inside big mask."""
        big = np.ones((10, 10), dtype=bool)
        small = np.zeros((10, 10), dtype=bool)
        small[3:7, 3:7] = True  # 16 pixels
        # intersection=16, union=100
        assert FastSNOWPipeline._mask_iou(big, small) == pytest.approx(16.0 / 100.0)

    def test_both_empty(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(0.0)

    def test_one_empty(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(0.0)

    def test_both_full(self):
        a = np.ones((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(1.0)

    def test_single_pixel_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:5, 0:5] = True   # 25 pixels
        b[4, 4] = True        # 1 pixel
        # intersection=1, union=25
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(1.0 / 25.0)

    def test_asymmetric_sizes(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:2, 0:2] = True  # 4 pixels
        b[0:8, 0:8] = True  # 64 pixels
        # intersection=4, union=64
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(4.0 / 64.0)

    def test_commutative(self):
        """IoU(a, b) == IoU(b, a)."""
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:6, :] = True
        b[3:10, :] = True
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(
            FastSNOWPipeline._mask_iou(b, a)
        )

    def test_perfect_complement(self):
        """a and ~a have no overlap when a has pixels."""
        a = np.zeros((10, 10), dtype=bool)
        a[0:5, :] = True
        b = ~a
        assert FastSNOWPipeline._mask_iou(a, b) == pytest.approx(0.0)
