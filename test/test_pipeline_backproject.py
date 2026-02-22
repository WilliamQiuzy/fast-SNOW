"""Tests for _backproject_mask_points (Step 4 internals)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline

from conftest import make_K, make_depth, make_mask, relaxed_config


@pytest.fixture
def pipe():
    return FastSNOWPipeline(config=relaxed_config())


class TestBackprojectBasic:

    def test_identity_transform(self, pipe):
        """K=eye, T_cw=eye → p_world = depth * [u, v, 1]."""
        h, w = 4, 4
        mask = np.zeros((h, w), dtype=bool)
        mask[2, 3] = True  # single pixel
        depth = np.full((h, w), 2.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)  # T_cw = inv(T_wc=eye) = eye

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape == (1, 3)
        # p_cam = depth * K_inv @ [u, v, 1] = 2 * [3, 2, 1] = [6, 4, 2]
        np.testing.assert_allclose(pts[0], [6.0, 4.0, 2.0], atol=1e-5)

    def test_known_intrinsics(self, pipe):
        """Verify manually computed 3D point with real intrinsics."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True
        depth = np.full((h, w), 10.0, dtype=np.float32)
        K = make_K(fx=500.0, fy=500.0, cx=32.0, cy=32.0)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        # At (u=32, v=32), K_inv @ [32, 32, 1] = [0, 0, 1/1] (principal point)
        # p_cam = 10 * [0, 0, 1] = [0, 0, 10]
        assert pts.shape == (1, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 10.0], atol=1e-4)

    def test_single_pixel_off_center(self, pipe):
        """Pixel away from principal point."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 42] = True  # u=42, v=32
        depth = np.full((h, w), 5.0, dtype=np.float32)
        K = make_K(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        # K_inv @ [42, 32, 1] → [(42-32)/100, 0, 1] = [0.1, 0, 1]
        # p_cam = 5 * [0.1, 0, 1] = [0.5, 0, 5]
        np.testing.assert_allclose(pts[0], [0.5, 0.0, 5.0], atol=1e-4)


class TestDepthFiltering:

    def test_confidence_filters_low(self, pipe):
        h, w = 8, 8
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)
        conf = np.full((h, w), 0.01, dtype=np.float32)  # all below default thresh

        pipe.config.depth_filter.conf_thresh = 0.5
        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, conf)
        assert pts.shape[0] == 0

    def test_partial_confidence(self, pipe):
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 3.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)
        conf = np.zeros((h, w), dtype=np.float32)
        conf[0, 0] = 1.0  # only 1 pixel passes

        pipe.config.depth_filter.conf_thresh = 0.5
        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, conf)
        assert pts.shape[0] == 1

    def test_nan_depth_skipped(self, pipe):
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        depth[1, 1] = np.nan
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape[0] == h * w - 1

    def test_inf_depth_skipped(self, pipe):
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        depth[0, 0] = np.inf
        depth[0, 1] = -np.inf
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape[0] == h * w - 2

    def test_zero_depth_skipped(self, pipe):
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        depth[2, 2] = 0.0
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape[0] == h * w - 1

    def test_negative_depth_skipped(self, pipe):
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        depth[3, 3] = -1.0
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape[0] == h * w - 1

    def test_no_conf_defaults_to_ones(self, pipe):
        """When depth_conf_t is None, all pixels with valid depth pass."""
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 3.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape[0] == h * w


class TestBackprojectValidation:

    def test_mask_depth_shape_mismatch(self, pipe):
        mask = np.ones((8, 8), dtype=bool)
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            pipe._backproject_mask_points(mask, depth, np.eye(3), np.eye(4), None)

    def test_conf_shape_mismatch(self, pipe):
        mask = np.ones((8, 8), dtype=bool)
        depth = np.ones((8, 8), dtype=np.float32)
        conf = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            pipe._backproject_mask_points(mask, depth, np.eye(3), np.eye(4), conf)

    def test_mask_not_2d(self, pipe):
        mask = np.ones((4, 4, 3), dtype=bool)
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            pipe._backproject_mask_points(mask, depth, np.eye(3), np.eye(4), None)

    def test_empty_mask_returns_empty(self, pipe):
        mask = np.zeros((8, 8), dtype=bool)
        depth = np.ones((8, 8), dtype=np.float32)
        pts = pipe._backproject_mask_points(mask, depth, np.eye(3), np.eye(4), None)
        assert pts.shape == (0, 3)


class TestTransforms:

    def test_translation(self, pipe):
        """T_wc with translation only → world coords shifted."""
        h, w = 4, 4
        mask = np.zeros((h, w), dtype=bool)
        mask[0, 0] = True
        depth = np.full((h, w), 1.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 10.0  # translate x by 10 in world→camera
        T_cw = np.linalg.inv(T_wc)

        pts_shifted = pipe._backproject_mask_points(mask, depth, K, T_cw, None)

        # Without translation
        T_cw_id = np.eye(4, dtype=np.float64)
        pts_origin = pipe._backproject_mask_points(mask, depth, K, T_cw_id, None)

        # The shift should appear in world x coordinate
        assert pts_shifted[0, 0] != pytest.approx(pts_origin[0, 0], abs=0.1)

    def test_rotation_90_yaw(self, pipe):
        """90° yaw rotation should swap axes."""
        h, w = 4, 4
        mask = np.zeros((h, w), dtype=bool)
        mask[0, 0] = True
        depth = np.full((h, w), 1.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)

        yaw = math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 0] = c; T_wc[0, 1] = -s
        T_wc[1, 0] = s; T_wc[1, 1] = c
        T_cw = np.linalg.inv(T_wc)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.shape == (1, 3)

    def test_output_dtype_float32(self, pipe):
        """Spec §5.5: output should be float32."""
        h, w = 4, 4
        mask = np.ones((h, w), dtype=bool)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        T_cw = np.eye(4, dtype=np.float64)

        pts = pipe._backproject_mask_points(mask, depth, K, T_cw, None)
        assert pts.dtype == np.float32
