"""Tests for DA3Result dataclass (no model loading, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.vision.perception.da3_wrapper import DA3Result


class TestDA3Result:

    def _make_result(self, T_wc=None, is_metric=True):
        h, w = 64, 64
        if T_wc is None:
            T_wc = np.eye(4, dtype=np.float64)
        return DA3Result(
            depth=np.ones((h, w), dtype=np.float32),
            K=np.eye(3, dtype=np.float64),
            T_wc=T_wc,
            depth_conf=np.ones((h, w), dtype=np.float32),
            is_metric=is_metric,
        )

    def test_T_cw_is_inverse_of_T_wc(self):
        T_wc = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        r = self._make_result(T_wc=T_wc)
        T_cw = r.T_cw
        identity = T_wc @ T_cw
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-10)

    def test_T_cw_identity(self):
        r = self._make_result()
        np.testing.assert_allclose(r.T_cw, np.eye(4), atol=1e-10)

    def test_is_metric_default_true(self):
        r = self._make_result()
        assert r.is_metric is True

    def test_is_metric_false(self):
        r = self._make_result(is_metric=False)
        assert r.is_metric is False

    def test_depth_shape(self):
        r = self._make_result()
        assert r.depth.shape == (64, 64)

    def test_K_shape(self):
        r = self._make_result()
        assert r.K.shape == (3, 3)

    def test_T_wc_shape(self):
        r = self._make_result()
        assert r.T_wc.shape == (4, 4)

    def test_depth_conf_shape(self):
        r = self._make_result()
        assert r.depth_conf.shape == (64, 64)

    def test_T_cw_with_rotation_and_translation(self):
        """Verify T_cw correctly inverts a non-trivial T_wc."""
        import math
        angle = math.pi / 4  # 45 degrees
        c, s = math.cos(angle), math.sin(angle)
        T_wc = np.array([
            [c, -s, 0, 5],
            [s, c, 0, -3],
            [0, 0, 1, 10],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        r = self._make_result(T_wc=T_wc)
        # T_cw @ T_wc should be identity
        product = r.T_cw @ T_wc
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_T_cw_translation_is_camera_position(self):
        """T_cw[:3, 3] gives the camera position in world coordinates."""
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 10.0  # world→camera: translate x by 10
        r = self._make_result(T_wc=T_wc)
        # Camera at world position (-10, 0, 0) (opposite of T_wc translation for pure translation)
        np.testing.assert_allclose(r.T_cw[0, 3], -10.0, atol=1e-10)
