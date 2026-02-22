"""Tests for CentroidToken, ShapeToken, and their builders."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.reasoning.tokens.geometry_tokens import (
    CentroidToken,
    ShapeToken,
    build_centroid_token,
    build_shape_token,
)


# ── CentroidToken ──────────────────────────────────────────────────────────

class TestBuildCentroidToken:

    def test_basic_mean(self):
        pts = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]], dtype=float)
        c = build_centroid_token(pts)
        assert c.x == pytest.approx(4.0)
        assert c.y == pytest.approx(5.0)
        assert c.z == pytest.approx(6.0)

    def test_single_point(self):
        pts = np.array([[1.5, -2.5, 3.5]], dtype=float)
        c = build_centroid_token(pts)
        assert c.x == pytest.approx(1.5)
        assert c.y == pytest.approx(-2.5)
        assert c.z == pytest.approx(3.5)

    def test_negative_coords(self):
        pts = np.array([[-10, -20, -30], [-30, -40, -50]], dtype=float)
        c = build_centroid_token(pts)
        assert c.x == pytest.approx(-20.0)
        assert c.y == pytest.approx(-30.0)
        assert c.z == pytest.approx(-40.0)

    def test_large_cloud_precision(self):
        rng = np.random.RandomState(42)
        pts = rng.randn(10_000, 3).astype(np.float64) * 100
        c = build_centroid_token(pts)
        expected = pts.mean(axis=0)
        assert c.x == pytest.approx(expected[0], abs=1e-6)
        assert c.y == pytest.approx(expected[1], abs=1e-6)
        assert c.z == pytest.approx(expected[2], abs=1e-6)

    def test_returns_python_float(self):
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        c = build_centroid_token(pts)
        assert isinstance(c.x, float)
        assert isinstance(c.y, float)
        assert isinstance(c.z, float)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="shape.*\\(N, 3\\)"):
            build_centroid_token(np.array([1.0, 2.0, 3.0]))

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="shape.*\\(N, 3\\)"):
            build_centroid_token(np.ones((5, 4)))

    def test_empty_array_warns_or_errors(self):
        """(0, 3) array: numpy mean returns nan — function should still work
        (the caller is responsible for filtering empty point clouds)."""
        pts = np.zeros((0, 3), dtype=float)
        # The function doesn't explicitly guard empty; numpy produces RuntimeWarning
        with pytest.warns(RuntimeWarning):
            c = build_centroid_token(pts)
        assert np.isnan(c.x)


# ── ShapeToken ─────────────────────────────────────────────────────────────

class TestBuildShapeToken:

    def test_basic_stats(self):
        pts = np.array([[0, 0, 0], [4, 6, 8]], dtype=float)
        s = build_shape_token(pts)
        assert s.x_mu == pytest.approx(2.0)
        assert s.y_mu == pytest.approx(3.0)
        assert s.z_mu == pytest.approx(4.0)
        assert s.x_min == pytest.approx(0.0)
        assert s.x_max == pytest.approx(4.0)
        assert s.y_min == pytest.approx(0.0)
        assert s.y_max == pytest.approx(6.0)
        assert s.z_min == pytest.approx(0.0)
        assert s.z_max == pytest.approx(8.0)

    def test_single_point_sigma_zero(self):
        pts = np.array([[3.0, -1.0, 7.0]])
        s = build_shape_token(pts)
        assert s.x_sigma == pytest.approx(0.0)
        assert s.y_sigma == pytest.approx(0.0)
        assert s.z_sigma == pytest.approx(0.0)
        assert s.x_min == s.x_max == pytest.approx(3.0)

    def test_two_points_predictable_sigma(self):
        pts = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
        s = build_shape_token(pts)
        # numpy std (ddof=0) of [0, 2] = 1.0
        assert s.x_sigma == pytest.approx(1.0)
        assert s.y_sigma == pytest.approx(0.0)
        assert s.z_sigma == pytest.approx(0.0)

    def test_collinear_along_z(self):
        pts = np.array([[5, 5, 0], [5, 5, 10], [5, 5, 20]], dtype=float)
        s = build_shape_token(pts)
        assert s.x_sigma == pytest.approx(0.0)
        assert s.y_sigma == pytest.approx(0.0)
        assert s.z_sigma > 0.0
        assert s.z_min == pytest.approx(0.0)
        assert s.z_max == pytest.approx(20.0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            build_shape_token(np.ones((5,)))
        with pytest.raises(ValueError):
            build_shape_token(np.ones((5, 2)))

    def test_shape_has_12_fields(self):
        pts = np.ones((3, 3))
        s = build_shape_token(pts)
        field_names = [f.name for f in s.__dataclass_fields__.values()]
        assert len(field_names) == 12


# ── Frozen behaviour ───────────────────────────────────────────────────────

class TestFrozen:

    def test_centroid_frozen(self):
        c = CentroidToken(x=1.0, y=2.0, z=3.0)
        with pytest.raises(AttributeError):
            c.x = 99.0  # type: ignore[misc]

    def test_shape_frozen(self):
        s = build_shape_token(np.array([[0, 0, 0], [1, 1, 1]], dtype=float))
        with pytest.raises(AttributeError):
            s.x_mu = 99.0  # type: ignore[misc]

    def test_centroid_equality(self):
        a = CentroidToken(x=1.0, y=2.0, z=3.0)
        b = CentroidToken(x=1.0, y=2.0, z=3.0)
        assert a == b

    def test_centroid_hashable(self):
        c = CentroidToken(x=1.0, y=2.0, z=3.0)
        assert {c}  # can be put in a set
