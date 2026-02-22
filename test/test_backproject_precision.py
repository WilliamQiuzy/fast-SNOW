"""Precision tests for _backproject_mask_points.

Every test hand-computes the expected 3D world coordinates and compares
against the pipeline's output.  This is the single most critical
numeric path: depth + K + T_wc → 3D world points → centroid / shape.

Backprojection math (pipeline lines 644-654):
    ray = K_inv @ [u, v, 1]^T
    p_cam = ray * depth
    p_world = T_cw @ [p_cam; 1]          (T_cw = inv(T_wc))

For K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]:
    K_inv @ [u, v, 1] = [(u-cx)/fx,  (v-cy)/fy,  1]

So p_cam = depth * [(u-cx)/fx,  (v-cy)/fy,  1].
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline

from conftest import relaxed_config


# ── Helpers ──────────────────────────────────────────────────────────────

def _pipe() -> FastSNOWPipeline:
    return FastSNOWPipeline(config=relaxed_config())


def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _T_wc_from_rotation_translation(
    R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Build T_wc (world→camera) from 3x3 rotation and 3-vector translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _single_pixel_mask(h: int, w: int, row: int, col: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[row, col] = True
    return mask


def _uniform_depth(h: int, w: int, d: float) -> np.ndarray:
    return np.full((h, w), d, dtype=np.float32)


def _expected_p_cam(u: int, v: int, depth: float,
                    fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Hand-computed camera-frame point for a single pixel."""
    return np.array([
        depth * (u - cx) / fx,
        depth * (v - cy) / fy,
        depth,
    ], dtype=np.float64)


def _expected_p_world(p_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform camera point to world using T_cw = inv(T_wc)."""
    T_cw = np.linalg.inv(T_wc)
    p_h = np.append(p_cam, 1.0)
    return (T_cw @ p_h)[:3]


# ── Single-pixel exact tests ────────────────────────────────────────────

class TestSinglePixelExact:
    """Each test specifies exactly 1 masked pixel, known K, known depth,
    known T_wc, and verifies the 3D output to high precision."""

    def test_principal_point_identity_pose(self):
        """Pixel at principal point, K with cx=cy=center, T_wc=I.
        Expected: p_cam = [0, 0, depth], p_world = same."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 32)  # row=v=32, col=u=32
        K = _K(fx, fy, cx, cy)
        T_wc = np.eye(4, dtype=np.float64)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.linalg.inv(T_wc), None
        )
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 10.0], atol=1e-5)

    def test_off_center_pixel(self):
        """Pixel at (u=42, v=32), fx=fy=100, cx=cy=32, depth=5."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 42)  # row=v=32, col=u=42
        K = _K(fx, fy, cx, cy)
        T_wc = np.eye(4, dtype=np.float64)

        expected = _expected_p_cam(u=42, v=32, depth=depth,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
        # expected = [5*(42-32)/100, 5*(32-32)/100, 5] = [0.5, 0.0, 5.0]
        assert expected == pytest.approx([0.5, 0.0, 5.0])

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts[0], expected, atol=1e-5)

    def test_corner_pixel_top_left(self):
        """Pixel (u=0, v=0), cx=32, cy=32, fx=fy=200, depth=8."""
        h, w, depth = 64, 64, 8.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 0, 0)  # v=0, u=0
        K = _K(fx, fy, cx, cy)

        expected = _expected_p_cam(0, 0, depth, fx, fy, cx, cy)
        # = [8*(0-32)/200, 8*(0-32)/200, 8] = [-1.28, -1.28, 8.0]
        assert expected[0] == pytest.approx(-1.28)
        assert expected[1] == pytest.approx(-1.28)
        assert expected[2] == pytest.approx(8.0)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts[0], expected, atol=1e-4)

    def test_corner_pixel_bottom_right(self):
        """Pixel (u=63, v=63), verify symmetry with top-left."""
        h, w, depth = 64, 64, 8.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 63, 63)  # v=63, u=63
        K = _K(fx, fy, cx, cy)

        expected = _expected_p_cam(63, 63, depth, fx, fy, cx, cy)
        # = [8*(63-32)/200, 8*(63-32)/200, 8] = [1.24, 1.24, 8.0]
        assert expected[0] == pytest.approx(1.24)
        assert expected[1] == pytest.approx(1.24)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts[0], expected, atol=1e-4)

    def test_asymmetric_focal_length(self):
        """fx != fy: verify x and y scale independently."""
        h, w, depth = 64, 64, 6.0
        fx, fy, cx, cy = 300.0, 600.0, 32.0, 32.0
        # Pixel at (u=62, v=62): offset = 30 from principal point
        mask = _single_pixel_mask(h, w, 62, 62)
        K = _K(fx, fy, cx, cy)

        expected = _expected_p_cam(62, 62, depth, fx, fy, cx, cy)
        # x = 6*(62-32)/300 = 6*30/300 = 0.6
        # y = 6*(62-32)/600 = 6*30/600 = 0.3
        # z = 6.0
        assert expected[0] == pytest.approx(0.6)
        assert expected[1] == pytest.approx(0.3)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts[0], expected, atol=1e-4)

    def test_asymmetric_principal_point(self):
        """cx != cy: verify the center offset is applied correctly."""
        h, w, depth = 64, 64, 4.0
        fx, fy = 100.0, 100.0
        cx, cy = 20.0, 40.0  # off-center
        mask = _single_pixel_mask(h, w, 40, 20)  # v=40, u=20 → at principal point

        expected = _expected_p_cam(20, 40, depth, fx, fy, cx, cy)
        # = [4*(20-20)/100, 4*(40-40)/100, 4] = [0, 0, 4]
        np.testing.assert_allclose(expected, [0.0, 0.0, 4.0])

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth),
            _K(fx, fy, cx, cy), np.eye(4), None
        )
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 4.0], atol=1e-5)

    def test_varying_depth_per_pixel(self):
        """Two pixels at different depths: verify each gets its own depth."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True  # principal point
        mask[32, 42] = True  # 10 pixels to the right

        depth = np.full((h, w), 5.0, dtype=np.float32)
        depth[32, 32] = 10.0
        depth[32, 42] = 20.0

        pts = _pipe()._backproject_mask_points(
            mask, depth, _K(fx, fy, cx, cy), np.eye(4), None
        )
        assert pts.shape == (2, 3)

        # Principal point: p = [10*(32-32)/100, 10*(32-32)/100, 10] = [0, 0, 10]
        # Right pixel:     p = [20*(42-32)/100, 20*(32-32)/100, 20] = [2, 0, 20]
        # Sort by z for stable comparison
        pts_sorted = pts[pts[:, 2].argsort()]
        np.testing.assert_allclose(pts_sorted[0], [0.0, 0.0, 10.0], atol=1e-4)
        np.testing.assert_allclose(pts_sorted[1], [2.0, 0.0, 20.0], atol=1e-4)


# ── Pose transform tests ────────────────────────────────────────────────

class TestPoseTransforms:
    """Verify that camera pose (T_wc) correctly transforms 3D points."""

    def test_pure_translation_x(self):
        """Camera translated +10 in x → world points shift by -10 in x."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 32)  # principal point
        K = _K(fx, fy, cx, cy)

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 10.0  # world→camera: translate x

        p_cam = _expected_p_cam(32, 32, depth, fx, fy, cx, cy)
        p_world = _expected_p_world(p_cam, T_wc)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-4)
        # p_cam = [0, 0, 5], T_cw = inv([[1,0,0,10],...]) → shifts x by -10
        assert pts[0, 0] == pytest.approx(-10.0, abs=1e-4)

    def test_pure_translation_z(self):
        """Camera translated +3 in z."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 32)
        K = _K(fx, fy, cx, cy)

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[2, 3] = 3.0

        p_cam = np.array([0.0, 0.0, 10.0])
        p_world = _expected_p_world(p_cam, T_wc)
        # T_cw shifts z by -3 → world z = 10 - 3 = 7
        assert p_world[2] == pytest.approx(7.0)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-4)

    def test_90_degree_yaw_rotation(self):
        """Camera rotated 90° yaw (around z-axis).
        Rotation R_z(90°) applied in T_wc swaps x↔y in world coords."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        # Pixel at (u=42, v=32) → p_cam = [0.5, 0, 5]
        mask = _single_pixel_mask(h, w, 32, 42)
        K = _K(fx, fy, cx, cy)

        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        T_wc = _T_wc_from_rotation_translation(R, np.zeros(3))

        p_cam = np.array([0.5, 0.0, 5.0])
        p_world = _expected_p_world(p_cam, T_wc)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-4)

    def test_180_degree_yaw(self):
        """Camera facing backwards: 180° yaw."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 42)
        K = _K(fx, fy, cx, cy)

        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        T_wc = _T_wc_from_rotation_translation(R, np.zeros(3))

        p_cam = _expected_p_cam(42, 32, depth, fx, fy, cx, cy)
        p_world = _expected_p_world(p_cam, T_wc)

        # 180° yaw: p_cam.x=0.5 → p_world.x=-0.5, p_cam.y=0 → p_world.y=0
        assert p_world[0] == pytest.approx(-0.5, abs=1e-6)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-4)

    def test_45_degree_pitch(self):
        """Camera pitched down 45° (rotation around x-axis)."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 32)
        K = _K(fx, fy, cx, cy)

        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        # Rotation around x-axis
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        T_wc = _T_wc_from_rotation_translation(R, np.zeros(3))

        p_cam = np.array([0.0, 0.0, 10.0])  # principal point
        p_world = _expected_p_world(p_cam, T_wc)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-4)

    def test_combined_rotation_and_translation(self):
        """Non-trivial pose: 30° yaw + translation (5, -3, 10)."""
        h, w, depth = 64, 64, 7.0
        fx, fy, cx, cy = 300.0, 300.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 40, 50)  # v=40, u=50
        K = _K(fx, fy, cx, cy)

        angle = math.pi / 6  # 30°
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        t = np.array([5.0, -3.0, 10.0])
        T_wc = _T_wc_from_rotation_translation(R, t)

        p_cam = _expected_p_cam(50, 40, depth, fx, fy, cx, cy)
        p_world = _expected_p_world(p_cam, T_wc)

        T_cw = np.linalg.inv(T_wc)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        np.testing.assert_allclose(pts[0], p_world, atol=1e-3)


# ── Multi-pixel centroid accuracy ────────────────────────────────────────

class TestMultiPixelCentroid:
    """Verify that the mean of backprojected points matches hand computation."""

    def test_2x2_block_centroid(self):
        """2x2 pixel block at known position, uniform depth."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[30:32, 30:32] = True  # 4 pixels: (v,u) = (30,30),(30,31),(31,30),(31,31)
        K = _K(fx, fy, cx, cy)

        # Manually compute all 4 camera-frame points
        expected_pts = []
        for v in [30, 31]:
            for u in [30, 31]:
                expected_pts.append(_expected_p_cam(u, v, depth, fx, fy, cx, cy))
        expected_pts = np.array(expected_pts)
        expected_centroid = expected_pts.mean(axis=0)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        assert pts.shape == (4, 3)
        actual_centroid = pts.mean(axis=0)
        np.testing.assert_allclose(actual_centroid, expected_centroid, atol=1e-4)

    def test_symmetric_block_around_principal(self):
        """Symmetric block centered on principal point → centroid at (0, 0, depth)."""
        h, w, depth = 64, 64, 8.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        # 4 pixels symmetric around (32, 32): (31,31),(31,32),(32,31),(32,32)
        # Note: pixel centers are at integer coords, so this is not perfectly
        # symmetric. True center is at (31.5, 31.5), offset -0.5 from (32, 32).
        mask = np.zeros((h, w), dtype=bool)
        mask[31:33, 31:33] = True
        K = _K(fx, fy, cx, cy)

        expected_pts = []
        for v in [31, 32]:
            for u in [31, 32]:
                expected_pts.append(_expected_p_cam(u, v, depth, fx, fy, cx, cy))
        expected_mean = np.array(expected_pts).mean(axis=0)
        # Mean u = 31.5, mean v = 31.5 → x = 8*(31.5-32)/200 = -0.02, same for y
        assert expected_mean[0] == pytest.approx(-0.02)
        assert expected_mean[1] == pytest.approx(-0.02)
        assert expected_mean[2] == pytest.approx(8.0)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts.mean(axis=0), expected_mean, atol=1e-4)

    def test_horizontal_strip_centroid(self):
        """Row of 10 pixels: centroid should be at the row's center."""
        h, w, depth = 64, 64, 6.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 20:30] = True  # u = 20..29, v = 32
        K = _K(fx, fy, cx, cy)

        expected_xs = [depth * (u - cx) / fx for u in range(20, 30)]
        expected_centroid_x = np.mean(expected_xs)
        expected_centroid_y = depth * (32 - cy) / fy
        expected_centroid_z = depth

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        centroid = pts.mean(axis=0)
        assert centroid[0] == pytest.approx(expected_centroid_x, abs=1e-4)
        assert centroid[1] == pytest.approx(expected_centroid_y, abs=1e-4)
        assert centroid[2] == pytest.approx(expected_centroid_z, abs=1e-4)

    def test_depth_gradient_affects_centroid_z(self):
        """Linear depth gradient along rows: centroid z != uniform depth."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[30:35, 32] = True  # 5 pixels in a column at u=32 (principal point x)
        K = _K(fx, fy, cx, cy)

        depth = np.full((h, w), 10.0, dtype=np.float32)
        # Set specific depths for the masked pixels
        for v in range(30, 35):
            depth[v, 32] = float(v)  # depth = 30, 31, 32, 33, 34

        pts = _pipe()._backproject_mask_points(
            mask, depth, K, np.eye(4), None
        )
        assert pts.shape == (5, 3)
        # z values should be exactly the depths (principal point x → x=0)
        z_values = sorted(pts[:, 2].tolist())
        np.testing.assert_allclose(z_values, [30, 31, 32, 33, 34], atol=1e-3)
        # Centroid z = mean(30..34) = 32
        assert pts.mean(axis=0)[2] == pytest.approx(32.0, abs=1e-3)


# ── Focal length effects ────────────────────────────────────────────────

class TestFocalLengthEffect:
    """Verify that focal length correctly scales the spatial spread."""

    def test_wider_fov_larger_spread(self):
        """Smaller focal length (wider FOV) → larger x spread at same depth."""
        h, w, depth = 64, 64, 10.0
        cx, cy = 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 0] = True    # leftmost pixel
        mask[32, 63] = True   # rightmost pixel

        # Narrow FOV (large fx)
        pts_narrow = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth),
            _K(500, 500, cx, cy), np.eye(4), None
        )
        spread_narrow = pts_narrow[:, 0].max() - pts_narrow[:, 0].min()

        # Wide FOV (small fx)
        pts_wide = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth),
            _K(100, 100, cx, cy), np.eye(4), None
        )
        spread_wide = pts_wide[:, 0].max() - pts_wide[:, 0].min()

        assert spread_wide > spread_narrow * 4  # fx ratio is 5:1

    def test_exact_spread_ratio(self):
        """For pixels equidistant from cx, spread scales as 1/fx."""
        h, w, depth = 64, 64, 10.0
        cx, cy = 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True  # u=22, offset = 10 from cx
        mask[32, 42] = True  # u=42, offset = 10 from cx

        fx_a, fx_b = 200.0, 400.0

        pts_a = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth),
            _K(fx_a, fx_a, cx, cy), np.eye(4), None
        )
        pts_b = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth),
            _K(fx_b, fx_b, cx, cy), np.eye(4), None
        )

        spread_a = abs(pts_a[0, 0] - pts_a[1, 0])
        spread_b = abs(pts_b[0, 0] - pts_b[1, 0])

        # spread ∝ 1/fx → ratio should be fx_b/fx_a = 2.0
        ratio = spread_a / spread_b
        assert ratio == pytest.approx(2.0, abs=1e-4)


# ── Numerical precision ─────────────────────────────────────────────────

class TestNumericalPrecision:
    """Verify float32 cast doesn't destroy accuracy for realistic ranges."""

    def test_float32_precision_near_origin(self):
        """Points near origin: float32 has ~7 decimal digits, so sub-mm precision."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 33)
        K = _K(fx, fy, cx, cy)

        expected = _expected_p_cam(33, 32, depth, fx, fy, cx, cy)
        # = [5*(33-32)/500, 0, 5] = [0.01, 0, 5]
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        assert pts.dtype == np.float32
        np.testing.assert_allclose(pts[0], expected, atol=1e-5)

    def test_float32_precision_at_large_distance(self):
        """Points at 100m range: float32 gives ~0.01m precision."""
        h, w, depth = 64, 64, 100.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 40, 50)
        K = _K(fx, fy, cx, cy)

        expected = _expected_p_cam(50, 40, depth, fx, fy, cx, cy)
        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        # At depth=100, expected.x = 100*(50-32)/500 = 3.6
        assert expected[0] == pytest.approx(3.6)
        # float32 at magnitude ~100 has precision ~1e-5 → sub-cm
        np.testing.assert_allclose(pts[0], expected, atol=0.01)

    def test_float32_precision_with_translation(self):
        """Large translation: T_wc with t=[1000, 0, 0]."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 32)
        K = _K(fx, fy, cx, cy)

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 1000.0
        T_cw = np.linalg.inv(T_wc)

        p_cam = np.array([0.0, 0.0, 10.0])
        p_world = _expected_p_world(p_cam, T_wc)
        # p_world.x = 0 - 1000 = -1000

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        # float32 at magnitude 1000: precision ~0.0001
        np.testing.assert_allclose(pts[0], p_world, atol=0.1)

    def test_computation_in_float64_internally(self):
        """Even though output is float32, internal math uses float64.
        Verify by checking a case where float32 intermediate would fail."""
        h, w = 64, 64
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = _single_pixel_mask(h, w, 32, 33)
        K = _K(fx, fy, cx, cy)

        # depth = 1e6: p_cam.x = 1e6 * 1/500 = 2000
        # float32 can represent 2000 exactly, but intermediate products
        # like K_inv @ uv1 must be precise.
        depth_big = np.full((h, w), 1e6, dtype=np.float32)
        pts = _pipe()._backproject_mask_points(
            mask, depth_big, K, np.eye(4), None
        )
        expected_x = 1e6 * (33 - 32) / 500  # = 2000.0
        assert pts[0, 0] == pytest.approx(expected_x, rel=1e-5)


# ── Consistency checks ──────────────────────────────────────────────────

class TestConsistency:
    """Cross-checks that backprojection is self-consistent."""

    def test_doubling_depth_doubles_coordinates(self):
        """With K=eye, T=eye: doubling depth doubles all world coordinates."""
        h, w = 8, 8
        mask = _single_pixel_mask(h, w, 3, 5)
        K = np.eye(3, dtype=np.float64)

        pts_d5 = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, 5.0), K, np.eye(4), None
        )
        pts_d10 = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, 10.0), K, np.eye(4), None
        )
        np.testing.assert_allclose(pts_d10[0], pts_d5[0] * 2, atol=1e-5)

    def test_inverse_transform_roundtrip(self):
        """Backproject and then project back should recover (u, v, d)."""
        h, w, depth = 64, 64, 7.5
        fx, fy, cx, cy = 300.0, 300.0, 32.0, 32.0
        u, v = 45, 20
        mask = _single_pixel_mask(h, w, v, u)
        K = _K(fx, fy, cx, cy)

        angle = math.pi / 5
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        t = np.array([2.0, -1.0, 3.0])
        T_wc = _T_wc_from_rotation_translation(R, t)
        T_cw = np.linalg.inv(T_wc)

        pts_world = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, T_cw, None
        )
        # Project back: p_cam = T_wc @ p_world_h
        p_world_h = np.append(pts_world[0].astype(np.float64), 1.0)
        p_cam = (T_wc @ p_world_h)[:3]

        # Recover pixel: [u', v', 1] = (1/d) * K @ p_cam
        recovered_uv1 = K @ p_cam / p_cam[2]
        assert recovered_uv1[0] == pytest.approx(u, abs=0.1)  # float32 rounding
        assert recovered_uv1[1] == pytest.approx(v, abs=0.1)
        assert p_cam[2] == pytest.approx(depth, abs=0.1)

    def test_all_pixels_same_depth_form_plane(self):
        """All masked pixels at same depth should lie on a plane z=depth
        (in camera frame with T_wc=I)."""
        h, w, depth = 32, 32, 15.0
        fx, fy, cx, cy = 200.0, 200.0, 16.0, 16.0
        mask = np.ones((h, w), dtype=bool)
        K = _K(fx, fy, cx, cy)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        # All z values should be exactly 'depth'
        np.testing.assert_allclose(pts[:, 2], depth, atol=1e-3)

    def test_mirrored_pixels_symmetric_x(self):
        """Two pixels equidistant from cx should have symmetric x."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        offset = 10
        mask = np.zeros((h, w), dtype=bool)
        mask[32, int(cx - offset)] = True  # u = 22
        mask[32, int(cx + offset)] = True  # u = 42
        K = _K(fx, fy, cx, cy)

        pts = _pipe()._backproject_mask_points(
            mask, _uniform_depth(h, w, depth), K, np.eye(4), None
        )
        xs = sorted(pts[:, 0].tolist())
        assert xs[0] == pytest.approx(-xs[1], abs=1e-5)
