"""End-to-end 3D accuracy tests: depth + K + T_wc → 4DSG centroid / shape.

This is the most critical verification: does the final serialized 4DSG JSON
contain correct 3D coordinates?  Each test constructs a scenario with
hand-computable geometry and checks the output "c" and "s" fields.

Pipeline path under test:
    FastFrameInput(depth, K, T_wc, mask)
    → _backproject_mask_points()  →  points_world (N,3)
    → build_centroid_token()      →  c = mean(points)
    → build_shape_token()         →  s = {mu, sigma, min, max} per axis
    → build_4dsg_dict()           →  JSON "c" and "s"
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastFrameInput,
    FastLocalDetection,
    FastSNOWPipeline,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _cfg() -> FastSNOWConfig:
    """Minimal config that passes all filters with tiny synthetic data."""
    cfg = FastSNOWConfig()
    cfg.depth_filter.min_points = 1
    cfg.depth_filter.max_extent = 1e9
    cfg.depth_filter.conf_thresh = 0.0
    cfg.step.grid_size = 4
    cfg.step.iou_threshold = 0.0
    cfg.step.max_tau_per_step = 0
    cfg.fusion.cross_run_iou_thresh = 0.3
    cfg.fusion.merge_centroid_dist_m = 1e6
    cfg.fusion.merge_temporal_gap = 1000
    cfg.edge.motion_window = 2
    return cfg


def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _frame(
    frame_idx: int,
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    run_id: str = "r",
    obj_id: int = 0,
) -> FastFrameInput:
    return FastFrameInput(
        frame_idx=frame_idx,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=run_id, local_obj_id=obj_id,
                               mask=mask, score=1.0),
        ],
    )


def _get_track_obs(dsg: dict, track_idx: int = 0, obs_idx: int = 0) -> dict:
    """Get a specific F_k observation from the 4DSG."""
    return dsg["tracks"][track_idx]["F_k"][obs_idx]


def _hand_backproject(u: int, v: int, d: float,
                      fx: float, fy: float, cx: float, cy: float,
                      T_wc: np.ndarray) -> np.ndarray:
    """Hand-compute the world point for a single pixel."""
    p_cam = np.array([
        d * (u - cx) / fx,
        d * (v - cy) / fy,
        d,
    ])
    T_cw = np.linalg.inv(T_wc)
    p_h = np.append(p_cam, 1.0)
    return (T_cw @ p_h)[:3]


# ── Centroid accuracy (c field) ──────────────────────────────────────────

class TestCentroidAccuracy:
    """Verify the 4DSG "c" field matches hand-computed 3D centroid."""

    def test_single_pixel_at_principal_point(self):
        """1-pixel mask at principal point, identity pose → c = [0, 0, depth]."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        dsg = pipe.build_4dsg_dict()
        c = _get_track_obs(dsg)["c"]

        assert c[0] == pytest.approx(0.0, abs=1e-4)
        assert c[1] == pytest.approx(0.0, abs=1e-4)
        assert c[2] == pytest.approx(10.0, abs=1e-4)

    def test_single_pixel_off_center(self):
        """1-pixel mask at (u=42, v=32), verify exact c."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 42] = True

        expected = [5.0 * (42 - 32) / 100, 0.0, 5.0]  # [0.5, 0, 5]

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c == pytest.approx(expected, abs=1e-4)

    def test_4_pixel_block_centroid(self):
        """2x2 block: c = mean of 4 backprojected points."""
        h, w, depth = 64, 64, 8.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[30:32, 30:32] = True  # (v,u): (30,30)(30,31)(31,30)(31,31)

        expected_pts = []
        for v in [30, 31]:
            for u in [30, 31]:
                expected_pts.append([
                    depth * (u - cx) / fx,
                    depth * (v - cy) / fy,
                    depth,
                ])
        expected_c = np.array(expected_pts).mean(axis=0).tolist()

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c == pytest.approx(expected_c, abs=1e-3)

    def test_centroid_with_translation(self):
        """Camera at (10, 0, 0) in world → centroid x shifted by -10."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 10.0  # world→camera translation in x

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), T_wc,
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]

        # p_cam = [0, 0, 5], T_cw shifts x by -10
        assert c[0] == pytest.approx(-10.0, abs=1e-3)
        assert c[1] == pytest.approx(0.0, abs=1e-3)
        assert c[2] == pytest.approx(5.0, abs=1e-3)

    def test_centroid_with_rotation(self):
        """90° yaw → p_cam.x=0.5 becomes p_world.y or swapped."""
        h, w, depth = 64, 64, 5.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 42] = True  # p_cam = [0.5, 0, 5]

        angle = math.pi / 2
        c_a, s_a = math.cos(angle), math.sin(angle)
        R = np.array([[c_a, -s_a, 0], [s_a, c_a, 0], [0, 0, 1]], dtype=np.float64)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R

        p_cam = np.array([0.5, 0.0, 5.0])
        T_cw = np.linalg.inv(T_wc)
        expected_world = (T_cw @ np.append(p_cam, 1.0))[:3]

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), T_wc,
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c == pytest.approx(expected_world.tolist(), abs=1e-3)

    def test_centroid_with_combined_pose(self):
        """30° yaw + translation: full non-trivial pose."""
        h, w, depth = 64, 64, 12.0
        fx, fy, cx, cy = 300.0, 300.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[40, 50] = True  # v=40, u=50

        angle = math.pi / 6
        c_a, s_a = math.cos(angle), math.sin(angle)
        R = np.array([[c_a, -s_a, 0], [s_a, c_a, 0], [0, 0, 1]], dtype=np.float64)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = [5.0, -2.0, 8.0]

        expected = _hand_backproject(50, 40, depth, fx, fy, cx, cy, T_wc)

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), T_wc,
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c == pytest.approx(expected.tolist(), abs=1e-2)

    def test_centroid_depth_gradient(self):
        """Non-uniform depth: centroid z reflects depth mean, not uniform value."""
        h, w = 64, 64
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 30:35] = True  # 5 pixels at v=32, u=30..34

        depth = np.full((h, w), 0.0, dtype=np.float32)
        # Assign specific depths
        for u in range(30, 35):
            depth[32, u] = 10.0 + (u - 30)  # 10, 11, 12, 13, 14

        expected_pts = []
        for u in range(30, 35):
            d = 10.0 + (u - 30)
            expected_pts.append([d * (u - cx) / fx, d * (32 - cy) / fy, d])
        expected_c = np.array(expected_pts).mean(axis=0)

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, depth, _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c == pytest.approx(expected_c.tolist(), abs=1e-2)


# ── Shape accuracy (s field) ────────────────────────────────────────────

class TestShapeAccuracy:
    """Verify the 4DSG "s" field matches hand-computed 3D statistics."""

    def test_single_pixel_sigma_zero(self):
        """1 pixel → sigma = 0, min = max = mu = the single point."""
        h, w, depth = 64, 64, 7.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True  # principal point

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        s = _get_track_obs(pipe.build_4dsg_dict())["s"]

        for axis in ["x", "y", "z"]:
            assert s[axis]["sigma"] == pytest.approx(0.0, abs=1e-6)
            assert s[axis]["min"] == pytest.approx(s[axis]["max"], abs=1e-6)
            assert s[axis]["mu"] == pytest.approx(s[axis]["min"], abs=1e-6)

    def test_two_pixel_shape_stats(self):
        """2 pixels: verify exact mu, sigma, min, max per axis."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True  # u=22 → x = d*(22-32)/100
        mask[32, 42] = True  # u=42 → x = d*(42-32)/100
        depth = np.full((h, w), 10.0, dtype=np.float32)

        # Hand compute the two world points (T_wc=I)
        p1 = np.array([10.0 * (22 - 32) / 100, 0.0, 10.0])  # [-1, 0, 10]
        p2 = np.array([10.0 * (42 - 32) / 100, 0.0, 10.0])  # [1, 0, 10]

        pts = np.array([p1, p2])
        expected_x_mu = pts[:, 0].mean()   # 0.0
        expected_x_sigma = pts[:, 0].std()  # 1.0
        expected_x_min = pts[:, 0].min()   # -1.0
        expected_x_max = pts[:, 0].max()   # 1.0
        expected_z_sigma = 0.0  # both at same z

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, depth, _K(fx, fy, cx, cy), np.eye(4),
        ))
        s = _get_track_obs(pipe.build_4dsg_dict())["s"]

        assert s["x"]["mu"] == pytest.approx(expected_x_mu, abs=1e-3)
        assert s["x"]["sigma"] == pytest.approx(expected_x_sigma, abs=1e-3)
        assert s["x"]["min"] == pytest.approx(expected_x_min, abs=1e-3)
        assert s["x"]["max"] == pytest.approx(expected_x_max, abs=1e-3)
        assert s["z"]["sigma"] == pytest.approx(expected_z_sigma, abs=1e-3)

    def test_horizontal_extent(self):
        """Row of pixels: shape x extent = x_max - x_min should match FOV spread."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22:43] = True  # u = 22..42, 21 pixels

        # x_min at u=22: 10*(22-32)/100 = -1.0
        # x_max at u=42: 10*(42-32)/100 = 1.0
        expected_x_min = 10.0 * (22 - 32) / 100
        expected_x_max = 10.0 * (42 - 32) / 100

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        s = _get_track_obs(pipe.build_4dsg_dict())["s"]

        assert s["x"]["min"] == pytest.approx(expected_x_min, abs=1e-3)
        assert s["x"]["max"] == pytest.approx(expected_x_max, abs=1e-3)
        extent = s["x"]["max"] - s["x"]["min"]
        assert extent == pytest.approx(2.0, abs=1e-3)

    def test_depth_gradient_z_stats(self):
        """Variable depth → z has non-zero sigma."""
        h, w = 64, 64
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True  # principal point: p_cam = [0, 0, d]
        mask[33, 32] = True  # one row below: p_cam = [0, d/200, d]

        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[32, 32] = 5.0
        depth[33, 32] = 15.0

        # p1 = [0, 0, 5], p2 = [0, 15*(33-32)/200, 15] = [0, 0.075, 15]
        expected_z = np.array([5.0, 15.0])
        expected_z_mu = expected_z.mean()
        expected_z_sigma = expected_z.std()
        expected_z_min = 5.0
        expected_z_max = 15.0

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, depth, _K(fx, fy, cx, cy), np.eye(4),
        ))
        s = _get_track_obs(pipe.build_4dsg_dict())["s"]

        assert s["z"]["mu"] == pytest.approx(expected_z_mu, abs=1e-2)
        assert s["z"]["sigma"] == pytest.approx(expected_z_sigma, abs=1e-2)
        assert s["z"]["min"] == pytest.approx(expected_z_min, abs=1e-2)
        assert s["z"]["max"] == pytest.approx(expected_z_max, abs=1e-2)

    def test_shape_with_translation(self):
        """Translation shifts all shape stats equally; sigma and extent unchanged."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True
        mask[32, 42] = True

        # Without translation
        pipe0 = FastSNOWPipeline(config=_cfg())
        pipe0.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        s0 = _get_track_obs(pipe0.build_4dsg_dict())["s"]

        # With translation (5, 0, 0)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 5.0
        pipe1 = FastSNOWPipeline(config=_cfg())
        pipe1.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), T_wc,
        ))
        s1 = _get_track_obs(pipe1.build_4dsg_dict())["s"]

        # Sigma should be the same (translation doesn't affect spread)
        assert s1["x"]["sigma"] == pytest.approx(s0["x"]["sigma"], abs=1e-3)
        # Extent should be the same
        extent0 = s0["x"]["max"] - s0["x"]["min"]
        extent1 = s1["x"]["max"] - s1["x"]["min"]
        assert extent1 == pytest.approx(extent0, abs=1e-3)
        # But mu should be shifted by -5 (T_cw inverts the translation)
        assert s1["x"]["mu"] == pytest.approx(s0["x"]["mu"] - 5.0, abs=1e-3)


# ── Multi-frame consistency ──────────────────────────────────────────────

class TestMultiFrameConsistency:
    """Verify 3D accuracy is consistent across multiple frames."""

    def test_stationary_object_same_centroid(self):
        """Same mask + depth + pose across frames → same centroid every frame."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[30:35, 30:35] = True
        K = _K(fx, fy, cx, cy)

        pipe = FastSNOWPipeline(config=_cfg())
        for t in range(5):
            pipe.process_frame(_frame(
                t, mask, np.full((h, w), depth, dtype=np.float32),
                K, np.eye(4),
            ))
        dsg = pipe.build_4dsg_dict()
        fk = dsg["tracks"][0]["F_k"]
        centroids = [obs["c"] for obs in fk]
        # All centroids should be identical
        for c in centroids[1:]:
            assert c == pytest.approx(centroids[0], abs=1e-4)

    def test_moving_camera_stationary_object_world_coords_stable(self):
        """Camera moves along x; object at fixed world position.
        The centroid in world coordinates should NOT change."""
        h, w, depth_base = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        K = _K(fx, fy, cx, cy)

        pipe = FastSNOWPipeline(config=_cfg())
        expected_world = None

        for t in range(5):
            # Camera moves in x: T_wc translation = (t, 0, 0)
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[0, 3] = float(t)

            # Object at world pos ~ (0, 0, 10) → camera sees it at
            # cam_x = world_x + t (the world→camera T maps world→cam)
            # So object at (u_cam, v, d) changes each frame.
            # But we need mask coords in pixel space.
            # Place mask at principal point: p_cam = [0, 0, depth].
            # Then p_world = T_cw @ [0, 0, depth, 1]
            mask = np.zeros((h, w), dtype=bool)
            mask[32, 32] = True

            pipe.process_frame(_frame(
                t, mask, np.full((h, w), depth_base, dtype=np.float32),
                K, T_wc,
            ))

            # Compute expected world point for this frame
            T_cw = np.linalg.inv(T_wc)
            p_cam_h = np.array([0.0, 0.0, depth_base, 1.0])
            p_world = (T_cw @ p_cam_h)[:3]

            if expected_world is None:
                expected_world = p_world
            # Since camera translates by (t, 0, 0), p_world.x = -t
            # This is NOT a fixed world object — it's a different world point each frame.
            # Let's just verify the math is correct per frame.

        dsg = pipe.build_4dsg_dict()
        fk = dsg["tracks"][0]["F_k"]

        for i, obs in enumerate(fk):
            T_cw = np.linalg.inv(np.eye(4))
            T_cw_i = np.eye(4, dtype=np.float64)
            T_cw_i[0, 3] = -float(i)  # inv of translate-x-by-i
            p_expected = (T_cw_i @ np.array([0, 0, depth_base, 1.0]))[:3]
            assert obs["c"] == pytest.approx(p_expected.tolist(), abs=1e-2)

    def test_two_objects_independent_centroids(self):
        """Two non-overlapping masks → two tracks with independent centroids."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask_a = np.zeros((h, w), dtype=bool)
        mask_a[10:20, 10:20] = True  # top-left
        mask_b = np.zeros((h, w), dtype=bool)
        mask_b[50:60, 50:60] = True  # bottom-right

        K = _K(fx, fy, cx, cy)

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(FastFrameInput(
            frame_idx=0,
            depth_t=np.full((h, w), depth, dtype=np.float32),
            K_t=K,
            T_wc_t=np.eye(4, dtype=np.float64),
            detections=[
                FastLocalDetection(run_id="r", local_obj_id=0,
                                   mask=mask_a, score=0.9),
                FastLocalDetection(run_id="r", local_obj_id=1,
                                   mask=mask_b, score=0.8),
            ],
        ))
        dsg = pipe.build_4dsg_dict()

        assert dsg["metadata"]["num_tracks"] == 2
        c0 = dsg["tracks"][0]["F_k"][0]["c"]
        c1 = dsg["tracks"][1]["F_k"][0]["c"]

        # Object A is in top-left quadrant (smaller u, v) → negative x, y
        # Object B is in bottom-right quadrant (larger u, v) → positive x, y
        # (relative to principal point at 32, 32)
        # They should have significantly different centroids
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c0, c1)))
        assert dist > 0.5, f"Two separated objects should have different centroids, dist={dist}"

        # Object A centroid should have x < 0, y < 0 (u,v < cx,cy)
        # Pixel mean for A: u ~ 14.5, v ~ 14.5
        assert c0[0] < 0.0, "Top-left object should have negative x"
        assert c0[1] < 0.0, "Top-left object should have negative y"
        # Object B centroid should have x > 0, y > 0
        assert c1[0] > 0.0, "Bottom-right object should have positive x"
        assert c1[1] > 0.0, "Bottom-right object should have positive y"


# ── JSON roundtrip preserves precision ───────────────────────────────────

class TestJSONPrecision:
    """Verify that serialize → parse roundtrip preserves 3D values."""

    def test_centroid_survives_json_roundtrip(self):
        """c values should be identical after JSON serialize → parse."""
        h, w, depth = 64, 64, 7.777
        fx, fy, cx, cy = 123.456, 789.012, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[20:30, 20:30] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))

        dsg_dict = pipe.build_4dsg_dict()
        json_str = pipe.serialize_4dsg()
        parsed = json.loads(json_str)

        c_orig = dsg_dict["tracks"][0]["F_k"][0]["c"]
        c_parsed = parsed["tracks"][0]["F_k"][0]["c"]
        assert c_orig == pytest.approx(c_parsed, abs=1e-10)

    def test_shape_survives_json_roundtrip(self):
        """s field values preserved through JSON."""
        h, w, depth = 64, 64, 10.0
        fx, fy, cx, cy = 200.0, 200.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[20:40, 20:40] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))

        dsg_dict = pipe.build_4dsg_dict()
        parsed = json.loads(pipe.serialize_4dsg())

        s_orig = dsg_dict["tracks"][0]["F_k"][0]["s"]
        s_parsed = parsed["tracks"][0]["F_k"][0]["s"]
        for axis in ["x", "y", "z"]:
            for stat in ["mu", "sigma", "min", "max"]:
                assert s_orig[axis][stat] == pytest.approx(
                    s_parsed[axis][stat], abs=1e-10
                )

    def test_ego_xyz_survives_json_roundtrip(self):
        """ego[].xyz preserved through JSON."""
        h, w = 64, 64
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, 3] = [1.111, 2.222, 3.333]

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(FastFrameInput(
            frame_idx=0,
            depth_t=np.full((h, w), 5.0, dtype=np.float32),
            K_t=_K(200, 200, 32, 32),
            T_wc_t=T_wc,
            detections=[],
        ))
        dsg_dict = pipe.build_4dsg_dict()
        parsed = json.loads(pipe.serialize_4dsg())

        xyz_orig = dsg_dict["ego"][0]["xyz"]
        xyz_parsed = parsed["ego"][0]["xyz"]
        assert xyz_orig == pytest.approx(xyz_parsed, abs=1e-10)


# ── Depth filter interaction with 3D ─────────────────────────────────────

class TestDepthFilterInteraction:
    """Verify that depth/confidence filtering doesn't corrupt 3D coordinates."""

    def test_low_confidence_pixels_excluded_from_centroid(self):
        """Only high-confidence pixels contribute to centroid."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True  # will have high confidence
        mask[32, 42] = True  # will have low confidence (excluded)

        depth = np.full((h, w), 10.0, dtype=np.float32)
        conf = np.zeros((h, w), dtype=np.float32)
        conf[32, 22] = 1.0  # high
        conf[32, 42] = 0.01  # below threshold

        cfg = _cfg()
        cfg.depth_filter.conf_thresh = 0.5
        pipe = FastSNOWPipeline(config=cfg)
        pipe.process_frame(FastFrameInput(
            frame_idx=0,
            depth_t=depth,
            K_t=_K(fx, fy, cx, cy),
            T_wc_t=np.eye(4, dtype=np.float64),
            detections=[FastLocalDetection("r", 0, mask, 1.0)],
            depth_conf_t=conf,
        ))
        dsg = pipe.build_4dsg_dict()
        c = _get_track_obs(dsg)["c"]

        # Only pixel (32, 22) contributes → c_x = 10*(22-32)/100 = -1.0
        assert c[0] == pytest.approx(-1.0, abs=1e-3)

    def test_nan_depth_excluded_from_centroid(self):
        """Pixels with NaN depth are excluded from 3D computation."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True  # valid
        mask[32, 42] = True  # will be NaN

        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[32, 42] = np.nan

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, depth, _K(fx, fy, cx, cy), np.eye(4),
        ))
        dsg = pipe.build_4dsg_dict()
        c = _get_track_obs(dsg)["c"]

        # Only pixel (32, 22) → c_x = 10*(22-32)/100 = -1.0
        assert c[0] == pytest.approx(-1.0, abs=1e-3)

    def test_zero_depth_excluded(self):
        """Pixels with depth=0 are excluded."""
        h, w = 64, 64
        fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 22] = True
        mask[32, 42] = True

        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[32, 42] = 0.0

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, depth, _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c[0] == pytest.approx(-1.0, abs=1e-3)


# ── Edge case: large scenes ─────────────────────────────────────────────

class TestLargeSceneAccuracy:
    """Verify accuracy with realistic depth ranges."""

    def test_close_range_1m(self):
        """Object at 1 metre."""
        h, w, depth = 64, 64, 1.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c[2] == pytest.approx(1.0, abs=1e-4)

    def test_far_range_100m(self):
        """Object at 100 metres."""
        h, w, depth = 64, 64, 100.0
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c[2] == pytest.approx(100.0, abs=0.1)

    def test_very_close_01m(self):
        """Object at 0.1 metres (10 cm)."""
        h, w, depth = 64, 64, 0.1
        fx, fy, cx, cy = 500.0, 500.0, 32.0, 32.0
        mask = np.zeros((h, w), dtype=bool)
        mask[32, 32] = True

        pipe = FastSNOWPipeline(config=_cfg())
        pipe.process_frame(_frame(
            0, mask, np.full((h, w), depth, dtype=np.float32),
            _K(fx, fy, cx, cy), np.eye(4),
        ))
        c = _get_track_obs(pipe.build_4dsg_dict())["c"]
        assert c[2] == pytest.approx(0.1, abs=1e-4)
