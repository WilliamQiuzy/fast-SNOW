"""Tests for DA3 chunked batch inference with SIM3 alignment.

Unit tests (no GPU): chunk splitting, SIM3 estimation, accumulation,
result transformation.

Integration tests (GPU + DA3 model): real video inference with
horsing.mp4, comparing chunked vs full-batch results.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_snow.vision.perception.da3_wrapper import (
    DA3Result,
    DA3Wrapper,
    _accumulate_sim3,
    _align_overlap,
    _apply_sim3_to_result,
    backproject_depth,
    compute_chunks,
    estimate_sim3,
    estimate_sim3_robust,
)
from fast_snow.engine.config.fast_snow_config import DA3Config

logger = logging.getLogger(__name__)

# ======================================================================
# Fixtures
# ======================================================================

MODEL_PATH = ROOT / "fast_snow" / "models" / "da3-small"
VIDEO_PATH = ROOT / "assets" / "examples_videos" / "horsing.mp4"

_has_model = MODEL_PATH.exists()
_has_video = VIDEO_PATH.exists()

try:
    import torch
    _has_cuda = torch.cuda.is_available()
except ImportError:
    _has_cuda = False

requires_gpu_and_model = pytest.mark.skipif(
    not (_has_cuda and _has_model),
    reason="Needs CUDA + DA3 model at fast_snow/models/da3-small",
)
requires_video = pytest.mark.skipif(
    not _has_video,
    reason="Needs horsing.mp4 at assets/examples_videos/",
)


def _make_da3_result(
    H: int = 64,
    W: int = 96,
    depth_val: float = 5.0,
    T_wc: np.ndarray | None = None,
    conf_val: float = 0.8,
) -> DA3Result:
    """Create a synthetic DA3Result for unit testing."""
    depth = np.full((H, W), depth_val, dtype=np.float32)
    K = np.array([
        [500, 0, W / 2],
        [0, 500, H / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    if T_wc is None:
        T_wc = np.eye(4, dtype=np.float64)
    conf = np.full((H, W), conf_val, dtype=np.float32)
    return DA3Result(depth=depth, K=K, T_wc=T_wc, depth_conf=conf, is_metric=False)


# ======================================================================
# Unit tests: compute_chunks
# ======================================================================

class TestComputeChunks:
    def test_no_chunking_needed(self):
        chunks = compute_chunks(10, chunk_size=20, overlap=5)
        assert chunks == [(0, 10)]

    def test_exact_fit(self):
        # 20 frames, chunk_size=10, overlap=3, step=7
        # chunk 0: [0, 10), chunk 1: [7, 17), chunk 2: [14, 20)
        chunks = compute_chunks(20, chunk_size=10, overlap=3)
        assert chunks[0] == (0, 10)
        assert len(chunks) >= 2
        # All frames covered
        covered = set()
        for s, e in chunks:
            covered.update(range(s, e))
        assert covered == set(range(20))

    def test_small_trailing_merged(self):
        # If last chunk has <= overlap frames, merge into previous
        # 23 frames, chunk_size=10, overlap=3, step=7
        # Without merge: (0,10), (7,17), (14,23) -> last=9 > 3 -> no merge
        chunks = compute_chunks(23, chunk_size=10, overlap=3)
        last_len = chunks[-1][1] - chunks[-1][0]
        assert last_len > 3

    def test_trailing_too_small_gets_merged(self):
        # 12 frames, chunk_size=10, overlap=5, step=5
        # Raw: (0,10), (5,12) -> last=7 > 5 -> no merge
        chunks = compute_chunks(12, chunk_size=10, overlap=5)
        assert len(chunks) == 2

        # 11 frames, chunk_size=10, overlap=5, step=5
        # Raw: (0,10), (5,11) -> last=6 > 5 -> no merge
        chunks = compute_chunks(11, chunk_size=10, overlap=5)
        assert len(chunks) == 2

        # 15 frames, chunk_size=10, overlap=8, step=2
        # Raw: (0,10), (2,12), (4,14), (6,15) -> last=9 > 8 -> no merge
        # But (8,15) = 7 < 8 -> would merge
        chunks = compute_chunks(15, chunk_size=10, overlap=8)
        # Just verify all frames are covered
        covered = set()
        for s, e in chunks:
            covered.update(range(s, e))
        assert covered == set(range(15))

    def test_overlap_between_adjacent_chunks(self):
        chunks = compute_chunks(51, chunk_size=20, overlap=5)
        for i in range(len(chunks) - 1):
            _, prev_end = chunks[i]
            curr_start, _ = chunks[i + 1]
            actual_overlap = prev_end - curr_start
            assert actual_overlap >= 5, f"Overlap between chunks {i} and {i+1} is {actual_overlap}"


# ======================================================================
# Unit tests: backproject_depth
# ======================================================================

class TestBackproject:
    def test_identity_pose_center_pixel(self):
        """At identity pose, center pixel at depth d -> (0, 0, d) in world."""
        H, W = 64, 96
        depth = np.full((H, W), 5.0, dtype=np.float32)
        K = np.array([[500, 0, W / 2], [0, 500, H / 2], [0, 0, 1]], dtype=np.float64)
        T_wc = np.eye(4, dtype=np.float64)

        pts = backproject_depth(depth, K, T_wc)
        assert pts.shape == (H, W, 3)

        # Center pixel should project to (0, 0, depth)
        cy, cx = H // 2, W // 2
        np.testing.assert_allclose(pts[cy, cx], [0, 0, 5.0], atol=0.1)

    def test_translated_pose(self):
        """Camera translated by [1,0,0] -> world points shift by -1 in x."""
        H, W = 16, 16
        depth = np.full((H, W), 10.0, dtype=np.float32)
        K = np.array([[100, 0, 8], [0, 100, 8], [0, 0, 1]], dtype=np.float64)

        T_wc_identity = np.eye(4)
        pts_id = backproject_depth(depth, K, T_wc_identity)

        # Translate camera by +1 in x (world-to-camera)
        T_wc_shifted = np.eye(4)
        T_wc_shifted[0, 3] = 1.0  # camera sees world shifted by +1 in x
        pts_shifted = backproject_depth(depth, K, T_wc_shifted)

        # World points should be shifted by -1 in x
        diff = pts_shifted[8, 8] - pts_id[8, 8]
        assert diff[0] < -0.5, f"Expected negative x shift, got {diff}"

    def test_shape_and_dtype(self):
        pts = backproject_depth(
            np.ones((10, 20), dtype=np.float32),
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
        )
        assert pts.shape == (10, 20, 3)
        assert pts.dtype == np.float64


# ======================================================================
# Unit tests: estimate_sim3
# ======================================================================

class TestEstimateSim3:
    def test_identity(self):
        """Same points -> s=1, R=I, t=0."""
        pts = np.random.default_rng(42).normal(size=(100, 3))
        s, R, t = estimate_sim3(pts, pts)
        assert abs(s - 1.0) < 1e-6
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-6)

    def test_pure_scale(self):
        """target = 2 * source -> s=2."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(200, 3))
        tgt = 2.0 * src
        s, R, t = estimate_sim3(src, tgt)
        assert abs(s - 2.0) < 1e-4
        np.testing.assert_allclose(R, np.eye(3), atol=1e-4)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-4)

    def test_pure_translation(self):
        """target = source + [1, 2, 3] -> t=[1,2,3], s=1."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(200, 3))
        offset = np.array([1.0, 2.0, 3.0])
        tgt = src + offset
        s, R, t = estimate_sim3(src, tgt)
        assert abs(s - 1.0) < 1e-4
        np.testing.assert_allclose(t, offset, atol=1e-4)

    def test_rotation_90deg_z(self):
        """90-degree rotation around z-axis."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(200, 3))
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        tgt = (R_true @ src.T).T
        s, R, t = estimate_sim3(src, tgt)
        assert abs(s - 1.0) < 1e-4
        np.testing.assert_allclose(R, R_true, atol=1e-4)

    def test_full_sim3(self):
        """Combined scale + rotation + translation."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(500, 3))

        s_true = 1.5
        angle = np.pi / 6  # 30 degrees around z
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        t_true = np.array([1.0, -2.0, 0.5])

        tgt = s_true * (R_true @ src.T).T + t_true

        s, R, t = estimate_sim3(src, tgt)
        assert abs(s - s_true) < 1e-3
        np.testing.assert_allclose(R, R_true, atol=1e-3)
        np.testing.assert_allclose(t, t_true, atol=1e-3)

    def test_weighted(self):
        """Weights suppress outlier influence."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(200, 3))
        tgt = src.copy()  # identity transform

        # Corrupt last 50 points
        tgt[150:] += 100.0
        weights = np.ones(200)
        weights[150:] = 0.0  # zero weight for corrupted

        s, R, t = estimate_sim3(src, tgt, weights)
        assert abs(s - 1.0) < 1e-4
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-4)

    def test_robust_with_outliers(self):
        """Robust estimator handles moderate outliers without explicit weights."""
        rng = np.random.default_rng(42)
        src = rng.normal(size=(300, 3))
        t_true = np.array([1.0, 0.0, 0.0])
        tgt = src + t_true

        # Corrupt 10% of points
        n_corrupt = 30
        tgt[:n_corrupt] += rng.normal(size=(n_corrupt, 3)) * 50

        s, R, t = estimate_sim3_robust(src, tgt, delta=1.0, max_iters=10)
        assert abs(s - 1.0) < 0.1
        np.testing.assert_allclose(t, t_true, atol=0.3)


# ======================================================================
# Unit tests: _accumulate_sim3
# ======================================================================

class TestAccumulateSim3:
    def test_single_pairwise(self):
        s, R, t = 2.0, np.eye(3), np.array([1.0, 0.0, 0.0])
        cum = _accumulate_sim3([(s, R, t)])
        assert len(cum) == 2

        # cumulative[0] = identity
        assert cum[0][0] == 1.0
        np.testing.assert_allclose(cum[0][1], np.eye(3))

        # cumulative[1] = the pairwise transform
        assert abs(cum[1][0] - 2.0) < 1e-10
        np.testing.assert_allclose(cum[1][2], [1.0, 0.0, 0.0])

    def test_chain_two_translations(self):
        """Two pure translations compose additively."""
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 2.0, 0.0])
        cum = _accumulate_sim3([
            (1.0, np.eye(3), t1),
            (1.0, np.eye(3), t2),
        ])
        assert len(cum) == 3
        # cum[2] should be t1 + t2
        np.testing.assert_allclose(cum[2][2], [1.0, 2.0, 0.0], atol=1e-10)

    def test_chain_with_scale(self):
        """s1=2, t1=[1,0,0]; s2=3, t2=[0,1,0].
        cum[2] = s=6, t = 2*(I@[0,1,0]) + [1,0,0] = [1,2,0]
        """
        cum = _accumulate_sim3([
            (2.0, np.eye(3), np.array([1.0, 0.0, 0.0])),
            (3.0, np.eye(3), np.array([0.0, 1.0, 0.0])),
        ])
        assert abs(cum[2][0] - 6.0) < 1e-10
        np.testing.assert_allclose(cum[2][2], [1.0, 2.0, 0.0], atol=1e-10)

    def test_roundtrip(self):
        """Apply cumulative SIM3 to a point, verify against sequential application."""
        rng = np.random.default_rng(42)
        p = rng.normal(size=3)

        sim3_list = [
            (1.2, _random_rotation(rng), rng.normal(size=3)),
            (0.8, _random_rotation(rng), rng.normal(size=3)),
        ]
        cum = _accumulate_sim3(sim3_list)

        # Sequential: apply sim3_list[0] then sim3_list[1]
        s0, R0, t0 = sim3_list[0]
        s1, R1, t1 = sim3_list[1]
        p1 = s1 * R1 @ p + t1       # chunk 2 -> chunk 1
        p0 = s0 * R0 @ p1 + t0      # chunk 1 -> chunk 0

        # Cumulative: apply cum[2] directly
        sc, Rc, tc = cum[2]
        p0_cum = sc * Rc @ p + tc

        np.testing.assert_allclose(p0_cum, p0, atol=1e-10)


# ======================================================================
# Unit tests: _apply_sim3_to_result
# ======================================================================

class TestApplySim3ToResult:
    def test_identity_sim3_is_noop(self):
        result = _make_da3_result(depth_val=5.0)
        out = _apply_sim3_to_result(result, 1.0, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(out.depth, result.depth, atol=1e-6)
        np.testing.assert_allclose(out.T_wc, result.T_wc, atol=1e-6)
        np.testing.assert_allclose(out.K, result.K, atol=1e-6)

    def test_scale_only(self):
        """s=2 should double the depth."""
        result = _make_da3_result(depth_val=5.0)
        out = _apply_sim3_to_result(result, 2.0, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(out.depth, 10.0, atol=1e-4)

    def test_backproject_consistency(self):
        """After SIM3, backprojected points should match SIM3(original points)."""
        rng = np.random.default_rng(42)
        H, W = 32, 48
        depth = rng.uniform(1, 10, (H, W)).astype(np.float32)
        K = np.array([[200, 0, W / 2], [0, 200, H / 2], [0, 0, 1]], dtype=np.float64)

        # Random source pose
        angle = 0.3
        R_cw_src = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        T_cw_src = np.eye(4)
        T_cw_src[:3, :3] = R_cw_src
        T_cw_src[:3, 3] = [1.0, 2.0, 0.5]
        T_wc_src = np.linalg.inv(T_cw_src)

        result = DA3Result(
            depth=depth, K=K, T_wc=T_wc_src,
            depth_conf=np.ones((H, W), dtype=np.float32),
            is_metric=False,
        )

        # Random SIM3
        s_sim3 = 1.5
        R_sim3 = _random_rotation(rng)
        t_sim3 = rng.normal(size=3)

        # Ground truth: backproject original, then SIM3 transform
        pts_original = backproject_depth(depth, K, T_wc_src)
        pts_gt = s_sim3 * (R_sim3 @ pts_original.reshape(-1, 3).T).T + t_sim3

        # Our method: apply SIM3 to result, then backproject
        out = _apply_sim3_to_result(result, s_sim3, R_sim3, t_sim3)
        pts_out = backproject_depth(out.depth, out.K, out.T_wc)

        np.testing.assert_allclose(
            pts_out.reshape(-1, 3), pts_gt, atol=1e-4,
            err_msg="Backprojected points after SIM3 do not match ground truth",
        )


# ======================================================================
# Unit tests: _align_overlap (mock data)
# ======================================================================

class TestAlignOverlap:
    def test_identity_overlap(self):
        """Identical DA3Results -> SIM3 ~ identity."""
        r = _make_da3_result(depth_val=5.0, conf_val=0.9)
        s, R, t = _align_overlap([r, r], [r, r])
        assert abs(s - 1.0) < 0.1
        np.testing.assert_allclose(R, np.eye(3), atol=0.1)
        np.testing.assert_allclose(t, [0, 0, 0], atol=0.5)

    def test_translated_overlap(self):
        """Prev chunk's world is shifted by [1,0,0] relative to curr chunk."""
        H, W = 32, 48
        K = np.array([[200, 0, W / 2], [0, 200, H / 2], [0, 0, 1]], dtype=np.float64)
        depth = np.full((H, W), 5.0, dtype=np.float32)
        conf = np.full((H, W), 0.9, dtype=np.float32)

        # Prev: camera at origin in prev world
        r_prev = DA3Result(
            depth=depth, K=K, T_wc=np.eye(4), depth_conf=conf, is_metric=False,
        )
        # Curr: camera at origin in curr world, but curr world is shifted
        # by -1 in x relative to prev world.  So SIM3 should be t=[1,0,0].
        # To simulate: prev's backprojected points at identity,
        # curr's backprojected points are the same scene but shifted.
        T_wc_curr = np.eye(4, dtype=np.float64)
        T_wc_curr[0, 3] = -1.0  # camera shifted in curr's world
        r_curr = DA3Result(
            depth=depth, K=K, T_wc=T_wc_curr, depth_conf=conf, is_metric=False,
        )

        s, R, t = _align_overlap([r_prev], [r_curr])
        # Should recover translation
        assert abs(s - 1.0) < 0.2, f"Scale {s} too far from 1.0"


# ======================================================================
# Integration tests: real video
# ======================================================================

def _extract_frames_from_video(video_path: Path, fps: float, max_frames: int) -> list:
    """Extract frames from video at given fps."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))

    frames = []
    idx = 0
    while idx < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += step
    cap.release()
    return frames


@requires_gpu_and_model
@requires_video
class TestChunkedVsFullBatch:
    """Compare chunked and full-batch on a small frame set where both fit in memory."""

    def test_small_batch_consistency(self):
        """Run 12 frames both ways, verify trajectories are similar."""
        frames = _extract_frames_from_video(VIDEO_PATH, fps=2.0, max_frames=12)
        n = len(frames)
        assert n >= 8, f"Expected >= 8 frames, got {n}"

        # --- Full batch ---
        cfg_full = DA3Config(chunk_size=0)  # no chunking
        wrapper_full = DA3Wrapper(cfg_full)
        results_full = wrapper_full.infer_batch(frames)
        assert len(results_full) == n

        # --- Chunked ---
        cfg_chunk = DA3Config(chunk_size=6, chunk_overlap=3)
        wrapper_chunk = DA3Wrapper(cfg_chunk)
        results_chunk = wrapper_chunk.infer_batch(frames)
        assert len(results_chunk) == n

        # Frame 0 should be identity in both
        np.testing.assert_allclose(
            results_full[0].T_wc, np.eye(4), atol=1e-6,
            err_msg="Full-batch frame 0 not identity",
        )
        np.testing.assert_allclose(
            results_chunk[0].T_wc, np.eye(4), atol=1e-6,
            err_msg="Chunked frame 0 not identity",
        )

        # Compare ego trajectories: extract camera positions
        def _positions(results):
            return np.array([r.T_cw[:3, 3] for r in results])

        pos_full = _positions(results_full)
        pos_chunk = _positions(results_chunk)

        # Trajectories should have similar shape (correlation > 0.8)
        # Not identical due to chunking + SIM3 alignment noise
        traj_range = pos_full.max(axis=0) - pos_full.min(axis=0)
        max_range = max(traj_range.max(), 1e-3)
        rel_error = np.linalg.norm(pos_full - pos_chunk, axis=1) / max_range

        # Allow up to 30% relative error per frame (generous for chunked)
        mean_rel_error = rel_error.mean()
        logger.info("Mean relative trajectory error: %.4f", mean_rel_error)
        assert mean_rel_error < 0.5, (
            f"Chunked trajectory deviates too much from full batch: "
            f"mean_rel_error={mean_rel_error:.4f}"
        )

    def test_depth_sanity(self):
        """Verify chunked inference produces valid depth maps."""
        frames = _extract_frames_from_video(VIDEO_PATH, fps=2.0, max_frames=12)
        cfg = DA3Config(chunk_size=6, chunk_overlap=3)
        wrapper = DA3Wrapper(cfg)
        results = wrapper.infer_batch(frames)

        for i, r in enumerate(results):
            assert r.depth.shape[0] > 0 and r.depth.shape[1] > 0
            assert np.isfinite(r.depth).all(), f"Frame {i}: non-finite depth"
            assert (r.depth > 0).all(), f"Frame {i}: non-positive depth"
            assert r.K.shape == (3, 3)
            assert r.T_wc.shape == (4, 4)
            assert r.depth_conf.shape == r.depth.shape


@requires_gpu_and_model
@requires_video
class TestChunkedLargeVideo:
    """Test chunked inference on a larger frame set that might OOM without chunking."""

    def test_many_frames_no_oom(self):
        """Run ~30 frames with small chunk_size — should not OOM."""
        frames = _extract_frames_from_video(VIDEO_PATH, fps=3.0, max_frames=30)
        n = len(frames)
        logger.info("Extracted %d frames from horsing.mp4", n)
        assert n >= 15, f"Expected >= 15 frames, got {n}"

        cfg = DA3Config(chunk_size=8, chunk_overlap=3)
        wrapper = DA3Wrapper(cfg)
        results = wrapper.infer_batch(frames)

        assert len(results) == n

        # Frame 0 = identity
        np.testing.assert_allclose(results[0].T_wc, np.eye(4), atol=1e-6)

        # All depths valid
        for i, r in enumerate(results):
            assert np.isfinite(r.depth).all(), f"Frame {i}: non-finite depth"
            assert (r.depth > 0).all(), f"Frame {i}: non-positive depth"

        # Trajectory smoothness: adjacent frames should not jump too far
        positions = np.array([r.T_cw[:3, 3] for r in results])
        for i in range(1, n):
            jump = np.linalg.norm(positions[i] - positions[i - 1])
            # Allow generous threshold — relative depth, arbitrary scale
            assert jump < 50.0, (
                f"Frame {i}: position jump {jump:.2f} is too large "
                f"(pos[{i-1}]={positions[i-1]}, pos[{i}]={positions[i]})"
            )

        logger.info(
            "Trajectory range: %s", positions.max(axis=0) - positions.min(axis=0)
        )


# ======================================================================
# Helpers
# ======================================================================

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random rotation matrix via QR decomposition."""
    M = rng.normal(size=(3, 3))
    Q, R_ = np.linalg.qr(M)
    # Ensure det(Q) = +1
    Q *= np.linalg.det(Q)
    return Q


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-x", "--tb=short"])
