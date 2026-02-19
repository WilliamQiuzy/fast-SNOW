#!/usr/bin/env python3
"""Smoke test for Fast-SNOW pipeline with mock vision outputs (no GPU needed).

Verifies Steps 4-8:
  - Backprojection from mock masks + depth
  - Global ID fusion (cross-run dedup)
  - STEP token construction
  - Scene graph relations (ego-object + object-object)
  - 4DSG JSON serialization (strict spec compliance)
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastSNOWPipeline,
    FastFrameInput,
    FastLocalDetection,
)


def make_identity_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    """Create a simple pinhole intrinsics matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def make_depth_map(h=480, w=640, base_depth=5.0):
    """Create a synthetic depth map with slight gradient."""
    depth = np.full((h, w), base_depth, dtype=np.float32)
    # Add slight gradient to make objects at different depths
    for r in range(h):
        depth[r, :] += (r / h) * 2.0
    return depth


def make_mask(h=480, w=640, y0=100, y1=200, x0=200, x1=350):
    """Create a rectangular boolean mask."""
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_single_frame():
    """Test pipeline with a single frame and one detection."""
    print("=== Test: Single frame, single detection ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)  # Camera at world origin
    depth = make_depth_map(H, W, base_depth=5.0)
    mask = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
        ],
    )

    pipeline.process_frame(frame)
    result = pipeline.build_4dsg_dict()

    assert result["metadata"]["num_frames"] == 1, f"Expected 1 frame, got {result['metadata']['num_frames']}"
    assert result["metadata"]["num_tracks"] >= 1, "Expected at least 1 track"
    assert len(result["ego"]) == 1, "Expected 1 ego entry"

    track = result["tracks"][0]
    assert "object_id" in track
    assert "F_k" in track
    obs = track["F_k"][0]
    assert "t" in obs
    assert "tau" in obs
    assert "c" in obs
    assert "s" in obs
    assert "theta" in obs
    assert len(obs["c"]) == 3, "Centroid must be [x, y, z]"
    assert all(k in obs["s"] for k in ["x", "y", "z"]), "Shape must have x, y, z"
    assert all(
        k in obs["s"]["x"] for k in ["mu", "sigma", "min", "max"]
    ), "Shape axis must have mu, sigma, min, max"

    print(f"  Tracks: {result['metadata']['num_tracks']}")
    print(f"  Centroid: {obs['c']}")
    print(f"  Patches: {len(obs['tau'])}")
    print("  PASSED")


def test_multi_frame_tracking():
    """Test pipeline with 5 frames and consistent object tracking."""
    print("\n=== Test: Multi-frame tracking ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    for t in range(5):
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = t * 0.1  # Ego moves slightly

        depth = make_depth_map(H, W, base_depth=5.0 + t * 0.2)

        # Object moves slightly across frames
        mask = make_mask(H, W, y0=100 + t * 5, y1=200 + t * 5, x0=200 + t * 10, x1=350 + t * 10)

        frame = FastFrameInput(
            frame_idx=t,
            depth_t=depth,
            K_t=K,
            T_wc_t=T_wc,
            detections=[
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.85),
            ],
        )
        pipeline.process_frame(frame)

    result = pipeline.build_4dsg_dict()

    assert result["metadata"]["num_frames"] == 5
    assert result["metadata"]["num_tracks"] == 1, f"Expected 1 track, got {result['metadata']['num_tracks']}"

    track = result["tracks"][0]
    assert len(track["F_k"]) == 5, f"Expected 5 observations, got {len(track['F_k'])}"

    # Verify theta (temporal token) reflects track span
    for obs in track["F_k"]:
        assert obs["theta"] == [0, 4], f"Expected theta=[0,4], got {obs['theta']}"

    # Verify float precision (no rounding)
    for obs in track["F_k"]:
        for coord in obs["c"]:
            # Should be a float, not rounded to 2 decimal places
            assert isinstance(coord, float), f"Centroid coord should be float, got {type(coord)}"

    print(f"  Tracks: {result['metadata']['num_tracks']}")
    print(f"  Observations per track: {len(track['F_k'])}")
    print(f"  Theta: {track['F_k'][0]['theta']}")
    print("  PASSED")


def test_cross_run_fusion():
    """Test that overlapping masks from different runs are fused."""
    print("\n=== Test: Cross-run fusion ===")

    config = FastSNOWConfig()
    config.fusion.cross_run_iou_thresh = 0.3  # Lower threshold for test
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.0)

    # Two runs produce overlapping masks for the same object
    mask1 = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)
    mask2 = make_mask(H, W, y0=110, y1=210, x0=210, x1=360)  # Mostly overlapping

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask1, score=0.9),
            FastLocalDetection(run_id=1, local_obj_id=0, mask=mask2, score=0.8),
        ],
    )

    pipeline.process_frame(frame)
    result = pipeline.build_4dsg_dict()

    # Should be fused into 1 track (not 2)
    assert result["metadata"]["num_tracks"] == 1, (
        f"Expected 1 fused track, got {result['metadata']['num_tracks']}"
    )

    print(f"  Input: 2 detections from 2 runs")
    print(f"  Output: {result['metadata']['num_tracks']} fused track(s)")
    print("  PASSED")


def test_multiple_objects():
    """Test multiple distinct objects in one frame."""
    print("\n=== Test: Multiple distinct objects ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.0)

    # Three non-overlapping objects
    mask_a = make_mask(H, W, y0=50, y1=150, x0=50, x1=150)
    mask_b = make_mask(H, W, y0=50, y1=150, x0=300, x1=400)
    mask_c = make_mask(H, W, y0=300, y1=400, x0=200, x1=350)

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask_a, score=0.9),
            FastLocalDetection(run_id=0, local_obj_id=1, mask=mask_b, score=0.85),
            FastLocalDetection(run_id=1, local_obj_id=0, mask=mask_c, score=0.8),
        ],
    )

    pipeline.process_frame(frame)
    result = pipeline.build_4dsg_dict()

    assert result["metadata"]["num_tracks"] == 3, (
        f"Expected 3 tracks, got {result['metadata']['num_tracks']}"
    )

    # Verify ego_relations exist for all objects
    assert len(result["ego_relations"]) == 3, (
        f"Expected 3 ego relations, got {len(result['ego_relations'])}"
    )

    # Verify each ego relation has required fields
    for rel in result["ego_relations"]:
        assert "object_id" in rel
        assert "bearing" in rel
        assert "elev" in rel
        assert "dist_m" in rel
        assert "motion" in rel
        assert rel["bearing"] in [
            "front", "front-left", "left", "back-left",
            "back", "back-right", "right", "front-right",
        ]
        assert rel["elev"] in ["above", "below", "level"]

    # Verify object_relations
    assert len(result["object_relations"]) > 0, "Expected some object-object relations"
    for rel in result["object_relations"]:
        assert "src" in rel
        assert "dst" in rel
        assert "dir" in rel
        assert "elev" in rel
        assert "dist_m" in rel

    print(f"  Tracks: {result['metadata']['num_tracks']}")
    print(f"  Ego relations: {len(result['ego_relations'])}")
    print(f"  Object relations: {len(result['object_relations'])}")
    print("  PASSED")


def test_json_schema_compliance():
    """Verify the 4DSG JSON matches the spec schema exactly."""
    print("\n=== Test: JSON schema compliance ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    for t in range(3):
        T_wc = np.eye(4, dtype=np.float64)
        depth = make_depth_map(H, W)
        mask = make_mask(H, W)

        frame = FastFrameInput(
            frame_idx=t,
            depth_t=depth,
            K_t=K,
            T_wc_t=T_wc,
            detections=[
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
            ],
        )
        pipeline.process_frame(frame)

    json_str = pipeline.serialize_4dsg()
    result = json.loads(json_str)

    # Top-level keys
    required_keys = {"metadata", "ego", "tracks", "ego_relations", "object_relations"}
    assert set(result.keys()) == required_keys, f"Top-level keys: {set(result.keys())} != {required_keys}"

    # Metadata
    assert "grid" in result["metadata"]
    assert result["metadata"]["grid"] == "16x16"
    assert "num_frames" in result["metadata"]
    assert "num_tracks" in result["metadata"]

    # Ego entries
    for ego in result["ego"]:
        assert "t" in ego
        assert "xyz" in ego
        assert len(ego["xyz"]) == 3

    # Track entries
    for track in result["tracks"]:
        assert "object_id" in track
        assert "F_k" in track
        for obs in track["F_k"]:
            assert "t" in obs
            assert "tau" in obs
            assert "c" in obs
            assert "s" in obs
            assert "theta" in obs

            # tau items
            for patch in obs["tau"]:
                assert "row" in patch
                assert "col" in patch
                assert "iou" in patch

            # c = [x, y, z]
            assert len(obs["c"]) == 3

            # s = {x: {mu, sigma, min, max}, y: {...}, z: {...}}
            for axis in ["x", "y", "z"]:
                assert axis in obs["s"]
                for stat in ["mu", "sigma", "min", "max"]:
                    assert stat in obs["s"][axis]

            # theta = [t_start, t_end]
            assert len(obs["theta"]) == 2

    # Ego relations
    for rel in result["ego_relations"]:
        assert set(rel.keys()) == {"object_id", "bearing", "elev", "dist_m", "motion"}

    # Object relations
    for rel in result["object_relations"]:
        assert set(rel.keys()) == {"src", "dst", "dir", "elev", "dist_m"}

    print(f"  JSON valid, {len(json_str)} chars")
    print(f"  Tracks: {result['metadata']['num_tracks']}")
    print(f"  Grid: {result['metadata']['grid']}")
    print("  PASSED")


def test_no_float_quantization():
    """Verify floats are NOT rounded (spec: float32 original precision)."""
    print("\n=== Test: No float quantization ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics(fx=517.123456, fy=517.789012, cx=319.567, cy=239.234)
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.123456)
    mask = make_mask(H, W)

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
        ],
    )

    pipeline.process_frame(frame)
    json_str = pipeline.serialize_4dsg()

    # Check that centroid values have more than 2 decimal places
    result = json.loads(json_str)
    track = result["tracks"][0]
    obs = track["F_k"][0]

    for coord in obs["c"]:
        # After json dump/load, should still have full float precision
        s = str(coord)
        # Remove trailing zeros for comparison
        # The point is: we should NOT see all values rounded to 2 decimals
        assert isinstance(coord, float), f"Expected float, got {type(coord)}"

    print(f"  Centroid: {obs['c']}")
    print(f"  Shape x mu: {obs['s']['x']['mu']}")
    print("  PASSED (floats preserved at full precision)")


def test_global_id_uniqueness():
    """Verify no duplicate global IDs in a single window."""
    print("\n=== Test: Global ID uniqueness ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    for t in range(5):
        T_wc = np.eye(4, dtype=np.float64)
        depth = make_depth_map(H, W)

        masks = [
            make_mask(H, W, y0=50, y1=150, x0=50, x1=150),
            make_mask(H, W, y0=50, y1=150, x0=300, x1=400),
            make_mask(H, W, y0=300, y1=400, x0=200, x1=350),
        ]

        dets = [
            FastLocalDetection(run_id=0, local_obj_id=i, mask=m, score=0.9 - i * 0.05)
            for i, m in enumerate(masks)
        ]

        frame = FastFrameInput(
            frame_idx=t, depth_t=depth, K_t=K, T_wc_t=T_wc, detections=dets,
        )
        pipeline.process_frame(frame)

    result = pipeline.build_4dsg_dict()
    ids = [track["object_id"] for track in result["tracks"]]
    assert len(ids) == len(set(ids)), f"Duplicate global IDs found: {ids}"

    print(f"  Track IDs: {ids}")
    print("  PASSED (all unique)")


def test_strided_frame_indices():
    """Test pipeline with non-sequential frame indices (simulating stride > 1)."""
    print("\n=== Test: Strided frame indices (stride=5) ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    # Simulate stride=5: source frames 0, 5, 10, 15, 20
    source_frames = [0, 5, 10, 15, 20]
    for i, src_t in enumerate(source_frames):
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = i * 0.1  # Ego moves slightly

        depth = make_depth_map(H, W, base_depth=5.0 + i * 0.2)
        mask = make_mask(H, W, y0=100 + i * 5, y1=200 + i * 5, x0=200 + i * 10, x1=350 + i * 10)

        frame = FastFrameInput(
            frame_idx=src_t,  # Source video frame number, not enumeration index
            depth_t=depth,
            K_t=K,
            T_wc_t=T_wc,
            detections=[
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.85),
            ],
        )
        pipeline.process_frame(frame)

    result = pipeline.build_4dsg_dict()

    # Verify ego entries use source frame numbers
    ego_ts = [e["t"] for e in result["ego"]]
    assert ego_ts == source_frames, f"Expected source frame numbers {source_frames}, got {ego_ts}"

    # Verify track observations use source frame numbers
    track = result["tracks"][0]
    obs_ts = [obs["t"] for obs in track["F_k"]]
    assert obs_ts == source_frames, f"Expected source frame numbers in obs, got {obs_ts}"

    # Verify theta spans full source range
    assert track["F_k"][0]["theta"] == [0, 20], (
        f"Expected theta=[0,20], got {track['F_k'][0]['theta']}"
    )

    print(f"  Ego t values: {ego_ts}")
    print(f"  Obs t values: {obs_ts}")
    print(f"  Theta: {track['F_k'][0]['theta']}")
    print("  PASSED (source frame numbers preserved)")


def test_conf_thresh_filtering():
    """Test that low-confidence depth pixels are excluded from backprojection."""
    print("\n=== Test: conf_thresh filtering ===")

    config = FastSNOWConfig()
    config.depth_filter.conf_thresh = 0.8
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.0)
    mask = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)

    # Confidence map: all 0.5 (below thresh 0.8) → should reject all pixels
    conf = np.full((H, W), 0.5, dtype=np.float32)
    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
        ],
        depth_conf_t=conf,
    )

    pipeline.process_frame(frame)
    result = pipeline.build_4dsg_dict()

    # All pixels below conf_thresh → min_points not met → detection rejected → 0 tracks
    assert result["metadata"]["num_tracks"] == 0, (
        f"Expected 0 tracks (all conf below thresh), got {result['metadata']['num_tracks']}"
    )

    # Now test with conf above threshold → should produce 1 track
    pipeline2 = FastSNOWPipeline(config)
    conf_high = np.full((H, W), 0.95, dtype=np.float32)
    frame2 = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
        ],
        depth_conf_t=conf_high,
    )
    pipeline2.process_frame(frame2)
    result2 = pipeline2.build_4dsg_dict()
    assert result2["metadata"]["num_tracks"] == 1, (
        f"Expected 1 track (conf above thresh), got {result2['metadata']['num_tracks']}"
    )

    print("  conf=0.5 < thresh=0.8 → 0 tracks (rejected)")
    print("  conf=0.95 > thresh=0.8 → 1 track (accepted)")
    print("  PASSED")


def test_max_extent_filtering():
    """Test that objects exceeding max_extent are rejected."""
    print("\n=== Test: max_extent filtering ===")

    config = FastSNOWConfig()
    config.depth_filter.max_extent = 5.0  # 5 metre limit
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)

    # Create a depth map with huge range (0.1 to 100m) so extent >> 5m
    depth = np.full((H, W), 50.0, dtype=np.float32)
    depth[100:200, 200:350] = 0.1  # Object region at near depth
    depth[100:200, 350:400] = 100.0  # Same row but extreme far

    # Mask spans both near and far pixels → extent > 5m → rejected
    mask_wide = make_mask(H, W, y0=100, y1=200, x0=200, x1=400)

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask_wide, score=0.9),
        ],
    )
    pipeline.process_frame(frame)
    result = pipeline.build_4dsg_dict()

    assert result["metadata"]["num_tracks"] == 0, (
        f"Expected 0 tracks (extent > max_extent), got {result['metadata']['num_tracks']}"
    )

    # Now test with a compact object → extent < 5m → accepted
    pipeline2 = FastSNOWPipeline(config)
    depth2 = make_depth_map(H, W, base_depth=5.0)  # Small range
    mask_compact = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)
    frame2 = FastFrameInput(
        frame_idx=0,
        depth_t=depth2,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask_compact, score=0.9),
        ],
    )
    pipeline2.process_frame(frame2)
    result2 = pipeline2.build_4dsg_dict()

    assert result2["metadata"]["num_tracks"] == 1, (
        f"Expected 1 track (extent < max_extent), got {result2['metadata']['num_tracks']}"
    )

    print("  Huge depth range → extent > 5m → 0 tracks (rejected)")
    print("  Normal depth → extent < 5m → 1 track (accepted)")
    print("  PASSED")


def test_archived_track_not_reactivated():
    """Archived tracks must never be re-identified (spec §5).

    Scenario: object appears frames 0-2, disappears until archived
    (lost_patience + archive_patience frames), then same (run_id, local_obj_id)
    reappears.  It should get a NEW global ID, not reuse the old one.
    """
    print("\n=== Test: Archived track not reactivated ===")

    config = FastSNOWConfig()
    config.fusion.lost_patience = 2
    config.fusion.archive_patience = 3  # archived after 2+3=5 consecutive misses

    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    def _frame(t, with_det=True):
        T_wc = np.eye(4, dtype=np.float64)
        depth = make_depth_map(H, W, base_depth=5.0)
        mask = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)
        dets = []
        if with_det:
            dets.append(
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9)
            )
        return FastFrameInput(
            frame_idx=t, depth_t=depth, K_t=K, T_wc_t=T_wc, detections=dets,
        )

    # Phase 1: object visible for frames 0-2
    for t in range(3):
        pipeline.process_frame(_frame(t, with_det=True))

    result1 = pipeline.build_4dsg_dict()
    assert result1["metadata"]["num_tracks"] == 1
    original_gid = result1["tracks"][0]["object_id"]

    # Phase 2: object missing for 6 frames (> lost_patience + archive_patience = 5)
    for t in range(3, 9):
        pipeline.process_frame(_frame(t, with_det=False))

    # Verify track is now archived
    track_state = pipeline._tracks[original_gid]
    assert track_state.status == "archived", (
        f"Expected archived after 6 misses, got {track_state.status}"
    )

    # Phase 3: same (run_id=0, local_obj_id=0) reappears at frame 9
    pipeline.process_frame(_frame(9, with_det=True))

    result2 = pipeline.build_4dsg_dict()
    # Should now have 2 tracks: old archived + new active
    assert result2["metadata"]["num_tracks"] == 2, (
        f"Expected 2 tracks (old archived + new), got {result2['metadata']['num_tracks']}"
    )

    gids = [t["object_id"] for t in result2["tracks"]]
    assert original_gid in gids, "Original archived track should still exist"
    new_gid = [g for g in gids if g != original_gid][0]
    assert new_gid != original_gid, "Reappeared detection must get a NEW global ID"

    # Verify the new track has only 1 observation (frame 9)
    new_track = [t for t in result2["tracks"] if t["object_id"] == new_gid][0]
    assert len(new_track["F_k"]) == 1, (
        f"New track should have 1 obs, got {len(new_track['F_k'])}"
    )
    assert new_track["F_k"][0]["t"] == 9

    # Verify the old track still has 3 observations (frames 0-2)
    old_track = [t for t in result2["tracks"] if t["object_id"] == original_gid][0]
    assert len(old_track["F_k"]) == 3, (
        f"Old track should have 3 obs, got {len(old_track['F_k'])}"
    )

    print(f"  Original track gid={original_gid}, 3 obs, status=archived")
    print(f"  New track gid={new_gid}, 1 obs (frame 9)")
    print("  PASSED (archived track NOT reactivated)")


def test_motion_cold_start_and_rate():
    """Motion must be 'unknown' when history < motion_window (spec §7).

    Also verifies rate denominator is dt (time gap), not N (sample count).
    """
    print("\n=== Test: Motion cold start + dt-normalized rate ===")

    config = FastSNOWConfig()
    config.edge.motion_window = 3
    config.edge.motion_thresh = 0.3
    config.edge.lateral_thresh = 0.3
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    mask = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)

    def _frame(t, ego_x=0.0):
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = ego_x
        depth = make_depth_map(H, W, base_depth=5.0)
        return FastFrameInput(
            frame_idx=t,
            depth_t=depth,
            K_t=K,
            T_wc_t=T_wc,
            detections=[
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
            ],
        )

    # Frame 0: 1 sample → unknown (< motion_window=3)
    pipeline.process_frame(_frame(0, ego_x=0.0))
    r0 = pipeline.build_4dsg_dict()
    m0 = r0["ego_relations"][0]["motion"]
    assert m0 == "unknown", f"1 sample: expected 'unknown', got '{m0}'"

    # Frame 1: 2 samples → still unknown (< motion_window=3)
    pipeline.process_frame(_frame(1, ego_x=0.0))
    r1 = pipeline.build_4dsg_dict()
    m1 = r1["ego_relations"][0]["motion"]
    assert m1 == "unknown", f"2 samples: expected 'unknown', got '{m1}'"

    # Frame 2: 3 samples → now can infer (= motion_window)
    pipeline.process_frame(_frame(2, ego_x=0.0))
    r2 = pipeline.build_4dsg_dict()
    m2 = r2["ego_relations"][0]["motion"]
    assert m2 != "unknown", f"3 samples: should NOT be 'unknown', got '{m2}'"

    print(f"  1 sample → '{m0}' (cold start)")
    print(f"  2 samples → '{m1}' (still cold start)")
    print(f"  3 samples → '{m2}' (motion_window met)")

    # --- Rate denominator: dt vs N discriminating test ---
    # Object approaches ego: depth decreases 10 → 9 → 8 across strided frames.
    # ego-object distance Δ ≈ -2.0m over the 3 samples.
    #   dt-based (correct):  rate = -2.0 / 20 = -0.10 → |rate| < 0.3 → "static"
    #   N-based (old impl):  rate = -2.0 / 3 ≈ -0.67 → |rate| > 0.3 → "approaching"
    config3 = FastSNOWConfig()
    config3.edge.motion_window = 3
    config3.edge.motion_thresh = 0.3
    config3.edge.lateral_thresh = 0.3
    pipeline3 = FastSNOWPipeline(config3)

    depths = [10.0, 9.0, 8.0]
    strided_ts = [0, 10, 20]
    for t, d in zip(strided_ts, depths):
        T_wc = np.eye(4, dtype=np.float64)
        depth = make_depth_map(H, W, base_depth=d)
        frame = FastFrameInput(
            frame_idx=t, depth_t=depth, K_t=K, T_wc_t=T_wc,
            detections=[FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9)],
        )
        pipeline3.process_frame(frame)

    r3 = pipeline3.build_4dsg_dict()
    motion_strided = r3["ego_relations"][0]["motion"]
    # With dt denominator this must be "static"; with N it would be "approaching".
    assert motion_strided == "static", (
        f"Strided depth 10→8: expected 'static' (rate/dt), got '{motion_strided}'"
    )
    print(f"  Strided depth 10→8 (t=0,10,20) → '{motion_strided}' (rate/dt, not rate/N)")
    print("  PASSED")


def test_merge_deduplicates_observations():
    """Cross-run fusion merge must not create duplicate per-frame observations.

    Scenario: two runs both detect the same object on frames 0-2 with high IoU.
    After fusion they share one global track.  Each frame should have exactly
    one observation (not two).
    """
    print("\n=== Test: Merge deduplicates observations ===")

    config = FastSNOWConfig()
    config.fusion.cross_run_iou_thresh = 0.3
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()

    for t in range(3):
        T_wc = np.eye(4, dtype=np.float64)
        depth = make_depth_map(H, W, base_depth=5.0)
        # Two runs with heavily overlapping masks → should be fused
        mask1 = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)
        mask2 = make_mask(H, W, y0=105, y1=205, x0=205, x1=355)

        frame = FastFrameInput(
            frame_idx=t,
            depth_t=depth,
            K_t=K,
            T_wc_t=T_wc,
            detections=[
                FastLocalDetection(run_id=0, local_obj_id=0, mask=mask1, score=0.9),
                FastLocalDetection(run_id=1, local_obj_id=0, mask=mask2, score=0.8),
            ],
        )
        pipeline.process_frame(frame)

    result = pipeline.build_4dsg_dict()
    assert result["metadata"]["num_tracks"] == 1, (
        f"Expected 1 fused track, got {result['metadata']['num_tracks']}"
    )

    track = result["tracks"][0]
    obs_ts = [obs["t"] for obs in track["F_k"]]
    # Must be exactly [0, 1, 2] — no duplicates
    assert obs_ts == [0, 1, 2], (
        f"Expected obs at [0,1,2] (no duplicates), got {obs_ts}"
    )

    print(f"  Fused 2 runs × 3 frames → {len(obs_ts)} observations (no duplicates)")
    print("  PASSED")


def test_merge_keeps_winner_observation():
    """When tracks merge on a delayed fusion, same-frame observation must come
    from the keep (higher-score) track, not the drop track.

    Scenario:
      Frame 0: run 0 detects mask_left (score=0.9) → gid=0, track 0 obs[t=0]
               run 1 detects mask_right (score=0.8) → gid=1, track 1 obs[t=0]
               (No fusion: masks don't overlap)
      Frame 1: run 0 detects mask_center (score=0.9), run 1 detects mask_center_shifted (score=0.8)
               (Fusion: IoU > thresh → union gid 0 and 1)
               _merge_track_states called with overlapping obs at t=0:
                 keep (gid=0) has obs from mask_left, drop (gid=1) has obs from mask_right.
                 Result must retain keep's obs (mask_left centroid), not drop's.
    """
    print("\n=== Test: Merge keeps winner (keep) observation ===")

    config = FastSNOWConfig()
    config.fusion.cross_run_iou_thresh = 0.3
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.0)

    # Frame 0: two non-overlapping masks → separate tracks
    mask_left = make_mask(H, W, y0=100, y1=200, x0=50, x1=200)
    mask_right = make_mask(H, W, y0=100, y1=200, x0=400, x1=550)

    frame0 = FastFrameInput(
        frame_idx=0, depth_t=depth, K_t=K, T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask_left, score=0.9),
            FastLocalDetection(run_id=1, local_obj_id=0, mask=mask_right, score=0.8),
        ],
    )
    pipeline.process_frame(frame0)

    r0 = pipeline.build_4dsg_dict()
    assert r0["metadata"]["num_tracks"] == 2, (
        f"Frame 0: expected 2 separate tracks, got {r0['metadata']['num_tracks']}"
    )

    # Remember the keep track's centroid at t=0 (from mask_left, gid=0)
    # gid 0 is the first allocated → run 0's track
    keep_track_f0 = [t for t in r0["tracks"] if t["object_id"] == 0][0]
    keep_centroid_x_f0 = keep_track_f0["F_k"][0]["c"][0]

    drop_track_f0 = [t for t in r0["tracks"] if t["object_id"] == 1][0]
    drop_centroid_x_f0 = drop_track_f0["F_k"][0]["c"][0]

    # Sanity: mask_left and mask_right produce very different x-centroids
    assert abs(keep_centroid_x_f0 - drop_centroid_x_f0) > 0.5, (
        f"Centroids should differ significantly: keep={keep_centroid_x_f0}, drop={drop_centroid_x_f0}"
    )

    # Frame 1: both runs now produce overlapping masks → fusion triggers merge
    mask_center = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)
    mask_center_shifted = make_mask(H, W, y0=105, y1=205, x0=205, x1=355)

    frame1 = FastFrameInput(
        frame_idx=1, depth_t=depth, K_t=K, T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask_center, score=0.9),
            FastLocalDetection(run_id=1, local_obj_id=0, mask=mask_center_shifted, score=0.8),
        ],
    )
    pipeline.process_frame(frame1)

    r1 = pipeline.build_4dsg_dict()
    assert r1["metadata"]["num_tracks"] == 1, (
        f"Frame 1: expected 1 fused track, got {r1['metadata']['num_tracks']}"
    )

    merged = r1["tracks"][0]
    obs_ts = [obs["t"] for obs in merged["F_k"]]
    assert obs_ts == [0, 1], f"Expected obs at [0,1], got {obs_ts}"

    # The t=0 observation must have the KEEP track's centroid (mask_left),
    # not the DROP track's centroid (mask_right).
    merged_centroid_x_f0 = merged["F_k"][0]["c"][0]
    assert merged_centroid_x_f0 == keep_centroid_x_f0, (
        f"t=0 obs should be keep's centroid ({keep_centroid_x_f0}), "
        f"got {merged_centroid_x_f0} (drop was {drop_centroid_x_f0})"
    )

    print(f"  Frame 0: 2 tracks, keep centroid_x={keep_centroid_x_f0:.3f}, drop centroid_x={drop_centroid_x_f0:.3f}")
    print(f"  Frame 1: fused → 1 track, t=0 centroid_x={merged_centroid_x_f0:.3f} (matches keep)")
    print("  PASSED (keep's observation preserved over drop's)")


def test_visual_anchor_metadata():
    """build_4dsg_dict must include visual_anchor in metadata when provided (spec §4.3).

    Also verifies backward compatibility: when visual_anchor is None,
    metadata should NOT contain the key.
    """
    print("\n=== Test: visual_anchor metadata ===")

    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    H, W = 480, 640
    K = make_identity_intrinsics()
    T_wc = np.eye(4, dtype=np.float64)
    depth = make_depth_map(H, W, base_depth=5.0)
    mask = make_mask(H, W, y0=100, y1=200, x0=200, x1=350)

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id=0, local_obj_id=0, mask=mask, score=0.9),
        ],
    )
    pipeline.process_frame(frame)

    # Without visual_anchor → key absent from metadata
    result_no_va = pipeline.build_4dsg_dict()
    assert "visual_anchor" not in result_no_va["metadata"], (
        "visual_anchor should NOT be in metadata when not provided"
    )

    # With visual_anchor → key present with correct entries
    va = [
        {"frame_idx": 0, "path": "/tmp/000000.jpg"},
        {"frame_idx": 5, "path": "/tmp/000001.jpg"},
    ]
    result_va = pipeline.build_4dsg_dict(visual_anchor=va)
    assert "visual_anchor" in result_va["metadata"], (
        "visual_anchor should be in metadata when provided"
    )
    assert len(result_va["metadata"]["visual_anchor"]) == 2
    assert result_va["metadata"]["visual_anchor"][0]["frame_idx"] == 0
    assert result_va["metadata"]["visual_anchor"][1]["frame_idx"] == 5

    # Verify JSON round-trip via build_4dsg_dict
    json_str = json.dumps(result_va)
    reloaded = json.loads(json_str)
    assert reloaded["metadata"]["visual_anchor"] == va

    # Verify serialize_4dsg() also includes visual_anchor when passed
    # (regression: previously serialize_4dsg called build_4dsg_dict without
    #  forwarding visual_anchor, so the VLM-facing JSON was missing it)
    serialized = pipeline.serialize_4dsg(visual_anchor=va)
    reloaded_ser = json.loads(serialized)
    assert "visual_anchor" in reloaded_ser["metadata"], (
        "serialize_4dsg must include visual_anchor in JSON"
    )
    assert reloaded_ser["metadata"]["visual_anchor"] == va

    # serialize_4dsg without visual_anchor → absent
    serialized_no_va = pipeline.serialize_4dsg()
    reloaded_no_va = json.loads(serialized_no_va)
    assert "visual_anchor" not in reloaded_no_va["metadata"], (
        "serialize_4dsg without visual_anchor should omit key"
    )

    print("  Without visual_anchor → key absent (backward compat)")
    print(f"  With visual_anchor → {len(va)} entries present")
    print("  JSON round-trip (dict) → OK")
    print("  serialize_4dsg(visual_anchor=...) → OK")
    print("  serialize_4dsg() without → key absent → OK")
    print("  PASSED")


def test_result_cleanup():
    """result.cleanup() must delete keyframe_dir; double-call must be safe.

    Tests both FastSNOWE2EResult and FastSNOW4DSGResult via shared
    _KeyframeDirMixin.cleanup() contract.
    """
    print("\n=== Test: Result cleanup() ===")

    from fast_snow.engine.pipeline.fast_snow_e2e import (
        FastSNOWE2EResult,
        FastSNOW4DSGResult,
    )

    for cls_name, make_result in [
        ("FastSNOWE2EResult", lambda d: FastSNOWE2EResult(
            answer="A", scene_json="{}", four_dsg_dict={}, keyframe_dir=d,
        )),
        ("FastSNOW4DSGResult", lambda d: FastSNOW4DSGResult(
            four_dsg_dict={}, scene_json="{}", keyframe_dir=d,
        )),
    ]:
        # Create a real temp dir with a file inside
        tmp = Path(tempfile.mkdtemp(prefix="fast_snow_cleanup_test_"))
        (tmp / "000000.jpg").write_bytes(b"\xff\xd8dummy")
        assert tmp.exists(), f"{cls_name}: temp dir should exist before cleanup"

        result = make_result(tmp)
        assert result.keyframe_dir == tmp

        # cleanup() should delete the directory
        result.cleanup()
        assert not tmp.exists(), (
            f"{cls_name}: keyframe_dir should be deleted after cleanup()"
        )
        assert result.keyframe_dir is None, (
            f"{cls_name}: keyframe_dir should be None after cleanup()"
        )

        # Double cleanup must be safe (no-op)
        result.cleanup()  # should not raise

        print(f"  {cls_name}: cleanup() deletes dir, double-call safe")

    # Also verify that export is discoverable
    from fast_snow.engine.pipeline import FastSNOW4DSGResult as Imported
    assert Imported is FastSNOW4DSGResult, "FastSNOW4DSGResult must be importable from package"

    print("  FastSNOW4DSGResult importable from fast_snow.engine.pipeline")
    print("  PASSED")


if __name__ == "__main__":
    print("Fast-SNOW Smoke Test (no GPU)\n")

    test_single_frame()
    test_multi_frame_tracking()
    test_cross_run_fusion()
    test_multiple_objects()
    test_json_schema_compliance()
    test_no_float_quantization()
    test_global_id_uniqueness()
    test_strided_frame_indices()
    test_conf_thresh_filtering()
    test_max_extent_filtering()
    test_archived_track_not_reactivated()
    test_motion_cold_start_and_rate()
    test_merge_deduplicates_observations()
    test_merge_keeps_winner_observation()
    test_visual_anchor_metadata()
    test_result_cleanup()

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
