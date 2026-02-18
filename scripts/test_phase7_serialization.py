#!/usr/bin/env python3
"""Test Phase 7 serialization determinism and correctness.

This script verifies that:
1. Serialization is deterministic (same input → same output)
2. All lists are properly sorted
3. Precision is correct (2 decimal places)
4. Field order is consistent
5. Frame indices are correctly used and distinct (Issue #1 fix)
6. Frame indices are properly propagated through the pipeline (Issue #1 fix)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from fast_snow.reasoning.graph.scene_graph import SceneGraph, SceneNode, SceneEdge
from fast_snow.reasoning.graph.temporal_linking import TemporalTrack, TemporalWindow
from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph
from fast_snow.reasoning.tokens.step_encoding import STEPToken
from fast_snow.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken
from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken
from fast_snow.reasoning.vlm.prompt_builder import (
    serialize_4dsg_json_strict,
    Phase7SerializationConfig,
)


def create_test_4dsg() -> FourDSceneGraph:
    """Create a test 4DSG with known properties for validation."""

    # Create STEP tokens for multiple tracks
    step_tokens = []

    # Track 0: 3 frames
    for frame in [0, 1, 2]:
        step = STEPToken(
            patch_tokens=[
                PatchToken(row=i, col=j, iou=0.8)
                for i in range(3) for j in range(3)
            ],
            centroid=CentroidToken(x=5.123, y=10.456, z=0.789),
            shape=ShapeToken(
                x_mu=5.0, x_sigma=0.5, x_min=4.0, x_max=6.0,
                y_mu=10.5, y_sigma=0.8, y_min=9.0, y_max=12.0,
                z_mu=0.75, z_sigma=0.3, z_min=0.0, z_max=1.5
            ),
            temporal=TemporalToken(t_start=0, t_end=2),
        )
        step_tokens.append(step)

    # CRITICAL: Provide frame_indices to test Issue #1 fix
    track0 = TemporalTrack(track_id=0, steps=step_tokens, frame_indices=[0, 1, 2])

    # Track 1: 2 frames (different order to test sorting)
    step_tokens_1 = []
    for frame in [1, 0]:  # Intentionally reversed order
        step = STEPToken(
            patch_tokens=[PatchToken(row=0, col=0, iou=0.9)],
            centroid=CentroidToken(x=15.111, y=20.222, z=0.333),
            shape=ShapeToken(
                x_mu=15.0, x_sigma=0.4, x_min=14.0, x_max=16.0,
                y_mu=20.0, y_sigma=0.5, y_min=19.0, y_max=21.0,
                z_mu=0.5, z_sigma=0.2, z_min=0.0, z_max=1.0
            ),
            temporal=TemporalToken(t_start=0, t_end=1),
        )
        step_tokens_1.append(step)

    # CRITICAL: Provide frame_indices (intentionally unsorted to test sorting)
    track1 = TemporalTrack(track_id=1, steps=step_tokens_1, frame_indices=[1, 0])

    # Temporal window (intentionally unsorted to test sorting)
    temporal_window = TemporalWindow(tracks={1: track1, 0: track0})  # Reversed order

    # Spatial graphs with edges
    # Create nodes from STEP tokens
    node0 = SceneNode(
        node_id=0,
        step=track0.steps[0],
        position=np.array([5.123, 10.456, 0.789]),
    )
    node1 = SceneNode(
        node_id=1,
        step=track1.steps[0],
        position=np.array([15.111, 20.222, 0.333]),
    )

    # Create edge (intentionally add multiple with different distances to test sorting)
    edge1 = SceneEdge(
        src=0,
        dst=1,
        distance=14.142,  # Second smallest distance
        direction=np.array([0.7071, 0.7071, 0.0]),
    )

    edge2 = SceneEdge(
        src=1,
        dst=0,
        distance=5.0,  # Smallest distance
        direction=np.array([-0.7071, -0.7071, 0.0]),
    )

    edge3 = SceneEdge(
        src=0,
        dst=1,
        distance=20.0,  # Largest distance
        direction=np.array([1.0, 0.0, 0.0]),
    )

    graph0 = SceneGraph(
        frame_idx=0,
        nodes=[node0, node1],
        edges=[edge1, edge2, edge3],  # Intentionally unsorted
        ego_pose=None,
    )

    spatial_graphs = [graph0]

    # Ego poses (intentionally unsorted to test sorting)
    ego_poses = {
        2: [2.0, 0.4, 0.0],
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.2, 0.0],
    }

    return FourDSceneGraph(
        spatial_graphs=spatial_graphs,
        temporal_window=temporal_window,
        ego_poses=ego_poses,
    )


def test_determinism():
    """Test that serialization is deterministic."""
    print("=" * 60)
    print("TEST 1: Determinism")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    # Serialize multiple times
    json1 = serialize_4dsg_json_strict(graph, config)
    json2 = serialize_4dsg_json_strict(graph, config)
    json3 = serialize_4dsg_json_strict(graph, config)

    # Convert to JSON strings for comparison
    str1 = json.dumps(json1, sort_keys=True, indent=2)
    str2 = json.dumps(json2, sort_keys=True, indent=2)
    str3 = json.dumps(json3, sort_keys=True, indent=2)

    if str1 == str2 == str3:
        print("✅ PASS: Serialization is deterministic")
        return True
    else:
        print("❌ FAIL: Serialization is not deterministic")
        print(f"First run:\n{str1[:200]}...")
        print(f"Second run:\n{str2[:200]}...")
        return False


def test_sorting():
    """Test that all lists are properly sorted."""
    print("\n" + "=" * 60)
    print("TEST 2: List Sorting")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    result = serialize_4dsg_json_strict(graph, config)

    all_passed = True

    # Test 1: Ego trajectory sorted by frame
    ego_frames = [t["frame"] for t in result["ego_agent"]["trajectory"]]
    if ego_frames == sorted(ego_frames):
        print("✅ PASS: Ego trajectory sorted by frame")
    else:
        print(f"❌ FAIL: Ego trajectory not sorted: {ego_frames}")
        all_passed = False

    # Test 2: Objects sorted by object_id
    object_ids = [o["object_id"] for o in result["objects"]]
    if object_ids == sorted(object_ids):
        print("✅ PASS: Objects sorted by object_id")
    else:
        print(f"❌ FAIL: Objects not sorted: {object_ids}")
        all_passed = False

    # Test 3: Track steps sorted by frame
    for obj in result["objects"]:
        frames = [s["frame"] for s in obj["track"]]
        if frames == sorted(frames):
            print(f"✅ PASS: Object {obj['object_id']} track sorted by frame")
        else:
            print(f"❌ FAIL: Object {obj['object_id']} track not sorted: {frames}")
            all_passed = False

    # Test 4: Spatial relations sorted by distance
    distances = [r["distance"] for r in result["spatial_relations"]]
    if distances == sorted(distances):
        print("✅ PASS: Spatial relations sorted by distance")
    else:
        print(f"❌ FAIL: Spatial relations not sorted: {distances}")
        all_passed = False

    return all_passed


def test_precision():
    """Test that precision is correct (2 decimal places)."""
    print("\n" + "=" * 60)
    print("TEST 3: Precision")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    result = serialize_4dsg_json_strict(graph, config)

    all_passed = True

    # Check ego positions
    for traj in result["ego_agent"]["trajectory"]:
        for val in traj["position"]:
            if len(str(val).split(".")[-1]) <= 2:
                pass  # OK
            else:
                print(f"❌ FAIL: Ego position has >2 decimals: {val}")
                all_passed = False
                break

    # Check object centroids
    for obj in result["objects"]:
        for step in obj["track"]:
            for val in step["centroid"]:
                decimal_part = str(val).split(".")[-1] if "." in str(val) else ""
                if len(decimal_part) <= 2:
                    pass  # OK
                else:
                    print(f"❌ FAIL: Centroid has >2 decimals: {val}")
                    all_passed = False
                    break

    # Check distances
    for rel in result["spatial_relations"]:
        val = rel["distance"]
        decimal_part = str(val).split(".")[-1] if "." in str(val) else ""
        if len(decimal_part) <= 2:
            pass  # OK
        else:
            print(f"❌ FAIL: Distance has >2 decimals: {val}")
            all_passed = False

    if all_passed:
        print("✅ PASS: All values have ≤2 decimal places")

    return all_passed


def test_field_order():
    """Test that field order is consistent."""
    print("\n" + "=" * 60)
    print("TEST 4: Field Order")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    result = serialize_4dsg_json_strict(graph, config)

    # Expected top-level order
    expected_top = ["metadata", "ego_agent", "objects", "spatial_relations"]
    actual_top = list(result.keys())

    if actual_top == expected_top:
        print(f"✅ PASS: Top-level fields in correct order: {actual_top}")
    else:
        print(f"❌ FAIL: Top-level fields incorrect")
        print(f"  Expected: {expected_top}")
        print(f"  Actual:   {actual_top}")
        return False

    # Expected metadata order
    expected_meta = ["num_frames", "num_objects", "temporal_window"]
    actual_meta = list(result["metadata"].keys())

    if actual_meta == expected_meta:
        print(f"✅ PASS: Metadata fields in correct order")
    else:
        print(f"❌ FAIL: Metadata fields incorrect")
        print(f"  Expected: {expected_meta}")
        print(f"  Actual:   {actual_meta}")
        return False

    return True


def test_frame_indices_consistency():
    """Test that frame indices are correctly used and distinct across steps.

    This is a critical test for Issue #1 fix: Previously, all steps in a track
    would have the same frame value (t_end), which was incorrect.
    Now we should see distinct frame values from frame_indices.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Frame Indices Consistency (Issue #1 Fix)")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    result = serialize_4dsg_json_strict(graph, config)

    all_passed = True

    # Test that each track has distinct frame values
    for obj in result["objects"]:
        track_frames = [step["frame"] for step in obj["track"]]

        # Check that frames are distinct (if track has multiple steps)
        if len(track_frames) > 1:
            if len(track_frames) == len(set(track_frames)):
                print(f"✅ PASS: Object {obj['object_id']} has distinct frames: {track_frames}")
            else:
                print(f"❌ FAIL: Object {obj['object_id']} has duplicate frames: {track_frames}")
                print(f"  This indicates frame_indices is not being used correctly!")
                all_passed = False

        # Check that frames are in valid range
        for frame in track_frames:
            if not isinstance(frame, int):
                print(f"❌ FAIL: Object {obj['object_id']} has non-integer frame: {frame}")
                all_passed = False
            elif frame < 0:
                print(f"❌ FAIL: Object {obj['object_id']} has negative frame: {frame}")
                all_passed = False

    # Verify specific test case: Object 0 should have frames [0, 1, 2]
    obj0 = [o for o in result["objects"] if o["object_id"] == 0][0]
    obj0_frames = [step["frame"] for step in obj0["track"]]
    expected_frames = [0, 1, 2]

    if obj0_frames == expected_frames:
        print(f"✅ PASS: Object 0 frames match expected: {obj0_frames}")
    else:
        print(f"❌ FAIL: Object 0 frames incorrect")
        print(f"  Expected: {expected_frames}")
        print(f"  Actual:   {obj0_frames}")
        all_passed = False

    return all_passed


def test_frame_indices_propagation():
    """Test that frame_indices are correctly propagated from Track to TemporalTrack.

    This verifies the fix for Issue #1: frame_indices must be preserved through
    the build_temporal_window_from_tracker() pipeline.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Frame Indices Propagation (Issue #1 Fix)")
    print("=" * 60)

    from fast_snow.reasoning.graph.object_tracker import Track
    from fast_snow.reasoning.graph.temporal_linking import build_temporal_window_from_tracker

    all_passed = True

    # Create test tracks with explicit frame_indices
    step_tokens = []
    for frame in [0, 2, 5, 7]:  # Non-consecutive frames
        step = STEPToken(
            patch_tokens=[PatchToken(row=0, col=0, iou=0.8)],
            centroid=CentroidToken(x=1.0, y=2.0, z=3.0),
            shape=ShapeToken(
                x_mu=1.0, x_sigma=0.1, x_min=0.9, x_max=1.1,
                y_mu=2.0, y_sigma=0.1, y_min=1.9, y_max=2.1,
                z_mu=3.0, z_sigma=0.1, z_min=2.9, z_max=3.1
            ),
            temporal=TemporalToken(t_start=0, t_end=7),
        )
        step_tokens.append(step)

    track = Track(
        track_id=42,
        steps=step_tokens,
        frame_indices=[0, 2, 5, 7],  # Explicit frame indices
    )

    # Build temporal window
    tracks = {42: track}
    temporal_window = build_temporal_window_from_tracker(tracks)

    # Verify frame_indices were preserved
    temporal_track = temporal_window.tracks[42]

    if temporal_track.frame_indices is not None:
        print(f"✅ PASS: frame_indices preserved in TemporalTrack")

        if temporal_track.frame_indices == [0, 2, 5, 7]:
            print(f"✅ PASS: frame_indices values correct: {temporal_track.frame_indices}")
        else:
            print(f"❌ FAIL: frame_indices values incorrect")
            print(f"  Expected: [0, 2, 5, 7]")
            print(f"  Actual:   {temporal_track.frame_indices}")
            all_passed = False
    else:
        print(f"❌ FAIL: frame_indices is None in TemporalTrack")
        all_passed = False

    # Now test serialization uses these frame_indices
    from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph
    from fast_snow.reasoning.graph.scene_graph import SceneGraph, SceneNode

    # Create minimal 4DSG for serialization test
    node = SceneNode(
        node_id=42,
        step=step_tokens[0],
        position=np.array([1.0, 2.0, 3.0]),
    )

    graph0 = SceneGraph(
        frame_idx=0,
        nodes=[node],
        edges=[],
        ego_pose=None,
    )

    four_dsg = FourDSceneGraph(
        spatial_graphs=[graph0],
        temporal_window=temporal_window,
        ego_poses={0: [0.0, 0.0, 0.0]},
    )

    # Serialize and check
    config = Phase7SerializationConfig()
    result = serialize_4dsg_json_strict(four_dsg, config)

    # Find object 42 in serialized output
    obj42 = [o for o in result["objects"] if o["object_id"] == 42][0]
    serialized_frames = [step["frame"] for step in obj42["track"]]

    if serialized_frames == [0, 2, 5, 7]:
        print(f"✅ PASS: Serialized frames match frame_indices: {serialized_frames}")
    else:
        print(f"❌ FAIL: Serialized frames don't match frame_indices")
        print(f"  Expected: [0, 2, 5, 7]")
        print(f"  Actual:   {serialized_frames}")
        all_passed = False

    return all_passed


def print_sample_output():
    """Print a sample serialized output for visual inspection."""
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    graph = create_test_4dsg()
    config = Phase7SerializationConfig()

    result = serialize_4dsg_json_strict(graph, config)
    print(json.dumps(result, indent=2))


def main():
    """Run all tests."""
    print("\n" + "🔬" * 30)
    print("Phase 7 Serialization Test Suite")
    print("🔬" * 30 + "\n")

    results = []

    # Original tests
    results.append(("Determinism", test_determinism()))
    results.append(("Sorting", test_sorting()))
    results.append(("Precision", test_precision()))
    results.append(("Field Order", test_field_order()))

    # Critical fix validation tests
    results.append(("Frame Indices Consistency", test_frame_indices_consistency()))
    results.append(("Frame Indices Propagation", test_frame_indices_propagation()))

    print_sample_output()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
