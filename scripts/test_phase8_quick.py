#!/usr/bin/env python3
"""Phase 8 Quick Smoke Test.

Minimal test to quickly verify Phase 8 pipeline is working.
This is useful for rapid iteration and debugging.

Usage:
    python scripts/test_phase8_quick.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_phase8_end2end import (
    create_mock_4dsg,
    create_mock_step_token,
    get_test_scenarios,
    MockVLM,
)
from fast_snow.reasoning.vlm.prompt_builder import serialize_4dsg_strict, Phase7SerializationConfig
from fast_snow.reasoning.eval.answer_postprocess import extract_multiple_choice_answer
import json


def test_mock_data_generation():
    """Quick test: Mock data generation."""
    print("=" * 60)
    print("TEST 1: Mock Data Generation")
    print("=" * 60)

    try:
        four_dsg, tracks = create_mock_4dsg(num_objects=3, num_frames=10, seed=42)

        assert len(tracks) == 3, f"Expected 3 tracks, got {len(tracks)}"
        assert len(four_dsg.temporal_window.tracks) == 3, f"Expected 3 temporal tracks, got {len(four_dsg.temporal_window.tracks)}"

        # Check frame indices
        for track_id, track in tracks.items():
            assert track.frame_indices == list(range(10)), f"Track {track_id} frame indices incorrect"

        for track_id, temporal_track in four_dsg.temporal_window.tracks.items():
            assert temporal_track.frame_indices is not None, f"Temporal track {track_id} missing frame_indices"
            assert temporal_track.frame_indices == list(range(10)), f"Temporal track {track_id} frame indices incorrect"

        # Check spatial graphs contain edges/relations (non-trivial structure)
        graphs_with_edges = [g for g in four_dsg.spatial_graphs if len(g.edges) > 0]
        assert graphs_with_edges, "Expected at least one spatial graph with edges"
        assert any(
            edge.relation is not None
            for g in graphs_with_edges
            for edge in g.edges
        ), "Expected at least one semantic relation on edges"

        print("✓ Mock 4DSG created successfully")
        print(f"  - {len(tracks)} tracks")
        print(f"  - {len(four_dsg.temporal_window.tracks)} temporal tracks")
        print(f"  - {len(four_dsg.ego_poses)} ego poses")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase7_serialization():
    """Quick test: Phase 7 strict serialization."""
    print("\n" + "=" * 60)
    print("TEST 2: Phase 7 Strict Serialization")
    print("=" * 60)

    try:
        four_dsg, _ = create_mock_4dsg(num_objects=3, num_frames=10, seed=42)
        config = Phase7SerializationConfig()
        scene_json = serialize_4dsg_strict(four_dsg, config)

        # Validate JSON
        scene_dict = json.loads(scene_json)

        # Check required fields
        assert "metadata" in scene_dict, "Missing 'metadata'"
        assert "ego_agent" in scene_dict, "Missing 'ego_agent'"
        assert "objects" in scene_dict, "Missing 'objects'"
        assert "spatial_relations" in scene_dict, "Missing 'spatial_relations'"

        # Check metadata
        metadata = scene_dict["metadata"]
        assert metadata["num_frames"] == 10, f"num_frames should be 10, got {metadata['num_frames']}"
        assert metadata["temporal_window"] == 10, f"temporal_window should be 10, got {metadata['temporal_window']}"

        # Check objects have correct structure
        for obj in scene_dict["objects"]:
            assert "object_id" in obj, "Object missing 'object_id'"
            assert "track" in obj, "Object missing 'track'"

            for step in obj["track"]:
                assert "frame" in step, "Step missing 'frame'"
                assert isinstance(step["frame"], int), f"frame should be int, got {type(step['frame'])}"
                assert "centroid" in step, "Step missing 'centroid'"
                assert len(step["centroid"]) == 3, f"centroid should have 3 values, got {len(step['centroid'])}"

        print("✓ Serialization successful")
        print(f"  - JSON size: {len(scene_json)} chars")
        print(f"  - Objects: {len(scene_dict['objects'])}")
        print(f"  - Spatial relations: {len(scene_dict['spatial_relations'])}")

        # Test determinism
        scene_json_2 = serialize_4dsg_strict(four_dsg, config)
        assert scene_json == scene_json_2, "Serialization not deterministic!"
        print("✓ Determinism verified")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_extraction():
    """Quick test: Answer extraction."""
    print("\n" + "=" * 60)
    print("TEST 3: Answer Extraction")
    print("=" * 60)

    choices = {"A": "0", "B": "1", "C": "2", "D": "3"}

    test_cases = [
        ("A", "A"),
        ("The answer is B", "B"),
        ("C", "C"),
        ("(D)", "D"),
    ]

    try:
        for response, expected in test_cases:
            result = extract_multiple_choice_answer(response, choices)
            assert result == expected, f"Expected '{expected}', got '{result}' for '{response}'"
            print(f"✓ '{response}' → '{result}'")

        print("✓ All extractions correct")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_mock_vlm():
    """Quick test: Mock VLM."""
    print("\n" + "=" * 60)
    print("TEST 4: Mock VLM")
    print("=" * 60)

    try:
        vlm = MockVLM(accuracy=0.75, response_format="direct", seed=42)

        prompt = """
Question: How many cars?
A. 0
B. 1
C. 2
D. 3
"""

        response = vlm.infer_text(prompt)
        print(f"✓ VLM response: '{response}'")

        assert response in ["A", "B", "C", "D"], f"Invalid response: {response}"
        print(f"✓ Response is valid choice")

        # Test multiple calls with same seed should be consistent
        vlm2 = MockVLM(accuracy=0.75, response_format="direct", seed=42)
        response2 = vlm2.infer_text(prompt)
        assert response == response2, "Non-deterministic VLM!"
        print(f"✓ Determinism verified")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_single():
    """Quick test: Single end-to-end scenario."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Single Scenario")
    print("=" * 60)

    try:
        # Get first scenario
        scenario = get_test_scenarios()[0]
        print(f"Scenario: {scenario.name}")
        print(f"Question: {scenario.question}")

        # Create 4DSG
        four_dsg, _ = create_mock_4dsg(
            num_objects=scenario.num_objects,
            num_frames=scenario.num_frames,
            seed=42,
        )
        print(f"✓ Created 4DSG")

        # Serialize
        config = Phase7SerializationConfig()
        scene_json = serialize_4dsg_strict(four_dsg, config)
        print(f"✓ Serialized ({len(scene_json)} chars)")

        # Build prompt
        from fast_snow.reasoning.eval.answer_postprocess import format_question_with_choices
        question_formatted = format_question_with_choices(
            question=scenario.question,
            choices=scenario.choices,
            include_instruction=True,
        )

        prompt = f"""You are a spatial reasoning assistant.

Scene: {scene_json[:200]}...

Question: {question_formatted}"""

        print(f"✓ Built prompt ({len(prompt)} chars)")

        # VLM inference
        vlm = MockVLM(accuracy=1.0, response_format="direct", seed=42)
        response = vlm.infer_text(prompt)
        print(f"✓ VLM response: '{response}'")

        # Extract answer
        prediction = extract_multiple_choice_answer(response, scenario.choices)
        print(f"✓ Extracted: '{prediction}'")

        # Note: We can't check correctness since mock VLM doesn't understand the scene
        # But we can verify the pipeline runs without errors
        print(f"  Ground truth: '{scenario.answer}'")
        print(f"  Prediction: '{prediction}'")

        print("✓ End-to-end pipeline completed successfully")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all quick tests."""
    print("\n" + "🚀" * 30)
    print("Phase 8 Quick Smoke Test")
    print("🚀" * 30 + "\n")

    tests = [
        ("Mock Data Generation", test_mock_data_generation),
        ("Phase 7 Serialization", test_phase7_serialization),
        ("Answer Extraction", test_answer_extraction),
        ("Mock VLM", test_mock_vlm),
        ("End-to-End Single", test_end_to_end_single),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)

    if all_passed:
        print("🎉 All quick tests PASSED!")
        print("\nPhase 8 pipeline basics are working correctly.")
        print("\nNext step: Run full tests:")
        print("  python scripts/test_phase8_end2end.py")
        return 0
    else:
        print("⚠️  Some tests FAILED")
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
