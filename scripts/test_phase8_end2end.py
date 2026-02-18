#!/usr/bin/env python3
"""Phase 8 End-to-End Integration Test (No SAM2 Required).

This script provides a complete integration test of the Phase 8 evaluation
pipeline using mock data. It validates:

1. **Mock Data Generation**: Creating realistic point clouds, clusters, tracks
2. **4DSG Construction**: Building complete 4D scene graphs from mock data
3. **Phase 7 Serialization**: JSON serialization with strict mode
4. **VLM Simulation**: Mock VLM responses with configurable behavior
5. **Answer Extraction**: Testing all answer extraction strategies
6. **Metrics Computation**: Full evaluation metrics calculation
7. **Error Handling**: Graceful degradation and error reporting

Usage:
    # Run full end-to-end test
    python scripts/test_phase8_end2end.py

    # Run with verbose output
    python scripts/test_phase8_end2end.py --verbose

    # Test specific scenario
    python scripts/test_phase8_end2end.py --scenario ego_centric
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_snow.reasoning.tokens.step_encoding import STEPToken
from fast_snow.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken
from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken
# Note: We don't need to import scene_graph directly since we use higher-level APIs
from fast_snow.reasoning.graph.object_tracker import Track, ObjectTracker, TrackerConfig
from fast_snow.reasoning.graph.temporal_linking import TemporalTrack, build_temporal_window_from_tracker
from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph, build_four_d_scene_graph
from fast_snow.reasoning.vlm.prompt_builder import (
    serialize_4dsg_strict,
    Phase7SerializationConfig,
)
from fast_snow.reasoning.eval.answer_postprocess import (
    extract_multiple_choice_answer,
    format_question_with_choices,
)
from fast_snow.reasoning.eval.metrics import Phase8Metrics, EvaluationResult, compare_with_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Mock Data Generation
# =============================================================================

def create_mock_point_cloud(
    num_points: int = 1000,
    scene_type: str = "driving",
    seed: int = 42,
) -> np.ndarray:
    """Create realistic mock point cloud data.

    Args:
        num_points: Number of points
        scene_type: Type of scene ("driving", "indoor", "outdoor")
        seed: Random seed

    Returns:
        Point cloud array of shape (num_points, 3)
    """
    rng = np.random.default_rng(seed)

    if scene_type == "driving":
        # Driving scene: Road plane + cars + pedestrians
        points = []

        # Road plane (y ≈ 0, wide x range)
        road = rng.normal(loc=[0, 0, 0], scale=[10, 0.1, 5], size=(num_points // 2, 3))
        points.append(road)

        # Car 1: Front left
        car1 = rng.normal(loc=[5, 1.5, 10], scale=[0.8, 0.6, 2.0], size=(num_points // 6, 3))
        points.append(car1)

        # Car 2: Front right
        car2 = rng.normal(loc=[-5, 1.5, 12], scale=[0.8, 0.6, 2.0], size=(num_points // 6, 3))
        points.append(car2)

        # Pedestrian
        ped = rng.normal(loc=[3, 0.8, 8], scale=[0.3, 0.4, 0.3], size=(num_points // 6, 3))
        points.append(ped)

        return np.vstack(points)

    elif scene_type == "indoor":
        # Indoor scene: Floor + furniture
        return rng.normal(loc=[0, 0, 0], scale=[3, 2, 3], size=(num_points, 3))

    else:
        # Generic outdoor scene
        return rng.normal(loc=[0, 0, 0], scale=[5, 3, 5], size=(num_points, 3))


def create_mock_step_token(
    object_id: int,
    frame_idx: int,
    centroid: np.ndarray,
    extent: np.ndarray,
    num_patches: int = 24,
) -> STEPToken:
    """Create a mock STEP token.

    Args:
        object_id: Object ID
        frame_idx: Frame index
        centroid: 3D centroid position
        extent: 3D bounding box extent
        num_patches: Number of patches

    Returns:
        STEPToken instance
    """
    # Create centroid token
    centroid_token = CentroidToken(
        x=float(centroid[0]),
        y=float(centroid[1]),
        z=float(centroid[2]),
    )

    # Create shape token (use extent to approximate shape)
    shape_token = ShapeToken(
        x_mu=float(centroid[0]),
        x_sigma=float(extent[0] / 6.0),  # Approximate sigma from extent
        x_min=float(centroid[0] - extent[0] / 2),
        x_max=float(centroid[0] + extent[0] / 2),
        y_mu=float(centroid[1]),
        y_sigma=float(extent[1] / 6.0),
        y_min=float(centroid[1] - extent[1] / 2),
        y_max=float(centroid[1] + extent[1] / 2),
        z_mu=float(centroid[2]),
        z_sigma=float(extent[2] / 6.0),
        z_min=float(centroid[2] - extent[2] / 2),
        z_max=float(centroid[2] + extent[2] / 2),
    )

    # Create mock patch tokens (simplified)
    patch_tokens = [
        PatchToken(row=i % 4, col=i // 4, iou=0.8)
        for i in range(min(num_patches, 16))  # Max 4x4 grid
    ]

    # Create temporal token
    temporal_token = TemporalToken(
        t_start=frame_idx,
        t_end=frame_idx,
    )

    return STEPToken(
        patch_tokens=patch_tokens,
        centroid=centroid_token,
        shape=shape_token,
        temporal=temporal_token,
    )


def create_mock_track(
    track_id: int,
    num_frames: int = 10,
    motion_type: str = "static",
    seed: int = 42,
) -> Track:
    """Create a mock object track.

    Args:
        track_id: Unique track ID
        num_frames: Number of frames in track
        motion_type: Motion type ("static", "forward", "left", "right")
        seed: Random seed

    Returns:
        Track instance
    """
    rng = np.random.default_rng(seed + track_id)

    # Initial position
    if motion_type == "static":
        base_pos = rng.uniform([-5, 0.5, 5], [5, 2, 15])
    elif motion_type == "forward":
        base_pos = rng.uniform([-3, 1, 8], [3, 2, 12])
    elif motion_type == "left":
        base_pos = rng.uniform([3, 1, 10], [6, 2, 15])
    elif motion_type == "right":
        base_pos = rng.uniform([-6, 1, 10], [-3, 2, 15])
    else:
        base_pos = rng.uniform([-5, 0.5, 5], [5, 2, 15])

    # Motion delta per frame
    if motion_type == "static":
        delta = np.array([0, 0, 0])
    elif motion_type == "forward":
        delta = np.array([0, 0, 0.5])  # Moving forward
    elif motion_type == "left":
        delta = np.array([-0.3, 0, 0.2])  # Moving left and forward
    elif motion_type == "right":
        delta = np.array([0.3, 0, 0.2])  # Moving right and forward
    else:
        delta = np.array([0, 0, 0])

    # Generate steps
    steps = []
    frame_indices = []

    for i in range(num_frames):
        centroid = base_pos + delta * i + rng.normal(0, 0.05, 3)  # Add noise
        extent = rng.uniform([1.5, 1.4, 4.0], [2.0, 1.8, 4.8])  # Car-like size

        step = create_mock_step_token(
            object_id=track_id,
            frame_idx=i,
            centroid=centroid,
            extent=extent,
            num_patches=24,
        )

        steps.append(step)
        frame_indices.append(i)

    return Track(
        track_id=track_id,
        steps=steps,
        frame_indices=frame_indices,
    )


def create_mock_4dsg(
    num_objects: int = 3,
    num_frames: int = 10,
    scene_type: str = "driving",
    seed: int = 42,
) -> Tuple[FourDSceneGraph, Dict[int, Track]]:
    """Create a complete mock 4DSG.

    Args:
        num_objects: Number of tracked objects
        num_frames: Number of frames
        scene_type: Scene type
        seed: Random seed

    Returns:
        Tuple of (FourDSceneGraph, tracks dict)
    """
    logger.info(f"Creating mock 4DSG: {num_objects} objects, {num_frames} frames")

    # Create mock tracks
    tracks = {}
    motion_types = ["static", "forward", "left", "right"]

    for obj_id in range(num_objects):
        motion = motion_types[obj_id % len(motion_types)]
        track = create_mock_track(
            track_id=obj_id,
            num_frames=num_frames,
            motion_type=motion,
            seed=seed + obj_id,
        )
        tracks[obj_id] = track

    # Build temporal window from tracker
    # First, we need to create a mock ObjectTracker
    tracker_config = TrackerConfig()
    tracker = ObjectTracker(config=tracker_config)

    # Add tracks to tracker (simulate tracking result)
    for track_id, track in tracks.items():
        # ObjectTracker expects tracks to be built frame-by-frame
        # For mock, we'll directly populate its internal state
        tracker.state.tracks[track_id] = track

    # Build temporal window
    temporal_window = build_temporal_window_from_tracker(tracker.state.tracks)

    # Create mock ego poses (Dict[int, List[float]])
    ego_poses = {}
    for i in range(num_frames):
        # Ego moves forward at 2 m/s (simplified to position only)
        ego_poses[i] = [0.0, 0.0, float(i * 2.0)]  # [x, y, z] position

    # Create mock spatial graphs (one per frame) using real node/edge logic
    # This exercises spatial relation generation in build_scene_graph.
    from fast_snow.reasoning.graph.scene_graph import build_scene_graph, SceneGraphConfig

    track_frame_maps: Dict[int, Dict[int, STEPToken]] = {}
    for track_id, track in tracks.items():
        track_frame_maps[track_id] = dict(zip(track.frame_indices, track.steps))

    spatial_graphs = []
    sg_config = SceneGraphConfig()  # default config (relations enabled)
    for i in range(num_frames):
        frame_steps: Dict[int, STEPToken] = {}
        for track_id, frame_map in track_frame_maps.items():
            if i in frame_map:
                frame_steps[track_id] = frame_map[i]

        sg = build_scene_graph(
            steps=frame_steps,
            frame_idx=i,
            ego_pose=None,
            config=sg_config,
        )
        spatial_graphs.append(sg)

    # Build 4DSG (direct construction with mock spatial graphs)
    four_dsg = FourDSceneGraph(
        spatial_graphs=spatial_graphs,  # Mock spatial graphs for num_frames
        temporal_window=temporal_window,
        ego_poses=ego_poses,
    )

    logger.info(f"Created 4DSG with {len(four_dsg.temporal_window.tracks)} objects")

    return four_dsg, tracks


# =============================================================================
# Mock VLM Interface
# =============================================================================

class MockVLM:
    """Mock VLM interface for testing without real API calls.

    This simulates VLM behavior with configurable response patterns.
    """

    def __init__(
        self,
        accuracy: float = 0.75,
        response_format: str = "direct",
        seed: int = 42,
    ):
        """Initialize mock VLM.

        Args:
            accuracy: Probability of correct answer (0-1)
            response_format: Format of response ("direct", "verbose", "noisy")
            seed: Random seed for determinism
        """
        self.accuracy = accuracy
        self.response_format = response_format
        self.rng = np.random.default_rng(seed)
        self.call_count = 0

    def infer_text(self, prompt: str) -> str:
        """Mock text inference.

        Args:
            prompt: Input prompt (contains question)

        Returns:
            Mock VLM response
        """
        self.call_count += 1

        # Extract ground truth from prompt if present
        # In real eval, we don't have ground truth in prompt
        # For testing, we'll randomly pick an answer

        # Parse choices from prompt
        choices = self._extract_choices_from_prompt(prompt)

        if not choices:
            return "Unable to parse question"

        # Determine if this should be correct
        is_correct = self.rng.random() < self.accuracy

        if is_correct:
            # Return first choice (assume it's correct for mock)
            answer = list(choices.keys())[0]
        else:
            # Return random incorrect answer
            answer = self.rng.choice(list(choices.keys()))

        # Format response based on response_format
        if self.response_format == "direct":
            return answer
        elif self.response_format == "verbose":
            return f"Based on the scene analysis, I believe the answer is {answer}. This conclusion is drawn from..."
        elif self.response_format == "noisy":
            # Add some noise to test extraction robustness
            prefixes = [
                f"Let me analyze this carefully... The answer is {answer}",
                f"After considering the spatial relationships, I choose {answer}",
                f"({answer})",
                f"Answer: {answer}",
                f"The correct option is {answer}.",
            ]
            return self.rng.choice(prefixes)
        else:
            return answer

    def _extract_choices_from_prompt(self, prompt: str) -> Dict[str, str]:
        """Extract choices from prompt."""
        # Simple extraction - look for "A. ...", "B. ...", etc.
        import re
        choices = {}
        pattern = r"([A-D])\.\s+([^\n]+)"
        for match in re.finditer(pattern, prompt):
            letter, text = match.groups()
            choices[letter] = text.strip()
        return choices


# =============================================================================
# Test Scenarios
# =============================================================================

@dataclass
class TestScenario:
    """A test scenario definition."""
    name: str
    question_type: str
    question: str
    choices: Dict[str, str]
    answer: str
    num_objects: int = 3
    num_frames: int = 10


def get_test_scenarios() -> List[TestScenario]:
    """Get predefined test scenarios."""
    return [
        TestScenario(
            name="ego_centric_car_count",
            question_type="ego-centric",
            question="How many cars are in front of the ego vehicle?",
            choices={"A": "0", "B": "1", "C": "2", "D": "3"},
            answer="C",
            num_objects=3,
        ),
        TestScenario(
            name="exo_centric_position",
            question_type="exo-centric",
            question="What is the relative position of car A to car B?",
            choices={"A": "Left", "B": "Right", "C": "Front", "D": "Behind"},
            answer="A",
            num_objects=4,
        ),
        TestScenario(
            name="directional_motion",
            question_type="directional",
            question="Which direction is the vehicle moving?",
            choices={"A": "Forward", "B": "Backward", "C": "Left", "D": "Right"},
            answer="A",
            num_objects=2,
        ),
        TestScenario(
            name="false_positive_check",
            question_type="false_positive",
            question="Is there a bicycle behind the ego vehicle?",
            choices={"A": "Yes", "B": "No", "C": "Cannot determine", "D": "Multiple"},
            answer="B",
            num_objects=3,
        ),
    ]


# =============================================================================
# End-to-End Test Pipeline
# =============================================================================

def run_single_scenario_test(
    scenario: TestScenario,
    vlm: MockVLM,
    seed: int = 42,
) -> EvaluationResult:
    """Run end-to-end test for a single scenario.

    Args:
        scenario: Test scenario
        vlm: Mock VLM interface
        seed: Random seed

    Returns:
        EvaluationResult
    """
    logger.info(f"Testing scenario: {scenario.name}")

    # Step 1: Create mock 4DSG
    four_dsg, tracks = create_mock_4dsg(
        num_objects=scenario.num_objects,
        num_frames=scenario.num_frames,
        seed=seed,
    )

    # Step 2: Serialize with Phase 7 strict mode
    config = Phase7SerializationConfig()
    scene_json = serialize_4dsg_strict(four_dsg, config)

    logger.info(f"Serialized 4DSG: {len(scene_json)} characters")

    # Verify JSON is valid
    try:
        scene_dict = json.loads(scene_json)
        logger.info(f"JSON valid: {len(scene_dict.get('objects', []))} objects")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return EvaluationResult(
            sample_id=scenario.name,
            category=scenario.question_type,
            question=scenario.question,
            prediction="",
            ground_truth=scenario.answer,
            is_correct=False,
            raw_response=f"JSON Error: {e}",
        )

    # Step 3: Format question
    question_formatted = format_question_with_choices(
        question=scenario.question,
        choices=scenario.choices,
        include_instruction=True,
    )

    # Step 4: Build full prompt
    full_prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

Scene Data (JSON):
{scene_json}

Question: {question_formatted}"""

    # Step 5: VLM inference (mock)
    response = vlm.infer_text(full_prompt)
    logger.info(f"VLM response: '{response}'")

    # Step 6: Extract answer
    prediction = extract_multiple_choice_answer(
        response=response,
        choices=scenario.choices,
    )
    logger.info(f"Extracted prediction: '{prediction}'")

    # Step 7: Check correctness
    is_correct = prediction.upper() == scenario.answer.upper()

    result = EvaluationResult(
        sample_id=scenario.name,
        category=scenario.question_type,
        question=scenario.question,
        prediction=prediction,
        ground_truth=scenario.answer,
        is_correct=is_correct,
        raw_response=response,
    )

    logger.info(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    return result


def run_full_pipeline_test(
    scenarios: List[TestScenario],
    vlm_accuracy: float = 0.75,
    vlm_response_format: str = "noisy",
    seed: int = 42,
) -> Phase8Metrics:
    """Run full end-to-end pipeline test.

    Args:
        scenarios: List of test scenarios
        vlm_accuracy: Mock VLM accuracy
        vlm_response_format: Mock VLM response format
        seed: Random seed

    Returns:
        Phase8Metrics with results
    """
    logger.info("=" * 60)
    logger.info("Phase 8 End-to-End Pipeline Test")
    logger.info("=" * 60)
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"VLM accuracy: {vlm_accuracy:.2%}")
    logger.info(f"VLM format: {vlm_response_format}")
    logger.info("=" * 60)

    # Create mock VLM
    vlm = MockVLM(
        accuracy=vlm_accuracy,
        response_format=vlm_response_format,
        seed=seed,
    )

    # Initialize metrics
    metrics = Phase8Metrics()
    metrics.config = {
        "model": "MockVLM",
        "accuracy": vlm_accuracy,
        "response_format": vlm_response_format,
        "seed": seed,
    }
    metrics.environment = {
        "test_type": "end-to-end",
        "mock_data": True,
    }

    # Run each scenario
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n[{i}/{len(scenarios)}] Testing: {scenario.name}")

        result = run_single_scenario_test(scenario, vlm, seed=seed + i)
        metrics.add_result(result)

    # Compute metrics
    metrics.compute_metrics()

    logger.info("\n" + "=" * 60)
    logger.info(f"Mock VLM called {vlm.call_count} times")
    logger.info("=" * 60)

    return metrics


# =============================================================================
# Component-Level Tests
# =============================================================================

def test_serialization_determinism(num_trials: int = 5) -> bool:
    """Test that serialization is deterministic.

    Args:
        num_trials: Number of trials to run

    Returns:
        True if all trials produce identical output
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Serialization Determinism")
    logger.info("=" * 60)

    outputs = []

    for i in range(num_trials):
        four_dsg, _ = create_mock_4dsg(num_objects=3, num_frames=10, seed=42)
        config = Phase7SerializationConfig()
        output = serialize_4dsg_strict(four_dsg, config)
        outputs.append(output)
        logger.info(f"Trial {i+1}: {len(output)} chars, hash={hash(output)}")

    # Check all outputs are identical
    all_same = all(out == outputs[0] for out in outputs)

    if all_same:
        logger.info("✓ PASS: All outputs identical (deterministic)")
    else:
        logger.error("✗ FAIL: Outputs differ (non-deterministic)")
        # Show differences
        for i, out in enumerate(outputs[1:], 1):
            if out != outputs[0]:
                logger.error(f"  Trial {i+1} differs from trial 1")

    return all_same


def test_frame_indices_preservation() -> bool:
    """Test that frame indices are correctly preserved through the pipeline.

    Returns:
        True if frame indices are preserved correctly
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Frame Indices Preservation")
    logger.info("=" * 60)

    # Create 4DSG with known frame indices
    four_dsg, tracks = create_mock_4dsg(num_objects=2, num_frames=10, seed=42)

    # Check frame indices in tracks
    for track_id, track in tracks.items():
        expected_frames = list(range(10))
        actual_frames = track.frame_indices

        if actual_frames != expected_frames:
            logger.error(f"✗ Track {track_id}: frame_indices mismatch")
            logger.error(f"  Expected: {expected_frames}")
            logger.error(f"  Actual: {actual_frames}")
            return False

    # Check frame indices in temporal tracks
    for track_id, temporal_track in four_dsg.temporal_window.tracks.items():
        if temporal_track.frame_indices is None:
            logger.error(f"✗ Temporal track {track_id}: frame_indices is None")
            return False

        expected_frames = list(range(10))
        actual_frames = temporal_track.frame_indices

        if actual_frames != expected_frames:
            logger.error(f"✗ Temporal track {track_id}: frame_indices mismatch")
            logger.error(f"  Expected: {expected_frames}")
            logger.error(f"  Actual: {actual_frames}")
            return False

    # Serialize and check JSON
    config = Phase7SerializationConfig()
    scene_json = serialize_4dsg_strict(four_dsg, config)
    scene_dict = json.loads(scene_json)

    # Check frame indices in JSON
    for obj in scene_dict.get("objects", []):
        for step in obj.get("track", []):
            frame = step.get("frame")
            if frame is None:
                logger.error("✗ JSON: 'frame' field missing in step")
                return False
            if not isinstance(frame, int):
                logger.error(f"✗ JSON: 'frame' is not int: {type(frame)}")
                return False
            if frame < 0 or frame >= 10:
                logger.error(f"✗ JSON: 'frame' out of range: {frame}")
                return False

    logger.info("✓ PASS: Frame indices correctly preserved")
    return True


def test_scene_graph_relations() -> bool:
    """Test that spatial graphs contain edges with semantic relations."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Scene Graph Relations")
    logger.info("=" * 60)

    try:
        four_dsg, _ = create_mock_4dsg(num_objects=3, num_frames=3, seed=42)

        # At least one frame should have edges if there are multiple objects
        graphs_with_edges = [g for g in four_dsg.spatial_graphs if len(g.edges) > 0]
        if not graphs_with_edges:
            logger.error("✗ FAIL: No edges found in any spatial graph")
            return False

        # Ensure at least one edge has a semantic relation
        has_relation = False
        for g in graphs_with_edges:
            for edge in g.edges:
                if edge.relation is not None:
                    has_relation = True
                    break
            if has_relation:
                break

        if not has_relation:
            logger.error("✗ FAIL: No semantic relations found on edges")
            return False

        logger.info("✓ PASS: Spatial graph edges include semantic relations")
        return True
    except Exception as e:
        logger.error(f"✗ FAIL: Scene graph relations test failed: {e}")
        return False


def test_answer_extraction_robustness() -> bool:
    """Test answer extraction with various response formats.

    Returns:
        True if all extractions are correct
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Answer Extraction Robustness")
    logger.info("=" * 60)

    choices = {"A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4"}

    test_cases = [
        ("A", "A", "Direct letter"),
        ("B", "B", "Direct letter B"),
        ("The answer is C", "C", "Answer is C"),
        ("I believe D is correct", "D", "Belief D"),
        ("(A)", "A", "Parenthesis"),
        ("[B]", "B", "Bracket"),
        ("Let me think... C", "C", "Trailing C"),
        ("Answer: D", "D", "Answer: D"),
        ("invalid text", "", "No answer"),
    ]

    all_passed = True

    for response, expected, description in test_cases:
        result = extract_multiple_choice_answer(response, choices)
        if result == expected:
            logger.info(f"✓ {description}: '{response}' → '{result}'")
        else:
            logger.error(f"✗ {description}: '{response}' → '{result}' (expected '{expected}')")
            all_passed = False

    if all_passed:
        logger.info("✓ PASS: All extractions correct")
    else:
        logger.error("✗ FAIL: Some extractions incorrect")

    return all_passed


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 End-to-End Integration Test (No SAM2 Required)"
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Test specific scenario only"
    )
    parser.add_argument(
        "--vlm-accuracy",
        type=float,
        default=0.75,
        help="Mock VLM accuracy (0-1)"
    )
    parser.add_argument(
        "--vlm-format",
        default="noisy",
        choices=["direct", "verbose", "noisy"],
        help="Mock VLM response format"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-component-tests",
        action="store_true",
        help="Skip component-level tests"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "🧪" * 30)
    print("Phase 8 End-to-End Integration Test")
    print("(No SAM2 Required - Mock Data)")
    print("🧪" * 30 + "\n")

    all_tests_passed = True

    # Component-level tests
    if not args.skip_component_tests:
        print("\n" + "=" * 60)
        print("COMPONENT-LEVEL TESTS")
        print("=" * 60)

        tests = [
            ("Serialization Determinism", test_serialization_determinism),
            ("Frame Indices Preservation", test_frame_indices_preservation),
            ("Scene Graph Relations", test_scene_graph_relations),
            ("Answer Extraction Robustness", test_answer_extraction_robustness),
        ]

        for test_name, test_func in tests:
            try:
                passed = test_func()
                if not passed:
                    all_tests_passed = False
            except Exception as e:
                logger.error(f"✗ {test_name} raised exception: {e}")
                all_tests_passed = False

    # Full pipeline test
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    scenarios = get_test_scenarios()

    # Filter by scenario if specified
    if args.scenario:
        target = args.scenario.lower().replace("_", "-")
        scenarios = [
            s for s in scenarios
            if s.name.lower().replace("_", "-") == target
            or s.question_type.lower().replace("_", "-") == target
        ]
        if not scenarios:
            logger.error(f"No scenarios found matching: {args.scenario}")
            return 1

    try:
        metrics = run_full_pipeline_test(
            scenarios=scenarios,
            vlm_accuracy=args.vlm_accuracy,
            vlm_response_format=args.vlm_format,
            seed=args.seed,
        )

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(metrics)
        print()
        print("=" * 60)
        print(compare_with_baseline(metrics))
        print("=" * 60)

        # Save results if requested
        if args.output:
            metrics.save_json(args.output, include_results=True)
            logger.info(f"Results saved to {args.output}")

        # Check if results match expected accuracy
        expected_accuracy = args.vlm_accuracy
        actual_accuracy = metrics.overall_accuracy
        tolerance = 0.2  # Allow 20% deviation due to randomness

        if abs(actual_accuracy - expected_accuracy) > tolerance:
            logger.warning(f"Accuracy deviation: expected ≈{expected_accuracy:.2%}, got {actual_accuracy:.2%}")

    except Exception as e:
        logger.error(f"Pipeline test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_tests_passed:
        print("✓ All tests PASSED")
        print("\nPhase 8 pipeline is working correctly with mock data!")
        print("Next steps:")
        print("  1. Test with real VLM API (set GOOGLE_AI_API_KEY)")
        print("  2. Test with real VLM4D dataset")
        print("  3. Run full evaluation: bash scripts/run_phase8_eval.sh")
        return 0
    else:
        print("✗ Some tests FAILED")
        print("\nPlease review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
