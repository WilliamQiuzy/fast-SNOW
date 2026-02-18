#!/usr/bin/env python3
"""VLM4D Evaluation Runner Script.

This script runs VLM4D evaluation using SNOW pipeline with various VLM backends.

Usage:
    # Using Google AI Studio (free, recommended)
    export GOOGLE_AI_API_KEY=your_api_key
    python scripts/run_vlm4d_eval.py --backend google_ai

    # Using HuggingFace Inference API
    export HF_TOKEN=your_token
    python scripts/run_vlm4d_eval.py --backend huggingface

    # Using local vLLM server
    python scripts/run_vlm4d_eval.py --backend openai --api-base http://localhost:8000/v1

    # Test mode (no VLM, just pipeline test)
    python scripts/run_vlm4d_eval.py --test-mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_snow.reasoning.vlm.inference import VLMConfig, VLMInterface, create_vlm_interface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_vlm_connection(vlm: VLMInterface) -> bool:
    """Test VLM connection with a simple query."""
    logger.info("Testing VLM connection...")
    try:
        response = vlm.infer_text("What is 2 + 2? Answer with just the number.")
        logger.info(f"VLM response: {response}")
        return True
    except Exception as e:
        logger.error(f"VLM connection test failed: {e}")
        return False


def create_sample_vlm4d_data() -> Dict:
    """Create sample VLM4D data for testing."""
    return [
        {
            "id": "test_001",
            "video": "http://example.com/video1.mp4",
            "question_type": "ego-centric",
            "question": "How many cars are in front of the ego vehicle?",
            "choices": {"A": "0", "B": "1", "C": "2", "D": "3"},
            "answer": "B"
        },
        {
            "id": "test_002",
            "video": "http://example.com/video2.mp4",
            "question_type": "directional",
            "question": "Which direction is the pedestrian walking?",
            "choices": {"A": "Left", "B": "Right", "C": "Forward", "D": "Backward"},
            "answer": "A"
        },
        {
            "id": "test_003",
            "video": "http://example.com/video3.mp4",
            "question_type": "exo-centric",
            "question": "What is the relative position of the bicycle to the car?",
            "choices": {"A": "In front", "B": "Behind", "C": "Left", "D": "Right"},
            "answer": "C"
        },
    ]


def run_simple_eval(
    vlm: VLMInterface,
    samples: list,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run a simple evaluation without video processing.

    This tests the VLM's ability to answer questions given a text description
    of a scene (simulating what SNOW would provide via 4DSG serialization).
    """
    from fast_snow.reasoning.eval.vlm4d import VLM4DMetrics, extract_answer_from_response, check_answer

    metrics = VLM4DMetrics()

    if max_samples:
        samples = samples[:max_samples]

    for i, sample in enumerate(samples):
        # Create a mock scene description (in real eval, this comes from 4DSG)
        scene_context = """
=== Scene Description ===
Frame 0-5, 5 objects tracked

Object ID 0: Car
  - Position: (5.2, 10.3, 0.5)
  - Velocity: (0.5, 0.2, 0.0) m/s
  - Relation to ego: in_front, 10.3m away

Object ID 1: Pedestrian
  - Position: (-2.1, 8.5, 0.0)
  - Velocity: (-0.3, 0.0, 0.0) m/s
  - Relation to ego: front_left, 8.8m away

Object ID 2: Bicycle
  - Position: (-5.0, 3.2, 0.0)
  - Velocity: (0.0, 0.5, 0.0) m/s
  - Relation to ego: left, 5.9m away
"""

        # Build question with choices
        question_text = f"{scene_context}\n\nQuestion: {sample['question']}\n"
        for key, text in sample['choices'].items():
            question_text += f"{key}. {text}\n"
        question_text += "\nAnswer with just the letter (A, B, C, or D):"

        try:
            response = vlm.infer_text(question_text)
            prediction = extract_answer_from_response(response, sample['choices'])
            is_correct = check_answer(prediction, sample['answer'])

            metrics.add_prediction(
                sample_id=sample['id'],
                category=sample['question_type'],
                prediction=prediction,
                ground_truth=sample['answer'],
                is_correct=is_correct,
            )

            logger.info(
                f"Sample {i+1}/{len(samples)}: "
                f"pred={prediction}, gt={sample['answer']}, "
                f"correct={is_correct}"
            )

        except Exception as e:
            logger.error(f"Error on sample {sample['id']}: {e}")
            metrics.add_prediction(
                sample_id=sample['id'],
                category=sample['question_type'],
                prediction="",
                ground_truth=sample['answer'],
                is_correct=False,
            )

    metrics.compute_final_metrics()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run VLM4D evaluation")
    parser.add_argument(
        "--backend",
        choices=["pipeline", "model", "google_ai", "huggingface", "openai"],
        default="google_ai",
        help="VLM backend to use"
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model ID (defaults based on backend)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (will use env vars if not provided)"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL (for openai backend)"
    )
    parser.add_argument(
        "--vlm4d-json",
        type=Path,
        default=None,
        help="Path to VLM4D JSON file (real_mc.json or synthetic_mc.json)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with sample data (no actual VLM calls)"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Just test VLM connection and exit"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("VLM4D Evaluation Runner")
    print("=" * 60)
    print()

    if args.test_mode:
        print("Running in TEST MODE (no VLM calls)")
        print()

        # Create sample data
        samples = create_sample_vlm4d_data()
        print(f"Created {len(samples)} sample questions")

        # Simulate evaluation results
        from fast_snow.reasoning.eval.vlm4d import VLM4DMetrics
        metrics = VLM4DMetrics()

        for sample in samples:
            # Simulate correct answers for demo
            metrics.add_prediction(
                sample_id=sample['id'],
                category=sample['question_type'],
                prediction=sample['answer'],
                ground_truth=sample['answer'],
                is_correct=True,
            )

        metrics.compute_final_metrics()
        print()
        print(metrics)
        return 0

    # Create VLM interface
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_id or '(default)'}")
    print()

    try:
        vlm = create_vlm_interface(
            backend=args.backend,
            model_id=args.model_id,
            api_key=args.api_key,
            api_base_url=args.api_base,
        )
    except Exception as e:
        logger.error(f"Failed to create VLM interface: {e}")
        return 1

    # Test connection
    if not test_vlm_connection(vlm):
        print("\nVLM connection test failed. Please check your API key and settings.")
        print("\nFor Google AI Studio:")
        print("  1. Get API key from https://aistudio.google.com/apikey")
        print("  2. export GOOGLE_AI_API_KEY=your_key")
        print("\nFor HuggingFace:")
        print("  1. Get token from https://huggingface.co/settings/tokens")
        print("  2. export HF_TOKEN=your_token")
        return 1

    if args.test_connection:
        print("\nVLM connection test successful!")
        return 0

    # Run evaluation
    print("\nRunning evaluation...")

    if args.vlm4d_json and args.vlm4d_json.exists():
        # Load real VLM4D data
        from fast_snow.data.loaders.vlm4d import load_vlm4d_json
        samples_raw = load_vlm4d_json(args.vlm4d_json)
        samples = [
            {
                "id": s.sample_id,
                "video": s.video,
                "question_type": s.question_type,
                "question": s.question,
                "choices": s.choices,
                "answer": s.answer,
            }
            for s in samples_raw
        ]
        print(f"Loaded {len(samples)} samples from {args.vlm4d_json}")
    else:
        # Use sample data
        samples = create_sample_vlm4d_data()
        print(f"Using {len(samples)} sample questions (no VLM4D JSON provided)")

    metrics = run_simple_eval(vlm, samples, max_samples=args.max_samples)

    print()
    print("=" * 60)
    print(metrics)
    print("=" * 60)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nSaved results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
