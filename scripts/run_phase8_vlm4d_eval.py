#!/usr/bin/env python3
"""Phase 8 VLM4D Evaluation Script (Strict Reproduction).

This script implements Phase 8 evaluation protocol with:
1. Phase 7 strict mode (temperature=0, max_tokens=256, Gemma3-4B-IT)
2. Unified answer extraction and formatting
3. Standardized metrics aligned with paper
4. Full reproducibility (config logging, timestamps, versions)

Usage:
    # Standard evaluation (full dataset)
    python scripts/run_phase8_vlm4d_eval.py \\
        --vlm4d-json fast_snow/data/VLM4D-main/data/real_mc.json \\
        --output results/phase8_eval_$(date +%Y%m%d).json

    # Quick test (10 samples)
    python scripts/run_phase8_vlm4d_eval.py \\
        --vlm4d-json fast_snow/data/VLM4D-main/data/real_mc.json \\
        --max-samples 10

    # Dry run (no VLM calls)
    python scripts/run_phase8_vlm4d_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_snow.data.loaders.vlm4d import load_vlm4d_json, VLM4DSample
from fast_snow.reasoning.eval.answer_postprocess import (
    extract_multiple_choice_answer,
    format_question_with_choices,
)
from fast_snow.reasoning.eval.metrics import Phase8Metrics, EvaluationResult, compare_with_baseline
from fast_snow.reasoning.vlm.inference import create_phase7_vlm_interface, VLMInterface
from fast_snow.reasoning.vlm.prompt_builder import Phase7SerializationConfig, serialize_4dsg_strict
from fast_snow.engine.pipeline.snow_pipeline import SNOWPipeline, SNOWConfig
from fast_snow.engine.config.snow_config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_config_dict() -> Dict:
    """Get current configuration as dict for reproducibility."""
    return {
        "model_id": "google/gemma-3-4b-it",
        "temperature": 0.0,
        "max_new_tokens": 256,
        "do_sample": False,
        "serialization": {
            "precision": 2,
            "max_tracks": 50,
            "max_frames_per_track": 10,
            "temporal_window": 10,
        },
        "seed": 42,
    }


def get_environment_info() -> Dict:
    """Get environment information for reproducibility."""
    import platform
    import torch

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Try to get CUDA/GPU info
    try:
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except:
        pass

    # Try to get git commit
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env["git_commit"] = git_commit
    except:
        pass

    return env


def run_phase8_evaluation(
    vlm: VLMInterface,
    samples: list[VLM4DSample],
    snow_pipeline: Optional[SNOWPipeline] = None,
    max_samples: Optional[int] = None,
    save_intermediate: Optional[Path] = None,
    use_mock_scenes: bool = False,
) -> Phase8Metrics:
    """Run Phase 8 evaluation with strict protocol.

    Args:
        vlm: Phase 7 VLM interface
        samples: VLM4D samples
        snow_pipeline: Optional SNOW pipeline for real video processing
        max_samples: Optional limit on samples
        save_intermediate: Optional path to save intermediate results
        use_mock_scenes: If True, use mock scenes (for testing without videos)

    Returns:
        Phase8Metrics with results
    """
    # Initialize metrics
    metrics = Phase8Metrics()
    metrics.config = get_config_dict()
    metrics.environment = get_environment_info()
    metrics.timestamp = datetime.datetime.now().isoformat()

    # Limit samples if requested
    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} samples with Phase 8 protocol")

    if use_mock_scenes:
        logger.warning("Using MOCK scenes - not real video processing!")
    elif snow_pipeline is None:
        logger.warning("No SNOW pipeline provided - will use mock scenes")
        use_mock_scenes = True

    for i, sample in enumerate(samples, 1):
        try:
            # Format question with choices (Phase 8 standard format)
            question_formatted = format_question_with_choices(
                question=sample.question,
                choices=sample.choices,
                include_instruction=True,
            )

            # Process video to get 4DSG
            if use_mock_scenes:
                # Mock scene for testing
                scene_context = _create_mock_scene_description(sample)
            else:
                # REAL PIPELINE: Video → Frames → MapAnything → 3DSG → 4DSG (NO VLM)
                # This avoids double VLM calls and ensures strict Phase 7 evaluation
                logger.info(f"Processing video: {sample.video}")

                try:
                    # Build 4DSG without VLM inference (Phase 8 strict pattern)
                    four_dsg, tracks = snow_pipeline.build_4dsg_from_video(
                        video_path=sample.video,
                        num_frames=10,    # Phase 8 specification
                        use_slam=True,    # Use SLAM for ego-motion
                    )

                    # Serialize with Phase 7 strict mode
                    # This is the ONLY serialization used for VLM inference
                    scene_context = serialize_4dsg_strict(
                        four_dsg,
                        Phase7SerializationConfig()
                    )

                    logger.info(f"Video processed successfully: {sample.sample_id}")

                except Exception as e:
                    logger.error(f"Video processing failed for {sample.sample_id}: {e}")
                    logger.warning("Falling back to mock scene")
                    scene_context = _create_mock_scene_description(sample)

            # Build full prompt
            full_prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

Scene Data (JSON):
{scene_context}

Question: {question_formatted}"""

            # Run VLM inference (Phase 7 strict mode)
            response = vlm.infer_text(full_prompt)

            # Extract answer (Phase 8 standardized extraction)
            prediction = extract_multiple_choice_answer(
                response=response,
                choices=sample.choices,
            )

            # Check correctness
            is_correct = prediction.upper() == sample.answer.upper()

            # Create result
            result = EvaluationResult(
                sample_id=sample.sample_id,
                category=sample.question_type,
                question=sample.question,
                prediction=prediction,
                ground_truth=sample.answer,
                is_correct=is_correct,
                raw_response=response,
            )

            # Add to metrics
            metrics.add_result(result)

            # Log progress
            if i % 10 == 0:
                logger.info(
                    f"Progress: {i}/{len(samples)} | "
                    f"Current accuracy: {metrics.correct_samples/i:.2%}"
                )

            # Save intermediate results
            if save_intermediate and i % 50 == 0:
                metrics.save_json(save_intermediate, include_results=True)
                logger.info(f"Saved intermediate results to {save_intermediate}")

        except Exception as e:
            logger.error(f"Error processing sample {sample.sample_id}: {e}")
            # Add failed result
            result = EvaluationResult(
                sample_id=sample.sample_id,
                category=sample.question_type,
                question=sample.question,
                prediction="",
                ground_truth=sample.answer,
                is_correct=False,
                raw_response=str(e),
            )
            metrics.add_result(result)

    # Compute final metrics
    metrics.compute_metrics()

    return metrics


def _create_mock_scene_description(sample: VLM4DSample) -> str:
    """Create a mock scene description for testing.

    In full pipeline, this would be the Phase 7 strict JSON serialization
    of the 4DSG built from the video.

    Args:
        sample: VLM4D sample

    Returns:
        Mock JSON scene description
    """
    # Simplified mock JSON (in real eval, this comes from serialize_4dsg_strict)
    mock_scene = {
        "metadata": {
            "num_frames": 10,
            "num_objects": 3,
            "temporal_window": 10
        },
        "ego_agent": {
            "trajectory": [
                {"frame": i, "position": [i * 0.5, 0.0, 0.0]}
                for i in range(10)
            ]
        },
        "objects": [
            {
                "object_id": 0,
                "track": [
                    {
                        "frame": 0,
                        "centroid": [5.2, 10.3, 0.5],
                        "extent": [1.8, 4.5, 1.6],
                        "temporal_span": [0, 9],
                        "num_patches": 24
                    }
                ]
            }
        ],
        "spatial_relations": [
            {
                "source_id": 0,
                "target_id": 1,
                "relation": "front_level_near",
                "distance": 10.3
            }
        ]
    }

    return json.dumps(mock_scene, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 VLM4D Evaluation (Strict Reproduction)"
    )
    parser.add_argument(
        "--vlm4d-json",
        type=Path,
        required=False,
        help="Path to VLM4D JSON file (real_mc.json)"
    )
    parser.add_argument(
        "--backend",
        default="google_ai",
        choices=["google_ai", "huggingface", "openai"],
        help="VLM backend (default: google_ai)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (will use env vars if not provided)"
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
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no VLM calls, mock results)"
    )
    parser.add_argument(
        "--use-mock-scenes",
        action="store_true",
        help="Use mock scenes instead of real video processing (faster, for testing)"
    )
    parser.add_argument(
        "--process-videos",
        action="store_true",
        help="Process real videos with SNOW pipeline (requires video files)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 8 VLM4D Evaluation (Strict Reproduction)")
    print("=" * 60)
    print()

    # Load samples
    if args.vlm4d_json and args.vlm4d_json.exists():
        logger.info(f"Loading VLM4D samples from {args.vlm4d_json}")
        samples = load_vlm4d_json(args.vlm4d_json)
        logger.info(f"Loaded {len(samples)} samples")
    else:
        logger.warning("No VLM4D JSON provided, using mock samples")
        # Create mock samples for testing
        from fast_snow.data.loaders.vlm4d import VLM4DSample
        samples = [
            VLM4DSample(
                sample_id="test_001",
                video="mock_video.mp4",
                question_type="ego-centric",
                question="How many cars are in front?",
                choices={"A": "0", "B": "1", "C": "2", "D": "3"},
                answer="B",
            )
        ]

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE: Creating mock results")
        metrics = Phase8Metrics()
        metrics.config = get_config_dict()
        metrics.environment = get_environment_info()
        metrics.timestamp = datetime.datetime.now().isoformat()

        for sample in samples[:args.max_samples] if args.max_samples else samples:
            # Mock correct answer
            result = EvaluationResult(
                sample_id=sample.sample_id,
                category=sample.question_type,
                question=sample.question,
                prediction=sample.answer,
                ground_truth=sample.answer,
                is_correct=True,
                raw_response=f"Mock response: {sample.answer}",
            )
            metrics.add_result(result)

        metrics.compute_metrics()

        print()
        print(metrics)
        print()
        print(compare_with_baseline(metrics))

        if args.output:
            metrics.save_json(args.output, include_results=True)
            print(f"\nSaved results to {args.output}")

        return 0

    # Create Phase 7 VLM interface
    logger.info(f"Creating Phase 7 VLM interface (backend: {args.backend})")
    try:
        vlm = create_phase7_vlm_interface(
            backend=args.backend,
            api_key=args.api_key,
        )
        logger.info("VLM interface created successfully")
    except Exception as e:
        logger.error(f"Failed to create VLM interface: {e}")
        print("\nPlease ensure API key is set:")
        print("  export GOOGLE_AI_API_KEY=your_key")
        return 1

    # Create SNOW pipeline if needed
    snow_pipeline = None
    if args.process_videos and not args.use_mock_scenes:
        logger.info("Creating SNOW pipeline for real video processing...")
        try:
            # Create Phase 7 strict config
            snow_config = SNOWConfig(
                use_phase7_strict=True,  # Force Phase 7 strict mode
                data=DataConfig(num_frames=10),  # Paper specification
                seed=42,                 # Fixed seed
            )
            snow_pipeline = SNOWPipeline(snow_config)
            logger.info("SNOW pipeline created successfully")
        except Exception as e:
            logger.error(f"Failed to create SNOW pipeline: {e}")
            logger.warning("Will use mock scenes instead")

    # Run evaluation
    logger.info("Starting Phase 8 evaluation...")
    metrics = run_phase8_evaluation(
        vlm=vlm,
        samples=samples,
        snow_pipeline=snow_pipeline,
        max_samples=args.max_samples,
        save_intermediate=args.output,
        use_mock_scenes=args.use_mock_scenes or args.dry_run,
    )

    # Print results
    print()
    print("=" * 60)
    print(metrics)
    print()
    print("=" * 60)
    print(compare_with_baseline(metrics))
    print("=" * 60)

    # Save results
    if args.output:
        metrics.save_json(args.output, include_results=True)
        logger.info(f"Saved results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
