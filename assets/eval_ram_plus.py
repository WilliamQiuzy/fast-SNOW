#!/usr/bin/env python3
"""RAM++ evaluation script for Fast-SNOW.

Processes videos in assets/examples_videos/ at 1 FPS sampling:
  1. Extracts per-frame tags and saves to a unified JSON file.
  2. Measures per-frame inference time.
  3. Prints a human-friendly summary: old tags, new tags vs previous frame.

Usage:
    python assets/eval_ram_plus.py                          # all videos
    python assets/eval_ram_plus.py --videos car.mp4 duck.mp4  # specific videos
    python assets/eval_ram_plus.py --device cpu              # force CPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_snow.engine.config.fast_snow_config import RAMPlusConfig
from fast_snow.vision.perception.ram_wrapper import RAMPlusWrapper

VIDEO_DIR = PROJECT_ROOT / "assets" / "examples_videos"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "eval_results"


def extract_frames_at_1fps(
    video_path: Path,
) -> tuple[float, list[tuple[int, float, np.ndarray]]]:
    """Extract frames at 1 FPS from a video using timestamp-based sampling.

    Returns:
        (fps, frames) where frames is a list of
        (source_frame_idx, timestamp_sec, rgb_image) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Video reports FPS={fps}, cannot determine sampling rate: {video_path}"
        )

    frames = []
    next_target_sec = 0.0
    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        timestamp_sec = frame_idx / fps
        if timestamp_sec >= next_target_sec:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((frame_idx, timestamp_sec, rgb))
            next_target_sec += 1.0  # fixed cadence: avoids drift
        frame_idx += 1
    cap.release()

    print(f"  Video: {video_path.name}")
    print(f"    FPS: {fps:.2f}, total frames: {total_frames}, "
          f"sampled: {len(frames)} frames (timestamp-based 1 FPS)")
    return fps, frames


def _make_cuda_sync(device: str):
    """Return a sync callable if device is CUDA, else a no-op."""
    if not device.startswith("cuda"):
        return lambda: None
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return lambda: None


def evaluate_video(
    wrapper: RAMPlusWrapper,
    video_path: Path,
    device: str = "cuda",
) -> dict:
    """Run RAM++ on every sampled frame and collect results."""

    sync = _make_cuda_sync(device)
    fps, frames = extract_frames_at_1fps(video_path)
    per_frame_results = []
    raw_times: list[float] = []
    prev_tags_set: set[str] = set()

    for i, (src_idx, ts_sec, rgb) in enumerate(frames):
        sync()
        t0 = time.perf_counter()
        tags = wrapper.infer(rgb)
        sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_times.append(elapsed_ms)
        tags_set = set(tags)
        new_tags = sorted(tags_set - prev_tags_set)
        old_tags = sorted(tags_set & prev_tags_set)

        per_frame_results.append({
            "sample_idx": i,
            "source_frame_idx": src_idx,
            "timestamp_sec": round(ts_sec, 3),
            "tags": tags,
            "new_tags": new_tags,
            "old_tags": old_tags,
            "inference_ms": round(elapsed_ms, 2),
        })
        prev_tags_set = tags_set

    # Compute timing statistics from raw (unrounded) values
    timing_stats = {
        "mean_ms": round(float(np.mean(raw_times)), 2) if raw_times else 0,
        "median_ms": round(float(np.median(raw_times)), 2) if raw_times else 0,
        "min_ms": round(float(min(raw_times)), 2) if raw_times else 0,
        "max_ms": round(float(max(raw_times)), 2) if raw_times else 0,
        "std_ms": round(float(np.std(raw_times)), 2) if raw_times else 0,
    }

    return {
        "video": video_path.name,
        "fps": round(fps, 2),
        "num_sampled_frames": len(per_frame_results),
        "timing": timing_stats,
        "frames": per_frame_results,
    }


def print_human_summary(result: dict) -> None:
    """Print a human-friendly per-frame tag summary."""
    video = result["video"]
    timing = result["timing"]

    print(f"\n{'=' * 70}")
    print(f"  {video}")
    print(f"{'=' * 70}")
    print(f"  Sampled frames : {result['num_sampled_frames']}")
    print(f"  Inference time : {timing['mean_ms']:.1f} ms/frame (mean), "
          f"{timing['median_ms']:.1f} ms (median), "
          f"min {timing['min_ms']:.1f} / max {timing['max_ms']:.1f} ms")
    print()

    for i, fr in enumerate(result["frames"]):
        ts = fr["timestamp_sec"]
        src = fr["source_frame_idx"]
        tags = fr["tags"]
        new = fr["new_tags"]
        old = fr["old_tags"]
        ms = fr["inference_ms"]

        print(f"  t={ts:.1f}s (frame {src})  [{ms:.0f}ms]")
        print(f"    All tags ({len(tags)}): {', '.join(tags)}")
        if i == 0:
            print(f"    (first frame — all tags are new)")
        else:
            if new:
                print(f"    + New tags ({len(new)}): {', '.join(new)}")
            else:
                print(f"    + New tags: (none)")
            if old:
                print(f"    = Old tags ({len(old)}): {', '.join(old)}")
            else:
                print(f"    = Old tags: (none)")
        print()


def main():
    parser = argparse.ArgumentParser(description="RAM++ evaluation on example videos")
    parser.add_argument(
        "--videos", nargs="*", default=None,
        help="Video filenames (e.g. car.mp4 duck.mp4). Default: all in examples_videos/",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for RAM++ inference (default: cuda)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: assets/eval_results/ram_plus_eval.json)",
    )
    args = parser.parse_args()

    # Resolve video paths
    if args.videos:
        video_paths = [VIDEO_DIR / v for v in args.videos]
        for vp in video_paths:
            if not vp.exists():
                print(f"ERROR: Video not found: {vp}", file=sys.stderr)
                sys.exit(1)
    else:
        video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
        if not video_paths:
            print(f"ERROR: No .mp4 files found in {VIDEO_DIR}", file=sys.stderr)
            sys.exit(1)

    # Initialize RAM++
    cfg = RAMPlusConfig(device=args.device)
    wrapper = RAMPlusWrapper(cfg)
    print("Loading RAM++ model...")
    wrapper.load()
    print("RAM++ model loaded.\n")

    # Evaluate each video
    all_results = []
    for vp in video_paths:
        result = evaluate_video(wrapper, vp, device=args.device)
        all_results.append(result)
        print_human_summary(result)

    # Save unified results
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "ram_plus_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
