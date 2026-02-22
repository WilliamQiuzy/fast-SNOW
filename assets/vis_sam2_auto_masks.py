#!/usr/bin/env python3
"""Visualise SAM2 AutomaticMaskGenerator on video frames.

Runs SAM2s (sam2.1_hiera_small) class-agnostic mask generation on
the first N frames and draws all masks with per-frame timing.

Usage:
  python assets/vis_sam2_auto_masks.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets" / "examples_videos" / "horse-human.mp4"
SAM2_CKPT = PROJECT_ROOT / "fast_snow" / "vision" / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "sam2s_mask_vis_horse_human"
NUM_FRAMES = 5
TARGET_FPS = 10.0


def extract_first_n_frames(video_path: Path, n: int, target_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS={fps}")
    sample_interval = 1.0 / target_fps
    next_target = 0.0
    src_idx = 0
    frames = []
    while len(frames) < n:
        ret, bgr = cap.read()
        if not ret:
            break
        ts = src_idx / fps
        if ts + 1e-9 >= next_target:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((src_idx, ts, rgb))
            while ts + 1e-9 >= next_target:
                next_target += sample_interval
        src_idx += 1
    cap.release()
    return frames


def draw_all_masks(rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Draw all SAM2 auto-generated masks on one image."""
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]
    n_masks = len(masks)
    if n_masks == 0:
        return canvas

    # Sort by area descending so small masks are drawn on top
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

    rng = np.random.default_rng(42)
    colours = rng.integers(60, 230, size=(n_masks, 3), dtype=np.uint8)

    # Draw semi-transparent overlays
    canvas_f = canvas.astype(np.float32)
    for i, m in enumerate(masks_sorted):
        seg = m["segmentation"]  # (H, W) bool
        colour = colours[i].astype(np.float32)
        canvas_f[seg] = 0.55 * canvas_f[seg] + 0.45 * colour
    canvas = canvas_f.clip(0, 255).astype(np.uint8)

    # Draw contours
    for i, m in enumerate(masks_sorted):
        seg = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = tuple(int(c) for c in colours[i])
        cv2.drawContours(canvas, contours, -1, colour, 2)

    # Mask count label
    label = f"{n_masks} masks"
    cv2.putText(canvas, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    print(f"Video:   {VIDEO_PATH}")
    print(f"Ckpt:    {SAM2_CKPT}")
    print(f"Config:  {SAM2_CFG}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Frames:  {NUM_FRAMES}, target_fps={TARGET_FPS}")
    print()

    frames = extract_first_n_frames(VIDEO_PATH, NUM_FRAMES, TARGET_FPS)
    print(f"Extracted {len(frames)} frames\n")

    # Build SAM2 model + auto mask generator
    print("Loading SAM2s...")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(
        SAM2_CFG,
        ckpt_path=str(SAM2_CKPT),
        device="cuda",
        mode="eval",
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.85,
        min_mask_region_area=75,
    )
    print("SAM2s loaded.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = []

    for i, (src_idx, ts, rgb) in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        masks = mask_generator.generate(rgb)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        canvas = draw_all_masks(rgb, masks)

        out_path = OUTPUT_DIR / f"frame_{i:03d}_src{src_idx:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  frame {i:03d} (src {src_idx:04d}, t={ts:.2f}s) "
              f"masks={len(masks):03d}  time={elapsed:.3f}s -> {out_path.name}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Timing summary (SAM2s AutomaticMaskGenerator):")
    print(f"  Frames:  {len(timings)}")
    print(f"  Total:   {sum(timings):.3f}s")
    print(f"  Mean:    {np.mean(timings):.3f}s")
    print(f"  Min:     {np.min(timings):.3f}s")
    print(f"  Max:     {np.max(timings):.3f}s")
    print(f"  FPS:     {len(timings)/sum(timings):.2f}")
    print(f"{'='*50}")
    print(f"\nSaved {len(frames)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
