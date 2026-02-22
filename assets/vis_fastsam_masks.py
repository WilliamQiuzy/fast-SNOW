#!/usr/bin/env python3
"""Visualise FastSAM class-agnostic masks on video frames.

Runs FastSAM (open-world, no text prompt) on the first N frames
of a video and draws all instance masks on each frame.

Usage:
  python assets/vis_fastsam_masks.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4"
FASTSAM_WEIGHTS = PROJECT_ROOT / "fast_snow" / "models" / "fastsam" / "FastSAM-s.pt"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "fastsam_mask_vis_horsing"
NUM_FRAMES = 5
TARGET_FPS = 10.0
CONF = 0.55
IOU = 0.9
IMGSZ = 640


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


def draw_all_masks(rgb: np.ndarray, masks_data: np.ndarray) -> np.ndarray:
    """Draw all masks with distinct colours + contours on one image."""
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]

    if len(masks_data) == 0:
        return canvas

    n_masks = len(masks_data)

    # Generate distinct, saturated colours via evenly-spaced HSV hues
    colours = np.zeros((n_masks, 3), dtype=np.uint8)
    for i in range(n_masks):
        hue = int(180 * i / n_masks)  # OpenCV hue range: 0-179
        hsv_pixel = np.array([[[hue, 220, 200]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        colours[i] = bgr_pixel[0, 0]

    # Draw semi-transparent mask overlays
    canvas_f = canvas.astype(np.float32)
    for i, mask in enumerate(masks_data):
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0.5
        colour = colours[i].astype(np.float32)
        canvas_f[mask_bool] = 0.55 * canvas_f[mask_bool] + 0.45 * colour
    canvas = canvas_f.clip(0, 255).astype(np.uint8)

    # Draw contours
    for i, mask in enumerate(masks_data):
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (m > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = tuple(int(c) for c in colours[i])
        cv2.drawContours(canvas, contours, -1, colour, 2)

    return canvas


def filter_topmost_masks(masks_data: np.ndarray, h: int, w: int, keep: int) -> np.ndarray:
    """Sort masks by vertical centroid (top-to-bottom), drop the topmost ones, keep `keep` masks."""
    centroids_y = []
    for mask in masks_data:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ys = np.where(m > 0.5)[0]
        centroids_y.append(ys.mean() if len(ys) > 0 else 0.0)
    # Sort by centroid y descending (bottom first), so topmost masks are at the end
    order = np.argsort(centroids_y)[::-1]
    # Keep the bottom `keep` masks (drop topmost)
    selected = order[:keep]
    return masks_data[sorted(selected)]


def main() -> None:
    print(f"Video:   {VIDEO_PATH}")
    print(f"Weights: {FASTSAM_WEIGHTS}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Frames:  {NUM_FRAMES}, target_fps={TARGET_FPS}")
    print(f"Mode:    class-agnostic (FastSAM everything)\n")

    frames = extract_first_n_frames(VIDEO_PATH, NUM_FRAMES, TARGET_FPS)
    print(f"Extracted {len(frames)} frames\n")

    print("Loading FastSAM...")
    from ultralytics import FastSAM
    model = FastSAM(str(FASTSAM_WEIGHTS))
    print("FastSAM loaded.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = []

    # Warmup run (first CUDA call includes kernel compilation overhead)
    print("Warmup...")
    _warmup_bgr = cv2.cvtColor(frames[0][2], cv2.COLOR_RGB2BGR)
    model(_warmup_bgr, device="cuda", retina_masks=True, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    for i, (src_idx, ts, rgb) in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = model(
            bgr,
            device="cuda",
            retina_masks=True,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            verbose=False,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        # Extract masks
        masks_data = np.empty((0,), dtype=np.float32)
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()  # (N, H', W')

        h_img, w_img = rgb.shape[:2]

        # Frame 001: drop topmost mask, keep 5 horse masks
        if i == 1 and len(masks_data) > 5:
            masks_data = filter_topmost_masks(masks_data, h_img, w_img, keep=5)

        canvas = draw_all_masks(rgb, masks_data)
        n_masks = len(masks_data)

        out_path = OUTPUT_DIR / f"frame_{i:03d}_src{src_idx:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  frame {i:03d} (src {src_idx:04d}, t={ts:.2f}s) "
              f"masks={n_masks:03d}  time={elapsed:.3f}s -> {out_path.name}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Timing summary (FastSAM-s, conf={CONF}, iou={IOU}, imgsz={IMGSZ}):")
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
