#!/usr/bin/env python3
"""Draw a single red mask on the rightmost horse in frame_001 of horsing.mp4.

Usage:
  python assets/vis_single_horse_red.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4"
FASTSAM_WEIGHTS = PROJECT_ROOT / "fast_snow" / "models" / "fastsam" / "FastSAM-s.pt"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "single_horse_red.png"

# We need frame index 1 (src frame 3) from the original vis script
TARGET_FPS = 10.0
FRAME_INDEX = 1  # 0-indexed, same as frame_001 in the vis output

CONF = 0.55
IOU = 0.9
IMGSZ = 640


def extract_frame(video_path: Path, frame_index: int, target_fps: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = 1.0 / target_fps
    next_target = 0.0
    src_idx = 0
    count = 0
    rgb = None
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        ts = src_idx / fps
        if ts + 1e-9 >= next_target:
            if count == frame_index:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                break
            count += 1
            while ts + 1e-9 >= next_target:
                next_target += sample_interval
        src_idx += 1
    cap.release()
    if rgb is None:
        raise RuntimeError(f"Could not extract frame {frame_index}")
    return rgb


def main() -> None:
    print("Extracting frame...")
    rgb = extract_frame(VIDEO_PATH, FRAME_INDEX, TARGET_FPS)
    h, w = rgb.shape[:2]
    print(f"Frame shape: {h}x{w}")

    print("Loading FastSAM...")
    from ultralytics import FastSAM
    model = FastSAM(str(FASTSAM_WEIGHTS))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Warmup
    model(bgr, device="cuda", retina_masks=True, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()

    # Run detection
    print("Running FastSAM...")
    results = model(bgr, device="cuda", retina_masks=True, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)

    if not results or results[0].masks is None:
        raise RuntimeError("No masks detected")

    masks_data = results[0].masks.data.cpu().numpy()  # (N, H', W')
    print(f"Detected {len(masks_data)} masks")

    # Find the rightmost horse mask by x-centroid
    # Filter: horse-sized (area 0.005-0.025), centroid in lower half (y > h*0.3)
    candidates = []
    for i, mask in enumerate(masks_data):
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0.5
        area_ratio = mask_bool.sum() / (h * w)
        ys, xs = np.where(mask_bool)
        if len(ys) == 0:
            continue
        cx = xs.mean()
        cy = ys.mean()
        print(f"  mask {i}: centroid=({cx:.0f}, {cy:.0f}), area={area_ratio:.3f}")
        # Horse masks: reasonable size, in lower portion of frame
        if area_ratio < 0.005 or area_ratio > 0.025:
            continue
        if cy < h * 0.3:
            continue  # skip masks in upper part (railing, sky, etc.)
        candidates.append((i, cx, cy, mask_bool))

    if not candidates:
        raise RuntimeError("No suitable masks found")

    # Pick the one with the largest x-centroid (rightmost)
    candidates.sort(key=lambda t: t[1], reverse=True)
    chosen_idx, cx, cy, chosen_mask = candidates[0]
    print(f"\nChosen: mask {chosen_idx} (rightmost, centroid x={cx:.0f})")

    # Draw on canvas: red overlay + red contour
    canvas = bgr.copy()
    canvas_f = canvas.astype(np.float32)

    # Red in BGR = (0, 0, 255)
    red = np.array([0, 0, 255], dtype=np.float32)
    canvas_f[chosen_mask] = 0.50 * canvas_f[chosen_mask] + 0.50 * red
    canvas = canvas_f.clip(0, 255).astype(np.uint8)

    # Draw contour
    binary = chosen_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
