#!/usr/bin/env python3
"""Visualise YOLO detections frame-by-frame and save annotated images.

Usage:
  python assets/vis_yolo_bboxes.py --videos car.mp4
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the detector and frame extractor from the YOLO eval script
from assets.eval_yolo_bbox_sam3_masks import (
    YoloBBoxDetector,
    _normalize_names,
    extract_frames_at_fps,
)

VIDEO_DIR = PROJECT_ROOT / "assets" / "examples_videos"
OUTPUT_ROOT = PROJECT_ROOT / "assets" / "yolo_bbox_vis"

# Deterministic colour per class name
_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {}

def _colour_for_class(name: str) -> tuple[int, int, int]:
    if name not in _CLASS_COLOURS:
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        c = rng.integers(60, 230, size=3, dtype=np.uint8)
        _CLASS_COLOURS[name] = (int(c[0]), int(c[1]), int(c[2]))
    return _CLASS_COLOURS[name]


def _draw_bboxes(rgb: np.ndarray, detections) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    for d in detections:
        x1n, y1n, x2n, y2n = d.bbox_xyxy
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)
        colour = _colour_for_class(d.class_name)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"{d.class_name} {d.score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, lh + 4)
        cv2.rectangle(canvas, (x1, ty - lh - 4), (x1 + lw + 2, ty + 2), colour, -1)
        cv2.putText(canvas, label, (x1 + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO bbox visualisation")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-weights", default="yolo11n.pt")
    parser.add_argument("--detector-prompts", default="car,person,vehicle,tire")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.65)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    if args.videos:
        video_paths = [VIDEO_DIR / v for v in args.videos]
    else:
        video_paths = sorted(VIDEO_DIR.glob("*.mp4"))

    prompt_classes = _normalize_names(args.detector_prompts)
    detector = YoloBBoxDetector(
        model_name_or_path=args.detector_weights,
        prompt_classes=prompt_classes,
        conf=args.det_conf,
        iou=args.det_iou,
        imgsz=args.det_imgsz,
        device=args.device,
    )
    detector.load()
    print("YOLO loaded.\n")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for video_path in video_paths:
        print(f"Processing {video_path.name}")
        _, _, sampled = extract_frames_at_fps(
            video_path,
            target_fps=args.target_fps,
            max_duration_s=args.max_seconds,
        )
        out_dir = (out_root / video_path.stem).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, (src_idx, ts, rgb, jpeg_path) in enumerate(sampled):
            dets = detector.detect(rgb, frame_idx=i)
            canvas = _draw_bboxes(rgb, dets)
            classes = sorted(set(d.class_name for d in dets))
            out_path = out_dir / f"{video_path.stem}_{src_idx:06d}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"  frame {i:03d} (src {src_idx}) dets={len(dets):02d} "
                  f"classes={classes} -> {out_path.name}")
            try:
                jpeg_path.unlink()
            except OSError:
                pass
        try:
            Path(sampled[0][3]).parent.rmdir()
        except OSError:
            pass

    print(f"\nSaved to {out_root}")


if __name__ == "__main__":
    main()
