"""Analyze discovery efficiency: what objects are discovered, how many are duplicates.

Runs Phase 2a/2b only (no DA3, no 4DSG) to quickly profile discovery behavior.
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rose.engine.config.rose_config import (
    FastSAMConfig, ROSEConfig, SAM3Config, SamplingConfig,
)
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("discovery-diag")

VIDEO = ROOT / "benchmark" / "VLM4D-video" / "videos_synthetic" / "synth_001.mp4"


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def extract_frames(video_path, target_fps):
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / target_fps
    next_t = 0.0
    frames = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        t = idx / src_fps
        if t + 1e-9 >= next_t:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            while t + 1e-9 >= next_t:
                next_t += interval
        idx += 1
    cap.release()
    return frames


def save_frames_as_jpeg(frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
    return out_dir


def run_analysis(discovery_iou_thresh=0.3, max_discovery_total=15,
                 max_discovery_per_frame=5, discovery_stride=1):
    """Run discovery with given params and return stats."""

    config = ROSEConfig(
        sampling=SamplingConfig(target_fps=10.0),
        fastsam=FastSAMConfig(conf_threshold=0.55, iou_threshold=0.9,
                              discovery_iou_thresh=discovery_iou_thresh),
        sam3=SAM3Config(offload_state_to_cpu=False, offload_video_to_cpu=False,
                        max_discovery_total=max_discovery_total,
                        max_discovery_per_frame=max_discovery_per_frame,
                        chunk_size=-1),
    )

    frames = extract_frames(VIDEO, 10.0)
    n = len(frames)
    frame_dir = save_frames_as_jpeg(frames, Path(tempfile.mkdtemp(prefix="disc_diag_")))

    fastsam = FastSAMWrapper(config.fastsam)
    sam3 = SAM3SharedSessionManager(config.sam3)
    sam3.set_video_dir(frame_dir)

    stats = {
        "params": {
            "discovery_iou_thresh": discovery_iou_thresh,
            "max_discovery_total": max_discovery_total,
            "max_discovery_per_frame": max_discovery_per_frame,
            "discovery_stride": discovery_stride,
        },
        "n_frames": n,
        "fastsam_frame0_dets": 0,
        "sam3_init_masks": 0,
        "frame0_point_prompts": 0,
        "discovery_per_frame": [],
        "total_discovery": 0,
        "total_tracked_objects": 0,
        "phase2a_time": 0.0,
        "phase2b_time": 0.0,
        "phase2c_time": 0.0,
    }

    try:
        # Phase 2a
        t0 = time.time()
        dets_0 = fastsam.detect(frames[0])
        stats["fastsam_frame0_dets"] = len(dets_0)

        bboxes = [list(d.bbox_xywh_norm) for d in dets_0]
        if bboxes:
            _, init_masks = sam3.create_run_with_initial_bboxes(
                boxes_xywh=bboxes, box_labels=[1] * len(bboxes), frame_idx=0, tag="bbox",
            )
            stats["sam3_init_masks"] = len(init_masks)

            sam3._predictor.model.retain_feature_cache = True
            sam3._predictor.model.detector.retain_multigpu_buffer = True
            sam3._predictor.model.tracker.num_maskmem = 4
            sam3._predictor.model.tracker.memory_temporal_stride_for_eval = 2
            sam3._predictor.model.partial_propagation_stride = 2
            sam3._predictor.model.full_propagation_stride = 2
            sam3._predictor.model.vg_detection_stride = 2

            fastsam.unload()
            torch.cuda.empty_cache()

            for fidx in range(n):
                sam3.propagate_all(fidx)

            MAX_FRAME0_POINTS = 8
            remaining = list(range(1, len(dets_0)))
            remaining.sort(key=lambda i: dets_0[i].mask.sum(), reverse=True)
            remaining = remaining[:MAX_FRAME0_POINTS]
            stats["frame0_point_prompts"] = len(remaining)
            for i in remaining:
                bx, by, bw, bh = bboxes[i]
                sam3.add_object_point(0, (bx + bw/2, by + bh/2))
        else:
            fastsam.unload()

        stats["phase2a_time"] = time.time() - t0

        # Phase 2b: Discovery with detailed logging
        t0 = time.time()
        new_obj_count = 0

        if sam3.active_runs:
            for fidx in range(1, n):
                if discovery_stride > 1 and fidx % discovery_stride != 0:
                    continue
                if max_discovery_total > 0 and new_obj_count >= max_discovery_total:
                    break

                dets = fastsam.detect(frames[fidx])
                cached = sam3.propagate_all(fidx)

                frame_new = 0
                frame_detail = {
                    "frame": fidx,
                    "fastsam_dets": len(dets),
                    "sam3_cached": len(cached),
                    "new_objects": 0,
                    "max_iou_of_rejected": [],  # IoU values of dets that matched existing
                    "iou_of_accepted": [],       # IoU values of dets accepted as new
                }

                for det in dets:
                    if max_discovery_per_frame > 0 and frame_new >= max_discovery_per_frame:
                        break
                    if max_discovery_total > 0 and new_obj_count >= max_discovery_total:
                        break

                    # Compute IoU with all cached masks
                    ious = [mask_iou(det.mask, m.mask) for m in cached]
                    max_iou = max(ious) if ious else 0.0

                    if max_iou < discovery_iou_thresh:
                        # This is a "new" object
                        cy_px, cx_px = mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fidx, (cx_px / w, cy_px / h))
                        new_obj_count += 1
                        frame_new += 1
                        frame_detail["iou_of_accepted"].append(max_iou)
                    else:
                        frame_detail["max_iou_of_rejected"].append(max_iou)

                frame_detail["new_objects"] = frame_new
                if frame_new > 0:
                    stats["discovery_per_frame"].append(frame_detail)

        stats["total_discovery"] = new_obj_count
        stats["phase2b_time"] = time.time() - t0
        fastsam.unload()
        torch.cuda.empty_cache()

        # Phase 2c
        t0 = time.time()
        total_new = stats["frame0_point_prompts"] + new_obj_count
        stats["total_tracked_objects"] = stats["sam3_init_masks"] + total_new
        if total_new > 0:
            sam3.propagate_new_objects()
        stats["phase2c_time"] = time.time() - t0

    finally:
        try:
            sam3.end_all_runs()
        except Exception:
            pass
        shutil.rmtree(frame_dir, ignore_errors=True)

    return stats


def print_stats(stats):
    p = stats["params"]
    logger.info("=" * 70)
    logger.info("DISCOVERY ANALYSIS (iou_thresh=%.2f, max_total=%d, max_per_frame=%d, stride=%d)",
                p["discovery_iou_thresh"], p["max_discovery_total"],
                p["max_discovery_per_frame"], p["discovery_stride"])
    logger.info("=" * 70)
    logger.info("  Frames: %d", stats["n_frames"])
    logger.info("  FastSAM frame 0: %d detections", stats["fastsam_frame0_dets"])
    logger.info("  SAM3 init (bbox): %d masks", stats["sam3_init_masks"])
    logger.info("  Frame-0 point prompts: %d", stats["frame0_point_prompts"])
    logger.info("  Discovery: %d new objects", stats["total_discovery"])
    logger.info("  Total tracked: %d objects", stats["total_tracked_objects"])
    logger.info("")
    logger.info("  Discovery breakdown per frame:")
    for fd in stats["discovery_per_frame"]:
        accepted_ious = ", ".join(f"{v:.3f}" for v in fd["iou_of_accepted"])
        rejected_ious = fd["max_iou_of_rejected"]
        n_rejected = len(rejected_ious)
        avg_rejected = sum(rejected_ious) / n_rejected if n_rejected else 0
        logger.info("    frame %2d: fastsam=%d, cached=%d, new=%d, accepted_ious=[%s], rejected=%d (avg_iou=%.3f)",
                    fd["frame"], fd["fastsam_dets"], fd["sam3_cached"], fd["new_objects"],
                    accepted_ious, n_rejected, avg_rejected)
    logger.info("")
    logger.info("  TIMING: phase2a=%.2fs, phase2b=%.2fs, phase2c=%.2fs, total=%.2fs",
                stats["phase2a_time"], stats["phase2b_time"], stats["phase2c_time"],
                stats["phase2a_time"] + stats["phase2b_time"] + stats["phase2c_time"])
    logger.info("=" * 70)


if __name__ == "__main__":
    # Test 1: Current settings (baseline)
    logger.info("\n\n>>> TEST 1: BASELINE (current settings)")
    s1 = run_analysis(discovery_iou_thresh=0.3, max_discovery_total=15, max_discovery_per_frame=5)
    print_stats(s1)

    # Test 2: Higher IoU threshold (stricter = fewer false discoveries)
    logger.info("\n\n>>> TEST 2: Higher IoU threshold (0.15)")
    s2 = run_analysis(discovery_iou_thresh=0.15, max_discovery_total=15, max_discovery_per_frame=5)
    print_stats(s2)

    # Test 3: Much lower max_discovery_total
    logger.info("\n\n>>> TEST 3: max_discovery_total=5")
    s3 = run_analysis(discovery_iou_thresh=0.3, max_discovery_total=5, max_discovery_per_frame=2)
    print_stats(s3)

    # Test 4: Discovery on every 3rd frame (stride=3)
    logger.info("\n\n>>> TEST 4: Discovery stride=3")
    s4 = run_analysis(discovery_iou_thresh=0.3, max_discovery_total=15, max_discovery_per_frame=5, discovery_stride=3)
    print_stats(s4)

    # Test 5: Aggressive — high threshold + low budget + stride
    logger.info("\n\n>>> TEST 5: Aggressive (iou=0.15, total=8, stride=3)")
    s5 = run_analysis(discovery_iou_thresh=0.15, max_discovery_total=8, max_discovery_per_frame=3, discovery_stride=3)
    print_stats(s5)
