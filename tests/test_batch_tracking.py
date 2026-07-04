"""Lightweight batch tracking test on youtube000-004.

Runs the FastSAM + SAM3 pipeline on 5 videos and outputs:
  - Crop images for each tracked object
  - Summary statistics (track count, key object detection)

Usage:
    python tests/test_batch_tracking.py [--max_mask_frac 0.15] [--conf 0.55]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rose.engine.config.rose_config import (
    DA3Config, FastSAMConfig, ROSEConfig, SAM3Config, SamplingConfig,
)
from rose.engine.pipeline.rose_e2e import _crop_object_from_mask
from rose.engine.pipeline.rose_pipeline import (
    FastFrameInput, FastLocalDetection, ROSEPipeline,
)
from rose.vision.perception.da3_wrapper import DA3Wrapper
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_test")

VIDEOS = [
    ROOT / "benchmark" / "VLM4D-video" / "videos_real" / "youtube-vos" / f"youtube00{i}.mp4"
    for i in range(5)
]
OUT_ROOT = ROOT / "assets" / "batch_tracking_test"


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def extract_frames(video_path, target_fps):
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / target_fps
    next_t = 0.0
    frames, indices, timestamps = [], [], []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        t = idx / src_fps
        if t + 1e-9 >= next_t:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            indices.append(idx)
            timestamps.append(t)
            while t + 1e-9 >= next_t:
                next_t += interval
        idx += 1
    cap.release()
    return frames, indices, timestamps


def save_frames_as_jpeg(frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
    return out_dir


def run_one_video(video_path: Path, out_dir: Path, config: ROSEConfig,
                  max_init_det: int = 0, discovery_stride: int = 1):
    """Run pipeline on one video, return summary dict."""
    import torch

    video_name = video_path.stem
    t_total_start = time.time()

    # Extract frames
    frames, src_indices, timestamps = extract_frames(video_path, 10.0)
    n = len(frames)
    frame_dir = Path(tempfile.mkdtemp(prefix=f"rose_{video_name}_"))
    save_frames_as_jpeg(frames, frame_dir)

    try:
        # DA3
        da3 = DA3Wrapper(config.da3)
        da3_results = da3.infer_batch(frames)
        da3.unload()
        torch.cuda.empty_cache()

        # FastSAM frame 0
        fastsam = FastSAMWrapper(config.fastsam)
        dets_0 = fastsam.detect(frames[0])
        logger.info("[%s] FastSAM frame 0: %d detections (raw)", video_name, len(dets_0))

        # Limit initial detections: top N by mask area (largest first)
        if max_init_det > 0 and len(dets_0) > max_init_det:
            dets_0.sort(key=lambda d: d.mask.sum(), reverse=True)
            dets_0 = dets_0[:max_init_det]
            logger.info("[%s] Limited to top %d by area", video_name, max_init_det)

        # SAM3 init
        sam3 = SAM3SharedSessionManager(config.sam3)
        sam3.set_video_dir(frame_dir)

        bboxes = [list(d.bbox_xywh_norm) for d in dets_0]
        if bboxes:
            sam3.create_run_with_initial_bboxes(
                boxes_xywh=bboxes, box_labels=[1] * len(bboxes),
                frame_idx=0, tag="fastsam_bbox",
            )
            sam3._predictor.model.retain_feature_cache = True
            sam3._predictor.model.detector.retain_multigpu_buffer = True
            sam3._predictor.model.tracker.num_maskmem = config.sam3.num_maskmem
            sam3._predictor.model.tracker.memory_temporal_stride_for_eval = config.sam3.memory_temporal_stride

            # Pre-compute backbone features to skip VG detector during propagation
            sam3.precompute_backbone_features(vg_stride=config.sam3.vg_stride)

            fastsam.unload()
            torch.cuda.empty_cache()

            # Propagate
            for fidx in range(n):
                sam3.propagate_all(fidx)

            # Add point prompts for additional bboxes
            if len(bboxes) > 1:
                for i in range(1, len(bboxes)):
                    bx, by, bw, bh = bboxes[i]
                    sam3.add_object_point(0, (bx + bw / 2.0, by + bh / 2.0))
        else:
            fastsam.unload()

        # Discovery
        discovery_thresh = config.fastsam.discovery_iou_thresh
        min_mask_frac = config.sam3.discovery_min_mask_frac
        max_disc = config.sam3.max_discovery_per_frame
        max_total = config.sam3.max_discovery_total
        new_obj_count = 0

        if sam3.active_runs:
            disc_frames = list(range(discovery_stride, n, discovery_stride))
            logger.info("[%s] Discovery: checking %d frames (stride=%d)",
                        video_name, len(disc_frames), discovery_stride)
            for fidx in disc_frames:
                if max_total > 0 and new_obj_count >= max_total:
                    break
                dets = fastsam.detect(frames[fidx])
                # FastSAM already returns detections sorted by mask area
                # descending — larger objects are more likely foreground.
                cached = sam3.propagate_all(fidx)
                dh, dw = frames[fidx].shape[:2]
                frame_disc = 0
                for det in dets:
                    if max_disc > 0 and frame_disc >= max_disc:
                        break
                    if max_total > 0 and new_obj_count >= max_total:
                        break
                    if min_mask_frac > 0 and det.mask.sum() < min_mask_frac * dh * dw:
                        continue
                    if not any(mask_iou(det.mask, m.mask) >= discovery_thresh for m in cached):
                        cy_px, cx_px = mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fidx, (cx_px / w, cy_px / h))
                        new_obj_count += 1
                        frame_disc += 1

        logger.info("[%s] Discovery: %d new objects", video_name, new_obj_count)
        fastsam.unload()
        torch.cuda.empty_cache()

        # Partial propagation
        frame0_pt_count = max(0, len(bboxes) - 1) if bboxes else 0
        total_new = frame0_pt_count + new_obj_count
        if total_new > 0:
            sam3.propagate_new_objects()

        # Build 4DSG
        pipeline = ROSEPipeline(config)
        crop_pad = config.vlm.object_crop_padding
        crop_sz = config.vlm.object_crop_size
        best_crops = {}

        for fidx in range(n):
            sam3_masks = list(sam3.propagate_all(fidx))
            best = {}
            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                if key not in best or m.score > best[key].score:
                    best[key] = m
            sam3_masks = list(best.values())

            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                crop = _crop_object_from_mask(frames[fidx], m.mask, padding=crop_pad, size=crop_sz)
                brightness = float(crop.mean())
                prev = best_crops.get(key)
                if prev is None or (m.score, brightness) > (prev[1], prev[3]):
                    best_crops[key] = (crop, m.score, src_indices[fidx], brightness)

            detections = [
                FastLocalDetection(run_id=m.run_id, local_obj_id=m.obj_id_local,
                                   mask=m.mask, score=m.score)
                for m in sam3_masks
            ]
            fi = FastFrameInput(
                frame_idx=src_indices[fidx],
                depth_t=da3_results[fidx].depth,
                K_t=da3_results[fidx].K,
                T_wc_t=da3_results[fidx].T_wc,
                detections=detections,
                depth_conf_t=da3_results[fidx].depth_conf,
                depth_is_metric=da3_results[fidx].is_metric,
                timestamp_s=timestamps[fidx],
            )
            pipeline.process_frame(fi)

        # Save crops
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        object_crops = {}
        for key, gid in pipeline._local_to_global.items():
            if key in best_crops and gid not in object_crops:
                crop_rgb, _score, src_idx, _bright = best_crops[key]
                if crop_rgb.mean() < 1.0:
                    continue
                crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                object_crops[gid] = {"path": str(crop_path), "source_frame_idx": src_idx}

        # Dedup
        n_before = len(object_crops)
        object_crops = pipeline.merge_duplicate_tracks(object_crops)
        n_after = len(object_crops)

        # Build 4DSG dict
        dsg = pipeline.build_4dsg_dict(object_crops=object_crops)
        dsg_json = json.dumps(dsg, indent=2, ensure_ascii=False, sort_keys=False)
        (out_dir / "4dsg.json").write_text(dsg_json)

        sam3.end_all_runs()

        elapsed = time.time() - t_total_start
        return {
            "video": video_name,
            "frames": n,
            "tracks_raw": n_before,
            "tracks_dedup": n_after,
            "discovery": new_obj_count,
            "time": elapsed,
            "dsg": dsg,
        }

    finally:
        try:
            sam3.end_all_runs()
        except Exception:
            pass
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--max_mask_frac", type=float, default=0.15)
    parser.add_argument("--max_discovery_total", type=int, default=25)
    parser.add_argument("--max_discovery_per_frame", type=int, default=5)
    parser.add_argument("--discovery_min_mask_frac", type=float, default=0.005)
    parser.add_argument("--max_init_det", type=int, default=0,
                        help="Max initial detections from frame 0 (0=unlimited, top N by confidence)")
    parser.add_argument("--discovery_stride", type=int, default=1,
                        help="Run discovery every N frames (1=every frame, 5=every 5th)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for SAM3")
    parser.add_argument("--vg_stride", type=int, default=25,
                        help="VG detection stride (0=skip all VG)")
    parser.add_argument("--num_maskmem", type=int, default=7,
                        help="SAM3 tracker memory bank size")
    args = parser.parse_args()

    config = ROSEConfig(
        sampling=SamplingConfig(target_fps=10.0),
        da3=DA3Config(chunk_size=0, chunk_overlap=5),
        fastsam=FastSAMConfig(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_mask_frac=args.max_mask_frac,
        ),
        sam3=SAM3Config(
            offload_state_to_cpu=True,
            offload_video_to_cpu=True,
            max_discovery_total=args.max_discovery_total,
            max_discovery_per_frame=args.max_discovery_per_frame,
            discovery_min_mask_frac=args.discovery_min_mask_frac,
            enable_compile=args.compile,
            vg_stride=args.vg_stride,
            num_maskmem=args.num_maskmem,
            chunk_size=-1,
        ),
    )

    logger.info("Config: conf=%.2f, iou=%.1f, max_mask_frac=%.2f, "
                "max_disc_total=%d, max_disc_frame=%d, min_mask_frac=%.3f",
                args.conf, args.iou, args.max_mask_frac,
                args.max_discovery_total, args.max_discovery_per_frame,
                args.discovery_min_mask_frac)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    results = []
    for video_path in VIDEOS:
        video_name = video_path.stem
        out_dir = OUT_ROOT / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Processing %s", video_name)
        logger.info("=" * 60)

        result = run_one_video(video_path, out_dir, config,
                              max_init_det=args.max_init_det,
                              discovery_stride=args.discovery_stride)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("BATCH TRACKING SUMMARY")
    print(f"Config: conf={args.conf}, max_mask_frac={args.max_mask_frac}, "
          f"max_disc={args.max_discovery_total}, init={args.max_init_det}, "
          f"stride={args.discovery_stride}")
    print("=" * 80)
    print(f"{'Video':12s} {'Frames':>6s} {'Raw':>5s} {'Dedup':>5s} {'Disc':>5s} {'Time':>6s}  Tracks")
    print("-" * 80)
    for r in results:
        tracks = r["dsg"]["tracks"]
        track_summary = []
        for t in tracks:
            oid = t["object_id"]
            n_obs = len(t["F_k"])
            motion = t["motion"]
            mot_short = "M" if "moving" in motion else "S"
            track_summary.append(f"o{oid}({n_obs}{mot_short})")
        print(f"{r['video']:12s} {r['frames']:6d} {r['tracks_raw']:5d} "
              f"{r['tracks_dedup']:5d} {r['discovery']:5d} {r['time']:5.0f}s  "
              f"{', '.join(track_summary[:8])}"
              f"{'...' if len(track_summary) > 8 else ''}")
    print("=" * 80)
    print(f"Crops saved to: {OUT_ROOT}/*/crops/")
    print("Review crops to check if key objects (person, bird, penguin, bear, truck) are tracked.")


if __name__ == "__main__":
    main()
