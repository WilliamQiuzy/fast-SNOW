"""Profile each phase of the warm-pool pipeline on a few real videos.

Goal: figure out exactly where the 35-40s per video is going so we can
target the next round of optimizations.

Phases timed:
  - frame extraction (CPU + cv2)
  - DA3 batch (Phase 1)
  - SAM3 set_video_frames (image upload)
  - SAM3 backbone precompute (Phase 2a part 1)
  - SAM3 init + propagate (Phase 2a part 2)
  - FastSAM discovery loop (Phase 2b)
  - SAM3 partial propagation (Phase 2c)
  - mask collection (Phase 2d)
  - 4DSG construction (Phase 3)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import cv2
import numpy as np
from PIL import Image
import torch

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_e2e import (
    _any_mask_iou_above, _crop_object_from_mask, _mask_centroid,
)
from rose.engine.pipeline.rose_pipeline import (
    FastFrameInput, FastLocalDetection, ROSEPipeline,
)
from rose.vision.perception.da3_wrapper import DA3Wrapper
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import (
    SAM3SharedSessionManager,
)

VLM4D = ROOT / "benchmark" / "VLM4D-video"
HF = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"


def cfg():
    c = ROSEConfig()
    c.da3.model_path = "rose/models/da3-small"
    c.sampling.max_frames = 32
    c.sampling.target_fps = 10.0
    c.sam3.use_fa3 = True
    c.sam3.offload_state_to_cpu = False
    c.sam3.offload_video_to_cpu = False
    c.sam3.enable_compile = False
    return c


def extract_frames(video_path, target_fps, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames, ts = [], []
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or target_fps
    interval = 1.0 / target_fps
    next_t = 0.0
    si = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = si / src_fps
        si += 1
        if t + 1e-9 < next_t:
            continue
        while t + 1e-9 >= next_t:
            next_t += interval
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts.append(t)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, ts


def profile_one(video_path, da3, fastsam, sam3, config):
    times = {}

    t = time.time()
    frames, ts = extract_frames(video_path, config.sampling.target_fps, config.sampling.max_frames)
    pil = [Image.fromarray(f) for f in frames]
    times["extract"] = time.time() - t

    n = len(frames)

    # Phase 1: DA3 batch
    t = time.time()
    da3_results = da3.infer_batch(frames)
    torch.cuda.synchronize()
    times["da3_batch"] = time.time() - t

    # Phase 2 (under bf16 autocast)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Set frames
        t = time.time()
        sam3.set_video_frames(pil)
        torch.cuda.synchronize()
        times["sam3_set_frames"] = time.time() - t

        # FastSAM frame 0
        t = time.time()
        dets_0 = fastsam.detect(frames[0])
        torch.cuda.synchronize()
        times["fastsam_init"] = time.time() - t

        if dets_0:
            bboxes = [list(d.bbox_xywh_norm) for d in dets_0]
            t = time.time()
            _, init_masks = sam3.create_run_with_initial_bboxes(
                boxes_xywh=bboxes, box_labels=[1]*len(bboxes),
                frame_idx=0, tag="bbox",
            )
            torch.cuda.synchronize()
            times["sam3_init"] = time.time() - t

            sam3._predictor.model.retain_feature_cache = True

            # Backbone precompute
            t = time.time()
            sam3.precompute_backbone_features(vg_stride=config.sam3.vg_stride)
            torch.cuda.synchronize()
            times["sam3_backbone_precompute"] = time.time() - t

            # Full propagation
            t = time.time()
            sam3.propagate_all(n - 1)
            torch.cuda.synchronize()
            times["sam3_propagate_full"] = time.time() - t

            # Discovery loop (Phase 2b)
            t = time.time()
            stride = config.sam3.full_propagation_stride
            new_count = 0
            for fi in range(1, n):
                if stride > 1 and fi % stride != 0:
                    continue
                dets = fastsam.detect(frames[fi])
                cached = sam3.propagate_all(fi)
                for det in dets:
                    if not _any_mask_iou_above(det.mask, cached, config.fastsam.discovery_iou_thresh):
                        cy, cx = _mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fi, (cx/w, cy/h))
                        new_count += 1
            torch.cuda.synchronize()
            times["fastsam_discovery"] = time.time() - t

            # Phase 2c
            t = time.time()
            if new_count > 0:
                sam3.propagate_new_objects()
            torch.cuda.synchronize()
            times["sam3_propagate_new"] = time.time() - t

            # Mask collection
            t = time.time()
            mask_cache = {fi: list(sam3.propagate_all(fi)) for fi in range(n)}
            torch.cuda.synchronize()
            times["mask_collect"] = time.time() - t
        else:
            mask_cache = {}
            for k in ["sam3_init", "sam3_backbone_precompute", "sam3_propagate_full",
                      "fastsam_discovery", "sam3_propagate_new", "mask_collect"]:
                times[k] = 0.0

    # Phase 3: 4DSG construction (CPU)
    t = time.time()
    pipeline = ROSEPipeline(config)
    for fi in range(n):
        masks = mask_cache.get(fi, [])
        best = {}
        for m in masks:
            key = (m.run_id, m.obj_id_local)
            if key not in best or m.score > best[key].score:
                best[key] = m
        dets = [FastLocalDetection(run_id=m.run_id, local_obj_id=m.obj_id_local,
                                    mask=m.mask, score=m.score)
                for m in best.values()]
        fi_input = FastFrameInput(
            frame_idx=fi, depth_t=da3_results[fi].depth, K_t=da3_results[fi].K,
            T_wc_t=da3_results[fi].T_wc, detections=dets,
            depth_conf_t=da3_results[fi].depth_conf,
            depth_is_metric=da3_results[fi].is_metric, timestamp_s=ts[fi],
        )
        pipeline.process_frame(fi_input)
    fdsg = pipeline.build_4dsg_dict(object_crops={})  # skip crops for profile
    times["build_4dsg"] = time.time() - t

    # Cleanup
    sam3.end_all_runs()

    times["TOTAL"] = sum(times.values())
    return times, n


def main():
    rng = random.Random(42)
    videos = []
    for fname in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D / "QA" / fname) as f:
            for q in json.load(f):
                p = VLM4D / q["video"].replace(HF, "")
                if p.is_file() and p not in videos:
                    videos.append(p)
    sample = rng.sample(videos, 5)

    c = cfg()
    print("Loading models (warmup, NOT counted)...")
    da3 = DA3Wrapper(c.da3); da3.load()
    fastsam = FastSAMWrapper(c.fastsam); fastsam.load()
    sam3 = SAM3SharedSessionManager(c.sam3); sam3.load()
    print("Models loaded.\n")

    print("Throwaway run (NOT counted)...")
    profile_one(sample[0], da3, fastsam, sam3, c)
    print("Throwaway done.\n")

    runs = []
    for i, vp in enumerate(sample[1:], 1):
        print(f"\n--- Profile {i}: {vp.name} ---")
        times, n = profile_one(vp, da3, fastsam, sam3, c)
        times["video"] = vp.name
        times["n_frames"] = n
        runs.append(times)
        for k, v in times.items():
            if isinstance(v, float):
                print(f"  {k:30s} {v:6.2f}s")
        hz = n / times["TOTAL"] if times["TOTAL"] > 0 else 0
        print(f"  >> {n} frames in {times['TOTAL']:.2f}s = {hz:.2f} Hz")

    # Aggregate per-phase mean
    print(f"\n{'='*60}\n  Phase-by-phase MEAN (over {len(runs)} videos)\n{'='*60}")
    keys = ["extract", "da3_batch", "sam3_set_frames", "fastsam_init", "sam3_init",
            "sam3_backbone_precompute", "sam3_propagate_full", "fastsam_discovery",
            "sam3_propagate_new", "mask_collect", "build_4dsg", "TOTAL"]
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            mean = sum(vals)/len(vals)
            pct = 100 * mean / (sum(r["TOTAL"] for r in runs)/len(runs)) if k != "TOTAL" else 100
            print(f"  {k:30s} {mean:6.2f}s  ({pct:5.1f}%)")

    out = ROOT / "benchmark" / "profile_warm_pool.json"
    with open(out, "w") as f:
        json.dump(runs, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
