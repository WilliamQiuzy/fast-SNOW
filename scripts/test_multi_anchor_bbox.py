"""Experiment: multi-anchor batched bbox add in ONE session.

Tests the architectural change — instead of:
  Phase B-3: bbox-batched at frame 0
  Phase B-8: per-object point add at late anchors (NEW SESSION)

We do:
  Phase B-3': bbox-batched at EVERY anchor frame (ONE session via skip_reset)
  ONE propagate covers all anchors

Compares time + 4DSG quality vs the current baseline.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    return {
        "tracks": len(n),
        "long": sum(1 for x in n if x >= 20),
        "max": max(n, default=0),
        "total": sum(n),
    }


def run_baseline(pool, video):
    t = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=str(video), question=None))
    dt = time.time() - t
    return resp.four_dsg_dict if resp.status == "ok" else None, dt


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # Warm up the model with a single inference first
    print("Warm-up (discarded)...")
    run_baseline(pool, videos[0])

    print("\n=== BASELINE (current architecture) ===")
    base_results = {}
    for v in videos:
        sg, dt = run_baseline(pool, v)
        base_results[Path(v).stem] = (sg, dt)
        print(f"  {Path(v).stem}: {dt:.2f}s  quality={quality(sg)}")

    # --- Test that skip_reset works for multi-frame bbox ---
    print("\n=== TEST: multi-anchor bbox add empirical cost ===")
    # We can't easily replace the production pipeline; instead use a small
    # contrived test: call add_bboxes_batch_multi_frame directly on a fresh
    # SAM session and time the propagate.
    sam3 = pool._sam3
    from PIL import Image
    import cv2 as cv2_lib
    import numpy as np

    # Use Easy1 to set up a fresh SAM session
    cap = cv2_lib.VideoCapture(videos[0])
    frames = []
    while len(frames) < 32:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2_lib.cvtColor(f, cv2_lib.COLOR_BGR2RGB))
    cap.release()
    pil = [Image.fromarray(f) for f in frames[:32]]

    # Quick FastSAM at 4 anchors to get bboxes (smaller test)
    anchor_idxs = [0, 8, 16, 24]
    anchor_bboxes_by_frame = {}
    for fidx in anchor_idxs:
        dets = pool._fastsam.detect(frames[fidx])
        # Take top-3 by area for speed
        dets = sorted(dets, key=lambda d: d.mask.sum(), reverse=True)[:3]
        anchor_bboxes_by_frame[fidx] = [list(d.bbox_xywh_norm) for d in dets]
        print(f"  anchor frame {fidx}: {len(anchor_bboxes_by_frame[fidx])} bboxes")

    # Fresh SAM session
    sam3.end_all_runs()
    sam3.set_video_frames(pil)

    bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    bf16_ctx.__enter__()
    try:
        t0 = time.time()
        # Multi-anchor bbox add with skip_reset on subsequent calls
        result = sam3.add_bboxes_batch_multi_frame(anchor_bboxes_by_frame)
        t_add = time.time() - t0
        total_obj = sum(len(v) for v in result.values())
        print(f"\n  add_bboxes_batch_multi_frame: {t_add:.3f}s for {total_obj} obj across {len(anchor_idxs)} anchors")
        for fidx, oids in result.items():
            print(f"    frame {fidx}: {len(oids)} obj_ids assigned")

        t0 = time.time()
        sam3.propagate_new_objects()
        t_prop = time.time() - t0
        print(f"  propagate_new_objects: {t_prop:.3f}s")

        # Verify masks are produced across all frames for all objs
        coverage = defaultdict(int)
        for fidx in range(32):
            masks = sam3.propagate_all(fidx)
            for m in masks:
                if m.mask is not None and m.mask.any():
                    coverage[m.obj_id_local] += 1
        print(f"\n  Per-obj coverage (n_frames with non-empty mask):")
        for oid, cnt in sorted(coverage.items()):
            print(f"    obj_{oid}: {cnt}/32 frames")
    finally:
        bf16_ctx.__exit__(None, None, None)
        sam3.end_all_runs()


if __name__ == "__main__":
    main()
