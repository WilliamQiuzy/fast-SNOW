"""Experiment: skip Phase B-4 VG grounding by using point prompts at frame 0
instead of bbox prompts.

Hypothesis:
  - VG grounding (propagation_full) takes ~4s and is only needed because we
    use bbox prompts.
  - Point prompts (add_object_point) trigger SAM2 tracker directly,
    bypassing VG grounding.
  - If we register frame-0 detections as POINTS instead of BBOXES, we skip
    Phase B-4 entirely.

Test:
  - Use FastSAM bboxes at frame 0
  - For each bbox, get its center point
  - Register all as add_object_point (one-by-one, no batched API)
  - Skip propagation_full
  - Run propagation_partial directly
  - Compare time + mask coverage vs baseline

If this works, we save ~4s/video (~17% speedup) without quality loss.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
import cv2 as cv2_lib
import numpy as np
from PIL import Image

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    sam3 = pool._sam3

    video = "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4"

    # Warm up
    print("Warm-up...")
    pool.run_inference(InferenceRequest(video_path=video, question=None))

    # ── Extract frames ──
    cap = cv2_lib.VideoCapture(video)
    frames = []
    while len(frames) < 32:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2_lib.cvtColor(f, cv2_lib.COLOR_BGR2RGB))
    cap.release()
    pil = [Image.fromarray(f) for f in frames[:32]]

    # ── FastSAM at frame 0 ──
    dets = pool._fastsam.detect(frames[0])
    dets = [d for d in dets if not d.mask.sum() == 0][:8]
    print(f"FastSAM: {len(dets)} detections at frame 0")

    # ── Method A: Current baseline (bbox + VG grounding + refine + memory) ──
    print("\n=== Method A: BBOX path (VG grounding + memory) ===")
    sam3.end_all_runs()
    sam3.set_video_frames(pil)
    bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    bf16_ctx.__enter__()

    try:
        t_init = time.time()
        bboxes = [list(d.bbox_xywh_norm) for d in dets]
        obj_ids = sam3.add_bboxes_batch(bboxes, frame_idx=0)
        t_a_init = time.time() - t_init
        print(f"  add_bboxes_batch: {t_a_init:.3f}s, {len(obj_ids)} obj_ids")

        t_vg = time.time()
        sam3.propagate_new_objects()
        t_a_vg = time.time() - t_vg
        print(f"  propagation_full (VG grounding): {t_a_vg:.3f}s")

        # Refine each obj with point
        t_refine = time.time()
        for obj_id, det in zip(obj_ids, dets):
            bx, by, bw, bh = det.bbox_xywh_norm
            sam3.refine_object_with_point(obj_id=obj_id, frame_idx=0,
                                          point_xy=(bx + bw/2, by + bh/2))
        t_a_refine = time.time() - t_refine
        print(f"  refine with points: {t_a_refine:.3f}s")

        # Memory tracker propagation
        sam3._propagation_cache.clear()
        sam3._propagation_dirty = True
        t_mt = time.time()
        sam3.propagate_new_objects()
        t_a_mt = time.time() - t_mt
        print(f"  propagation_partial (memory tracker): {t_a_mt:.3f}s")

        # Coverage
        cov_a = defaultdict(int)
        for fidx in range(32):
            masks = sam3.propagate_all(fidx)
            for m in masks:
                if m.mask is not None and m.mask.any():
                    cov_a[m.obj_id_local] += 1
        print(f"  Coverage (frames with mask per obj): {dict(cov_a)}")
        t_method_a = t_a_init + t_a_vg + t_a_refine + t_a_mt
        print(f"  TOTAL Method A: {t_method_a:.3f}s")
    finally:
        bf16_ctx.__exit__(None, None, None)
        sam3.end_all_runs()

    # ── Method B: Skip VG, use points only ──
    print("\n=== Method B: POINT-only path (skip VG grounding) ===")
    sam3.end_all_runs()
    sam3.set_video_frames(pil)
    bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    bf16_ctx.__enter__()

    try:
        t_init = time.time()
        obj_ids_b = []
        for det in dets:
            bx, by, bw, bh = det.bbox_xywh_norm
            cx, cy = bx + bw/2, by + bh/2
            oid = sam3.add_object_point(0, (cx, cy))
            obj_ids_b.append(oid)
        t_b_init = time.time() - t_init
        print(f"  add_object_point × {len(obj_ids_b)}: {t_b_init:.3f}s")

        # Single propagation
        t_prop = time.time()
        sam3.propagate_new_objects()
        t_b_prop = time.time() - t_prop
        print(f"  propagation: {t_b_prop:.3f}s")

        cov_b = defaultdict(int)
        for fidx in range(32):
            masks = sam3.propagate_all(fidx)
            for m in masks:
                if m.mask is not None and m.mask.any():
                    cov_b[m.obj_id_local] += 1
        print(f"  Coverage: {dict(cov_b)}")
        t_method_b = t_b_init + t_b_prop
        print(f"  TOTAL Method B: {t_method_b:.3f}s")
    finally:
        bf16_ctx.__exit__(None, None, None)
        sam3.end_all_runs()

    # ── Compare ──
    print(f"\n=== COMPARISON ===")
    print(f"Method A (bbox+VG): {t_method_a:.3f}s")
    print(f"Method B (points only): {t_method_b:.3f}s")
    print(f"Savings: {t_method_a - t_method_b:.3f}s ({100*(t_method_a-t_method_b)/t_method_a:.1f}%)")
    total_cov_a = sum(cov_a.values())
    total_cov_b = sum(cov_b.values())
    print(f"Total mask-frames A: {total_cov_a}, B: {total_cov_b}")
    print(f"Quality ratio (B/A): {100*total_cov_b/max(total_cov_a, 1):.1f}%")


if __name__ == "__main__":
    main()
