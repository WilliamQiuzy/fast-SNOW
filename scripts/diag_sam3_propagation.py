"""Trace what SAM 3.1 multiplex actually produces per (obj_id, frame).

Hooks into WarmModelPool's mask_cache right after Phase B-8 / B-7.5 and
before our dedup, so we see the model's raw output per track.

For each obj_id, prints:
  - presence vector ('X' = mask, '.' = empty) across 32 frames
  - mask areas across frames (so we see drift/shrink/loss)
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import numpy as np
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else (
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4"
    )

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 50
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    pool = WarmModelPool(cfg)

    # Monkey-patch _dedup_mask_cache_by_trajectory to capture mask_cache BEFORE dedup
    raw_cache_holder = {}
    orig_dedup = WarmModelPool._dedup_mask_cache_by_trajectory
    def _capture_then_dedup(mc, **kw):
        # Deep enough copy: keep references to masks (cheap)
        raw_cache_holder["mc"] = {f: list(masks) for f, masks in mc.items()}
        return orig_dedup(mc, **kw)
    WarmModelPool._dedup_mask_cache_by_trajectory = staticmethod(_capture_then_dedup)

    pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    resp = pool.run_inference(InferenceRequest(video_path=video, question=None))
    if resp.status != "ok":
        print("FAILED:", resp.error_message); sys.exit(1)

    mc = raw_cache_holder.get("mc", {})
    n_frames = max(mc.keys()) + 1 if mc else 0

    # Group by (run_id, obj_id_local)
    presence = defaultdict(lambda: [None] * n_frames)
    for fidx, masks in mc.items():
        for m in masks:
            key = (m.run_id, m.obj_id_local)
            if m.mask is not None and m.mask.any():
                presence[key][fidx] = int(m.mask.sum())
            else:
                presence[key][fidx] = 0

    # Print: per-track presence row
    print(f"\nRaw SAM3.1 mask_cache trace — {n_frames} frames, {len(presence)} tracks\n")
    print(f"{'(run,obj)':>14}  pattern (X=mask, .=empty, -=missing)        | obs | min_area  max_area")
    print("-" * 95)

    for key in sorted(presence.keys(), key=lambda k: (k[0] if isinstance(k[0], int) else 99, k[1])):
        row = presence[key]
        chars = []
        areas = []
        for v in row:
            if v is None:
                chars.append("-")
            elif v == 0:
                chars.append(".")
            else:
                chars.append("X")
                areas.append(v)
        if not areas:
            min_a, max_a = 0, 0
        else:
            min_a, max_a = min(areas), max(areas)
        nobs = sum(1 for c in chars if c == "X")
        print(f"{str(key):>14}  {''.join(chars)} | {nobs:>3} | {min_a:>7}   {max_a:>7}")


if __name__ == "__main__":
    main()
