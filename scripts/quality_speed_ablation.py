"""Quality-vs-speed ablation: find the best tradeoff knob settings.

Tests each knob along its quality-cost curve, measures both time and a
quality metric (long-track count, total observations, max obs per track).
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

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


def run(pool, video, sam_overrides=None, sampling_overrides=None):
    saved_sam = {}; saved_sampling = {}
    if sam_overrides:
        for k, v in sam_overrides.items():
            saved_sam[k] = getattr(pool.config.sam3, k)
            setattr(pool.config.sam3, k, v)
    if sampling_overrides:
        for k, v in sampling_overrides.items():
            saved_sampling[k] = getattr(pool.config.sampling, k)
            setattr(pool.config.sampling, k, v)
    try:
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=str(video), question=None))
        dt = time.time() - t
        return (resp.four_dsg_dict, dt) if resp.status == "ok" else (None, dt)
    finally:
        for k, v in saved_sam.items():
            setattr(pool.config.sam3, k, v)
        for k, v in saved_sampling.items():
            setattr(pool.config.sampling, k, v)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # Warm up
    print("Warm-up...")
    run(pool, videos[0])

    # ── BASELINE ──
    print(f"\n{'CONFIG':<35} {'Easy1 t':<10} {'E1 long/max':<14} "
          f"{'Easy2 t':<10} {'E2 long/max':<14} {'avg t':<8}")
    print("-" * 100)

    configs = [
        ("baseline", None, None),

        # ── Knob 1: frame count ──
        ("max_frames=24", None, {"max_frames": 24}),
        ("max_frames=16", None, {"max_frames": 16}),
        ("max_frames=12", None, {"max_frames": 12}),

        # ── Knob 2: target_fps lower (longer time span per frame) ──
        ("target_fps=5,max=16", None, {"max_frames": 16, "target_fps": 5}),
        ("target_fps=8,max=24", None, {"max_frames": 24, "target_fps": 8}),

        # ── Knob 3: num_maskmem ──
        ("num_maskmem=5", {"num_maskmem": 5}, None),
        ("num_maskmem=3", {"num_maskmem": 3}, None),

        # ── Knob 4: memory_temporal_stride higher ──
        ("mem_stride=8", {"memory_temporal_stride": 8}, None),

        # ── Knob 5: max_active_tracks (cap obj count) ──
        ("max_active=15", {"max_active_tracks": 15}, None),
        ("max_active=10", {"max_active_tracks": 10}, None),
        ("max_active=8",  {"max_active_tracks": 8},  None),

        # ── Knob 6: max_init (fewer init objs) ──
        ("max_init=15", {"max_init_masks": 15}, None),
        ("max_init=10", {"max_init_masks": 10}, None),

        # ── Knob 7: anchor_stride (less FastSAM, fewer late discoveries) ──
        ("anchor_stride=8", {"anchor_stride": 8}, None),
        ("anchor_stride=16", {"anchor_stride": 16}, None),

        # ── COMBINATIONS ──
        ("max_frames=24+max_active=15", {"max_active_tracks": 15}, {"max_frames": 24}),
        ("max_frames=16+max_active=12", {"max_active_tracks": 12}, {"max_frames": 16}),
        ("AGGRESSIVE: f=16,mem=3,active=10", {"num_maskmem": 3, "max_active_tracks": 10},
         {"max_frames": 16}),
    ]

    results = []
    for cfg_name, ovs, ovp in configs:
        row = {"cfg": cfg_name}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp)
            q = quality(sg) if sg else {"tracks": 0, "long": 0, "max": 0, "total": 0}
            row[f"v{i}_time"] = dt
            row[f"v{i}_q"] = q
        avg_t = (row["v0_time"] + row["v1_time"]) / 2
        row["avg"] = avg_t
        results.append(row)
        print(f"{cfg_name:<35} "
              f"{row['v0_time']:<10.2f} "
              f"{row['v0_q']['long']}/{row['v0_q']['max']:<10} "
              f"{row['v1_time']:<10.2f} "
              f"{row['v1_q']['long']}/{row['v1_q']['max']:<10} "
              f"{avg_t:<8.2f}")

    # Sort by speed
    print(f"\n{'='*100}\nSORTED BY SPEED (best Hz first):")
    print(f"{'CONFIG':<35} {'avg t (s)':<10} {'Hz':<8} {'E1 long':<8} {'E2 long':<8}")
    print("-" * 80)
    baseline_long = next(r for r in results if r["cfg"] == "baseline")
    for r in sorted(results, key=lambda x: x["avg"]):
        hz = 1.0 / r["avg"] if r["avg"] > 0 else 0
        e1_drop = baseline_long['v0_q']['long'] - r['v0_q']['long']
        e2_drop = baseline_long['v1_q']['long'] - r['v1_q']['long']
        marker = " ✓" if (e1_drop <= 1 and e2_drop <= 1) else (" ⚠" if (e1_drop <= 2 and e2_drop <= 2) else " ✗")
        print(f"{r['cfg']:<35} {r['avg']:<10.2f} {hz:<8.4f} "
              f"{r['v0_q']['long']:<8} {r['v1_q']['long']:<8}{marker}")


if __name__ == "__main__":
    main()
