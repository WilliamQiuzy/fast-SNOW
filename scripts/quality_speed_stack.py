"""Stack-test: combine the per-knob safe winners from quality_speed_ablation.py.

Single knobs that gave ZERO quality loss vs baseline (E1 long=6, E2 long=4) at
max_frames=32:
    max_active_tracks=15   →  -11%
    num_maskmem=5          →   -9%
    num_maskmem=3          →   -8%
    max_init_masks=15      →   -6%
    memory_temporal_stride=8 → -2%

Plus the aggressive cap that lost 1/4 in E2 but gave -28%:
    max_active_tracks=8

We stack them and measure two Easy videos. Quality metrics:
    - long20: tracks with >=20 obs (only meaningful for max_frames>=20)
    - long_half: tracks with >= half(max_frames) obs (length-normalized)
    - total: sum of all obs across tracks
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, max_frames):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, max_frames // 2)
    return {
        "tracks": len(n),
        "long20": sum(1 for x in n if x >= 20),
        "long_half": sum(1 for x in n if x >= half),
        "max": max(n, default=0),
        "total": sum(n),
    }


def run(pool, video, sam_overrides=None, sampling_overrides=None):
    saved_sam, saved_sampling = {}, {}
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

    print("Warm-up..."); run(pool, videos[0])

    # Stacked configs. Each tuple: (label, sam_overrides, sampling_overrides, max_frames_for_metric)
    configs = [
        ("baseline", None, None, 32),

        # ── Tier 1: zero-loss stack ──
        ("STACK1: active15+memmem5",
            {"max_active_tracks": 15, "num_maskmem": 5}, None, 32),

        ("STACK2: active15+memmem5+init15",
            {"max_active_tracks": 15, "num_maskmem": 5, "max_init_masks": 15}, None, 32),

        ("STACK3: active15+memmem5+init15+stride8",
            {"max_active_tracks": 15, "num_maskmem": 5, "max_init_masks": 15,
             "memory_temporal_stride": 8}, None, 32),

        ("STACK4: active15+memmem3+init15+stride8",
            {"max_active_tracks": 15, "num_maskmem": 3, "max_init_masks": 15,
             "memory_temporal_stride": 8}, None, 32),

        # ── Tier 2: include max_active=8 (was -28% with -1/4 E2 quality) ──
        ("STACK5: active8+memmem5+init15+stride8",
            {"max_active_tracks": 8, "num_maskmem": 5, "max_init_masks": 15,
             "memory_temporal_stride": 8}, None, 32),

        ("STACK6: active8+memmem3+init15+stride8",
            {"max_active_tracks": 8, "num_maskmem": 3, "max_init_masks": 15,
             "memory_temporal_stride": 8}, None, 32),

        # ── Tier 3: also reduce max_frames; long20 metric not applicable ──
        ("STACK7: f24+active15+memmem5+init15+stride8",
            {"max_active_tracks": 15, "num_maskmem": 5, "max_init_masks": 15,
             "memory_temporal_stride": 8}, {"max_frames": 24}, 24),

        ("STACK8: f24+active10+memmem3+init15+stride8",
            {"max_active_tracks": 10, "num_maskmem": 3, "max_init_masks": 15,
             "memory_temporal_stride": 8}, {"max_frames": 24}, 24),
    ]

    print(f"\n{'CONFIG':<48} "
          f"{'E1 t':<7}{'E1 trk/lh/mx':<14}"
          f"{'E2 t':<7}{'E2 trk/lh/mx':<14}{'avg':<7}")
    print("-" * 110)
    results = []
    for label, ovs, ovp, mf in configs:
        row = {"cfg": label, "mf": mf}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp)
            q = quality(sg, mf) if sg else {"tracks": 0, "long20": 0, "long_half": 0, "max": 0, "total": 0}
            row[f"v{i}_t"] = dt; row[f"v{i}_q"] = q
        row["avg"] = (row["v0_t"] + row["v1_t"]) / 2
        results.append(row)
        print(f"{label:<48} "
              f"{row['v0_t']:<7.2f}"
              f"{row['v0_q']['tracks']}/{row['v0_q']['long_half']}/{row['v0_q']['max']:<10}"
              f"{row['v1_t']:<7.2f}"
              f"{row['v1_q']['tracks']}/{row['v1_q']['long_half']}/{row['v1_q']['max']:<10}"
              f"{row['avg']:<7.2f}")

    bl = next(r for r in results if r["cfg"] == "baseline")
    print(f"\n{'='*110}\nSORTED BY SPEED (long_half = tracks with >= max_frames/2 observations):")
    print(f"{'CONFIG':<48} {'avg':<7}{'Hz':<7}{'E1 lh':<7}{'E2 lh':<7}{'ΔE1':<6}{'ΔE2':<6}{'mark':<5}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["avg"]):
        hz = 1.0 / r["avg"] if r["avg"] > 0 else 0
        d1 = bl["v0_q"]["long_half"] - r["v0_q"]["long_half"]
        d2 = bl["v1_q"]["long_half"] - r["v1_q"]["long_half"]
        mark = "✓" if (d1 <= 1 and d2 <= 1) else ("⚠" if (d1 <= 2 and d2 <= 2) else "✗")
        print(f"{r['cfg']:<48} {r['avg']:<7.2f}{hz:<7.4f}"
              f"{r['v0_q']['long_half']:<7}{r['v1_q']['long_half']:<7}"
              f"{d1:<+6}{d2:<+6}{mark}")


if __name__ == "__main__":
    main()
