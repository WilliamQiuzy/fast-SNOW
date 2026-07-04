"""Round 3: compound the Round-2 winners.

Round 2 results on top of STACK8 (max_frames=24, active=10, maskmem=3, init=15, stride=8):
    +max_frames=16    9.60s  E1 5/5/16  E2 4/4/15  ←  -36% ZERO drop
    +max_frames=20   12.43s  E1 5/5/20  E2 4/4/20  ←  -17% ZERO drop
    +max_active=6    12.03s  E1 4/4    E2 3       ←  -20% -1 each ⚠
    +late_disc=off   12.73s  E1 4/4    E2 4/4     ←  -15% -1 E1
    +anchor=12       14.62s  E1 6/6    E2 5/5     ←  -3% +1 BOTH ✓✓ FREE
    +anchor=8        14.70s  E1 6/6    E2 4/4     ←  -2% +1 E1 ✓
    +max_active=8    13.80s  E1 5/5    E2 4/4     ←  -8% safe
    +vg_stride=0     14.49s  E1 5/5    E2 4/4     ←  -4% safe
    +maskmem=2       14.19s  E1 5/5    E2 4/4     ←  -6% safe

Hypothesis: max_frames=16 + anchor=12 + max_active=8 + late_disc=off should land near 7s.
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


# STACK8 floor
STACK8_SAM = {"max_active_tracks": 10, "num_maskmem": 3,
              "max_init_masks": 15, "memory_temporal_stride": 8}
STACK8_SAMP = {"max_frames": 24}


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return {"tracks": len(n),
            "long_half": sum(1 for x in n if x >= half),
            "max": max(n, default=0),
            "total": sum(n)}


def run(pool, video, sam_overrides=None, sampling_overrides=None, fusion_overrides=None):
    saved_sam, saved_samp, saved_fus = {}, {}, {}
    if sam_overrides:
        for k, v in sam_overrides.items():
            saved_sam[k] = getattr(pool.config.sam3, k); setattr(pool.config.sam3, k, v)
    if sampling_overrides:
        for k, v in sampling_overrides.items():
            saved_samp[k] = getattr(pool.config.sampling, k); setattr(pool.config.sampling, k, v)
    if fusion_overrides:
        for k, v in fusion_overrides.items():
            saved_fus[k] = getattr(pool.config.fusion, k); setattr(pool.config.fusion, k, v)
    try:
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=str(video), question=None))
        dt = time.time() - t
        return (resp.four_dsg_dict, dt) if resp.status == "ok" else (None, dt)
    finally:
        for k, v in saved_sam.items():
            setattr(pool.config.sam3, k, v)
        for k, v in saved_samp.items():
            setattr(pool.config.sampling, k, v)
        for k, v in saved_fus.items():
            setattr(pool.config.fusion, k, v)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.target_fps = 10.0

    # Apply STACK8 as floor
    for k, v in STACK8_SAM.items(): setattr(cfg.sam3, k, v)
    for k, v in STACK8_SAMP.items(): setattr(cfg.sampling, k, v)

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("Warm-up..."); run(pool, videos[0])

    # configs: label, sam_ov, samp_ov, fusion_ov, max_frames_metric
    configs = [
        ("STACK8_base",                                  None, None, None, 24),

        # Free wins individually
        ("+anchor=12",                                    {"anchor_stride": 12}, None, None, 24),

        # Build the candidates from f=16 floor up
        ("F16_BASE  (f=16)",                              None, {"max_frames": 16}, None, 16),
        ("F16+anchor=12",                                 {"anchor_stride": 12}, {"max_frames": 16}, None, 16),
        ("F16+anchor=12+late=off",                        {"anchor_stride": 12, "late_discovery_mode": "off"},
                                                          {"max_frames": 16}, None, 16),
        ("F16+anchor=12+late=off+active=8",               {"anchor_stride": 12, "late_discovery_mode": "off",
                                                           "max_active_tracks": 8},
                                                          {"max_frames": 16}, None, 16),
        ("F16+anchor=12+late=off+active=6",               {"anchor_stride": 12, "late_discovery_mode": "off",
                                                           "max_active_tracks": 6},
                                                          {"max_frames": 16}, None, 16),
        ("F16+anchor=12+late=off+vg=0+maskmem=2",         {"anchor_stride": 12, "late_discovery_mode": "off",
                                                           "vg_stride": 0, "num_maskmem": 2},
                                                          {"max_frames": 16}, None, 16),
        ("F16+anchor=12+late=off+vg=0+maskmem=2+active=8",{"anchor_stride": 12, "late_discovery_mode": "off",
                                                           "vg_stride": 0, "num_maskmem": 2,
                                                           "max_active_tracks": 8},
                                                          {"max_frames": 16}, None, 16),

        # Ultra-aggressive
        ("ULTRA1 (f=12+anchor=12+late=off+active=8)",     {"anchor_stride": 12, "late_discovery_mode": "off",
                                                           "max_active_tracks": 8},
                                                          {"max_frames": 12}, None, 12),
        ("ULTRA2 (f=12+anchor=8+late=off+active=8)",      {"anchor_stride": 8, "late_discovery_mode": "off",
                                                           "max_active_tracks": 8},
                                                          {"max_frames": 12}, None, 12),
        ("ULTRA3 (f=16+anchor=16+late=off+active=8)",     {"anchor_stride": 16, "late_discovery_mode": "off",
                                                           "max_active_tracks": 8},
                                                          {"max_frames": 16}, None, 16),

        # Sanity check that turning off post-dedup helps a bit
        ("F16+late=off+post_dedup=off",                   {"late_discovery_mode": "off"},
                                                          {"max_frames": 16},
                                                          {"enable_post_dedup": False}, 16),

        # min_track_obs is a fusion knob, sanity probe
        ("F16+min_track_obs=5",                           None, {"max_frames": 16},
                                                          {"min_track_observations": 5}, 16),
    ]

    print(f"\n{'CONFIG':<58} {'E1 t':<7}{'E1 trk/lh/mx':<14}{'E2 t':<7}{'E2 trk/lh/mx':<14}{'avg':<7}")
    print("-" * 115)
    results = []
    for label, ovs, ovp, ovf, mf in configs:
        row = {"cfg": label, "mf": mf}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp, ovf)
            q = quality(sg, mf) if sg else {"tracks": 0, "long_half": 0, "max": 0, "total": 0}
            row[f"v{i}_t"] = dt; row[f"v{i}_q"] = q
        row["avg"] = (row["v0_t"] + row["v1_t"]) / 2
        results.append(row)
        print(f"{label:<58} "
              f"{row['v0_t']:<7.2f}"
              f"{row['v0_q']['tracks']}/{row['v0_q']['long_half']}/{row['v0_q']['max']:<10}"
              f"{row['v1_t']:<7.2f}"
              f"{row['v1_q']['tracks']}/{row['v1_q']['long_half']}/{row['v1_q']['max']:<10}"
              f"{row['avg']:<7.2f}")

    floor = results[0]  # STACK8_base
    print(f"\n{'='*115}\nSORTED BY SPEED (Δlh relative to STACK8_base):")
    print(f"{'CONFIG':<58} {'avg':<7}{'Hz':<7}{'E1 lh':<7}{'E2 lh':<7}{'ΔE1':<6}{'ΔE2':<6}{'mark':<5}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: x["avg"]):
        hz = 1.0 / r["avg"] if r["avg"] > 0 else 0
        d1 = floor["v0_q"]["long_half"] - r["v0_q"]["long_half"]
        d2 = floor["v1_q"]["long_half"] - r["v1_q"]["long_half"]
        mark = "✓" if (d1 <= 1 and d2 <= 1) else ("⚠" if (d1 <= 2 and d2 <= 2) else "✗")
        print(f"{r['cfg']:<58} {r['avg']:<7.2f}{hz:<7.4f}"
              f"{r['v0_q']['long_half']:<7}{r['v1_q']['long_half']:<7}"
              f"{d1:<+6}{d2:<+6}{mark}")


if __name__ == "__main__":
    main()
