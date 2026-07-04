"""Round 2: explore additive knobs on top of STACK8 + then combine winners.

STACK8 base = (f=24, max_active=10, num_maskmem=3, max_init=15, stride=8) ~15s.
Biggest remaining cost: Phase B-8 in_session = 5-8s/video (50% of runtime).

Knobs to probe individually on top of STACK8:
  A  late_discovery_mode='off'      → skip B-8 entirely (-5-8s, quality risk)
  B  full_propagation_stride=10     → halve B-8 discovery candidates
  C  anchor_stride=8                → halve FastSAM anchors
  D  anchor_stride=12               → 3x fewer FastSAM anchors
  E  vg_stride=0                    → skip re-grounding during propagate
  F  max_frames=20                  → less spatial coverage but less compute
  G  max_frames=16                  → minimal frames
  H  max_active=8                   → tighter cap
  I  max_active=6                   → even tighter
  J  num_maskmem=2                  → smallest memory bank
  K  min_track_observations=5       → drop weak tracks earlier
  L  enable_post_dedup=False        → skip CPU dedup (small win)

Then a final round combining the winners.
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


# STACK8 settings applied as the new baseline
STACK8_SAM = {
    "max_active_tracks": 10,
    "num_maskmem": 3,
    "max_init_masks": 15,
    "memory_temporal_stride": 8,
}
STACK8_SAMP = {"max_frames": 24}


def quality(sg, max_frames):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, max_frames // 2)
    return {
        "tracks": len(n),
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


def merge(*dicts):
    out = {}
    for d in dicts:
        if d: out.update(d)
    return out


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.target_fps = 10.0

    # APPLY STACK8 as floor
    for k, v in STACK8_SAM.items(): setattr(cfg.sam3, k, v)
    for k, v in STACK8_SAMP.items(): setattr(cfg.sampling, k, v)
    print(f"Applied STACK8 base: SAM={STACK8_SAM}, SAMP={STACK8_SAMP}")

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("Warm-up..."); run(pool, videos[0])

    configs = [
        # ── floor ──
        ("STACK8_base", None, None, 24),

        # ── single-knob probes ──
        ("+late_disc=off",         {"late_discovery_mode": "off"}, None, 24),
        ("+full_prop_stride=10",   {"full_propagation_stride": 10}, None, 24),
        ("+anchor_stride=8",       {"anchor_stride": 8}, None, 24),
        ("+anchor_stride=12",      {"anchor_stride": 12}, None, 24),
        ("+vg_stride=0",           {"vg_stride": 0}, None, 24),
        ("+max_frames=20",         None, {"max_frames": 20}, 20),
        ("+max_frames=16",         None, {"max_frames": 16}, 16),
        ("+max_active=8",          {"max_active_tracks": 8}, None, 24),
        ("+max_active=6",          {"max_active_tracks": 6}, None, 24),
        ("+maskmem=2",             {"num_maskmem": 2}, None, 24),
        ("+min_track_obs=5",       {"min_track_observations": 5}, None, 24),
        ("+post_dedup=off",        {"enable_post_dedup": False}, None, 24),

        # ── compound candidates (built progressively from likely winners) ──
        ("COMBO1 +late_off +anchor8",
            {"late_discovery_mode": "off", "anchor_stride": 8}, None, 24),
        ("COMBO2 +late_off +anchor8 +active8",
            {"late_discovery_mode": "off", "anchor_stride": 8, "max_active_tracks": 8}, None, 24),
        ("COMBO3 +late_off +anchor8 +active8 +f=20",
            {"late_discovery_mode": "off", "anchor_stride": 8, "max_active_tracks": 8},
            {"max_frames": 20}, 20),
        ("COMBO4 +late_off +anchor8 +active8 +f=16",
            {"late_discovery_mode": "off", "anchor_stride": 8, "max_active_tracks": 8},
            {"max_frames": 16}, 16),
        ("COMBO5 +late_off +anchor12 +active6 +f=16",
            {"late_discovery_mode": "off", "anchor_stride": 12, "max_active_tracks": 6},
            {"max_frames": 16}, 16),
    ]

    print(f"\n{'CONFIG':<55} {'E1 t':<7}{'E1 trk/lh/mx':<14}{'E2 t':<7}{'E2 trk/lh/mx':<14}{'avg':<7}")
    print("-" * 110)
    results = []
    for label, ovs, ovp, mf in configs:
        row = {"cfg": label, "mf": mf}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp)
            q = quality(sg, mf) if sg else {"tracks": 0, "long_half": 0, "max": 0, "total": 0}
            row[f"v{i}_t"] = dt; row[f"v{i}_q"] = q
        row["avg"] = (row["v0_t"] + row["v1_t"]) / 2
        results.append(row)
        print(f"{label:<55} "
              f"{row['v0_t']:<7.2f}"
              f"{row['v0_q']['tracks']}/{row['v0_q']['long_half']}/{row['v0_q']['max']:<10}"
              f"{row['v1_t']:<7.2f}"
              f"{row['v1_q']['tracks']}/{row['v1_q']['long_half']}/{row['v1_q']['max']:<10}"
              f"{row['avg']:<7.2f}")

    floor = results[0]
    print(f"\n{'='*110}\nSORTED BY SPEED (Δlh relative to STACK8_base):")
    print(f"{'CONFIG':<55} {'avg':<7}{'Hz':<7}{'E1 lh':<7}{'E2 lh':<7}{'ΔE1':<6}{'ΔE2':<6}{'mark':<5}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["avg"]):
        hz = 1.0 / r["avg"] if r["avg"] > 0 else 0
        d1 = floor["v0_q"]["long_half"] - r["v0_q"]["long_half"]
        d2 = floor["v1_q"]["long_half"] - r["v1_q"]["long_half"]
        mark = "✓" if (d1 <= 1 and d2 <= 1) else ("⚠" if (d1 <= 2 and d2 <= 2) else "✗")
        print(f"{r['cfg']:<55} {r['avg']:<7.2f}{hz:<7.4f}"
              f"{r['v0_q']['long_half']:<7}{r['v1_q']['long_half']:<7}"
              f"{d1:<+6}{d2:<+6}{mark}")


if __name__ == "__main__":
    main()
