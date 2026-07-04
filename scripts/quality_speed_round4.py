"""Round 4: explore beyond ULTRA2 floor.

ULTRA2: f=12, anchor_stride=8, late_discovery=off, max_active=8,
        (plus inherited STACK8: maskmem=3, max_init=15, mem_stride=8)
        → 6.22s = 0.161 Hz, E1 4/4/12 (full coverage), E2 5/5/12 (full coverage)

What's left to attack?
  - max_frames=8  (extreme): half the frames again
  - max_active=4  : even fewer concurrent tracks
  - anchor_stride=4 : more anchors but might converge faster (less reset)
  - anchor_stride=1 : every frame is anchor — drastic
  - num_maskmem=1 : minimal memory bank
  - Combos: f=8+active=4 / f=10+active=6
  - vg_stride=0 : skip full VG re-grounding during prop

Also profile per-phase timing on ULTRA2 to know where remaining 6s goes.
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


# ULTRA2 floor
ULTRA2_SAM = {
    "max_active_tracks": 8,
    "num_maskmem": 3,
    "max_init_masks": 15,
    "memory_temporal_stride": 8,
    "anchor_stride": 8,
    "late_discovery_mode": "off",
}
ULTRA2_SAMP = {"max_frames": 12}


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return {"tracks": len(n),
            "long_half": sum(1 for x in n if x >= half),
            "max": max(n, default=0)}


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

    for k, v in ULTRA2_SAM.items(): setattr(cfg.sam3, k, v)
    for k, v in ULTRA2_SAMP.items(): setattr(cfg.sampling, k, v)

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("Warm-up..."); run(pool, videos[0])

    configs = [
        ("ULTRA2 floor",                                  None, None, None, 12),

        # New single-knob probes
        ("+max_frames=8",                                 None, {"max_frames": 8}, None, 8),
        ("+max_frames=10",                                None, {"max_frames": 10}, None, 10),
        ("+max_frames=16",                                None, {"max_frames": 16}, None, 16),
        ("+max_active=6",                                 {"max_active_tracks": 6}, None, None, 12),
        ("+max_active=4",                                 {"max_active_tracks": 4}, None, None, 12),
        ("+anchor_stride=4",                              {"anchor_stride": 4}, None, None, 12),
        ("+anchor_stride=12",                             {"anchor_stride": 12}, None, None, 12),
        ("+anchor_stride=16",                             {"anchor_stride": 16}, None, None, 12),
        ("+num_maskmem=1",                                {"num_maskmem": 1}, None, None, 12),
        ("+max_init=8",                                   {"max_init_masks": 8}, None, None, 12),
        ("+vg_stride=0",                                  {"vg_stride": 0}, None, None, 12),

        # Combos
        ("COMBO_A f=8+active=4",                          {"max_active_tracks": 4}, {"max_frames": 8}, None, 8),
        ("COMBO_B f=8+active=6+maskmem=2",                {"max_active_tracks": 6, "num_maskmem": 2},
                                                          {"max_frames": 8}, None, 8),
        ("COMBO_C f=10+active=6+vg=0",                    {"max_active_tracks": 6, "vg_stride": 0},
                                                          {"max_frames": 10}, None, 10),
        ("COMBO_D f=8+active=4+maskmem=1",                {"max_active_tracks": 4, "num_maskmem": 1},
                                                          {"max_frames": 8}, None, 8),
    ]

    print(f"\n{'CONFIG':<48} {'E1 t':<7}{'E1 trk/lh/mx':<14}{'E2 t':<7}{'E2 trk/lh/mx':<14}{'avg':<7}")
    print("-" * 105)
    results = []
    for label, ovs, ovp, ovf, mf in configs:
        row = {"cfg": label, "mf": mf}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp, ovf)
            q = quality(sg, mf) if sg else {"tracks": 0, "long_half": 0, "max": 0}
            row[f"v{i}_t"] = dt; row[f"v{i}_q"] = q
        row["avg"] = (row["v0_t"] + row["v1_t"]) / 2
        results.append(row)
        print(f"{label:<48} "
              f"{row['v0_t']:<7.2f}"
              f"{row['v0_q']['tracks']}/{row['v0_q']['long_half']}/{row['v0_q']['max']:<10}"
              f"{row['v1_t']:<7.2f}"
              f"{row['v1_q']['tracks']}/{row['v1_q']['long_half']}/{row['v1_q']['max']:<10}"
              f"{row['avg']:<7.2f}")

    floor = results[0]
    print(f"\n{'='*105}\nSORTED BY SPEED (Δ tracks/lh relative to ULTRA2 floor):")
    print(f"{'CONFIG':<48} {'avg':<7}{'Hz':<7}{'E1 trk/lh':<11}{'E2 trk/lh':<11}{'mark':<5}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: x["avg"]):
        hz = 1.0 / r["avg"] if r["avg"] > 0 else 0
        d_e1_lh = floor["v0_q"]["long_half"] - r["v0_q"]["long_half"]
        d_e2_lh = floor["v1_q"]["long_half"] - r["v1_q"]["long_half"]
        d_e1_trk = floor["v0_q"]["tracks"] - r["v0_q"]["tracks"]
        d_e2_trk = floor["v1_q"]["tracks"] - r["v1_q"]["tracks"]
        mark = "✓" if (d_e1_lh <= 1 and d_e2_lh <= 1) else ("⚠" if (d_e1_lh <= 2 and d_e2_lh <= 2) else "✗")
        print(f"{r['cfg']:<48} {r['avg']:<7.2f}{hz:<7.4f}"
              f"{r['v0_q']['tracks']}/{r['v0_q']['long_half']:<8}"
              f"{r['v1_q']['tracks']}/{r['v1_q']['long_half']:<8}"
              f"{mark}")


if __name__ == "__main__":
    main()
