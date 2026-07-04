"""Final: stack the safe wins, measure stability, pick the fastest no-quality-loss config.

Round 5 confirmed individual safe wins on top of FLOOR (f=10):
    +max_frames=9    5.05s
    +max_init=8      5.09s
    +vg_stride=0     5.21s
    +maskmem=1       5.24s
    FLOOR            5.27s

Stack them. Then run the winner 3× to confirm stability.
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


FLOOR_SAM = {
    "max_active_tracks": 8,
    "num_maskmem": 3,
    "max_init_masks": 15,
    "memory_temporal_stride": 8,
    "anchor_stride": 8,
    "late_discovery_mode": "off",
}
FLOOR_SAMP = {"max_frames": 10}


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return {"tracks": len(n),
            "long_half": sum(1 for x in n if x >= half),
            "max": max(n, default=0)}


def run(pool, video, sam_overrides=None, sampling_overrides=None):
    saved_sam, saved_samp = {}, {}
    if sam_overrides:
        for k, v in sam_overrides.items():
            saved_sam[k] = getattr(pool.config.sam3, k); setattr(pool.config.sam3, k, v)
    if sampling_overrides:
        for k, v in sampling_overrides.items():
            saved_samp[k] = getattr(pool.config.sampling, k); setattr(pool.config.sampling, k, v)
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


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.target_fps = 10.0

    for k, v in FLOOR_SAM.items(): setattr(cfg.sam3, k, v)
    for k, v in FLOOR_SAMP.items(): setattr(cfg.sampling, k, v)

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    print("Warm-up..."); run(pool, videos[0])

    configs = [
        # Final stacks, ordered most → least aggressive
        ("F9 ONLY",                                    None, {"max_frames": 9}, 9),
        ("F9 + max_init=8",                            {"max_init_masks": 8}, {"max_frames": 9}, 9),
        ("F9 + max_init=8 + vg=0",                     {"max_init_masks": 8, "vg_stride": 0},
                                                       {"max_frames": 9}, 9),
        ("F9 + max_init=8 + maskmem=1",                {"max_init_masks": 8, "num_maskmem": 1},
                                                       {"max_frames": 9}, 9),
        ("F9 + max_init=8 + maskmem=1 + vg=0",         {"max_init_masks": 8, "num_maskmem": 1,
                                                        "vg_stride": 0},
                                                       {"max_frames": 9}, 9),
        ("F9 + max_init=8 + maskmem=2 + vg=0",         {"max_init_masks": 8, "num_maskmem": 2,
                                                        "vg_stride": 0},
                                                       {"max_frames": 9}, 9),
        ("F8 + max_init=8 + maskmem=2 + vg=0",         {"max_init_masks": 8, "num_maskmem": 2,
                                                        "vg_stride": 0},
                                                       {"max_frames": 8}, 8),
    ]

    print(f"\n{'CONFIG':<48} {'E1 t':<7}{'E1 q':<14}{'E2 t':<7}{'E2 q':<14}{'avg':<7}")
    print("-" * 100)
    results = []
    for label, ovs, ovp, mf in configs:
        row = {"cfg": label, "mf": mf}
        for i, v in enumerate(videos):
            sg, dt = run(pool, v, ovs, ovp)
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

    # Find best ✓ (E2 tracks must stay 5/5)
    print(f"\n{'='*100}\nWinner candidates (E2 tracks==5 and E2 long_half==5):")
    keepers = [r for r in results if r["v1_q"]["tracks"] >= 5 and r["v1_q"]["long_half"] >= 5]
    keepers.sort(key=lambda x: x["avg"])
    for r in keepers:
        print(f"  {r['cfg']:<48} {r['avg']:.2f}s  ({1/r['avg']:.3f} Hz)")
    if not keepers:
        print("  (none — re-evaluate)")
        return

    # Stability check: best keeper × 5 repeats
    best = keepers[0]
    print(f"\n=== STABILITY: best config '{best['cfg']}' × 5 ===")
    bl_label = best["cfg"]
    # Re-derive overrides for the winner by label match
    overrides_map = {label: (ovs, ovp, mf) for label, ovs, ovp, mf in configs}
    ovs, ovp, mf = overrides_map[bl_label]
    times = []
    for k in range(5):
        t_e1, _ = (lambda: ((time.time()), 0))()
        _, dt0 = run(pool, videos[0], ovs, ovp)
        _, dt1 = run(pool, videos[1], ovs, ovp)
        avg = (dt0 + dt1) / 2
        times.append(avg)
        print(f"  run {k+1}: E1={dt0:.2f}  E2={dt1:.2f}  avg={avg:.2f}s")
    mean = sum(times) / len(times)
    stdev = (sum((x - mean)**2 for x in times) / len(times)) ** 0.5
    print(f"  → mean={mean:.2f}s  stdev={stdev:.2f}s  ({1/mean:.3f} Hz)")


if __name__ == "__main__":
    main()
