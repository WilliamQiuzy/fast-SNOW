"""Benchmark Fast-SAM2-style token pruning.

Runs two passes:
  1. Baseline (use_token_pruning=False): the locked 4.68s config.
  2. Pruned (use_token_pruning=True): same config + FastSAM saliency-driven
     token pruning of memory-attention queries.

Measures per-video time and 4DSG quality (track count, longest track) over
5 runs each.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def benchmark(pool, videos, label, runs=5):
    print(f"\n=== {label} ===")
    print(f"  warmup ...", end="", flush=True)
    t0 = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    print(f"  {time.time()-t0:.2f}s ({resp.status})")

    avgs, qs_e1, qs_e2 = [], [], []
    for k in range(runs):
        ts = []
        for i, v in enumerate(videos):
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            dt = time.time() - t0
            ts.append(dt)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            mf = pool.config.sampling.max_frames or 32
            q = quality(sg, mf) if sg else (0, 0, 0)
            if i == 0: qs_e1.append(q)
            else: qs_e2.append(q)
        avg = sum(ts) / 2
        avgs.append(avg)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {qs_e1[-1]}  E2={ts[1]:.2f}s {qs_e2[-1]}  avg={avg:.2f}s")

    mean = sum(avgs) / len(avgs)
    stdev = (sum((x - mean) ** 2 for x in avgs) / len(avgs)) ** 0.5
    print(f"  → mean={mean:.2f}s ± {stdev:.2f}s ({1/mean:.3f} Hz)")
    print(f"  → E1 quality: {qs_e1[0]} (consistent across runs)")
    print(f"  → E2 quality: {qs_e2[0]}")
    return mean, stdev, qs_e1[0], qs_e2[0]


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    # Start with token pruning OFF; toggle later
    cfg.sam3.use_token_pruning = False

    print("Loading pool with current rose_config defaults...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # ── Baseline ──
    pool.config.sam3.use_token_pruning = False
    base_mean, base_std, base_e1, base_e2 = benchmark(pool, videos, "BASELINE (no token pruning)")

    # ── Token pruning ON ──
    pool.config.sam3.use_token_pruning = True
    pool.config.sam3.token_prune_dilate_cells = 2
    pp_mean, pp_std, pp_e1, pp_e2 = benchmark(pool, videos, "TOKEN PRUNING (dilate=2)")

    # ── With wider dilation (safer) ──
    pool.config.sam3.token_prune_dilate_cells = 4
    pp4_mean, pp4_std, pp4_e1, pp4_e2 = benchmark(pool, videos, "TOKEN PRUNING (dilate=4)")

    # ── Aggressive (dilate=1) ──
    pool.config.sam3.token_prune_dilate_cells = 1
    pp1_mean, pp1_std, pp1_e1, pp1_e2 = benchmark(pool, videos, "TOKEN PRUNING (dilate=1, aggressive)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"{'Variant':<35} {'time':<14} {'Hz':<7} {'Δ vs base':<10}  {'E1 q':<14} {'E2 q':<14}")
    rows = [
        ("baseline",                   base_mean, base_std, base_e1, base_e2),
        ("token pruning dilate=4",     pp4_mean,  pp4_std,  pp4_e1,  pp4_e2),
        ("token pruning dilate=2",     pp_mean,   pp_std,   pp_e1,   pp_e2),
        ("token pruning dilate=1",     pp1_mean,  pp1_std,  pp1_e1,  pp1_e2),
    ]
    for name, mean, std, qe1, qe2 in rows:
        delta = (mean - base_mean) / base_mean * 100
        print(f"{name:<35} {mean:.2f}±{std:.2f}s   {1/mean:<7.3f} {delta:+.1f}%      "
              f"{str(qe1):<14} {str(qe2):<14}")


if __name__ == "__main__":
    main()
