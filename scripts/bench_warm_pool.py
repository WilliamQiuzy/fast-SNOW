"""Benchmark ROSE with the WarmModelPool — measures POST-warmup speed.

Bundles 4 optimizations in one config:
  1. Warm models (no per-video unload)
  2. SAM3 use_fa3=True (Flash Attention 3)
  3. SAM3 offload_state_to_cpu=False (keep state on H200, no CPU↔GPU copy)
  4. SAM3 enable_compile=True (optional)

Excludes warmup time:
  - Loads + DA3 batch-size warmup + (optionally) SAM3 compile warmup are
    counted SEPARATELY from per-video inference.
  - First 2 inferences are also THROWAWAY (in case any caches still cold).
  - Reported Hz is computed only over the measured (post-throwaway) runs.

Usage:
    python scripts/bench_warm_pool.py              # without compile
    python scripts/bench_warm_pool.py --compile    # with torch.compile
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

VLM4D_ROOT = ROOT / "benchmark" / "VLM4D-video"
HF_PREFIX = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"

N_THROWAWAY = 2
N_MEASURED = 12

MAX_FRAMES = 32
TARGET_FPS = 10.0


def hf_url_to_local(url: str) -> Path:
    return VLM4D_ROOT / url.replace(HF_PREFIX, "")


def make_config(use_compile: bool) -> ROSEConfig:
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sampling.max_frames = MAX_FRAMES
    cfg.sampling.target_fps = TARGET_FPS

    # All 4 optimizations
    cfg.sam3.use_fa3 = True
    cfg.sam3.offload_state_to_cpu = False     # H200 has plenty of VRAM
    cfg.sam3.offload_video_to_cpu = False
    cfg.sam3.enable_compile = use_compile

    cfg.seed = 42
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile (adds ~2-4 min warmup)")
    args = ap.parse_args()

    rng = random.Random(42)

    # Collect unique videos from mini-200
    videos = []
    for fname in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D_ROOT / "QA" / fname) as f:
            for q in json.load(f):
                p = hf_url_to_local(q["video"])
                if p.is_file() and p not in videos:
                    videos.append(p)
    sample = rng.sample(videos, min(N_THROWAWAY + N_MEASURED, len(videos)))
    throwaway = sample[:N_THROWAWAY]
    measured = sample[N_THROWAWAY:N_THROWAWAY + N_MEASURED]

    print(f"\n{'='*70}")
    print(f"Mode: {'COMPILE+FA3+WARM+NO-OFFLOAD' if args.compile else 'FA3+WARM+NO-OFFLOAD (no compile)'}")
    print(f"max_frames={MAX_FRAMES}, target_fps={TARGET_FPS}")
    print(f"Throwaway runs: {N_THROWAWAY},  Measured runs: {N_MEASURED}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Phase 1: Warmup (NOT counted in Hz)
    # ------------------------------------------------------------------
    cfg = make_config(use_compile=args.compile)
    pool = WarmModelPool(cfg)

    t_load = time.time()
    pool.load_all()
    load_dt = time.time() - t_load
    print(f"[warmup] Models loaded:        {load_dt:6.1f}s")

    t_cuda = time.time()
    pool.warmup_cuda()
    cuda_dt = time.time() - t_cuda
    print(f"[warmup] DA3 CUDA kernels:     {cuda_dt:6.1f}s  (sizes: {pool._da3_warmed_sizes})")

    if args.compile:
        t_comp = time.time()
        pool.warmup_compile()
        comp_dt = time.time() - t_comp
        print(f"[warmup] SAM3 torch.compile:   {comp_dt:6.1f}s")

    pool._status = "ready"
    print(f"[warmup] Total warmup time:    {time.time() - t_load:6.1f}s\n")

    # ------------------------------------------------------------------
    # Phase 2: Throwaway runs (NOT counted)
    # ------------------------------------------------------------------
    print("[throwaway] running 2 videos to warm any remaining JIT caches...")
    for i, vp in enumerate(throwaway, 1):
        try:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=str(vp)))
            print(f"  throwaway {i}: {vp.name:35s}  status={resp.status}  time={time.time()-t0:5.1f}s")
        except Exception as exc:
            print(f"  throwaway {i}: FAILED ({type(exc).__name__}: {str(exc)[:80]})")

    # ------------------------------------------------------------------
    # Phase 3: Measured runs
    # ------------------------------------------------------------------
    print(f"\n[measured] running {len(measured)} videos for timing...")
    rows = []
    for i, vp in enumerate(measured, 1):
        try:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=str(vp)))
            dt = time.time() - t0
            if resp.status != "ok":
                print(f"  [{i:>2}] {vp.name:35s}  FAILED: {resp.error_message}")
                continue
            md = (resp.four_dsg_dict or {}).get("metadata", {})
            n_frames = md.get("num_frames", 0)
            n_tracks = md.get("num_tracks", 0)
            hz = n_frames / dt if dt > 0 else 0.0
            rows.append({"video": vp.name, "n_frames": n_frames, "n_tracks": n_tracks,
                         "time_s": dt, "hz": hz})
            print(f"  [{i:>2}] {vp.name:35s}  frames={n_frames:>2}  tracks={n_tracks:>2}  time={dt:5.1f}s  Hz={hz:5.2f}")
        except Exception as exc:
            print(f"  [{i:>2}] {vp.name:35s}  FAILED: {type(exc).__name__}: {str(exc)[:80]}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    print(f"\n{'='*70}\n  POST-WARMUP THROUGHPUT\n{'='*70}")
    if rows:
        hzs = [r["hz"] for r in rows]
        ts = [r["n_tracks"] for r in rows]
        times = [r["time_s"] for r in rows]
        print(f"  Hz median:          {statistics.median(hzs):5.2f}")
        print(f"  Hz mean:            {statistics.mean(hzs):5.2f}")
        print(f"  Hz min / max:       {min(hzs):5.2f} / {max(hzs):5.2f}")
        print(f"  Inference time med: {statistics.median(times):5.1f}s")
        print(f"  num_tracks median:  {statistics.median(ts)}")
        print(f"  num_tracks min/max: {min(ts)} / {max(ts)}")

        by_t = defaultdict(list)
        for r in rows:
            by_t[r["n_tracks"]].append(r["hz"])
        print(f"\n  num_tracks  median Hz")
        for t in sorted(by_t):
            print(f"    {t:>3}  ({len(by_t[t]):>2} vids)  {statistics.median(by_t[t]):5.2f}")

        ten_hz = [r for r in rows if r["hz"] >= 10.0]
        if ten_hz:
            max_t = max(r["n_tracks"] for r in ten_hz)
            print(f"\n  ✓ {len(ten_hz)}/{len(rows)} videos at ≥ 10 Hz   max num_tracks at 10 Hz: {max_t}")
        else:
            print(f"\n  ✗ NO video at ≥ 10 Hz  (best: {max(hzs):.2f} Hz)")

    out = ROOT / "benchmark" / f"warm_pool_{'compile' if args.compile else 'nocompile'}.json"
    with open(out, "w") as f:
        json.dump({"mode": "compile" if args.compile else "nocompile",
                   "max_frames": MAX_FRAMES, "target_fps": TARGET_FPS,
                   "rows": rows}, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
