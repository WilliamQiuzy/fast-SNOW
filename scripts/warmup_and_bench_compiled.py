"""Compile + warm-up + mini-200 benchmark, end-to-end.

Steps:
  1. Set TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache
  2. Build pool with enable_compile=True (triggers SAM3.1 multiplex compile)
  3. Run SAM3's warm_up_compilation (default shapes)
  4. Run a few real mini-200 videos covering 15/17/18/32 frames to
     trigger any remaining shape-specific recompiles
  5. Run mini-200 sample (15 videos) and report Hz
"""
from __future__ import annotations

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
# Allow torch.compile to fall back to eager mode on unsupported patterns
# (e.g., SAM3.1 necks.py has `getattr(x, "tensors", x)` which dynamo can't
# trace through).  Without this, the entire compile aborts.
os.environ["TORCH_COMPILE_SUPPRESS_ERRORS"] = "1"
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

VLM4D = ROOT / "benchmark" / "VLM4D-video"
HF = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.offload_state_to_cpu = False
    cfg.sam3.offload_video_to_cpu = False
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = True   # ← critical
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 50
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("=" * 72, flush=True)
    print("STEP 1: Load models with compile=True", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    print(f"  load_all: {time.time() - t0:.1f}s", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("STEP 2: warmup_cuda (DA3 batch sizes)", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    pool.warmup_cuda()
    print(f"  warmup_cuda: {time.time() - t0:.1f}s", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("STEP 3: SAM3.1 multiplex warm_up_compilation (this is the long one)", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    pool.warmup_compile()
    print(f"  warmup_compile: {time.time() - t0:.1f}s", flush=True)
    pool._status = "ready"

    # Step 4: warm a few real shapes from mini-200
    print("\n" + "=" * 72, flush=True)
    print("STEP 4: Warm real-video shapes (15/17/18/32 frames)", flush=True)
    print("=" * 72, flush=True)
    rng = random.Random(42)
    videos = []
    for f in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D / "QA" / f) as fh:
            for q in json.load(fh):
                p = VLM4D / q["video"].replace(HF, "")
                if p.is_file() and p not in videos:
                    videos.append(p)
    sample = rng.sample(videos, 17)
    throwaway, measured = sample[:2], sample[2:]
    for vp in throwaway:
        t0 = time.time()
        pool.run_inference(InferenceRequest(video_path=str(vp)))
        print(f"  warm-up real video {vp.name}: {time.time() - t0:.1f}s", flush=True)

    # Step 5: Real benchmark
    print("\n" + "=" * 72, flush=True)
    print("STEP 5: Mini-200 benchmark with compiled model", flush=True)
    print("=" * 72, flush=True)
    rows = []
    for i, vp in enumerate(measured, 1):
        t0 = time.time()
        r = pool.run_inference(InferenceRequest(video_path=str(vp)))
        dt = time.time() - t0
        if r.status != "ok":
            print(f"  [{i:>2}] {vp.name}: FAILED {r.error_message[:200]}", flush=True)
            continue
        md = (r.four_dsg_dict or {}).get("metadata", {})
        n = md.get("num_frames", 0)
        t = md.get("num_tracks", 0)
        obs = sum(len(x.get("F_k", [])) for x in (r.four_dsg_dict or {}).get("tracks", []))
        hz = n / dt if dt > 0 else 0
        rows.append({"vp": vp.name, "n": n, "tracks": t, "obs": obs, "dt": dt, "hz": hz})
        print(f"  [{i:>2}] {vp.name:30s} frames={n:>2} tracks={t:>2} obs={obs:>3} time={dt:5.2f}s Hz={hz:5.2f}", flush=True)

    print("\n=== Compiled Mini-200 Summary ===", flush=True)
    if rows:
        hzs = [r["hz"] for r in rows]
        print(f"  Hz median: {statistics.median(hzs):.2f}", flush=True)
        print(f"  Hz mean:   {statistics.mean(hzs):.2f}", flush=True)
        print(f"  Hz min/max: {min(hzs):.2f} / {max(hzs):.2f}", flush=True)
        print(f"  >= 10 Hz: {sum(1 for h in hzs if h >= 10)}/{len(rows)}", flush=True)
        print(f"  >= 12 Hz: {sum(1 for h in hzs if h >= 12)}/{len(rows)}", flush=True)
    print("\nDone.  TORCHINDUCTOR_CACHE_DIR persisted at /tmp/torch_inductor_cache.", flush=True)


if __name__ == "__main__":
    main()
