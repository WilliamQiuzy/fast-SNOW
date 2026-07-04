"""Measure the FA3 patch alone (without fa3_everywhere). The patch also
unblocks the 19 *default-True* MultiheadAttention modules that were
silently using SDPA due to the upstream qkv-same-embed bug.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def count_fa3_during(pool, videos):
    """Instrument flash_attn_func + return total invocations across N videos."""
    import sam3.perflib.fa3 as fa3_mod
    orig = fa3_mod.flash_attn_func
    counts = Counter()
    def cf(q, k, v):
        counts["total"] += 1
        return orig(q, k, v)
    fa3_mod.flash_attn_func = cf
    # Re-bind in any module that aliased it
    import sys
    for modname in [m for m in sys.modules if "sam3" in m]:
        mod = sys.modules[modname]
        if hasattr(mod, "flash_attn_func") and mod.flash_attn_func is orig:
            mod.flash_attn_func = cf
    try:
        for v in videos:
            pool.run_inference(InferenceRequest(video_path=v, question=None))
    finally:
        fa3_mod.flash_attn_func = orig
    return counts["total"]


def bench(pool, videos, label, runs=8):
    print(f"\n=== {label} ===")
    pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    avgs = []
    for k in range(runs):
        ts, q_all = [], []
        for v in videos:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            ts.append(time.time() - t0)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            q_all.append(quality(sg, pool.config.sampling.max_frames or 32) if sg else (0,0,0))
        avg = sum(ts) / 2
        avgs.append(avg)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {q_all[0]}  E2={ts[1]:.2f}s {q_all[1]}  avg={avg:.2f}s")
    mean = sum(avgs)/len(avgs)
    stdev = (sum((x-mean)**2 for x in avgs)/len(avgs))**0.5
    print(f"  → mean={mean:.2f}s ± {stdev:.3f}s ({1/mean:.3f} Hz)")
    return mean, stdev


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = True
    cfg.sam3.compile_mask_decoder_transformer = False
    cfg.sam3.fa3_everywhere = False  # default shipped

    print("Loading (compile + patched MultiheadAttention, NO fa3_everywhere)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("\n# FA3 call count under default config (post-patch)")
    n = count_fa3_during(pool, videos)
    print(f"  FA3 calls per (E1+E2) run: {n}")

    bench(pool, videos, "Post-patch, default fa3_everywhere=False")


if __name__ == "__main__":
    main()
