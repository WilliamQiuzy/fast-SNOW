"""With the FA3 patches now in place, the decoder TwoWayTransformer Attention
modules actually do go through FA3 (when we set use_fa3=True on them).
Retry the decoder compile path now to see if quality holds AND speed wins.

Combinations:
  A. baseline (just compile_memory_encoder)
  B. + decoder use_fa3=True (no compile)  — establishes FA3-only baseline
  C. + decoder use_fa3=True + transformer.compile  — the previous "marginal" path
  D. + decoder use_fa3=True + transformer.compile (mode=default)
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
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


def bench(pool, videos, label, runs=6):
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
    cfg.sam3.fa3_everywhere = False
    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    dec = pool._sam3._predictor.model.tracker.model.sam_mask_decoder
    tfm = dec.transformer
    attn_modules = []
    for layer in tfm.layers:
        for name in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            attn_modules.append(getattr(layer, name))
    if hasattr(tfm, "final_attn_token_to_image"):
        attn_modules.append(tfm.final_attn_token_to_image)

    a_mean, a_std = bench(pool, videos, "A. baseline (encoder.compile only)")

    # Enable FA3 on decoder attentions
    for a in attn_modules: a.use_fa3 = True
    b_mean, b_std = bench(pool, videos, "B. + decoder use_fa3=True, no compile")

    tfm_orig = tfm.forward
    # C: compile with reduce-overhead
    try:
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
        c_mean, c_std = bench(pool, videos, "C. + transformer.compile (reduce-overhead)")
    finally:
        tfm.forward = tfm_orig

    # D: compile with default mode
    try:
        tfm.forward = torch.compile(tfm_orig, mode="default", dynamic=True)
        d_mean, d_std = bench(pool, videos, "D. + transformer.compile (default mode)")
    finally:
        tfm.forward = tfm_orig

    # Reset use_fa3 to False
    for a in attn_modules: a.use_fa3 = False

    print("\n" + "=" * 80)
    print(f"{'Variant':<55} {'time':<14} {'Hz':<8} {'Δ%':<8}")
    print("-" * 80)
    for name, m, s in [("A. encoder.compile only", a_mean, a_std),
                       ("B. + decoder use_fa3=True", b_mean, b_std),
                       ("C. + decoder compile (reduce-overhead)", c_mean, c_std),
                       ("D. + decoder compile (default)", d_mean, d_std)]:
        d = (m - a_mean)/a_mean*100 if a_mean else 0
        print(f"{name:<55} {m:.2f}±{s:.2f}s   {1/m:<8.3f} {d:+.2f}%")


if __name__ == "__main__":
    main()
