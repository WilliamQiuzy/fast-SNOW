"""Final clean A/B: encoder.compile vs encoder.compile + decoder(use_fa3+compile).

5 runs each for statistical power. The decoder add-on either gives a real ~1%
or it's noise.
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


def bench(pool, videos, label, runs=5):
    print(f"\n=== {label} ===")
    try:
        t0 = time.time()
        pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        print(f"  warmup: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  warmup err: {e}")
        return None
    avgs = []
    for k in range(runs):
        ts = []; q_all = []
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
    cfg.sam3.compile_memory_encoder = True  # Already-shipped default
    print("Loading (encoder.compile=True default)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    a_mean, a_std = bench(pool, videos, "A. baseline (encoder.compile only)")

    # Add the decoder optimization on top
    dec = pool._sam3._predictor.model.tracker.model.sam_mask_decoder
    tfm = dec.transformer
    attn_modules = []
    for layer in tfm.layers:
        for name in ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]:
            attn_modules.append(getattr(layer, name))
    if hasattr(tfm, "final_attn_token_to_image"):
        attn_modules.append(tfm.final_attn_token_to_image)
    for a in attn_modules:
        a.use_fa3 = True

    tfm_orig = tfm.forward
    tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
    try:
        b_mean, b_std = bench(pool, videos, "B. encoder.compile + decoder(use_fa3+compile)")
    finally:
        tfm.forward = tfm_orig
        for a in attn_modules: a.use_fa3 = False

    if a_mean and b_mean:
        delta_pct = (b_mean - a_mean) / a_mean * 100
        print(f"\nDelta: {b_mean:.3f} - {a_mean:.3f} = {b_mean-a_mean:+.3f}s ({delta_pct:+.2f}%)")
        combined_std = (a_std**2 + b_std**2)**0.5
        # 95% CI on the difference (approximate, assuming runs are independent)
        ci = 1.96 * combined_std
        print(f"  combined stdev: {combined_std:.3f}s; 95% CI on diff: ±{ci:.3f}s")
        if abs(b_mean - a_mean) > 2 * combined_std:
            print("  → SIGNIFICANT")
        else:
            print("  → NOT significant — within noise")


if __name__ == "__main__":
    main()
