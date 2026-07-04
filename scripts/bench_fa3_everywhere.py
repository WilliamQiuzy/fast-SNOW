"""A/B: baseline (use_fa3 mix as model loads) vs FA3-everywhere."""
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
    # Warm up with one full inference to absorb any first-call costs.
    try:
        pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    except Exception as e:
        print(f"  warmup err: {type(e).__name__}: {e}")
        return None
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


def count_fa3(root):
    c_t = c_f = 0
    for _, m in root.named_modules():
        if hasattr(m, "use_fa3"):
            if getattr(m, "use_fa3"): c_t += 1
            else: c_f += 1
    return c_t, c_f


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = True  # keep shipped default
    cfg.sam3.compile_mask_decoder_transformer = False
    cfg.sam3.fa3_everywhere = False  # baseline first

    print("=== Loading pool (fa3_everywhere=False) ===")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    root = pool._sam3._predictor.model
    t, f = count_fa3(root)
    print(f"  use_fa3 count: True={t}  False={f}")

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    a_mean, a_std = bench(pool, videos, "A. baseline (59 FA3, 20 SDPA)")

    # Flip remaining to FA3 in-place
    flipped = 0
    for _, m in root.named_modules():
        if hasattr(m, "use_fa3") and not getattr(m, "use_fa3"):
            m.use_fa3 = True; flipped += 1
    t2, f2 = count_fa3(root)
    print(f"\n=== Flipped {flipped} modules to FA3 → now True={t2}  False={f2} ===")

    b_mean, b_std = bench(pool, videos, "B. FA3-everywhere (79 FA3, 0 SDPA)")

    if a_mean and b_mean:
        delta = b_mean - a_mean
        delta_pct = delta / a_mean * 100
        print(f"\nDelta: {b_mean:.3f} - {a_mean:.3f} = {delta:+.3f}s ({delta_pct:+.2f}%)")
        # Pooled stdev for significance: if |delta| > 2 * combined_stdev → meaningful
        comb = (a_std**2 + b_std**2)**0.5
        if abs(delta) > 2 * comb:
            print(f"  → SIGNIFICANT (|delta|={abs(delta):.3f} > 2σ={2*comb:.3f})")
        else:
            print(f"  → within noise (|delta|={abs(delta):.3f} ≤ 2σ={2*comb:.3f})")


if __name__ == "__main__":
    main()
