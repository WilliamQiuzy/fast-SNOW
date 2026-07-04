"""Force PyTorch SDPA dispatcher to ALWAYS pick the flash kernel by disabling
math/mem-efficient backends. If shapes don't support flash, this errors.

Goal: ensure every SDPA call uses Flash Attention 2 (the model authors already
turned use_fa3=True on long-seqlen modules; the remaining 20 SDPA modules let
the dispatcher pick — usually flash, sometimes mem-efficient).
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def bench(pool, videos, label, runs=5, sdpa_context=None):
    print(f"\n=== {label} ===")
    try:
        if sdpa_context is not None:
            with sdpa_context():
                pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        else:
            pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    except Exception as e:
        print(f"  warmup err: {type(e).__name__}: {e}")
        return None
    avgs = []
    for k in range(runs):
        ts, q_all = [], []
        for v in videos:
            t0 = time.time()
            try:
                if sdpa_context is not None:
                    with sdpa_context():
                        resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
                else:
                    resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            except Exception as e:
                print(f"  run {k+1} crashed: {type(e).__name__}: {e}")
                return None
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

    a_mean, a_std = bench(pool, videos, "A. baseline (SDPA dispatcher chooses)")

    # B. Force flash attention only
    b_mean, b_std = bench(
        pool, videos, "B. SDPA forced to FLASH only",
        sdpa_context=lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION]),
    )

    # C. Force flash + cuDNN (cuDNN flash is sometimes faster)
    c_mean, c_std = bench(
        pool, videos, "C. SDPA forced to FLASH + CUDNN",
        sdpa_context=lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]),
    )

    print("\n" + "=" * 80)
    print(f"{'Variant':<40} {'time':<14} {'Hz':<8} {'Δ%':<8}")
    print("-" * 80)
    rows = [("A. SDPA dispatcher", a_mean, a_std)]
    if b_mean is not None: rows.append(("B. SDPA forced FLASH", b_mean, b_std))
    if c_mean is not None: rows.append(("C. SDPA forced FLASH + CUDNN", c_mean, c_std))
    for name, m, s in rows:
        d = (m - a_mean)/a_mean*100 if a_mean else 0
        print(f"{name:<40} {m:.2f}±{s:.2f}s   {1/m:<8.3f} {d:+.2f}%")


if __name__ == "__main__":
    main()
