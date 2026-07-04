"""Benchmark torch.compile on the SAM 3.1 memory encoder + measure quality.

Three configs:
  1. baseline (no compile)
  2. encoder.compile (the TransformerEncoderDecoupledCrossAttention.forward)
  3. encoder.compile + mask decoder compile (if it has stable shapes)
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


def benchmark(pool, videos, label, runs=5):
    print(f"\n=== {label} ===")
    print("  warmup ...", end="", flush=True)
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
    return mean, stdev, qs_e1[-1], qs_e2[-1]


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False  # Make sure no other compile path

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # ── (1) Baseline ──
    base_mean, base_std, base_e1, base_e2 = benchmark(pool, videos, "BASELINE (no compile)")

    # ── (2) Compile the encoder ──
    enc = pool._sam3._predictor.model.tracker.model.transformer.encoder
    enc_orig = enc.forward
    print("\nCompiling encoder.forward (reduce-overhead, dynamic) …", flush=True)
    enc.forward = torch.compile(enc.forward, mode="reduce-overhead", dynamic=True)
    enc_mean, enc_std, enc_e1, enc_e2 = benchmark(pool, videos, "ENCODER COMPILED")
    enc.forward = enc_orig  # restore

    # ── (3) Compile sam_mask_decoder too if present ──
    tracker_model = pool._sam3._predictor.model.tracker.model
    decoder = getattr(tracker_model, "sam_mask_decoder", None)
    enc.forward = torch.compile(enc.forward, mode="reduce-overhead", dynamic=True)
    dec_orig = None
    if decoder is not None:
        dec_orig = decoder.forward
        try:
            decoder.forward = torch.compile(decoder.forward, mode="reduce-overhead", dynamic=True)
            print("\nCompiling sam_mask_decoder.forward …", flush=True)
            ed_mean, ed_std, ed_e1, ed_e2 = benchmark(pool, videos, "ENCODER + DECODER COMPILED")
        except Exception as e:
            print(f"Decoder compile threw: {e}")
            ed_mean = ed_std = None
            ed_e1 = ed_e2 = (0, 0, 0)
        finally:
            decoder.forward = dec_orig
    enc.forward = enc_orig

    # ── Summary ──
    print("\n" + "=" * 90)
    print(f"{'Variant':<32} {'time':<16} {'Hz':<8} {'Δ%':<8} {'E1 q':<14} {'E2 q':<14}")
    rows = [("baseline", base_mean, base_std, base_e1, base_e2),
            ("encoder compiled", enc_mean, enc_std, enc_e1, enc_e2)]
    if ed_mean is not None:
        rows.append(("encoder+decoder compiled", ed_mean, ed_std, ed_e1, ed_e2))
    for name, m, s, qe1, qe2 in rows:
        delta = (m - base_mean) / base_mean * 100 if base_mean else 0
        print(f"{name:<32} {m:.2f}±{s:.2f}s  {1/m:<8.3f} {delta:+.1f}%   "
              f"{str(qe1):<14} {str(qe2):<14}")


if __name__ == "__main__":
    main()
