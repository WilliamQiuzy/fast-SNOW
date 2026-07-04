"""Try every compile path (torch.compile + TRT) on backbone / encoder /
decoder and report which combinations are wins.

Variants (each starts from a fresh "no-compile" state via uninstall/restore):
  1. baseline
  2. encoder.compile (inductor)   — already shipped as default
  3. encoder.compile (TRT)
  4. backbone.compile (inductor)
  5. backbone + encoder.compile (inductor)
  6. mask_decoder.compile (fullgraph=False) — try to fix silent failure
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
import torch_tensorrt  # noqa: F401 — registers TRT backends
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def benchmark(pool, videos, label, runs=4):
    print(f"\n=== {label} ===")
    try:
        t0 = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        print(f"  warmup: {time.time()-t0:.2f}s ({resp.status})")
    except Exception as e:
        print(f"  warmup failed: {type(e).__name__}: {e}")
        return None, None, (0, 0, 0), (0, 0, 0)

    avgs, qs_e1, qs_e2 = [], [], []
    for k in range(runs):
        ts = []; ok = True
        for i, v in enumerate(videos):
            t0 = time.time()
            try:
                resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            except Exception as e:
                print(f"  run {k+1} {('E1','E2')[i]} crashed: {type(e).__name__}: {e}")
                ok = False; break
            dt = time.time() - t0
            ts.append(dt)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            mf = pool.config.sampling.max_frames or 32
            q = quality(sg, mf) if sg else (0, 0, 0)
            if i == 0: qs_e1.append(q)
            else: qs_e2.append(q)
        if not ok:
            break
        avg = sum(ts) / 2
        avgs.append(avg)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {qs_e1[-1]}  E2={ts[1]:.2f}s {qs_e2[-1]}  avg={avg:.2f}s")
    if not avgs:
        return None, None, (0, 0, 0), (0, 0, 0)
    mean = sum(avgs) / len(avgs)
    stdev = (sum((x - mean) ** 2 for x in avgs) / len(avgs)) ** 0.5
    print(f"  → mean={mean:.2f}s ± {stdev:.2f}s ({1/mean:.3f} Hz)")
    return mean, stdev, qs_e1[-1] if qs_e1 else (0, 0, 0), qs_e2[-1] if qs_e2 else (0, 0, 0)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False  # manage manually

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    enc = pool._sam3._predictor.model.tracker.model.transformer.encoder
    tracker_model = pool._sam3._predictor.model.tracker.model
    decoder = getattr(tracker_model, "sam_mask_decoder", None)
    backbone = pool._sam3._predictor.model.detector.backbone

    results = []

    # 1. Baseline
    print(f"\n{'#'*60}\n# 1. BASELINE\n{'#'*60}")
    base_mean, base_std, base_e1, base_e2 = benchmark(pool, videos, "baseline (no compile)")
    results.append(("baseline", base_mean, base_std, base_e1, base_e2))

    def try_variant(name, install_fn, uninstall_fn):
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        try:
            install_fn()
        except Exception as e:
            print(f"  install threw: {type(e).__name__}: {e}")
            results.append((name, None, None, (0, 0, 0), (0, 0, 0)))
            return
        try:
            m, s, qe1, qe2 = benchmark(pool, videos, name)
            results.append((name, m, s, qe1, qe2))
        finally:
            try:
                uninstall_fn()
            except Exception as e:
                print(f"  uninstall threw: {e}")

    # 2. encoder.compile (inductor)
    enc_orig = enc.forward
    try_variant(
        "2. encoder.compile (inductor)",
        lambda: setattr(enc, "forward", torch.compile(enc_orig, mode="reduce-overhead", dynamic=True)),
        lambda: setattr(enc, "forward", enc_orig),
    )

    # 3. encoder.compile (TRT)
    try_variant(
        "3. encoder.compile (TRT)",
        lambda: setattr(enc, "forward", torch.compile(enc_orig, backend="torch_tensorrt", dynamic=True)),
        lambda: setattr(enc, "forward", enc_orig),
    )

    # 4. backbone.compile (inductor)
    bb_orig = backbone.forward
    try_variant(
        "4. backbone.compile (inductor)",
        lambda: setattr(backbone, "forward", torch.compile(bb_orig, mode="reduce-overhead", dynamic=True)),
        lambda: setattr(backbone, "forward", bb_orig),
    )

    # 5. backbone + encoder.compile (inductor)
    def install_both():
        backbone.forward = torch.compile(bb_orig, mode="reduce-overhead", dynamic=True)
        enc.forward = torch.compile(enc_orig, mode="reduce-overhead", dynamic=True)
    def uninstall_both():
        backbone.forward = bb_orig
        enc.forward = enc_orig
    try_variant("5. backbone+encoder.compile (inductor)", install_both, uninstall_both)

    # 6. mask_decoder.compile with fullgraph=False
    if decoder is not None:
        dec_orig = decoder.forward
        try_variant(
            "6. mask_decoder.compile (fullgraph=False)",
            lambda: setattr(decoder, "forward",
                            torch.compile(dec_orig, dynamic=True, fullgraph=False)),
            lambda: setattr(decoder, "forward", dec_orig),
        )

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Variant':<46} {'time':<16} {'Hz':<8} {'Δ%':<10} {'E1 q':<14} {'E2 q':<14}")
    print("-" * 100)
    bm = results[0][1]
    for name, m, s, qe1, qe2 in results:
        if m is None:
            print(f"{name:<46} FAILED")
            continue
        delta = (m - bm) / bm * 100 if bm else 0
        print(f"{name:<46} {m:.2f}±{s:.2f}s  {1/m:<8.3f} {delta:+6.1f}%   {str(qe1):<14} {str(qe2):<14}")


if __name__ == "__main__":
    main()
