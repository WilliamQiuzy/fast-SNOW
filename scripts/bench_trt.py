"""Benchmark TensorRT compilation of the SAM 3.1 memory encoder."""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
import torch_tensorrt  # noqa: F401 — registers torch._dynamo backends "tensorrt" and "torch_tensorrt"
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def benchmark(pool, videos, label, runs=5):
    print(f"\n=== {label} ===")
    t0 = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    print(f"  warmup: {time.time()-t0:.2f}s ({resp.status})")

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
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False  # we'll do it manually with TRT

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # ── Baseline ──
    base_mean, base_std, base_e1, base_e2 = benchmark(pool, videos, "BASELINE")

    # ── TRT compile via torch.compile backend="tensorrt" ──
    enc = pool._sam3._predictor.model.tracker.model.transformer.encoder
    enc_orig = enc.forward
    print("\nTry: torch.compile(encoder.forward, backend='tensorrt') …")
    try:
        enc.forward = torch.compile(enc.forward, backend="tensorrt", dynamic=True)
        trt_mean, trt_std, trt_e1, trt_e2 = benchmark(pool, videos, "TRT-compiled encoder")
    except Exception as e:
        print(f"TRT compile or run threw: {e}")
        import traceback; traceback.print_exc()
        trt_mean = trt_std = None
        trt_e1 = trt_e2 = (0, 0, 0)
    finally:
        enc.forward = enc_orig

    # ── Summary ──
    print("\n" + "=" * 90)
    print(f"{'Variant':<32} {'time':<16} {'Hz':<8} {'Δ%':<8} {'E1 q':<14} {'E2 q':<14}")
    rows = [("baseline", base_mean, base_std, base_e1, base_e2)]
    if trt_mean is not None:
        rows.append(("TRT-compiled encoder", trt_mean, trt_std, trt_e1, trt_e2))
    for name, m, s, qe1, qe2 in rows:
        delta = (m - base_mean) / base_mean * 100 if base_mean else 0
        print(f"{name:<32} {m:.2f}±{s:.2f}s  {1/m:<8.3f} {delta:+.1f}%   "
              f"{str(qe1):<14} {str(qe2):<14}")


if __name__ == "__main__":
    main()
