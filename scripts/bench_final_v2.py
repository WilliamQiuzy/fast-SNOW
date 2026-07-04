"""Re-verify the encoder+decoder compile combo with 10 runs."""
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


def bench(pool, videos, label, runs=10):
    print(f"\n=== {label} ===")
    # Two warmup runs to fully prime caches and trigger any compile
    for _ in range(2):
        try:
            pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        except Exception as e:
            print(f"  warmup err: {e}")
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


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    # Both compiles via load_all
    cfg.sam3.compile_memory_encoder = True
    cfg.sam3.compile_mask_decoder_transformer = True

    print("Loading (both compiles enabled via warm_server.load_all)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    bench(pool, videos, "warm_server.load_all both compiles")


if __name__ == "__main__":
    main()
