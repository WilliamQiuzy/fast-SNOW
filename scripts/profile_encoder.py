"""Profile: how much wall-time does the SAM 3.1 multiplex memory encoder
actually consume during a typical run? And does torch.compile help?

Strategy:
  1. Run the pipeline normally and count encoder-forward calls + total time.
  2. Monkey-patch the encoder.forward to time itself with CUDA events.
  3. Then try torch.compile and re-measure.
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


def install_encoder_timer(predictor, stats):
    enc = predictor.model.tracker.model.transformer.encoder
    cls = type(enc)
    orig = cls.forward

    def timed_forward(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig(self, *args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        stats["total_ms"] += start.elapsed_time(end)
        stats["calls"] += 1
        return out

    cls.forward = timed_forward
    return cls, orig


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False

    print("Loading pool...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    # Warm up
    print("Warm-up..."); pool.run_inference(InferenceRequest(video_path=videos[0], question=None))

    # ── Run with encoder timer installed ──
    print("\n=== Baseline (no compile) — measure encoder cost ===")
    stats = {"total_ms": 0.0, "calls": 0}
    cls, orig = install_encoder_timer(pool._sam3._predictor, stats)
    try:
        wall_times = []
        for k in range(5):
            stats_before = stats["total_ms"]
            calls_before = stats["calls"]
            t0 = time.time()
            for v in videos:
                pool.run_inference(InferenceRequest(video_path=v, question=None))
            wall = time.time() - t0
            wall_times.append(wall)
            d_ms = stats["total_ms"] - stats_before
            d_calls = stats["calls"] - calls_before
            print(f"  run {k+1}: wall={wall:.2f}s  encoder={d_ms/1000:.2f}s ({d_calls} calls)"
                  f"  encoder/wall={100*d_ms/1000/wall:.1f}%")
        avg_wall = sum(wall_times)/len(wall_times)/2
        avg_enc = stats["total_ms"]/1000/5/2
        print(f"  Avg per-video: wall={avg_wall:.2f}s  encoder={avg_enc:.2f}s ({100*avg_enc/avg_wall:.1f}%)")
    finally:
        cls.forward = orig

    # ── Now compile the encoder and re-measure ──
    print("\n=== Compile encoder (reduce-overhead mode) ===")
    enc = pool._sam3._predictor.model.tracker.model.transformer.encoder
    enc_orig = enc.forward
    try:
        enc.forward = torch.compile(enc.forward, mode="reduce-overhead", dynamic=True)
    except Exception as e:
        print(f"Compile failed: {e}")
        return

    # Re-install timer on top of compiled forward
    stats = {"total_ms": 0.0, "calls": 0}
    cls, orig = install_encoder_timer(pool._sam3._predictor, stats)
    try:
        # Warmup to trigger compile
        print("  warmup (triggers compile)...")
        try:
            t0 = time.time()
            pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
            print(f"  warmup {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"  warmup compile threw: {e}")
            return

        wall_times = []
        for k in range(5):
            stats_before = stats["total_ms"]
            calls_before = stats["calls"]
            t0 = time.time()
            for v in videos:
                pool.run_inference(InferenceRequest(video_path=v, question=None))
            wall = time.time() - t0
            wall_times.append(wall)
            d_ms = stats["total_ms"] - stats_before
            d_calls = stats["calls"] - calls_before
            print(f"  run {k+1}: wall={wall:.2f}s  encoder={d_ms/1000:.2f}s ({d_calls} calls)")
        avg_wall = sum(wall_times)/len(wall_times)/2
        avg_enc = stats["total_ms"]/1000/5/2
        print(f"  Avg per-video: wall={avg_wall:.2f}s  encoder={avg_enc:.2f}s")
    finally:
        cls.forward = orig
        enc.forward = enc_orig


if __name__ == "__main__":
    main()
