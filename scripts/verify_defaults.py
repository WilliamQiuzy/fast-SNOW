"""Verify the new defaults — no overrides. Should reproduce ~4.7s ± 0.05s.

Picks up rose_config.py's defaults directly.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def main():
    cfg = ROSEConfig()
    # Only set the absolute required runtime flags; everything else comes from
    # config defaults so we can see if the new defaults work end-to-end.
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False

    print(f"sampling.max_frames = {cfg.sampling.max_frames}")
    print(f"sam3.num_maskmem = {cfg.sam3.num_maskmem}")
    print(f"sam3.memory_temporal_stride = {cfg.sam3.memory_temporal_stride}")
    print(f"sam3.max_init_masks = {cfg.sam3.max_init_masks}")
    print(f"sam3.max_active_tracks = {cfg.sam3.max_active_tracks}")
    print(f"sam3.anchor_stride = {cfg.sam3.anchor_stride}")
    print(f"sam3.late_discovery_mode = {cfg.sam3.late_discovery_mode}")
    print(f"sam3.vg_stride = {cfg.sam3.vg_stride}")

    print("\nLoading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    mf = cfg.sampling.max_frames

    print("Warm-up...")
    t = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    print(f"  warmup: {time.time()-t:.2f}s ({resp.status})")

    print("\nBenchmark × 5 (E1, E2 alternating):")
    avgs = []
    for k in range(5):
        ts = []
        qs = []
        for v in videos:
            t = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            dt = time.time() - t
            ts.append(dt)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            qs.append(quality(sg, mf) if sg else (0, 0, 0))
        avg = sum(ts) / 2
        avgs.append(avg)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {qs[0]}  E2={ts[1]:.2f}s {qs[1]}  avg={avg:.2f}s")

    mean = sum(avgs) / len(avgs)
    stdev = (sum((x - mean)**2 for x in avgs) / len(avgs)) ** 0.5
    print(f"\n→ mean={mean:.2f}s  stdev={stdev:.2f}s  ({1/mean:.3f} Hz)")


if __name__ == "__main__":
    main()
