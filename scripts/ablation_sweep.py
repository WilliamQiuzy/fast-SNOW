"""Quick parameter ablation: anchor_stride and vg_stride.

Tests multiple configs in ONE process (shares warm pool, no re-load).
Records inference time + 4DSG quality metrics per config.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def measure_4dsg_quality(sg):
    tracks = sg["tracks"]
    n_obs = sorted([len(t["F_k"]) for t in tracks], reverse=True)
    return {
        "n_tracks": sg["metadata"]["num_tracks"],
        "n_obs": n_obs,
        "total_obs": sum(n_obs),
        "long_tracks": sum(1 for n in n_obs if n >= 20),
        "max_obs": max(n_obs, default=0),
    }


def run_with_cfg(pool, video, cfg_overrides):
    """Override cfg fields, run inference, restore."""
    saved = {}
    sam3_cfg = pool.config.sam3
    for k, v in cfg_overrides.items():
        saved[k] = getattr(sam3_cfg, k)
        setattr(sam3_cfg, k, v)
    try:
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=str(video), question=None))
        dt = time.time() - t
        if resp.status != "ok":
            return None, dt
        return resp.four_dsg_dict, dt
    finally:
        for k, v in saved.items():
            setattr(sam3_cfg, k, v)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading warm pool (one-time)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    configs = [
        ("baseline",        {}),
        ("anchor=8",         {"anchor_stride": 8}),
        ("vg=100",          {"vg_stride": 100}),
        ("anchor=8+vg=100", {"anchor_stride": 8, "vg_stride": 100}),
        ("max_active=20",   {"max_active_tracks": 20}),
        ("ALL combined",    {"anchor_stride": 8, "vg_stride": 100, "max_active_tracks": 20}),
    ]

    print(f"\n{'CONFIG':<24} {'VIDEO':<10} {'TIME (s)':<10} {'tracks':<8} {'long(n>=20)':<13} {'max_obs':<8} {'total_obs':<10}")
    print("-" * 90)

    for cfg_name, overrides in configs:
        for video in videos:
            stem = Path(video).stem
            sg, dt = run_with_cfg(pool, video, overrides)
            if sg is None:
                print(f"{cfg_name:<24} {stem:<10} FAILED")
                continue
            q = measure_4dsg_quality(sg)
            print(f"{cfg_name:<24} {stem:<10} {dt:<10.2f} {q['n_tracks']:<8} {q['long_tracks']:<13} {q['max_obs']:<8} {q['total_obs']:<10}")


if __name__ == "__main__":
    main()
