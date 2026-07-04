"""Aggressive single-GPU sweep — try every parameter knob that doesn't
break quality, measure on both videos.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg):
    tracks = sg["tracks"]
    n = sorted([len(t["F_k"]) for t in tracks], reverse=True)
    return {
        "n_tracks": sg["metadata"]["num_tracks"],
        "long": sum(1 for x in n if x >= 20),
        "max_obs": max(n, default=0),
        "total_obs": sum(n),
    }


def run(pool, video, overrides_sam=None, overrides_fusion=None):
    saved_sam = {}; saved_fusion = {}
    if overrides_sam:
        for k, v in overrides_sam.items():
            saved_sam[k] = getattr(pool.config.sam3, k)
            setattr(pool.config.sam3, k, v)
    if overrides_fusion:
        for k, v in overrides_fusion.items():
            saved_fusion[k] = getattr(pool.config.fusion, k)
            setattr(pool.config.fusion, k, v)
    try:
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=str(video), question=None))
        dt = time.time() - t
        return (resp.four_dsg_dict, dt) if resp.status == "ok" else (None, dt)
    finally:
        for k, v in saved_sam.items():
            setattr(pool.config.sam3, k, v)
        for k, v in saved_fusion.items():
            setattr(pool.config.fusion, k, v)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading pool...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    configs = [
        ("baseline",                       None, None),
        ("mem_stride=4",                   {"memory_temporal_stride": 4}, None),
        ("mem_stride=1",                   {"memory_temporal_stride": 1}, None),
        ("vg=200",                         {"vg_stride": 200}, None),
        ("anchor=6+vg=200",                {"anchor_stride": 6, "vg_stride": 200}, None),
        ("anchor=8+vg=200+mem=4",          {"anchor_stride": 8, "vg_stride": 200,
                                            "memory_temporal_stride": 4}, None),
        ("max_active=15+anchor=8",         {"anchor_stride": 8, "max_active_tracks": 15}, None),
        ("max_init=20",                    {"max_init_masks": 20}, None),
    ]

    # Warm-up first: run baseline once
    print("Warm-up (results discarded)...")
    run(pool, videos[0])

    print(f"\n{'CONFIG':<28} {'V':<8} {'time':<7} {'tracks':<7} {'long':<6} {'max':<5} {'obs':<5}")
    print("-" * 70)
    for cfg_name, ovs, ovf in configs:
        for v in videos:
            stem = Path(v).stem.replace("VLM4D-", "")
            sg, dt = run(pool, v, ovs, ovf)
            if sg is None:
                print(f"{cfg_name:<28} {stem:<8} FAILED")
                continue
            q = quality(sg)
            print(f"{cfg_name:<28} {stem:<8} {dt:<7.2f} {q['n_tracks']:<7} {q['long']:<6} {q['max_obs']:<5} {q['total_obs']:<5}")


if __name__ == "__main__":
    main()
