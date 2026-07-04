"""Benchmark ROSE pipeline with FA3 ON vs OFF on a single video.

Same thresholds, same video, same sampling — only `sam3.use_fa3` toggled.

Usage:
    python scripts/bench_fa3_vs_sdpa.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import flash_attn_interface
import sam3.perflib.fa3 as fa3_mod

# Spy on FA3 invocations
_FA3_CALLS = [0]
_orig_lib = flash_attn_interface.flash_attn_func
_orig_wrap = fa3_mod.flash_attn_func


def _spy_lib(*a, **kw):
    _FA3_CALLS[0] += 1
    return _orig_lib(*a, **kw)


def _spy_wrap(q, k, v):
    _FA3_CALLS[0] += 1
    return _orig_wrap(q, k, v)


flash_attn_interface.flash_attn_func = _spy_lib
fa3_mod.flash_attn_func = _spy_wrap

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_e2e import ROSEEndToEnd


VIDEO = "assets/examples_videos/horse-human.mp4"
MAX_FRAMES = 16
TARGET_FPS = 4.0


def make_config(use_fa3: bool) -> ROSEConfig:
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sampling.max_frames = MAX_FRAMES
    cfg.sampling.target_fps = TARGET_FPS
    cfg.sam3.use_fa3 = use_fa3
    # Identical thresholds in both runs:
    cfg.sam3.score_threshold_detection = 0.3
    cfg.depth_filter.conf_thresh = 0.5
    cfg.depth_filter.min_points = 50
    cfg.depth_filter.max_extent = 30.0
    cfg.fusion.cross_run_iou_thresh = 0.5
    cfg.fusion.merge_centroid_dist_m = 2.0
    cfg.fusion.lost_patience = 5
    cfg.fusion.archive_patience = 30
    cfg.step.grid_size = 16
    cfg.step.iou_threshold = 0.5
    cfg.seed = 42
    return cfg


def run_one(label: str, use_fa3: bool):
    print(f"\n{'='*70}\n  RUN: {label}  (use_fa3={use_fa3})\n{'='*70}")
    _FA3_CALLS[0] = 0
    cfg = make_config(use_fa3=use_fa3)
    e2e = ROSEEndToEnd(cfg)

    t_load = time.time()
    # Force model loads to factor out from inference time
    e2e._da3.load()
    e2e._fastsam.load()
    e2e._sam3.load()
    load_dt = time.time() - t_load

    fa3_after_load = _FA3_CALLS[0]

    t_inf = time.time()
    result = e2e.build_4dsg_from_video(VIDEO)
    inf_dt = time.time() - t_inf

    fa3_total = _FA3_CALLS[0]
    md = result.four_dsg_dict.get("metadata", {})
    n_frames = md.get("num_frames", 0)
    n_tracks = md.get("num_tracks", 0)
    json_size = len(result.scene_json)
    n_obs_total = sum(len(t.get("F_k", [])) for t in result.four_dsg_dict.get("tracks", []))

    result.cleanup()
    return {
        "label": label,
        "use_fa3": use_fa3,
        "load_time_s": round(load_dt, 1),
        "inference_time_s": round(inf_dt, 1),
        "n_frames": n_frames,
        "n_tracks": n_tracks,
        "n_obs_total": n_obs_total,
        "scene_json_chars": json_size,
        "fa3_calls_load": fa3_after_load,
        "fa3_calls_total": fa3_total,
        "fa3_calls_inference": fa3_total - fa3_after_load,
    }


def main():
    print(f"Video: {VIDEO}")
    print(f"Sampling: max_frames={MAX_FRAMES}, target_fps={TARGET_FPS}")

    off = run_one("FA3 OFF (PyTorch SDPA)", use_fa3=False)
    on = run_one("FA3 ON (Flash Attention 3, FP8)", use_fa3=True)

    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    keys = ["label", "use_fa3", "inference_time_s", "n_frames", "n_tracks",
            "n_obs_total", "scene_json_chars", "fa3_calls_inference"]
    rows = [off, on]
    widths = [max(len(str(r[k])) for r in rows + [{k: k}]) for k in keys]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(f"{k:<{w}}" for k, w in zip(keys, widths)))
    print(sep)
    for r in rows:
        print("  ".join(f"{str(r[k]):<{w}}" for k, w in zip(keys, widths)))

    speedup = off["inference_time_s"] / max(on["inference_time_s"], 1e-9)
    print(f"\nSpeedup (OFF/ON): {speedup:.2f}x")
    print(f"FA3 invocations during inference: OFF={off['fa3_calls_inference']}, ON={on['fa3_calls_inference']}")


if __name__ == "__main__":
    main()
