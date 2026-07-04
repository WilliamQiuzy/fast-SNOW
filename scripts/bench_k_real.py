"""K-vs-time on a series of REAL VLM4D background + N moving disks (K in
{5,10,20,30,40,50}).  Background is synth_350.mp4 (5 s, 121 frames @ 24 fps),
sampled at 10 fps (~50 frames) — same regime as our synthetic-disk bench, but
DA3, FastSAM, and SAM3 now operate on real-scene textures, lighting, and
motion instead of a black canvas.

Saves to benchmark/k_tracking_real.json so it does NOT clobber the synthetic
results in benchmark/k_tracking.json.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

VIDEO_DIR = Path("/tmp/synth_real")
TARGET_KS = [5, 10, 20, 30, 40, 50]
CONF_THRESHOLD = 0.40
N_TRIALS = 2
OUT_DIR = ROOT / "benchmark"
OUT_JSON = OUT_DIR / "k_tracking_real.json"


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = True
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 80
    cfg.sam3.num_maskmem = 7
    cfg.sampling.max_frames = None
    cfg.sampling.target_fps = 10.0
    cfg.fastsam.conf_threshold = CONF_THRESHOLD

    print("Loading warm pool (compile=True)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    big_video = VIDEO_DIR / f"synth_real_K{TARGET_KS[-1]:02d}.mp4"
    print(f"  global warmup on {big_video.name}...", flush=True)
    t = time.time()
    pool.run_inference(InferenceRequest(video_path=str(big_video), question=None))
    print(f"    warmup done in {time.time() - t:.1f}s", flush=True)

    rows = []
    for K_target in TARGET_KS:
        video = VIDEO_DIR / f"synth_real_K{K_target:02d}.mp4"
        if not video.is_file():
            print(f"  MISSING: {video}", flush=True)
            continue

        ts, K_obs, n_frames = [], [], 0
        for trial in range(N_TRIALS):
            t = time.time()
            resp = pool.run_inference(InferenceRequest(
                video_path=str(video), question=None,
            ))
            dt = time.time() - t
            if resp.status != "ok":
                print(f"  K_target={K_target}: FAILED {resp.error_message}", flush=True)
                break
            ts.append(dt)
            md = (resp.four_dsg_dict or {}).get("metadata", {})
            n_frames = md.get("num_frames", 0)
            K_obs.append(md.get("num_tracks", 0))
        if not ts:
            continue
        med_t = statistics.median(ts)
        med_K = int(statistics.median(K_obs))
        hz = n_frames / med_t if med_t > 0 else 0
        row = dict(K_target=K_target, K=med_K, frames=n_frames,
                   trials=ts, median_t=med_t, hz=hz,
                   conf_threshold=CONF_THRESHOLD)
        rows.append(row)
        print(f"  K_target={K_target:>2}  K_obs={med_K:>2}  median={med_t:5.2f}s  Hz={hz:5.2f}",
              flush=True)

        OUT_JSON.write_text(json.dumps({
            "_note": (f"K-tracking on REAL VLM4D background (synth_350.mp4) + "
                      f"N synthetic disks. FastSAM threshold {CONF_THRESHOLD}. "
                      f"Full 5 s video sampled at 10 fps (no truncation)."),
            "videos_dir": str(VIDEO_DIR),
            "background_video": "synth_350.mp4",
            "config": {
                "max_active_tracks": cfg.sam3.max_active_tracks,
                "target_fps": cfg.sampling.target_fps,
                "max_frames": cfg.sampling.max_frames,
                "anchor_stride": cfg.sam3.anchor_stride,
                "fastsam_conf_threshold": CONF_THRESHOLD,
            },
            "rows": rows,
        }, indent=2))

    print("\nFinal results:", flush=True)
    for r in rows:
        print(f"  K_target={r['K_target']:>2}  K_obs={r['K']:>2}  "
              f"t={r['median_t']:.2f}s  Hz={r['hz']:.2f}  "
              f"warm={r['trials'][1]:.2f}s",
              flush=True)
    print(f"\nSaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
