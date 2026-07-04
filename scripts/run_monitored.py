"""Run one video through the production pipeline with real-time GPU monitoring.

Logs:
  - util/power/memory time-series (CSV)
  - phase markers (B-1..B-8) alongside the timeline
  - top idle windows where GPU is < 5% busy

Output:
  /tmp/h200_<stem>.csv          — 50Hz samples
  /tmp/h200_<stem>.marks.csv    — labelled phase boundaries
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

from scripts.gpu_monitor import GPUMonitor

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <absolute-video-path>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1]).resolve()
    stem = path.stem

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 50
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("Loading warm pool (excluded from GPU profile)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s")

    # Start the monitor ONLY around the inference call.
    csv_path = f"/tmp/h200_{stem}.csv"
    mon = GPUMonitor(interval_hz=50, csv_path=csv_path)
    mon.start()
    mon.mark("inference_start")
    t = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=str(path), question=None))
    mon.mark("inference_end")
    dt = time.time() - t
    mon.stop()

    if resp.status != "ok":
        print("FAILED:", resp.error_message); sys.exit(1)

    fdsg = resp.four_dsg_dict
    print(f"\nInference: {dt:.2f}s  |  num_tracks={fdsg['metadata']['num_tracks']}")
    print(mon.summary())
    print(f"\nFull CSV at: {csv_path}")


if __name__ == "__main__":
    main()
