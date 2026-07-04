"""Run the same video 5 times in a row and measure per-call time.

Tells us if there's a real cold-vs-warm gap and what the steady-state is.
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


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else (
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4"
    )
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading + warming pool...")
    t_load = time.time()
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    print(f"  pool ready in {time.time()-t_load:.1f}s\n")

    print(f"Video: {Path(video).name}")
    print(f"{'run':<6} {'time (s)':<10} {'tracks':<8} {'note'}")
    print("-" * 50)
    times = []
    for i in range(n_runs):
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=video, question=None))
        dt = time.time() - t
        if resp.status != "ok":
            print(f"{i+1:<6} FAILED: {resp.error_message}")
            continue
        n_tracks = resp.four_dsg_dict["metadata"]["num_tracks"]
        note = "(cold)" if i == 0 else "(warm)"
        times.append(dt)
        print(f"{i+1:<6} {dt:<10.2f} {n_tracks:<8} {note}")
    if len(times) > 1:
        print()
        print(f"Cold-start (run 1): {times[0]:.2f}s")
        print(f"Mean warm (runs 2+): {sum(times[1:])/len(times[1:]):.2f}s")
        print(f"Min warm:             {min(times[1:]):.2f}s")
        print(f"Hz at steady state:   {1.0/min(times[1:]):.4f}")


if __name__ == "__main__":
    main()
