"""Trace per-(obj_id, frame) mask presence + size + score so we can see
exactly where each track is lost or recovered.  Pure read-only instrumentation
on the warm pool — does not alter any model behaviour.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import numpy as np
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else (
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4"
    )

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 50
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    pool = WarmModelPool(cfg)
    pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    resp = pool.run_inference(InferenceRequest(video_path=video, question=None))
    if resp.status != "ok":
        print("FAILED:", resp.error_message); sys.exit(1)

    # Pull mask_cache from the saved 4DSG observation list (run_id, obj_id_local
    # is lost by then — we need to instrument differently).  But we can already
    # see per-track completeness via the saved JSON.
    fdsg = resp.four_dsg_dict
    print(f"\n{Path(video).name}  |  num_frames={fdsg['metadata']['num_frames']}  "
          f"num_tracks={fdsg['metadata']['num_tracks']}\n")

    # Build per-track frame-presence map.  Note observations carry
    # source_frame_idx (== sampled idx in our sampling).  Frame indices are
    # in the F_k entries' timestamps so we need to map t→frame.
    # Easier: F_k entries have monotonic 't'; we have target_fps=10 so
    # frames are at t = 0, 0.1, 0.2 ...  But source frames at 24fps are at
    # 0, 1/24, 2/24, ...
    # We just count distinct 't' per track.
    for tr in fdsg["tracks"]:
        oid = tr["object_id"]
        fk = tr["F_k"]
        ts = sorted(set(round(o["t"], 3) for o in fk))
        # Find gaps in the t-series (a gap = at least one missed sampling)
        nob = tr["n_obs"] if "n_obs" in tr else len(fk)
        gaps = []
        if len(ts) >= 2:
            for a, b in zip(ts, ts[1:]):
                if (b - a) > 0.12 + 1e-3:  # > 1.2 sampling intervals
                    gaps.append((a, b, round(b - a, 2)))
        ext = tr["extent"]
        print(f"obj{oid:>3}  n={len(fk):>2}  span=[{ts[0]:.2f},{ts[-1]:.2f}]s  "
              f"ext={ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f}  "
              f"pos={tr['image_position']:<14}  gaps={gaps}")


if __name__ == "__main__":
    main()
