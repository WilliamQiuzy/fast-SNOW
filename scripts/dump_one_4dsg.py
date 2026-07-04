"""Run ONE video through the v38 production config and dump full 4DSG.

Single-video, single-process — no batching, no consecutive calls.  Pure
reference for 4DSG quality inspection.

Usage:
    python scripts/dump_one_4dsg.py videos_real/davis/basketball-game.mp4
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

VLM4D = ROOT / "benchmark" / "VLM4D-video"
OUT_DIR = ROOT / "benchmark" / "fdsg_one"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <relpath under VLM4D-video>", file=sys.stderr)
        sys.exit(1)
    rel = sys.argv[1]
    path = VLM4D / rel
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr); sys.exit(1)

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 50
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("Loading warm pool...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    print(f"\nProcessing {path.name}...", flush=True)
    t = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=str(path), question=None))
    dt = time.time() - t
    if resp.status != "ok":
        print(f"FAILED: {resp.error_message}", file=sys.stderr); sys.exit(1)

    fdsg = resp.four_dsg_dict
    meta = fdsg["metadata"]
    tracks = fdsg["tracks"]
    print(f"  inference: {dt:.2f}s, {meta['num_frames']} frames, "
          f"{meta['num_tracks']} tracks", flush=True)
    for tr in tracks:
        oid = tr.get("object_id", "?")
        n_obs = len(tr.get("F_k", []))
        theta = tr.get("theta", [0, 0])
        ext = tr.get("extent", [0, 0, 0])
        mot = tr.get("motion", "?")
        pos = tr.get("image_position", "?")
        print(f"    obj{oid}  obs={n_obs:>2}  t={theta[0]:.0f}-{theta[1]:.0f}s  "
              f"size={ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f}m  "
              f"img={pos}  motion={mot}",
              flush=True)

    out_json = OUT_DIR / f"{path.stem}.4dsg.json"
    out_json.write_text(json.dumps(fdsg, indent=2))
    print(f"\nSaved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
