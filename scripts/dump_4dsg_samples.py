"""Dump full 4DSG JSON for 2 VLM4D sample videos to inspect quality.

Same config as v38/v40 (the production-target setup).  Skips VLM —
we just want to see the 4DSG structure and verify it hasn't
regressed under the eager-mode + lean-prompt optimizations.
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
HF = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"

# Two contrasting samples:
#   - basketball-game.mp4: real-world, 6 tracks, fast-motion sports
#   - synth_241.mp4:       synthetic (Kubric/SAPIEN), 10 tracks, controlled motion
SAMPLES = [
    ("real",  "videos_real/davis/basketball-game.mp4"),
    ("synth", "videos_synthetic/synth_241.mp4"),
]

OUT_DIR_BASE = ROOT / "benchmark" / "fdsg_v45"
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
# OUT_DIR set later based on use_multiplex flag


def main():
    use_multiplex = "--multiplex" in sys.argv
    if "--multiplex" in sys.argv:
        sys.argv.remove("--multiplex")

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = use_multiplex
    # v45 final config: compile + num_maskmem=7 + max_tracks=15
    cfg.sam3.enable_compile = True if use_multiplex else False
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 15
    cfg.sam3.num_maskmem = 7
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0
    print(f"  use_multiplex={cfg.sam3.use_multiplex}, "
          f"compile={cfg.sam3.enable_compile}", flush=True)
    OUT_DIR = OUT_DIR_BASE / "multiplex_refined" if cfg.sam3.use_multiplex else OUT_DIR_BASE / "base_sam3"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading warm pool (eager)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    # Use the same pipelined path as bench (run_inference_pipelined) — the
    # sequential path (run_inference) currently truncates F_k observations
    # to ~3 entries per track on consecutive calls (state contamination bug).
    requests = []
    tags_paths = []
    for tag, rel in SAMPLES:
        p = VLM4D / rel
        if not p.is_file():
            print(f"  MISSING: {p}", flush=True)
            continue
        requests.append(InferenceRequest(video_path=str(p), question=None))
        tags_paths.append((tag, p))

    print(f"\nRunning {len(requests)} videos via pipelined path...", flush=True)
    t0 = time.time()
    responses = pool.run_inference_pipelined(requests)
    print(f"  pipelined wall: {time.time() - t0:.2f}s", flush=True)

    for (tag, path), resp in zip(tags_paths, responses):
        print(f"\n=== {tag}: {path.name} ===", flush=True)
        if resp.status != "ok":
            print(f"  FAILED: {resp.error_message}", flush=True)
            continue
        fdsg = resp.four_dsg_dict
        meta = fdsg.get("metadata", {})
        tracks = fdsg.get("tracks", [])
        print(f"  {meta.get('num_frames')} frames, {meta.get('num_tracks')} tracks",
              flush=True)
        for tr in tracks:
            oid = tr.get("object_id", "?")
            n_obs = len(tr.get("F_k", []))
            theta = tr.get("theta", [0, 0])
            ext = tr.get("extent", [0, 0, 0])
            mot = tr.get("motion", "?")
            pos = tr.get("image_position", "?")
            va = tr.get("visual_anchor")
            crop_path = (Path(va["path"]).name if va else "(none)")
            print(f"    obj{oid}  obs={n_obs:>2}  t={theta[0]:.0f}-{theta[1]:.0f}s  "
                  f"size={ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f}m  "
                  f"img={pos}  motion={mot}  crop={crop_path}",
                  flush=True)

        out_json = OUT_DIR / f"{tag}_{path.stem}.4dsg.json"
        out_json.write_text(json.dumps(fdsg, indent=2))
        print(f"  saved {out_json}", flush=True)
        if tracks and tracks[0].get("F_k"):
            print(f"  sample obs (obj{tracks[0].get('object_id', 0)}):", flush=True)
            print(json.dumps(tracks[0]["F_k"][:3], indent=4), flush=True)


if __name__ == "__main__":
    main()
