#!/usr/bin/env python3
"""E2E test: run DA3-Small pipeline, output ranking for a specific frame range."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd

VIDEO = PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4"
OUTPUT_JSON = PROJECT_ROOT / "assets" / "eval_results" / "e2e_3d_horsing_144f_da3_small.json"

# Need 144 sampled frames to reach source frame 429
cfg = FastSNOWConfig()
cfg.sampling.max_frames = 144
cfg.sampling.target_fps = 10.0

print(f"Video: {VIDEO.name}")
print(f"DA3 model: {cfg.da3.model_path}")
print(f"max_frames: {cfg.sampling.max_frames}")
print()

e2e = FastSNOWEndToEnd(cfg)

t0 = time.perf_counter()
result = e2e.build_4dsg_from_video(VIDEO)
elapsed = time.perf_counter() - t0

try:
    dsg = result.four_dsg_dict
    print(f"\nPipeline finished in {elapsed:.1f}s")
    meta = dsg.get("metadata", {})
    print(f"Frames: {meta.get('num_frames')}, Tracks: {meta.get('num_tracks')}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dsg, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUTPUT_JSON}")

    # Ranking for frames 357-429
    frames_data = {}
    for tr in dsg['tracks']:
        oid = tr['object_id']
        if oid > 4:
            continue
        for obs in tr['F_k']:
            fidx = obs['frame']
            if 357 <= fidx <= 429:
                c = obs['centroid_xyz']
                if fidx not in frames_data:
                    frames_data[fidx] = {}
                frames_data[fidx][oid] = (c[0], c[1], c[2])

    sorted_frames = sorted(frames_data.keys())
    print(f"\n{'='*78}")
    print(f"  z-depth ranking frames 357-429 (DA3-Small)")
    print(f"  Rank 1=closest, 5=farthest | LEFT to RIGHT by x")
    print(f"{'='*78}")

    prev_ranking = None
    changes = 0
    for fidx in sorted_frames:
        objs = frames_data[fidx]
        if len(objs) < 5:
            print(f"\n  Frame {fidx}: only {len(objs)} horses tracked, skipping")
            continue
        by_x = sorted(objs.items(), key=lambda kv: kv[1][0])
        z_sorted = sorted(objs.items(), key=lambda kv: kv[1][2])
        z_rank = {oid: r+1 for r, (oid, _) in enumerate(z_sorted)}

        cur = tuple(z_rank[oid] for oid, _ in by_x)
        changed = ""
        if prev_ranking is not None and cur != prev_ranking:
            changes += 1
            changed = " *"
        prev_ranking = cur

        print(f"\n  Frame {fidx:3d}{changed}")
        print(f"  {'':>8s}", end="")
        for oid, _ in by_x:
            print(f"   Obj{oid} ", end="")
        print()
        print(f"  {'x':>8s}", end="")
        for oid, (x, y, z) in by_x:
            print(f"  {x:+.3f}", end="")
        print()
        print(f"  {'z':>8s}", end="")
        for oid, (x, y, z) in by_x:
            print(f"  {z:.4f}", end="")
        print()
        print(f"  {'z rank':>8s}", end="")
        for oid, _ in by_x:
            print(f"    #{z_rank[oid]}  ", end="")
        print()

    total_transitions = max(0, len([f for f in sorted_frames if len(frames_data[f]) >= 5]) - 1)
    print(f"\n  Ranking changed in {changes}/{total_transitions} frame transitions")

finally:
    result.cleanup()
