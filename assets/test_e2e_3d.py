#!/usr/bin/env python3
"""Quick e2e test: YOLO + SAM3 + DA3 → 4DSG with 3D world coordinates."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd

VIDEO = PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4"
OUTPUT_JSON = PROJECT_ROOT / "assets" / "eval_results" / "e2e_3d_horsing_20f.json"

cfg = FastSNOWConfig()
cfg.sampling.max_frames = 20
cfg.sampling.target_fps = 10.0

print(f"Video: {VIDEO.name}")
print(f"Config: max_frames={cfg.sampling.max_frames}, target_fps={cfg.sampling.target_fps}")
print(f"DA3 model: {cfg.da3.model_path}")
print(f"SAM3 model: {cfg.sam3.model_path}")
print(f"YOLO model: {cfg.yolo.model_path}")
print()

e2e = FastSNOWEndToEnd(cfg)

t0 = time.perf_counter()
result = e2e.build_4dsg_from_video(VIDEO)
elapsed = time.perf_counter() - t0

try:
    dsg = result.four_dsg_dict
    print(f"\n{'='*60}")
    print(f"Pipeline finished in {elapsed:.1f}s")
    print(f"{'='*60}")

    meta = dsg.get("metadata", {})
    print(f"Frames: {meta.get('num_frames', '?')}")
    print(f"Tracks: {meta.get('num_tracks', '?')}")

    # Print ego poses
    ego = dsg.get("ego", [])
    print(f"\n--- Ego poses ({len(ego)} frames) ---")
    for e in ego[:5]:
        xyz = e["xyz"]
        print(f"  t={e['t']:3d}  xyz=({xyz[0]:8.3f}, {xyz[1]:8.3f}, {xyz[2]:8.3f})")
    if len(ego) > 5:
        print(f"  ... ({len(ego)-5} more)")

    # Print tracks with 3D coordinates
    tracks = dsg.get("tracks", [])
    print(f"\n--- Tracks ({len(tracks)} objects) ---")
    for tr in tracks:
        oid = tr["object_id"]
        fk = tr["F_k"]
        print(f"\n  Object {oid}: {len(fk)} observations")
        for obs in fk[:5]:
            c = obs["c"]
            s = obs["s"]
            n_patches = len(obs["tau"])
            print(
                f"    t={obs['t']:3d}  c=({c[0]:8.3f}, {c[1]:8.3f}, {c[2]:8.3f})  "
                f"tau={n_patches}  "
                f"extent_x={s['x']['max']-s['x']['min']:.2f}m  "
                f"extent_z={s['z']['max']-s['z']['min']:.2f}m"
            )
        if len(fk) > 5:
            print(f"    ... ({len(fk)-5} more)")

    # Print relations
    ego_rel = dsg.get("ego_relations", [])
    obj_rel = dsg.get("object_relations", [])
    print(f"\n--- Relations (latest frame) ---")
    print(f"  Ego-Object: {len(ego_rel)}")
    for r in ego_rel:
        print(f"    obj {r['object_id']}: {r['bearing']} {r['elev']} dist={r['dist_m']:.2f}m motion={r['motion']}")
    print(f"  Object-Object: {len(obj_rel)}")
    for r in obj_rel[:10]:
        print(f"    {r['src']}→{r['dst']}: {r['dir']} {r['elev']} dist={r['dist_m']:.2f}m")

    # Save full JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dsg, f, indent=2, ensure_ascii=False)
    print(f"\nFull 4DSG JSON saved to: {OUTPUT_JSON}")

finally:
    result.cleanup()
