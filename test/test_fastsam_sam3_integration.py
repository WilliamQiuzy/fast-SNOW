#!/usr/bin/env python3
"""GPU integration test: FastSAM + SAM3 two-pass → 4DSG.

Validates the full two-pass architecture on a real video:
  Phase 2a: FastSAM frame 0 → SAM3 init + full propagation
  Phase 2b: FastSAM per-frame discovery of new objects via IoU comparison
  Phase 2c: SAM3 partial propagation for new objects (merge with cached)
  Phase 2d: Build FastFrameInput per frame → CPU pipeline

Requires: CUDA GPU, FastSAM-s.pt, sam3.pt, da3-small weights, test video.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd

# Use a short video; limit to 5 frames for speed.
VIDEO = PROJECT_ROOT / "benchmark" / "VLM4D-video" / "videos_real" / "davis" / "bear.mp4"
if not VIDEO.exists():
    # Fallback: try blackswan
    VIDEO = PROJECT_ROOT / "benchmark" / "VLM4D-video" / "videos_real" / "davis" / "blackswan.mp4"

assert VIDEO.exists(), f"Test video not found: {VIDEO}"

cfg = FastSNOWConfig()
cfg.sampling.max_frames = 3
cfg.sampling.target_fps = 2.0

print(f"{'='*60}")
print(f"FastSAM + SAM3 Two-Pass Integration Test")
print(f"{'='*60}")
print(f"Video:    {VIDEO.name}")
print(f"Frames:   max_frames={cfg.sampling.max_frames}, target_fps={cfg.sampling.target_fps}")
print(f"FastSAM:  {cfg.fastsam.model_path} (conf={cfg.fastsam.conf_threshold}, iou={cfg.fastsam.iou_threshold})")
print(f"SAM3:     {cfg.sam3.model_path}")
print(f"DA3:      {cfg.da3.model_path}")
print(f"Discovery IoU threshold: {cfg.fastsam.discovery_iou_thresh}")
print()

e2e = FastSNOWEndToEnd(cfg)

t0 = time.perf_counter()
result = e2e.build_4dsg_from_video(VIDEO)
elapsed = time.perf_counter() - t0

try:
    dsg = result.four_dsg_dict
    meta = dsg.get("metadata", {})
    tracks = dsg.get("tracks", [])
    ego = dsg.get("ego", [])

    print(f"\n{'='*60}")
    print(f"Pipeline finished in {elapsed:.1f}s")
    print(f"{'='*60}")

    num_frames = meta.get("num_frames", 0)
    num_tracks = meta.get("num_tracks", 0)
    print(f"Frames processed: {num_frames}")
    print(f"Tracks detected:  {num_tracks}")
    print(f"Ego poses:        {len(ego)}")

    # --- Validation checks ---
    errors = []

    # 1. Must have processed frames
    if num_frames == 0:
        errors.append("FAIL: num_frames == 0")
    else:
        print(f"  [PASS] num_frames = {num_frames}")

    # 2. Must have detected at least 1 object
    if num_tracks == 0:
        errors.append("FAIL: num_tracks == 0 — FastSAM + SAM3 found nothing")
    else:
        print(f"  [PASS] num_tracks = {num_tracks}")

    # 3. Ego poses must match frame count
    if len(ego) != num_frames:
        errors.append(f"FAIL: ego poses ({len(ego)}) != num_frames ({num_frames})")
    else:
        print(f"  [PASS] ego poses match frame count")

    # 4. Each track must have F_k observations
    for tr in tracks:
        oid = tr["object_id"]
        fk = tr["F_k"]
        if len(fk) == 0:
            errors.append(f"FAIL: track {oid} has 0 observations")
        else:
            # Check that observations have required fields
            for obs in fk:
                for key in ["t", "c", "s", "theta", "tau"]:
                    if key not in obs:
                        errors.append(f"FAIL: track {oid} obs missing '{key}'")

    if not any("track" in e and "has 0 observations" in e for e in errors):
        print(f"  [PASS] all tracks have valid F_k observations")

    # 5. 3D coordinates must be present and non-zero somewhere
    has_3d = False
    for tr in tracks:
        for obs in tr["F_k"]:
            c = obs["c"]
            if any(abs(v) > 1e-6 for v in c):
                has_3d = True
                break
        if has_3d:
            break
    if not has_3d:
        errors.append("FAIL: all centroids are zero — DA3 integration broken")
    else:
        print(f"  [PASS] 3D centroids present (DA3 integration OK)")

    # 6. Visual anchors
    va = meta.get("visual_anchor", [])
    if len(va) == 0:
        errors.append("FAIL: no visual_anchor in metadata")
    else:
        valid_va = sum(1 for v in va if Path(v["path"]).exists())
        print(f"  [PASS] visual_anchor: {len(va)} entries ({valid_va} files exist)")

    # --- Print sample data ---
    print(f"\n--- Ego poses (first 3) ---")
    for e_entry in ego[:3]:
        xyz = e_entry["xyz"]
        print(f"  t={e_entry['t']:3d}  xyz=({xyz[0]:8.3f}, {xyz[1]:8.3f}, {xyz[2]:8.3f})")

    print(f"\n--- Tracks ({len(tracks)} objects) ---")
    for tr in tracks[:5]:
        oid = tr["object_id"]
        fk = tr["F_k"]
        print(f"\n  Object {oid}: {len(fk)} observations")
        for obs in fk[:3]:
            c = obs["c"]
            n_patches = len(obs["tau"])
            print(
                f"    t={obs['t']:3d}  c=({c[0]:8.3f}, {c[1]:8.3f}, {c[2]:8.3f})  "
                f"tau={n_patches}"
            )
        if len(fk) > 3:
            print(f"    ... ({len(fk)-3} more)")

    # --- Final verdict ---
    print(f"\n{'='*60}")
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\nRESULT: FAILED ({len(errors)} error(s))")
        sys.exit(1)
    else:
        print(f"RESULT: ALL CHECKS PASSED")
        print(f"FastSAM + SAM3 two-pass integration verified on {VIDEO.name}")
        print(f"  {num_frames} frames, {num_tracks} tracks, {elapsed:.1f}s total")
    print(f"{'='*60}")

finally:
    result.cleanup()
