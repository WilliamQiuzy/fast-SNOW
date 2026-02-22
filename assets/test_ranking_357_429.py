#!/usr/bin/env python3
"""z-depth ranking for frames 357-429 using DA3-Small + YOLO + SAM3 (25 frames only)."""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DA3_SRC = PROJECT_ROOT / "fast_snow" / "vision" / "da3" / "src"
sys.path.insert(0, str(DA3_SRC))

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastSNOWPipeline, FastFrameInput, FastLocalDetection,
)
from fast_snow.vision.perception.da3_wrapper import DA3Wrapper
from fast_snow.vision.perception.yolo_wrapper import YoloBBoxDetector
from fast_snow.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

VIDEO = str(PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4")
FRAME_START, FRAME_END, FRAME_STEP = 357, 429, 3

# Extract frames 357-429 from video and save as JPEGs for SAM3
cap = cv2.VideoCapture(VIDEO)
frame_dir = Path(tempfile.mkdtemp(prefix="fast_snow_357_"))
frames_rgb = []
source_indices = []
sam3_idx = 0

for fidx in range(FRAME_START, FRAME_END + 1, FRAME_STEP):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, bgr = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frames_rgb.append(rgb)
    source_indices.append(fidx)
    cv2.imwrite(str(frame_dir / f"{sam3_idx:06d}.jpg"), bgr)
    sam3_idx += 1

cap.release()
print(f"Extracted {len(frames_rgb)} frames ({FRAME_START}-{FRAME_END}) to {frame_dir}")

# Init models
cfg = FastSNOWConfig()
da3 = DA3Wrapper(cfg.da3)
yolo = YoloBBoxDetector(cfg.yolo)
sam3 = SAM3SharedSessionManager(cfg.sam3)
sam3.set_video_dir(frame_dir)

pipeline = FastSNOWPipeline(cfg)
sam3_initialized = False

t0 = time.perf_counter()
for i in range(len(frames_rgb)):
    img = frames_rgb[i]
    src_idx = source_indices[i]

    da3r = da3.infer(img)
    yolo_dets = yolo.detect(img, frame_idx=i)

    if not sam3_initialized:
        frame_masks = []
        cur_bboxes = [list(d.bbox_xywh_norm) for d in yolo_dets]
        if cur_bboxes:
            _, init_masks = sam3.create_run_with_initial_bboxes(
                boxes_xywh=cur_bboxes,
                box_labels=[1] * len(cur_bboxes),
                frame_idx=i, tag="yolo_bbox",
            )
            frame_masks.extend(init_masks)
            sam3_initialized = True
    else:
        frame_masks = list(sam3.propagate_all(i))

    best = {}
    for m in frame_masks:
        key = (m.run_id, m.obj_id_local)
        prev = best.get(key)
        if prev is None or m.score > prev.score:
            best[key] = m
    frame_masks = list(best.values())

    detections = [
        FastLocalDetection(
            run_id=m.run_id, local_obj_id=m.obj_id_local,
            mask=m.mask, score=m.score,
        )
        for m in frame_masks
    ]

    fi = FastFrameInput(
        frame_idx=src_idx,
        depth_t=da3r.depth,
        K_t=da3r.K,
        T_wc_t=da3r.T_wc,
        detections=detections,
        depth_conf_t=da3r.depth_conf,
    )
    pipeline.process_frame(fi)
    print(f"  frame {src_idx:3d} (sam3_idx={i:2d}): {len(detections)} masks")

elapsed = time.perf_counter() - t0
print(f"\nPipeline: {elapsed:.1f}s for {len(frames_rgb)} frames")

sam3.end_all_runs()

four_dsg = pipeline.build_4dsg_dict()
meta = four_dsg.get("metadata", {})
print(f"Tracks: {meta.get('num_tracks')}")

# Save JSON
out_json = PROJECT_ROOT / "assets" / "eval_results" / "e2e_3d_horsing_357_429_da3_small.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(four_dsg, f, indent=2, ensure_ascii=False)
print(f"Saved: {out_json}")

# Compute ranking
frames_data = {}
for tr in four_dsg['tracks']:
    oid = tr['object_id']
    for obs in tr['F_k']:
        fidx = obs['frame']
        c = obs['centroid_xyz']
        if fidx not in frames_data:
            frames_data[fidx] = {}
        frames_data[fidx][oid] = (c[0], c[1], c[2])

# Figure out which objects are the 5 main horses (exclude far background)
# Use median z to identify foreground vs background
all_objs = set()
for fdata in frames_data.values():
    all_objs.update(fdata.keys())

obj_median_z = {}
for oid in all_objs:
    zs = [frames_data[f][oid][2] for f in frames_data if oid in frames_data[f]]
    obj_median_z[oid] = np.median(zs)

# Top 5 closest objects by median z
fg_objs = sorted(obj_median_z.keys(), key=lambda o: obj_median_z[o])[:5]
print(f"\n5 foreground horses: {fg_objs} (median z: {[f'{obj_median_z[o]:.3f}' for o in fg_objs]})")

sorted_frames = sorted(frames_data.keys())

print(f"\n{'='*78}")
print(f"  z-depth ranking frames {FRAME_START}-{FRAME_END} (DA3-Small + SAM3 masks)")
print(f"  Rank 1=closest, 5=farthest | LEFT to RIGHT by x")
print(f"{'='*78}")

prev_ranking = None
changes = 0
valid = 0

for fidx in sorted_frames:
    objs = {o: frames_data[fidx][o] for o in fg_objs if o in frames_data[fidx]}
    if len(objs) < 5:
        print(f"\n  Frame {fidx}: only {len(objs)}/5 fg horses, skipping")
        continue

    by_x = sorted(objs.items(), key=lambda kv: kv[1][0])
    z_sorted = sorted(objs.items(), key=lambda kv: kv[1][2])
    z_rank = {oid: r + 1 for r, (oid, _) in enumerate(z_sorted)}

    cur = tuple(z_rank[oid] for oid, _ in by_x)
    changed = ""
    if prev_ranking is not None:
        valid += 1
        if cur != prev_ranking:
            changes += 1
            changed = " *"
    prev_ranking = cur

    print(f"\n  Frame {fidx:3d}{changed}")
    print(f"  {'':>8s}", end="")
    for oid, _ in by_x:
        print(f"  Obj{oid:d}  ", end="")
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

print(f"\n  Ranking changed in {changes}/{valid} frame transitions")

shutil.rmtree(frame_dir, ignore_errors=True)
