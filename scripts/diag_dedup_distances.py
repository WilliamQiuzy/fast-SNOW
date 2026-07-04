"""Dump the actual 2D trajectory distances between all track pairs to see
why the post-pipeline dedup isn't merging.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np

# Load tracks JSON saved by inspect_4dsg.py — has F_k + image_position only.
# But it doesn't have per-frame mask_center_2d.  So let me read the 4dsg JSON directly.
fdsg = json.load(open(ROOT / "benchmark/fdsg_one/VLM4D-Easy2_in_session.4dsg.json"))
tracks = fdsg["tracks"]

# Each track's F_k has list of {t, c=[x,y,z]} — 3D world centroids.
# These aren't 2D image-frame centroids that the dedup uses.
# But we can still compute 3D-trajectory distance as a proxy.

n = len(tracks)
print(f"{n} tracks  |  computing pairwise 3D-trajectory mean distance + n_shared\n")
print(f"{'pair':<8}  {'n_shared':<10}  mean_3d_dist  n_a   n_b   pos_a            pos_b")
print("-" * 90)

# Build per-track: t -> [x,y,z]
traj = []
for tr in tracks:
    fk = tr["F_k"]
    d = {round(o["t"], 3): np.array(o["c"], dtype=float) for o in fk}
    traj.append((tr["object_id"], d, tr["image_position"]))

pairs = []
for i, (oi, ti, pi) in enumerate(traj):
    for j, (oj, tj, pj) in enumerate(traj):
        if j <= i:
            continue
        shared = set(ti.keys()) & set(tj.keys())
        if len(shared) < 2:
            continue
        ds = [float(np.linalg.norm(ti[t_] - tj[t_])) for t_ in shared]
        mean_d = sum(ds) / len(ds)
        pairs.append((mean_d, oi, oj, len(shared), len(ti), len(tj), pi, pj))

# Sort by mean distance — smallest first (most likely dupes)
pairs.sort(key=lambda x: x[0])
for d, a, b, sh, na, nb, pa, pb in pairs[:25]:
    print(f"({a:>2},{b:>2})  {sh:>4}/{min(na,nb):<4}  {d:>11.3f}  {na:>3}  {nb:>3}   {pa:<14}  {pb}")
