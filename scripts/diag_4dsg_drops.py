"""Run a single video and report how many candidates/tracks survive each
filter stage in ROSE's pipeline.  Pure instrumentation — no edits to
production code."""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import numpy as np
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest
from rose.engine.pipeline import rose_pipeline as rp


# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------

stats = {
    "frames": 0,
    "raw_dets_per_frame": [],
    "candidates_per_frame": [],
    "dropped_min_points_per_frame": [],
    "dropped_max_extent_per_frame": [],
    "unique_gids_after_fuse": set(),
    "n_tracks_pre_dedup": 0,
    "n_tracks_post_crop_dedup": 0,
    "n_tracks_post_traj_dedup": 0,
    "n_tracks_in_4dsg": 0,
    "dropped_missing_crop": 0,
    "dropped_min_obs": 0,
    "dropped_max_extent_track": 0,
    "dropped_blob": 0,
}


# --- wrap _build_candidates to count input dets vs surviving candidates --
_orig_build = rp.ROSEPipeline._build_candidates


def _wrapped_build_candidates(self, frame, T_cw_t, K_inv, frame_ctx=None):
    n_in = len(frame.detections)
    # Count drops by re-implementing the filter checks
    cands = _orig_build(self, frame, T_cw_t, K_inv, frame_ctx=frame_ctx)
    n_out = len(cands)
    stats["frames"] += 1
    stats["raw_dets_per_frame"].append(n_in)
    stats["candidates_per_frame"].append(n_out)
    return cands


# --- wrap _fuse_candidates to track gid set after fusion ---
_orig_fuse = rp.ROSEPipeline._fuse_candidates


def _wrapped_fuse(self, frame_idx, candidates, depth_is_metric=True):
    winners = _orig_fuse(self, frame_idx, candidates, depth_is_metric)
    for gid in winners.keys():
        stats["unique_gids_after_fuse"].add(gid)
    return winners


# --- wrap merge_duplicate_tracks to count pre/post dedup ---
_orig_merge = rp.ROSEPipeline.merge_duplicate_tracks


def _wrapped_merge(self, object_crops):
    stats["n_tracks_pre_dedup"] = len(object_crops)
    # Re-implement to count crop-dedup vs traj-dedup separately would need
    # forking the method; for now just capture before/after.
    result = _orig_merge(self, object_crops)
    stats["n_tracks_post_traj_dedup"] = len(result)
    return result


# --- wrap build_4dsg_dict to count filter drops ---
_orig_build_4dsg = rp.ROSEPipeline.build_4dsg_dict


def _wrapped_build_4dsg(self, object_crops=None):
    # Count drops at each filter inside build_4dsg_dict by manual replay.
    if object_crops is None:
        object_crops = {}
    min_obs = int(getattr(self.config.fusion, "min_track_observations", 3))
    max_ext = float(getattr(self.config.fusion, "max_track_extent", 0.7))

    n_no_crop = 0
    n_low_obs = 0
    n_huge_ext = 0
    n_blob = 0
    n_pass = 0

    for gid in sorted(self._tracks.keys()):
        state = self._tracks[gid]
        if not state.observations:
            continue
        has_crop = gid in object_crops
        obs_sorted = sorted(state.observations, key=lambda x: x.frame_idx)

        if not has_crop:
            n_no_crop += 1
            continue
        if len(obs_sorted) < min_obs:
            n_low_obs += 1
            continue

        ex_x = [o.step.shape.x_max - o.step.shape.x_min for o in obs_sorted]
        ex_y = [o.step.shape.y_max - o.step.shape.y_min for o in obs_sorted]
        ex_z = [o.step.shape.z_max - o.step.shape.z_min for o in obs_sorted]
        extent = [float(np.median(ex_x)), float(np.median(ex_y)), float(np.median(ex_z))]
        n_huge = sum(1 for e in extent if e > 0.7)
        first_c = obs_sorted[0].step.centroid
        last_c = obs_sorted[-1].step.centroid
        disp_max = max(abs(last_c.x - first_c.x), abs(last_c.y - first_c.y), abs(last_c.z - first_c.z))
        if max(extent) > max_ext:
            n_huge_ext += 1
            continue
        if n_huge >= 2 and disp_max < 0.10:
            n_blob += 1
            continue
        n_pass += 1

    stats["dropped_missing_crop"] = n_no_crop
    stats["dropped_min_obs"] = n_low_obs
    stats["dropped_max_extent_track"] = n_huge_ext
    stats["dropped_blob"] = n_blob
    stats["n_tracks_in_4dsg"] = n_pass

    # Now call the real method
    return _orig_build_4dsg(self, object_crops=object_crops)


rp.ROSEPipeline._build_candidates = _wrapped_build_candidates
rp.ROSEPipeline._fuse_candidates = _wrapped_fuse
rp.ROSEPipeline.merge_duplicate_tracks = _wrapped_merge
rp.ROSEPipeline.build_4dsg_dict = _wrapped_build_4dsg


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <absolute-video-path>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1]).resolve()

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
    print(f"\ninference time: {dt:.2f}s", flush=True)

    print("\n========== DROP REPORT ==========")
    raw = sum(stats["raw_dets_per_frame"])
    cand = sum(stats["candidates_per_frame"])
    print(f"frames processed                : {stats['frames']}")
    print(f"raw SAM3 detections (sum/frame) : {raw}")
    print(f"surviving candidates after _build_candidates")
    print(f"  (= dets passing depth.min_points + max_extent filters): {cand}")
    print(f"  -> dropped (depth invalid / too few points / huge extent): {raw - cand}")
    print(f"unique global track IDs after fusion: {len(stats['unique_gids_after_fuse'])}")
    print(f"  (= candidates merged via cross_run IoU + centroid + temporal gap)")
    print()
    print(f"tracks entering merge_duplicate_tracks: {stats['n_tracks_pre_dedup']}")
    print(f"tracks after merge_duplicate_tracks   : {stats['n_tracks_post_traj_dedup']}")
    print(f"  -> dropped by dedup (crop-sim OR 2D-traj): {stats['n_tracks_pre_dedup'] - stats['n_tracks_post_traj_dedup']}")
    print()
    print(f"build_4dsg_dict filter breakdown:")
    print(f"  dropped: missing crop                : {stats['dropped_missing_crop']}")
    print(f"  dropped: < min_track_observations(3) : {stats['dropped_min_obs']}")
    print(f"  dropped: max_track_extent > 1.3      : {stats['dropped_max_extent_track']}")
    print(f"  dropped: stationary blob filter      : {stats['dropped_blob']}")
    print(f"  ---------------------------------------")
    print(f"  num_tracks in final 4DSG             : {stats['n_tracks_in_4dsg']}")
    print()
    print(f"reported num_tracks (sanity)            : {meta['num_tracks']}")


if __name__ == "__main__":
    main()
