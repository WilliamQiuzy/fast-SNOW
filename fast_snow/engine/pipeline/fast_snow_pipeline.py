"""Fast-SNOW runtime pipeline.

Implements the Fast-SNOW route in docs/roadmap/Fast-SNOW_IMPLEMENTATION.md:
- Step 4: mask + depth backprojection and geometric filtering
- Step 5: cross-run global ID fusion
- Step 6: STEP token construction
- Step 7: temporal tracks (F_k)
- Step 8: strict 1:1 JSON serialization

This module is model-agnostic: callers provide per-frame SAM3 detections and DA3 outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.reasoning.tokens.geometry_tokens import build_centroid_token, build_shape_token
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken, mask_to_patch_tokens
from fast_snow.reasoning.tokens.step_encoding import STEPToken, update_temporal_token
from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken


RunKey = Tuple[Hashable, Hashable]


@dataclass(frozen=True)
class FastLocalDetection:
    """One SAM3 detection from a single run in one frame."""

    run_id: Hashable
    local_obj_id: Hashable
    mask: np.ndarray  # (H, W) bool
    score: float = 1.0


@dataclass(frozen=True)
class FastFrameInput:
    """Per-frame inputs required by Fast-SNOW Step 4+.

    Attributes:
        frame_idx: Frame index.
        depth_t: DA3 depth map (H, W), meters.
        K_t: Camera intrinsics (3, 3).
        T_wc_t: DA3 world->camera transform (4, 4).
        detections: SAM3 detections from all active runs for this frame.
        depth_conf_t: DA3 depth confidence map (H, W), values in [0, 1].
            When None, all pixels are trusted (equivalent to ones_like(depth_t)).
        depth_is_metric: Whether depth_t is in absolute metres (True) or
            relative/arbitrary scale (False).  When False, max_extent
            filtering is skipped since the threshold is meaningless.
    """

    frame_idx: int
    depth_t: np.ndarray
    K_t: np.ndarray
    T_wc_t: np.ndarray
    detections: Sequence[FastLocalDetection]
    depth_conf_t: Optional[np.ndarray] = None
    depth_is_metric: bool = True
    timestamp_s: float = 0.0  # Physical timestamp in seconds from video start


@dataclass(frozen=True)
class _FrameObservation:
    frame_idx: int
    step: STEPToken


@dataclass
class _TrackState:
    track_id: int
    observations: List[_FrameObservation] = field(default_factory=list)
    status: str = "active"  # active|lost|archived
    missing_streak: int = 0
    last_seen_t: Optional[int] = None
    last_centroid: Optional[np.ndarray] = None

    def observe(self, frame_idx: int, step: STEPToken, centroid: np.ndarray) -> None:
        if self.status == "archived":
            raise RuntimeError(
                f"Bug: observe() called on archived track {self.track_id}. "
                f"Archived tracks must not be re-identified."
            )
        self.observations.append(_FrameObservation(frame_idx=frame_idx, step=step))
        self.last_seen_t = frame_idx
        self.last_centroid = centroid
        self.status = "active"
        self.missing_streak = 0

    def miss(self, cfg: FastSNOWConfig) -> None:
        if self.status == "archived":
            return
        self.missing_streak += 1
        if self.missing_streak >= cfg.fusion.lost_patience + cfg.fusion.archive_patience:
            self.status = "archived"
        elif self.missing_streak >= cfg.fusion.lost_patience:
            self.status = "lost"


@dataclass(frozen=True)
class _Candidate:
    run_id: Hashable
    local_obj_id: Hashable
    mask: np.ndarray
    score: float
    centroid_xyz: np.ndarray
    step: STEPToken


class FastSNOWPipeline:
    """Fast-SNOW pipeline implementation for precomputed model outputs."""

    def __init__(self, config: Optional[FastSNOWConfig] = None):
        self.config = config or FastSNOWConfig()
        self.reset()

    def reset(self) -> None:
        self._next_global_id = 0
        self._local_to_global: Dict[RunKey, int] = {}
        self._tracks: Dict[int, _TrackState] = {}
        self._ego_poses_cw: Dict[int, np.ndarray] = {}
        self._latest_frame_idx: Optional[int] = None

    def process_frames(self, frames: Iterable[FastFrameInput], reset: bool = True) -> None:
        """Process a sequence of frame inputs."""
        if reset:
            self.reset()

        for frame in sorted(frames, key=lambda x: x.frame_idx):
            self.process_frame(frame)

    def process_frame(self, frame: FastFrameInput) -> None:
        """Process one frame (Step 4-7)."""
        self._latest_frame_idx = frame.frame_idx
        T_cw_t = np.linalg.inv(frame.T_wc_t)
        self._ego_poses_cw[frame.frame_idx] = T_cw_t

        candidates = self._build_candidates(frame, T_cw_t)
        winners = self._fuse_candidates(frame.frame_idx, candidates)

        touched: set[int] = set()

        for gid, cand in winners.items():
            state = self._tracks.setdefault(gid, _TrackState(track_id=gid))
            state.observe(frame.frame_idx, cand.step, cand.centroid_xyz)
            touched.add(gid)

        for gid, state in self._tracks.items():
            if gid not in touched:
                state.miss(self.config)

    def build_4dsg_dict(
        self,
        visual_anchor: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Build strict Fast-SNOW 4DSG JSON object (Step 8).

        Args:
            visual_anchor: Optional list of keyframe references for VLM,
                each ``{"frame_idx": int, "path": str}``.  When provided the
                entries are included under ``metadata.visual_anchor`` so the
                VLM can correlate image content blocks with STEP tokens
                (spec §4.3, line 445).
        """
        grid = self.config.step.grid_size

        ego_entries: List[Dict[str, object]] = []
        for t in sorted(self._ego_poses_cw.keys()):
            T_cw = self._ego_poses_cw[t]
            xyz = T_cw[:3, 3]
            ego_entries.append({"t": t, "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])]})

        tracks_entries: List[Dict[str, object]] = []
        for gid in sorted(self._tracks.keys()):
            state = self._tracks[gid]
            if not state.observations:
                continue

            obs_sorted = sorted(state.observations, key=lambda x: x.frame_idx)
            t_start = obs_sorted[0].frame_idx
            t_end = obs_sorted[-1].frame_idx

            # Sliding window: keep only the most recent T observations
            # for VLM serialization (SNOW §4.2, Eq. 7).
            T = self.config.step.temporal_window
            if T > 0 and len(obs_sorted) > T:
                obs_sorted = obs_sorted[-T:]

            fk: List[Dict[str, object]] = []

            for obs in obs_sorted:
                step = update_temporal_token(obs.step, t_start=t_start, t_end=t_end)
                fk.append(self._step_to_json(obs.frame_idx, step))

            tracks_entries.append({"object_id": gid, "F_k": fk})

        metadata: Dict[str, object] = {
            "grid": f"{grid}x{grid}",
            "num_frames": len(self._ego_poses_cw),
            "num_tracks": len(tracks_entries),
            "coordinate_system": "World frame = first frame camera. X=right, Y=down, Z=forward. Scale: metres if metric model, otherwise relative.",
            "schema": {
                "ego[].xyz": "Camera position in world coordinates [x, y, z].",
                "tracks[].F_k[]": "Per-frame STEP token S_t^k (Eq. 4). Fields use paper symbols.",
                "tau": f"Image patch tokens: occupied cells in a {grid}x{grid} grid on the masked image. row/col = grid index; iou = mask-cell overlap.",
                "c": "Centroid token: 3D center of the object's point cloud in world coordinates [x, y, z].",
                "s": "Shape token: per-axis Gaussian statistics (mu, sigma, min, max) of the object's 3D point cloud.",
                "theta": "Temporal token: [t_start, t_end] track-level lifespan.",
            },
        }
        if visual_anchor is not None:
            metadata["visual_anchor"] = visual_anchor

        return {
            "metadata": metadata,
            "ego": ego_entries,
            "tracks": tracks_entries,
        }

    def serialize_4dsg(
        self,
        visual_anchor: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        """Serialize Step 8 JSON (no truncation, no quantization)."""
        return json.dumps(
            self.build_4dsg_dict(visual_anchor=visual_anchor),
            indent=2,
            sort_keys=False,
        )

    def _build_candidates(self, frame: FastFrameInput, T_cw_t: np.ndarray) -> List[_Candidate]:
        out: List[_Candidate] = []
        for det in frame.detections:
            # Score filtering is done in Step 3 (SAM3 wrapper);
            # Step 4 only applies geometric filters (conf/min_points/max_extent).
            mask = det.mask.astype(bool, copy=False)
            points_world = self._backproject_mask_points(
                mask=mask,
                depth_t=frame.depth_t,
                K_t=frame.K_t,
                T_cw_t=T_cw_t,
                depth_conf_t=frame.depth_conf_t,
            )
            if points_world.shape[0] < self.config.depth_filter.min_points:
                continue

            shape = build_shape_token(points_world)
            # max_extent filter only meaningful with metric depth;
            # relative depth has arbitrary scale → skip to avoid false rejection.
            if frame.depth_is_metric:
                extents = np.array(
                    [
                        shape.x_max - shape.x_min,
                        shape.y_max - shape.y_min,
                        shape.z_max - shape.z_min,
                    ],
                    dtype=float,
                )
                if np.any(extents > self.config.depth_filter.max_extent):
                    continue

            centroid = build_centroid_token(points_world)
            patch_tokens = mask_to_patch_tokens(
                mask,
                grid_size=self.config.step.grid_size,
                iou_threshold=self.config.step.iou_threshold,
            )
            step = STEPToken(
                patch_tokens=patch_tokens,
                centroid=centroid,
                shape=shape,
                temporal=TemporalToken(t_start=frame.frame_idx, t_end=frame.frame_idx),
            )

            out.append(
                _Candidate(
                    run_id=det.run_id,
                    local_obj_id=det.local_obj_id,
                    mask=mask,
                    score=float(det.score),
                    centroid_xyz=np.array([centroid.x, centroid.y, centroid.z], dtype=float),
                    step=step,
                )
            )
        return out

    def _fuse_candidates(self, frame_idx: int, candidates: Sequence[_Candidate]) -> Dict[int, _Candidate]:
        if not candidates:
            return {}

        # 1) Assign provisional global IDs by local (run_id, obj_id)
        #    Archived tracks do NOT participate in matching (spec §5):
        #    if the old gid maps to an archived track, allocate a fresh gid.
        candidate_gids: List[int] = []
        keys: List[RunKey] = []
        for cand in candidates:
            key = (cand.run_id, cand.local_obj_id)
            keys.append(key)
            gid = self._local_to_global.get(key)
            if gid is not None:
                track = self._tracks.get(gid)
                if track is not None and track.status == "archived":
                    gid = None  # force new allocation
            if gid is None:
                gid = self._allocate_global_id()
                self._local_to_global[key] = gid
            candidate_gids.append(gid)

        # 2) Cross-run fusion with score-desc greedy order
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(keep: int, drop: int) -> int:
            rk = find(keep)
            rd = find(drop)
            if rk != rd:
                parent[rd] = rk
            return rk

        for gid in candidate_gids:
            parent.setdefault(gid, gid)

        order = sorted(range(len(candidates)), key=lambda i: candidates[i].score, reverse=True)
        for i_pos, i in enumerate(order):
            ci = candidates[i]
            gi = find(candidate_gids[i])
            # Skip if this track is archived (should not participate in merging)
            ti = self._tracks.get(gi)
            if ti is not None and ti.status == "archived":
                continue
            for j in order[i_pos + 1 :]:
                cj = candidates[j]
                if ci.run_id == cj.run_id:
                    continue
                gj = find(candidate_gids[j])
                if gi == gj:
                    continue
                # Skip if target track is archived
                tj = self._tracks.get(gj)
                if tj is not None and tj.status == "archived":
                    continue

                iou = self._mask_iou(ci.mask, cj.mask)
                if iou <= self.config.fusion.cross_run_iou_thresh:
                    continue

                cdist = float(np.linalg.norm(ci.centroid_xyz - cj.centroid_xyz))
                if cdist >= self.config.fusion.merge_centroid_dist_m:
                    continue

                li = self._tracks[gi].last_seen_t if gi in self._tracks and self._tracks[gi].last_seen_t is not None else frame_idx
                lj = self._tracks[gj].last_seen_t if gj in self._tracks and self._tracks[gj].last_seen_t is not None else frame_idx
                if abs(li - lj) > self.config.fusion.merge_temporal_gap:
                    continue

                gi = union(gi, gj)

        # 3) Merge historical track states for merged global IDs
        for gid in sorted(set(candidate_gids)):
            root = find(gid)
            if gid != root:
                self._merge_track_states(root, gid)

        # Keep mapping consistent globally
        for key, gid in list(self._local_to_global.items()):
            self._local_to_global[key] = find(gid)

        # 4) Keep only highest-score candidate per fused global ID for this frame
        best_idx_for_gid: Dict[int, int] = {}
        for idx, cand in enumerate(candidates):
            gid = find(candidate_gids[idx])
            best_idx = best_idx_for_gid.get(gid)
            if best_idx is None or cand.score > candidates[best_idx].score:
                best_idx_for_gid[gid] = idx

        winners: Dict[int, _Candidate] = {}
        for gid, idx in best_idx_for_gid.items():
            winners[gid] = candidates[idx]

        # Update current local ID mapping to winning global IDs.
        for idx, key in enumerate(keys):
            self._local_to_global[key] = find(candidate_gids[idx])

        return winners

    def _merge_track_states(self, keep_gid: int, drop_gid: int) -> None:
        if keep_gid == drop_gid:
            return

        keep = self._tracks.get(keep_gid)
        drop = self._tracks.get(drop_gid)
        if drop is None:
            return

        if keep is None:
            keep = _TrackState(track_id=keep_gid)
            self._tracks[keep_gid] = keep

        # Merge and deduplicate observations: one per frame_idx.
        # Prefer keep's observation (higher-score winner) over drop's.
        obs_by_frame: Dict[int, _FrameObservation] = {}
        for obs in drop.observations:
            obs_by_frame[obs.frame_idx] = obs
        for obs in keep.observations:
            obs_by_frame[obs.frame_idx] = obs  # keep overwrites drop
        keep.observations = sorted(obs_by_frame.values(), key=lambda x: x.frame_idx)

        if keep.last_seen_t is None or (drop.last_seen_t is not None and drop.last_seen_t > keep.last_seen_t):
            keep.last_seen_t = drop.last_seen_t
            keep.last_centroid = drop.last_centroid

        status_rank = {"active": 0, "lost": 1, "archived": 2}
        keep.status = min((keep.status, drop.status), key=lambda s: status_rank.get(s, 99))
        keep.missing_streak = min(keep.missing_streak, drop.missing_streak)

        del self._tracks[drop_gid]

    def _step_to_json(
        self,
        frame_idx: int,
        step: STEPToken,
    ) -> Dict[str, object]:
        # Top-k patch selection: sort by IoU desc, keep at most max_tau_per_step.
        patches = list(step.patch_tokens)
        k = self.config.step.max_tau_per_step
        if k > 0 and len(patches) > k:
            patches.sort(key=lambda p: p.iou, reverse=True)
            patches = patches[:k]

        tau_list: List[Dict[str, object]] = []
        for p in patches:
            tau_list.append({
                "row": int(p.row),
                "col": int(p.col),
                "iou": float(p.iou),
            })

        return {
            "t": frame_idx,
            "tau": tau_list,
            "c": [
                float(step.centroid.x),
                float(step.centroid.y),
                float(step.centroid.z),
            ],
            "s": {
                "x": {
                    "mu": float(step.shape.x_mu),
                    "sigma": float(step.shape.x_sigma),
                    "min": float(step.shape.x_min),
                    "max": float(step.shape.x_max),
                },
                "y": {
                    "mu": float(step.shape.y_mu),
                    "sigma": float(step.shape.y_sigma),
                    "min": float(step.shape.y_min),
                    "max": float(step.shape.y_max),
                },
                "z": {
                    "mu": float(step.shape.z_mu),
                    "sigma": float(step.shape.z_sigma),
                    "min": float(step.shape.z_min),
                    "max": float(step.shape.z_max),
                },
            },
            "theta": [int(step.temporal.t_start), int(step.temporal.t_end)],
        }

    def _backproject_mask_points(
        self,
        mask: np.ndarray,
        depth_t: np.ndarray,
        K_t: np.ndarray,
        T_cw_t: np.ndarray,
        depth_conf_t: Optional[np.ndarray],
    ) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")
        if depth_t.ndim != 2:
            raise ValueError(f"depth_t must be 2D, got shape {depth_t.shape}")
        if mask.shape != depth_t.shape:
            raise ValueError(
                f"mask/depth shape mismatch: mask={mask.shape}, depth={depth_t.shape}"
            )

        # Guarantee confidence map: fallback to ones (trust all pixels)
        if depth_conf_t is None:
            depth_conf_t = np.ones_like(depth_t, dtype=np.float32)
        elif depth_conf_t.shape != depth_t.shape:
            raise ValueError(
                f"depth_conf_t shape mismatch: conf={depth_conf_t.shape}, depth={depth_t.shape}"
            )

        valid = mask & np.isfinite(depth_t) & (depth_t > 0.0)
        valid &= depth_conf_t >= self.config.depth_filter.conf_thresh

        v, u = np.nonzero(valid)
        if u.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        d = depth_t[v, u].astype(np.float64)
        uv1 = np.stack([u.astype(np.float64), v.astype(np.float64), np.ones_like(d)], axis=0)

        K_inv = np.linalg.inv(K_t.astype(np.float64))
        rays = K_inv @ uv1
        p_cam = rays * d  # (3, N)

        p_cam_h = np.vstack([p_cam, np.ones((1, p_cam.shape[1]), dtype=np.float64)])
        p_world_h = T_cw_t.astype(np.float64) @ p_cam_h
        # Cast to float32 for storage/serialization (spec §5.5)
        return p_world_h[:3, :].T.astype(np.float32)

    def _allocate_global_id(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

__all__ = [
    "FastSNOWPipeline",
    "FastFrameInput",
    "FastLocalDetection",
]
