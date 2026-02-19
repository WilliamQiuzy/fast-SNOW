"""Fast-SNOW runtime pipeline.

Implements the Fast-SNOW route in docs/roadmap/Fast-SNOW_IMPLEMENTATION.md:
- Step 4: mask + depth backprojection and geometric filtering
- Step 5: cross-run global ID fusion
- Step 6: STEP token construction
- Step 7: per-frame relation graph + temporal tracks
- Step 8: strict 1:1 JSON serialization

This module is model-agnostic: callers provide per-frame SAM3 detections and DA3 outputs.
"""

from __future__ import annotations

import json
import math
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
    """

    frame_idx: int
    depth_t: np.ndarray
    K_t: np.ndarray
    T_wc_t: np.ndarray
    detections: Sequence[FastLocalDetection]
    depth_conf_t: Optional[np.ndarray] = None


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
    # (frame_idx, ego-distance, ego-lateral)
    motion_samples: List[Tuple[int, float, float]] = field(default_factory=list)

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
        self._frame_relations: Dict[int, Tuple[List[Dict[str, object]], List[Dict[str, object]]]] = {}
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
        visible: Dict[int, _Candidate] = {}

        for gid, cand in winners.items():
            state = self._tracks.setdefault(gid, _TrackState(track_id=gid))
            state.observe(frame.frame_idx, cand.step, cand.centroid_xyz)
            touched.add(gid)
            visible[gid] = cand

        for gid, state in self._tracks.items():
            if gid not in touched:
                state.miss(self.config)

        ego_rel, obj_rel = self._compute_relations(frame.frame_idx, T_cw_t, visible)
        self._frame_relations[frame.frame_idx] = (ego_rel, obj_rel)

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
            fk: List[Dict[str, object]] = []

            for obs in obs_sorted:
                step = update_temporal_token(obs.step, t_start=t_start, t_end=t_end)
                fk.append(self._step_to_json(obs.frame_idx, step))

            tracks_entries.append({"object_id": gid, "F_k": fk})

        latest = self._latest_frame_idx
        if latest is not None and latest in self._frame_relations:
            ego_rel, obj_rel = self._frame_relations[latest]
        else:
            ego_rel, obj_rel = [], []

        metadata: Dict[str, object] = {
            "grid": f"{grid}x{grid}",
            "num_frames": len(self._ego_poses_cw),
            "num_tracks": len(tracks_entries),
        }
        if visual_anchor is not None:
            metadata["visual_anchor"] = visual_anchor

        return {
            "metadata": metadata,
            "ego": ego_entries,
            "tracks": tracks_entries,
            "ego_relations": ego_rel,
            "object_relations": obj_rel,
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

        # Merge and deduplicate motion_samples: one per frame_idx.
        # Prefer keep's samples (derived from winner candidate).
        ms_by_frame: Dict[int, Tuple[int, float, float]] = {}
        for s in drop.motion_samples:
            ms_by_frame[s[0]] = s
        for s in keep.motion_samples:
            ms_by_frame[s[0]] = s  # keep overwrites drop
        keep.motion_samples = sorted(ms_by_frame.values(), key=lambda x: x[0])

        if keep.last_seen_t is None or (drop.last_seen_t is not None and drop.last_seen_t > keep.last_seen_t):
            keep.last_seen_t = drop.last_seen_t
            keep.last_centroid = drop.last_centroid

        status_rank = {"active": 0, "lost": 1, "archived": 2}
        keep.status = min((keep.status, drop.status), key=lambda s: status_rank.get(s, 99))
        keep.missing_streak = min(keep.missing_streak, drop.missing_streak)

        del self._tracks[drop_gid]

    def _compute_relations(
        self,
        frame_idx: int,
        T_cw_t: np.ndarray,
        visible: Dict[int, _Candidate],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not visible:
            return [], []

        ego_xyz = T_cw_t[:3, 3]
        R_cw = T_cw_t[:3, :3]
        forward_world = R_cw @ np.array([0.0, 0.0, 1.0], dtype=float)
        yaw = float(np.arctan2(forward_world[1], forward_world[0]))
        c, s = math.cos(yaw), math.sin(yaw)
        R_ego = np.array([[c, -s], [s, c]], dtype=float)

        ego_relations: List[Dict[str, object]] = []
        for gid in sorted(visible.keys()):
            cand = visible[gid]
            delta = cand.centroid_xyz - ego_xyz
            d_xy_ego = R_ego.T @ delta[:2]
            angle = float(np.arctan2(d_xy_ego[1], d_xy_ego[0]))
            bearing = self._quantize_8_way(angle)
            elev = self._elev_relation(float(delta[2]))
            dist_m = float(np.linalg.norm(delta))

            track = self._tracks[gid]
            track.motion_samples.append((frame_idx, dist_m, float(d_xy_ego[1])))
            # Cap to motion_window to prevent unbounded growth
            window = max(2, self.config.edge.motion_window)
            if len(track.motion_samples) > window:
                track.motion_samples = track.motion_samples[-window:]
            motion = self._infer_motion(track)

            ego_relations.append(
                {
                    "object_id": gid,
                    "bearing": bearing,
                    "elev": elev,
                    "dist_m": dist_m,
                    "motion": motion,
                }
            )

        object_relations: List[Dict[str, object]] = []
        ids = sorted(visible.keys())
        k = max(1, self.config.edge.knn_k)

        for src in ids:
            src_c = visible[src].centroid_xyz
            neighbors: List[Tuple[float, int]] = []
            for dst in ids:
                if src == dst:
                    continue
                dst_c = visible[dst].centroid_xyz
                dist = float(np.linalg.norm(dst_c - src_c))
                neighbors.append((dist, dst))

            neighbors.sort(key=lambda x: x[0])
            for dist, dst in neighbors[:k]:
                dst_c = visible[dst].centroid_xyz
                delta = dst_c - src_c
                d_xy_ego = R_ego.T @ delta[:2]
                angle = float(np.arctan2(d_xy_ego[1], d_xy_ego[0]))
                obj_dir = self._quantize_8_way(angle)
                elev = self._elev_relation(float(delta[2]))

                object_relations.append(
                    {
                        "src": src,
                        "dst": dst,
                        "dir": obj_dir,
                        "elev": elev,
                        "dist_m": float(dist),
                    }
                )

        object_relations.sort(key=lambda x: (x["dist_m"], x["src"], x["dst"]))
        object_relations = object_relations[: self.config.serialization.max_obj_relations]

        return ego_relations, object_relations

    def _infer_motion(self, track: _TrackState) -> str:
        window = max(2, self.config.edge.motion_window)
        samples = track.motion_samples[-window:]
        # Cold start: need at least motion_window samples (spec §7)
        if len(samples) < window:
            return "unknown"

        t0, d0, l0 = samples[0]
        t1, d1, l1 = samples[-1]
        # Normalize by frame gap so sparse (e.g., 10Hz) sampling preserves motion magnitude.
        dt = max(1, t1 - t0)

        rate = (d1 - d0) / dt
        lateral_rate = abs(l1 - l0) / dt

        if rate < -self.config.edge.motion_thresh:
            return "approaching"
        if rate > self.config.edge.motion_thresh:
            return "receding"
        if lateral_rate > self.config.edge.lateral_thresh:
            return "lateral"
        return "static"

    def _step_to_json(self, frame_idx: int, step: STEPToken) -> Dict[str, object]:
        return {
            "t": frame_idx,
            "tau": [
                {"row": int(p.row), "col": int(p.col), "iou": float(p.iou)}
                for p in step.patch_tokens
            ],
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

    def _elev_relation(self, dz: float) -> str:
        if dz > self.config.edge.elev_thresh:
            return "above"
        if dz < -self.config.edge.elev_thresh:
            return "below"
        return "level"

    @staticmethod
    def _quantize_8_way(angle_rad: float) -> str:
        # Angle is in ego frame where 0 points to "front".
        deg = math.degrees(angle_rad)
        if -22.5 <= deg < 22.5:
            return "front"
        if 22.5 <= deg < 67.5:
            return "front-left"
        if 67.5 <= deg < 112.5:
            return "left"
        if 112.5 <= deg < 157.5:
            return "back-left"
        if deg >= 157.5 or deg < -157.5:
            return "back"
        if -157.5 <= deg < -112.5:
            return "back-right"
        if -112.5 <= deg < -67.5:
            return "right"
        return "front-right"


__all__ = [
    "FastSNOWPipeline",
    "FastFrameInput",
    "FastLocalDetection",
]
