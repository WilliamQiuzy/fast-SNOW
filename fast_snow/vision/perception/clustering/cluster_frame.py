"""Cluster a FrameData and sample SAM2 point prompts.

This module is the Phase-1 bridge between Phase-0 (data alignment) and
Phase-2 (SAM2 segmentation).  It takes a :class:`~data.schema.FrameData`,
runs HDBSCAN on ``points_world``, and for every cluster samples exactly
*m* representative 3-D points that will serve as SAM2 point prompts.

Paper reference (Section 3.1, Eq. 1)::

    R_t = HDBSCAN(U_t)
    V_t^k = UniformSample(R_t^k, m)   # m = 4

Noise points (label = -1) are collected separately so that downstream
iterative refinement (Phase 5) can re-cluster them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fast_snow.data.schema import FrameData
from fast_snow.vision.perception.clustering.hdbscan import (
    ClusterResult,
    HDBSCANConfig,
    cluster_points,
    sample_cluster_points,
)

# Paper default: m = 4 prompt points per cluster
DEFAULT_PROMPTS_PER_CLUSTER: int = 4


@dataclass
class ClusterPrompt:
    """A single cluster with its m sampled SAM2 prompt points.

    Attributes:
        cluster_id: HDBSCAN's original cluster label (may be non-contiguous).
        point_indices: Indices into ``FrameData.points_world`` for all
            points belonging to this cluster.
        points_world: (N_k, 3) cluster points in world frame.
        prompt_points_world: (m, 3) sampled points in world frame.
    """
    cluster_id: int
    point_indices: np.ndarray          # (N_k,) int indices
    points_world: np.ndarray           # (N_k, 3)
    prompt_points_world: np.ndarray    # (m, 3)


@dataclass
class ClusteredFrame:
    """Result of clustering + prompt sampling for one frame.

    Attributes:
        frame_idx: Corresponding ``FrameData.frame_idx``.
        cluster_result: Raw HDBSCAN output (labels + index lists).
        prompts: Per-cluster prompt data, keyed by cluster_id.
        noise_indices: Indices of points labelled as noise (-1).
        prompts_per_cluster: m value used for sampling.
    """
    frame_idx: int
    cluster_result: ClusterResult
    prompts: Dict[int, ClusterPrompt] = field(default_factory=dict)
    noise_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )
    prompts_per_cluster: int = DEFAULT_PROMPTS_PER_CLUSTER

    @property
    def num_clusters(self) -> int:
        return len(self.prompts)

    @property
    def num_noise_points(self) -> int:
        return self.noise_indices.shape[0]

    def all_prompt_points(self) -> np.ndarray:
        """Concatenate all prompt points across clusters. Shape (K*m, 3)."""
        if not self.prompts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(
            [p.prompt_points_world for p in self.prompts.values()], axis=0
        )


def cluster_frame(
    frame: FrameData,
    config: Optional[HDBSCANConfig] = None,
    prompts_per_cluster: int = DEFAULT_PROMPTS_PER_CLUSTER,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ClusteredFrame:
    """Cluster a frame's point cloud and sample prompt points.

    Args:
        frame: Input frame with ``points_world``.
        config: HDBSCAN hyperparameters (paper defaults if *None*).
        prompts_per_cluster: *m* – number of prompt points per cluster
            (paper: 4).
        seed: Random seed for prompt sampling.  Ignored if *rng* is
            provided.
        rng: Explicit :class:`numpy.random.Generator`.  Takes priority
            over *seed*.

    Returns:
        :class:`ClusteredFrame` with per-cluster prompts.
    """
    config = config or HDBSCANConfig()
    if rng is None:
        rng = np.random.default_rng(seed)

    pts = frame.points_world

    # Edge case: empty point cloud
    if pts.shape[0] == 0:
        return ClusteredFrame(
            frame_idx=frame.frame_idx,
            cluster_result=ClusterResult(
                labels=np.empty(0, dtype=np.intp),
                clusters=[],
            ),
            prompts={},
            noise_indices=np.empty(0, dtype=np.intp),
            prompts_per_cluster=prompts_per_cluster,
        )

    # 1. HDBSCAN clustering  (Eq. 1)
    cr = cluster_points(pts, config)

    # 2. Collect noise indices (label == -1)
    noise_mask = cr.labels == -1
    noise_indices = np.where(noise_mask)[0]

    # 3. Sample m prompt points per cluster
    sampled = sample_cluster_points(
        pts, cr.clusters, prompts_per_cluster, rng
    )

    # 4. Build ClusterPrompt per cluster (use HDBSCAN's original label)
    prompts: Dict[int, ClusterPrompt] = {}
    for idx_arr, prompt_pts in zip(cr.clusters, sampled):
        label = int(cr.labels[idx_arr[0]])
        prompts[label] = ClusterPrompt(
            cluster_id=label,
            point_indices=idx_arr,
            points_world=pts[idx_arr],
            prompt_points_world=prompt_pts,
        )

    return ClusteredFrame(
        frame_idx=frame.frame_idx,
        cluster_result=cr,
        prompts=prompts,
        noise_indices=noise_indices,
        prompts_per_cluster=prompts_per_cluster,
    )
