"""HDBSCAN clustering for point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

# Single source of truth for HDBSCAN hyper-parameters lives in
# config.snow_config.  Re-export here so that existing callers
# (``from fast_snow.vision.perception.clustering.hdbscan import HDBSCANConfig``)
# continue to work unchanged.
from fast_snow.engine.config.snow_config import HDBSCANConfig  # noqa: F401 – re-export

try:
    import hdbscan
except Exception as exc:  # pragma: no cover - optional dependency
    hdbscan = None
    _HDBSCAN_IMPORT_ERROR = exc
else:
    _HDBSCAN_IMPORT_ERROR = None


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    clusters: List[np.ndarray]


def _check_hdbscan_available() -> None:
    if hdbscan is None:
        raise ImportError(
            "hdbscan is not available. Install it to use point cloud clustering."
        ) from _HDBSCAN_IMPORT_ERROR


def cluster_points(
    points_xyz: np.ndarray,
    config: HDBSCANConfig,
) -> ClusterResult:
    """Cluster point cloud with HDBSCAN.

    Args:
        points_xyz: (N, 3) array of xyz points.
        config: clustering hyperparameters.

    Returns:
        ClusterResult with labels for all points and list of per-cluster indices.
    """
    _check_hdbscan_available()
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        metric=config.metric,
        cluster_selection_method=config.cluster_selection_method,
        allow_single_cluster=config.allow_single_cluster,
    )
    labels = clusterer.fit_predict(points_xyz)

    clusters: List[np.ndarray] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # noise
        idx = np.where(labels == label)[0]
        clusters.append(idx)

    return ClusterResult(labels=labels, clusters=clusters)


def sample_cluster_points(
    points_xyz: np.ndarray,
    clusters: Sequence[np.ndarray],
    samples_per_cluster: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Uniformly sample points per cluster.

    Always returns exactly ``samples_per_cluster`` points per cluster,
    using sampling-with-replacement when a cluster has fewer points.

    Returns list of sampled point arrays (shape (m, 3)).
    """
    if samples_per_cluster <= 0:
        raise ValueError("samples_per_cluster must be positive")

    sampled: List[np.ndarray] = []
    for idx in clusters:
        if idx.size == 0:
            continue
        replace = idx.size < samples_per_cluster
        sampled_idx = rng.choice(idx, size=samples_per_cluster, replace=replace)
        sampled.append(points_xyz[sampled_idx])
    return sampled
