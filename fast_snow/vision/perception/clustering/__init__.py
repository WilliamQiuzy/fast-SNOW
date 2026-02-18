"""Clustering module for point cloud segmentation.

Uses HDBSCAN for density-based clustering as described in SNOW paper.
"""

from fast_snow.vision.perception.clustering.hdbscan import (
    HDBSCANConfig,
    ClusterResult,
    cluster_points,
    sample_cluster_points,
)
from fast_snow.vision.perception.clustering.cluster_frame import (
    ClusterPrompt,
    ClusteredFrame,
    cluster_frame,
    DEFAULT_PROMPTS_PER_CLUSTER,
)

__all__ = [
    "HDBSCANConfig",
    "ClusterResult",
    "cluster_points",
    "sample_cluster_points",
    "ClusterPrompt",
    "ClusteredFrame",
    "cluster_frame",
    "DEFAULT_PROMPTS_PER_CLUSTER",
]
