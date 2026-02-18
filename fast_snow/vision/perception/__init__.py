"""Perception module for SNOW.

Includes clustering, segmentation, association, and refinement components.
"""

from fast_snow.vision.perception.clustering.hdbscan import (
    HDBSCANConfig,
    ClusterResult,
    cluster_points,
    sample_cluster_points,
)
from fast_snow.vision.perception.refinement.hhop_filter import (
    HHopConfig,
    ValidationResult,
    detect_implausible,
    filter_implausible,
    hhop_validate,
    validate_track,
)
from fast_snow.vision.perception.refinement.iterative_refinement import (
    RefinementConfig,
    RefinementResult,
    iterative_refinement,
    IterativeRefinementPipeline,
)

__all__ = [
    # Clustering
    "HDBSCANConfig",
    "ClusterResult",
    "cluster_points",
    "sample_cluster_points",
    # H-hop Refinement
    "HHopConfig",
    "ValidationResult",
    "detect_implausible",
    "filter_implausible",
    "hhop_validate",
    "validate_track",
    # Iterative Refinement
    "RefinementConfig",
    "RefinementResult",
    "iterative_refinement",
    "IterativeRefinementPipeline",
]
