r"""Iterative refinement for point cloud segmentation.

This module implements the iterative refinement loop from SNOW Algorithm 1:

    for n = 1 to N_iter do
        R_t = HDBSCAN(U_t)
        ... segmentation and STEP encoding ...
        Apply H-hop validation (H_hop rounds)
        U_t = P_t \ ∪_k R̂_t^k  (valid points)
        U_t ← U_t ∪ rejected_points  (rejected by H-hop go back to U_t)
        if U_t = ∅ then break
    end for

Key implementation detail (Phase 5 critical):
- Points from objects rejected by H-hop validation are returned to U_t
- This allows them to be re-clustered in the next iteration
- Paper: "geometrically implausible detections are filtered out and their
  points are returned to the unmapped set for potential re-assignment"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fast_snow.vision.perception.clustering.hdbscan import HDBSCANConfig, ClusterResult, cluster_points, sample_cluster_points
from fast_snow.vision.perception.refinement.hhop_filter import HHopConfig, filter_implausible
from fast_snow.reasoning.tokens.step_encoding import STEPToken


@dataclass
class RefinementConfig:
    """Configuration for iterative refinement.

    Paper reference: Algorithm 1
    Paper parameters: N_iter=1, H_hop=1
    """

    # Maximum iterations (Paper: N_iter=1)
    n_iter: int = 1

    # H-hop validation rounds per iteration (Paper: H_hop=1)
    h_hop: int = 1

    # Minimum points to continue refinement
    min_unmapped_points: int = 10

    # Samples per cluster for prompts (Paper: m=4)
    samples_per_cluster: int = 4

    # HDBSCAN config
    hdbscan_config: HDBSCANConfig = field(default_factory=HDBSCANConfig)

    # H-hop config
    hhop_config: HHopConfig = field(default_factory=HHopConfig)


@dataclass
class RefinementResult:
    """Result of iterative refinement for a single frame.

    Paper reference: Algorithm 1 output after N_iter iterations.
    """

    # Final STEP tokens (node_id -> STEPToken)
    step_tokens: Dict[int, STEPToken]

    # Points that could not be assigned (final U_t)
    unmapped_points: np.ndarray

    # Statistics
    iterations_run: int
    total_clusters_found: int
    total_rejected_by_hhop: int
    total_points_recycled: int  # Points returned to U_t due to H-hop rejection


# Type alias for the segmentation function
# Takes: points_xyz (N, 3), cluster_indices List[ndarray], frame_idx int
# Returns: Tuple of (Dict[int, STEPToken], Dict[int, ndarray])
#   - First dict maps cluster_id to STEP token
#   - Second dict maps cluster_id to point indices in original cloud
SegmentationFunc = Callable[
    [np.ndarray, List[np.ndarray], int],
    Tuple[Dict[int, STEPToken], Dict[int, np.ndarray]]
]


def iterative_refinement(
    points_xyz: np.ndarray,
    frame_idx: int,
    segment_fn: SegmentationFunc,
    config: Optional[RefinementConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> RefinementResult:
    """Run iterative refinement on a point cloud.

    This implements the core loop of SNOW Algorithm 1:
    1. Cluster unmapped points with HDBSCAN
    2. Sample prompt points from clusters
    3. Run segmentation (SAM2) and build STEP tokens
    4. Apply H-hop geometric validation
    5. Update unmapped points
    6. Repeat until convergence or max iterations

    Args:
        points_xyz: Full point cloud, shape (N, 3)
        frame_idx: Current frame index
        segment_fn: Function that takes (points, cluster_indices, frame_idx)
                   and returns Dict[int, STEPToken]
        config: Refinement configuration
        rng: Random generator for sampling

    Returns:
        RefinementResult with final STEP tokens and statistics
    """
    if config is None:
        config = RefinementConfig()

    if rng is None:
        rng = np.random.default_rng()

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    N = points_xyz.shape[0]

    # Initialize unmapped points as all points
    unmapped_mask = np.ones(N, dtype=bool)

    # Accumulated results
    all_step_tokens: Dict[int, STEPToken] = {}
    next_node_id = 0

    iterations_run = 0
    total_clusters = 0
    total_rejected = 0
    total_recycled = 0  # Track points returned to U_t

    for iteration in range(config.n_iter):
        iterations_run = iteration + 1

        # Get unmapped points
        unmapped_indices = np.where(unmapped_mask)[0]
        unmapped_points = points_xyz[unmapped_indices]

        if len(unmapped_points) < config.min_unmapped_points:
            break

        # Step 1: Cluster unmapped points
        try:
            cluster_result = cluster_points(unmapped_points, config.hdbscan_config)
        except Exception:
            # Clustering failed, use fallback: treat all unmapped points as one cluster
            # This ensures segment_fn and H-hop validation still run
            cluster_result = None

        # Handle clustering results with fallback
        if cluster_result is not None and cluster_result.clusters:
            # Normal case: HDBSCAN found clusters
            total_clusters += len(cluster_result.clusters)
            cluster_indices_original = [
                unmapped_indices[idx] for idx in cluster_result.clusters
            ]
        else:
            # Fallback case: No clusters found or clustering failed
            # Treat all unmapped points as a single cluster to allow H-hop validation
            if len(unmapped_points) >= config.min_unmapped_points:
                total_clusters += 1
                cluster_indices_original = [unmapped_indices]
            else:
                # Too few points even for fallback
                break

        # Step 2: Run segmentation and get STEP tokens
        # The segment_fn handles: prompt sampling, SAM2, mask-point association, STEP encoding
        # Returns: (tokens, point_indices) where point_indices[cluster_id] = array of point indices
        iteration_tokens, token_point_indices = segment_fn(points_xyz, cluster_indices_original, frame_idx)

        if not iteration_tokens:
            break

        # Step 3: H-hop geometric validation
        # Paper (Algorithm 1): Rejected points go back to U_t for next iteration
        for hop in range(config.h_hop):
            valid_tokens = filter_implausible(iteration_tokens, config.hhop_config)
            rejected_count = len(iteration_tokens) - len(valid_tokens)
            total_rejected += rejected_count

            # CRITICAL: Add rejected points back to unmapped_mask (回收机制)
            # Paper: "points from implausible detections are returned to U_t"
            rejected_ids = set(iteration_tokens.keys()) - set(valid_tokens.keys())
            for rejected_id in rejected_ids:
                if rejected_id in token_point_indices:
                    # Mark these points as unmapped again
                    rejected_points = token_point_indices[rejected_id]
                    unmapped_mask[rejected_points] = True
                    total_recycled += len(rejected_points)

            iteration_tokens = valid_tokens

        # Step 4: Assign node IDs and accumulate
        for step in iteration_tokens.values():
            all_step_tokens[next_node_id] = step
            next_node_id += 1

        # Step 5: Update unmapped points
        # Mark points that were successfully assigned (passed H-hop) in this iteration
        for cluster_id, step_token in iteration_tokens.items():
            if cluster_id in token_point_indices:
                unmapped_mask[token_point_indices[cluster_id]] = False

        # Check if all points mapped
        if not unmapped_mask.any():
            break

    # Get final unmapped points
    final_unmapped = points_xyz[unmapped_mask]

    return RefinementResult(
        step_tokens=all_step_tokens,
        unmapped_points=final_unmapped,
        iterations_run=iterations_run,
        total_clusters_found=total_clusters,
        total_rejected_by_hhop=total_rejected,
        total_points_recycled=total_recycled,
    )


def create_simple_segment_fn(
    masks: Sequence[np.ndarray],
    images: Sequence[np.ndarray],
    projection_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> SegmentationFunc:
    """Create a simple segmentation function for testing.

    In production, this should be replaced with SAM2-based segmentation.

    Args:
        masks: Pre-computed masks for each cluster
        images: Images for each camera
        projection_fn: Function that projects points to image coords

    Returns:
        A segmentation function compatible with iterative_refinement
    """
    from fast_snow.reasoning.tokens.step_encoding import build_step_token

    def segment_fn(
        points_xyz: np.ndarray,
        cluster_indices: List[np.ndarray],
        frame_idx: int,
    ) -> Tuple[Dict[int, STEPToken], Dict[int, np.ndarray]]:
        """Simple segmentation that creates STEP tokens from cluster points.

        Returns:
            Tuple of (tokens, point_indices) where:
            - tokens[cluster_id] = STEPToken
            - point_indices[cluster_id] = array of point indices in original cloud
        """
        tokens: Dict[int, STEPToken] = {}
        point_indices: Dict[int, np.ndarray] = {}

        for i, indices in enumerate(cluster_indices):
            if len(indices) == 0:
                continue

            cluster_points = points_xyz[indices]

            # Use provided mask or create dummy mask
            if i < len(masks):
                mask = masks[i]
            else:
                # Create dummy mask
                mask = np.zeros((256, 256), dtype=bool)
                mask[64:192, 64:192] = True

            try:
                step = build_step_token(
                    mask=mask,
                    points_xyz=cluster_points,
                    t_start=frame_idx,
                    t_end=frame_idx,
                )
                tokens[i] = step
                point_indices[i] = indices  # Store original point indices
            except Exception:
                continue

        return tokens, point_indices

    return segment_fn


class IterativeRefinementPipeline:
    """Pipeline for iterative refinement with SAM2 integration.

    This class manages the full refinement loop including:
    - Point cloud clustering
    - SAM2 segmentation with point prompts
    - Multi-view mask matching
    - STEP token creation
    - H-hop validation
    """

    def __init__(
        self,
        config: Optional[RefinementConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config or RefinementConfig()
        self.rng = rng or np.random.default_rng()

    def process_frame(
        self,
        points_xyz: np.ndarray,
        frame_idx: int,
        segment_fn: SegmentationFunc,
    ) -> RefinementResult:
        """Process a single frame with iterative refinement.

        Args:
            points_xyz: Point cloud (N, 3)
            frame_idx: Frame index
            segment_fn: Segmentation function

        Returns:
            RefinementResult
        """
        return iterative_refinement(
            points_xyz=points_xyz,
            frame_idx=frame_idx,
            segment_fn=segment_fn,
            config=self.config,
            rng=self.rng,
        )

    def process_sequence(
        self,
        point_clouds: Sequence[np.ndarray],
        segment_fn: SegmentationFunc,
    ) -> List[RefinementResult]:
        """Process a sequence of frames.

        Args:
            point_clouds: List of point clouds, each (N_t, 3)
            segment_fn: Segmentation function

        Returns:
            List of RefinementResult for each frame
        """
        results = []
        for frame_idx, points in enumerate(point_clouds):
            result = self.process_frame(points, frame_idx, segment_fn)
            results.append(result)
        return results
