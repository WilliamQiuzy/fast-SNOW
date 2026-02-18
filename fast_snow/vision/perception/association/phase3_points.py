"""Phase 3: 3D-2D Association and Object Point Set Construction.

This module implements the association of 3D points with 2D segmentation masks
across multiple camera views, producing object-level point clouds for STEP encoding.

Paper reference (Section 3.2):
    "Each 3D point p_i^t is assigned to mask m_t^{k,c} if its projection
    π(p_i^t, I_t^c) lies within the mask's support."

Workflow:
    1. Project all 3D points to each camera view
    2. Assign points to masks per view using point_mask.assign_points_to_masks()
    3. Merge point sets across views using cross-view matching results
    4. Construct object-level point clouds (cluster_id -> points_xyz)
    5. Update U_t (unmapped points that didn't get assigned to any mask)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from fast_snow.data.schema import FrameData
from fast_snow.data.transforms.projection import project_points
from fast_snow.vision.perception.association.point_mask import assign_points_to_masks
from fast_snow.vision.perception.segmentation.phase2_segment import SegmentationResult


@dataclass
class ObjectPointSet:
    """Point set for a single object after multi-view fusion.

    Attributes:
        cluster_id: HDBSCAN cluster label.
        point_indices: (N_k,) indices into the original point cloud.
        points_xyz: (N_k, 3) actual 3D coordinates (world frame).
        num_views: Number of camera views that contributed points.
        view_counts: Dict mapping cam_id to number of points from that view.
    """
    cluster_id: int
    point_indices: np.ndarray
    points_xyz: np.ndarray
    num_views: int = 0
    view_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class Phase3Result:
    """Complete Phase 3 output for one frame.

    Attributes:
        object_points: Mapping from cluster_id to ObjectPointSet.
        unmapped_indices: (M,) indices of points not assigned to any mask.
        unmapped_points: (M, 3) xyz coordinates of unmapped points.
        total_points: Total number of points in input.
        assigned_points: Number of points assigned to objects.
        unmapped_count: Number of unmapped points (for U_t).
    """
    object_points: Dict[int, ObjectPointSet] = field(default_factory=dict)
    unmapped_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    unmapped_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    total_points: int = 0
    assigned_points: int = 0
    unmapped_count: int = 0


def project_points_to_all_views(
    frame_data: FrameData,
) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """Project all 3D points to each camera view.

    Args:
        frame_data: FrameData with points_world and cameras.

    Returns:
        Dict mapping cam_id -> (xy, valid) where:
            - xy: (N, 2) pixel coordinates
            - valid: (N,) bool mask of valid projections
    """
    points_world = frame_data.points_world
    projections: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cam_id, camera in frame_data.cameras.items():
        proj_result = project_points(points_world, camera)
        projections[cam_id] = (proj_result.xy, proj_result.valid)

    return projections


def assign_points_per_view(
    projections: Dict[str, tuple[np.ndarray, np.ndarray]],
    seg_result: SegmentationResult,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Assign points to masks for each camera view.

    Args:
        projections: Dict of {cam_id: (xy, valid)} from project_points_to_all_views.
        seg_result: Phase 2 segmentation result with per_view masks.

    Returns:
        Dict mapping {cam_id: {cluster_id: point_indices}}.
        Only includes clusters that have valid masks in this view.
    """
    points_per_view: Dict[str, Dict[int, np.ndarray]] = {}

    for cam_id, (xy, valid) in projections.items():
        if cam_id not in seg_result.per_view:
            points_per_view[cam_id] = {}
            continue

        per_view_masks = seg_result.per_view[cam_id]

        # Extract masks in sorted cluster_id order
        # IMPORTANT: PerViewMasks.cluster_ids is sorted (guaranteed by property)
        # This ensures stable mapping between mask_idx and cluster_id
        cluster_ids = per_view_masks.cluster_ids
        masks_list = [per_view_masks.masks[cid].mask for cid in cluster_ids]

        # Verify sorting consistency (defensive programming)
        assert cluster_ids == sorted(cluster_ids), \
            f"cluster_ids not sorted in view {cam_id}: {cluster_ids}"

        if not masks_list:
            points_per_view[cam_id] = {}
            continue

        # Assign points to masks
        # mask_idx in association corresponds to index in cluster_ids
        association = assign_points_to_masks(xy, masks_list, valid)

        # Build mapping from cluster_id to point indices
        # mask_idx -> cluster_id mapping is guaranteed by sorted cluster_ids
        cluster_to_points: Dict[int, np.ndarray] = {}
        for mask_idx, cluster_id in enumerate(cluster_ids):
            point_indices = association.mask_to_points[mask_idx]
            if len(point_indices) > 0:
                cluster_to_points[cluster_id] = point_indices

        points_per_view[cam_id] = cluster_to_points

    return points_per_view


def merge_points_across_views(
    points_per_view: Dict[str, Dict[int, np.ndarray]],
    points_world: np.ndarray,
    seg_result: SegmentationResult,
) -> Dict[int, ObjectPointSet]:
    """Merge point sets across camera views for each cluster.

    Paper: Multiple views of the same cluster should contribute to the same
    object point set. Uses Phase 2 cross-view matching results to ensure
    consistency across views.

    Strategy:
        1. If cross_view matching is available, use it to validate that masks
           with the same cluster_id across views are actually matched.
        2. Only merge points from views where the cluster is confirmed by
           cross-view consistency.
        3. Deduplicate point indices across views.

    Args:
        points_per_view: {cam_id: {cluster_id: point_indices}}.
        points_world: (N, 3) original point cloud for extracting xyz.
        seg_result: Phase 2 segmentation result with cross_view matches.

    Returns:
        Dict mapping cluster_id -> ObjectPointSet with merged points.
    """
    # Collect all cluster IDs across all views
    all_cluster_ids: Set[int] = set()
    for cam_points in points_per_view.values():
        all_cluster_ids.update(cam_points.keys())

    object_points: Dict[int, ObjectPointSet] = {}

    # If cross-view matching is available, use it for consistency
    # Otherwise fall back to simple cluster_id merging
    use_cross_view = (seg_result.cross_view is not None and
                      len(seg_result.cross_view.per_view) >= 2)

    for cluster_id in sorted(all_cluster_ids):
        # Gather point indices from all views
        point_indices_list: List[np.ndarray] = []
        view_counts: Dict[str, int] = {}

        if use_cross_view:
            # Use cross-view matching to validate consistency
            # Only include points from views where this cluster is confirmed
            # by Hungarian matching across views
            for cam_id, cam_points in points_per_view.items():
                if cluster_id not in cam_points:
                    continue

                # Check if this cluster is consistently matched across views
                # For single-view clusters, always include them
                # For multi-view, verify they're matched by Hungarian
                is_valid = _validate_cross_view_cluster(
                    cluster_id, cam_id, seg_result.cross_view
                )

                if is_valid:
                    indices = cam_points[cluster_id]
                    point_indices_list.append(indices)
                    view_counts[cam_id] = len(indices)
        else:
            # Fallback: simple cluster_id merging (no cross-view validation)
            for cam_id, cam_points in points_per_view.items():
                if cluster_id in cam_points:
                    indices = cam_points[cluster_id]
                    point_indices_list.append(indices)
                    view_counts[cam_id] = len(indices)

        if not point_indices_list:
            continue

        # Concatenate and deduplicate
        all_indices = np.concatenate(point_indices_list)
        unique_indices = np.unique(all_indices)

        # Extract 3D coordinates
        points_xyz = points_world[unique_indices]

        object_points[cluster_id] = ObjectPointSet(
            cluster_id=cluster_id,
            point_indices=unique_indices,
            points_xyz=points_xyz,
            num_views=len(view_counts),
            view_counts=view_counts,
        )

    return object_points


def _validate_cross_view_cluster(
    cluster_id: int,
    cam_id: str,
    cross_view,
) -> bool:
    """Validate that a cluster is consistent across views using Hungarian matching.

    This function verifies that when a cluster appears in multiple views,
    the masks are actually matched by Hungarian algorithm (not just having
    the same cluster_id by coincidence).

    Args:
        cluster_id: Cluster ID to validate.
        cam_id: Camera ID where this cluster appears.
        cross_view: CrossViewMatches from Phase 2.

    Returns:
        True if the cluster is valid (single-view or matched across views).
        False if the cluster appears in multiple views but isn't matched.
    """
    # Find which view index this cam_id corresponds to
    try:
        view_idx = cross_view.cam_ids.index(cam_id)
    except (ValueError, AttributeError):
        # Camera not in cross-view matching, accept by default
        return True

    # Get the cluster_ids list for this view
    per_view = cross_view.per_view[view_idx]
    if cluster_id not in per_view.cluster_ids:
        return True  # Not in this view's masks, shouldn't happen but accept

    # Find which mask index this cluster_id corresponds to in this view
    try:
        mask_idx_in_view = per_view.cluster_ids.index(cluster_id)
    except ValueError:
        return True  # Shouldn't happen, but accept

    # Check if this cluster appears in other views
    other_views_with_cluster = []
    for other_view_idx, other_per_view in enumerate(cross_view.per_view):
        if other_view_idx == view_idx:
            continue
        if cluster_id in other_per_view.cluster_ids:
            other_mask_idx = other_per_view.cluster_ids.index(cluster_id)
            other_views_with_cluster.append((other_view_idx, other_mask_idx))

    # If this cluster only appears in current view (by cluster_id),
    # we still need to verify the mask isn't matched to other views with different cluster_ids
    if not other_views_with_cluster:
        # Check if this mask is matched to any masks in other views
        # If yes, it means cluster_id is inconsistent across views → reject
        matched_masks = cross_view.matched_masks

        for other_view_idx in range(len(cross_view.per_view)):
            if other_view_idx == view_idx:
                continue

            pair_key = (min(view_idx, other_view_idx), max(view_idx, other_view_idx))

            if pair_key not in matched_masks.matches:
                # Missing pair_key → data inconsistency, reject
                return False

            matches = matched_masks.matches[pair_key]

            # Check if our mask is matched to any mask in this other view
            for match in matches:
                if view_idx < other_view_idx:
                    if match[0] == mask_idx_in_view:
                        # Our mask is matched to match[1] in other view
                        # But that other mask doesn't have our cluster_id
                        # (otherwise it would be in other_views_with_cluster)
                        # This is cluster_id inconsistency → reject
                        return False
                else:
                    if match[1] == mask_idx_in_view:
                        # Our mask is matched to match[0] in other view
                        # Cluster_id inconsistency → reject
                        return False

        # Not matched to any other views - true single-view cluster
        return True

    # For multi-view clusters, verify Hungarian matching
    # Check if masks in current view and other views are actually matched
    matched_masks = cross_view.matched_masks

    for other_view_idx, other_mask_idx in other_views_with_cluster:
        # Determine the pair key (always use sorted order)
        if view_idx < other_view_idx:
            pair_key = (view_idx, other_view_idx)
            expected_match = (mask_idx_in_view, other_mask_idx)
        else:
            pair_key = (other_view_idx, view_idx)
            expected_match = (other_mask_idx, mask_idx_in_view)

        # Check if this pair was matched by Hungarian algorithm
        if pair_key not in matched_masks.matches:
            # This should NOT happen - Phase 2 always creates keys for all view pairs
            # If this occurs, it indicates data inconsistency → reject for safety
            # (In non-strict mode, we could 'continue', but strict alignment requires rejection)
            return False

        matches = matched_masks.matches[pair_key]

        # Verify that the masks are actually matched
        if expected_match not in matches:
            # This cluster appears in both views but masks are NOT matched
            # This means they might be different physical objects with same cluster_id
            # REJECT this cluster in current view
            return False

    # All cross-view matches verified, accept this cluster
    return True


def compute_unmapped_points(
    total_num_points: int,
    object_points: Dict[int, ObjectPointSet],
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unmapped points (U_t in paper) that weren't assigned to any mask.

    Paper reference: Points not assigned to any mask are returned to U_t for
    iterative refinement in subsequent iterations.

    Args:
        total_num_points: Total number of points in the input cloud.
        object_points: Mapping of cluster_id -> ObjectPointSet.
        points_world: (N, 3) original point cloud.

    Returns:
        Tuple of (unmapped_indices, unmapped_points):
            - unmapped_indices: (M,) indices of unmapped points
            - unmapped_points: (M, 3) xyz coordinates
    """
    # Collect all assigned point indices
    assigned_indices_set: Set[int] = set()
    for obj_points in object_points.values():
        assigned_indices_set.update(obj_points.point_indices.tolist())

    # Find unmapped indices
    all_indices = np.arange(total_num_points)
    assigned_mask = np.isin(all_indices, list(assigned_indices_set))
    unmapped_indices = all_indices[~assigned_mask]

    # Extract unmapped points
    unmapped_points = points_world[unmapped_indices]

    return unmapped_indices, unmapped_points


def run_phase3(
    frame_data: FrameData,
    seg_result: SegmentationResult,
) -> Phase3Result:
    """Run the complete Phase 3 pipeline for one frame.

    Steps:
        1. Project all 3D points to each camera view.
        2. Assign points to masks per view.
        3. Merge point sets across views (deduplicating by point index).
        4. Construct object-level point clouds.
        5. Compute unmapped points (U_t).

    Args:
        frame_data: FrameData with points_world and cameras.
        seg_result: Phase 2 segmentation result with per-view masks.

    Returns:
        Phase3Result with object_points and unmapped_points.
    """
    points_world = frame_data.points_world
    total_points = len(points_world)

    # Handle empty point cloud
    if total_points == 0:
        return Phase3Result(
            total_points=0,
            assigned_points=0,
            unmapped_count=0,
        )

    # Handle case with no cameras (fallback)
    if not frame_data.cameras:
        return Phase3Result(
            unmapped_indices=np.arange(total_points),
            unmapped_points=points_world.copy(),
            total_points=total_points,
            assigned_points=0,
            unmapped_count=total_points,
        )

    # Step 1: Project points to all views
    projections = project_points_to_all_views(frame_data)

    # Step 2: Assign points to masks per view
    points_per_view = assign_points_per_view(projections, seg_result)

    # Step 3: Merge points across views (using Phase 2 cross-view matching)
    object_points = merge_points_across_views(points_per_view, points_world, seg_result)

    # Step 4: Compute unmapped points (U_t)
    unmapped_indices, unmapped_points = compute_unmapped_points(
        total_points, object_points, points_world
    )

    # Step 5: Build result
    assigned_points = total_points - len(unmapped_indices)

    return Phase3Result(
        object_points=object_points,
        unmapped_indices=unmapped_indices,
        unmapped_points=unmapped_points,
        total_points=total_points,
        assigned_points=assigned_points,
        unmapped_count=len(unmapped_indices),
    )
