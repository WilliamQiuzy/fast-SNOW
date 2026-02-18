"""Phase 2: SAM2 segmentation from cluster prompt points.

Projects Phase-1 prompt points (world coords) into each camera view,
runs SAM2 per-view, matches masks across views with Hungarian IoU,
and selects the best mask per cluster for downstream STEP encoding.

Paper reference (Section 3.1)::

    M_t^{k,c} = SAM2(I_t^c, project(V_t^k, K^c, [R|t]^c))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import CameraModel
from fast_snow.data.schema import FrameData
from fast_snow.data.transforms.projection import project_points
from fast_snow.vision.perception.association.multiview_match import (
    MatchedMasks,
    match_masks_across_views,
)
from fast_snow.vision.perception.clustering.cluster_frame import ClusteredFrame
from fast_snow.vision.perception.segmentation.sam2_wrapper import SAM2Mask, SAM2Wrapper

# Import SAM2Config from the global config (has min_mask_score + multimask_output)
from fast_snow.engine.config.snow_config import SAM2Config


# =====================================================================
# Dataclasses
# =====================================================================

@dataclass(frozen=True)
class PromptProjection:
    """Projected prompt points for one cluster in one camera view.

    Attributes:
        cluster_id: HDBSCAN label.
        cam_id: Camera identifier.
        xy: (K, 2) valid pixel coordinates (K <= m).
        valid_count: Number of valid projected points (K).
        all_valid: Whether all m prompt points projected successfully.
    """
    cluster_id: int
    cam_id: str
    xy: np.ndarray
    valid_count: int
    all_valid: bool


@dataclass(frozen=True)
class ClusterMask:
    """Best SAM2 mask for one cluster in one camera view.

    Attributes:
        cluster_id: HDBSCAN label.
        cam_id: Camera identifier.
        mask: (H, W) bool array.
        score: SAM2 confidence score.
        prompt_xy: (K, 2) pixel prompts that produced this mask.
    """
    cluster_id: int
    cam_id: str
    mask: np.ndarray
    score: float
    prompt_xy: np.ndarray


@dataclass
class PerViewMasks:
    """All cluster masks for a single camera view."""
    cam_id: str
    masks: Dict[int, ClusterMask] = field(default_factory=dict)

    @property
    def mask_list(self) -> List[np.ndarray]:
        """Ordered list of mask arrays, sorted by cluster_id."""
        return [self.masks[k].mask for k in sorted(self.masks)]

    @property
    def cluster_ids(self) -> List[int]:
        """Sorted cluster IDs that have masks."""
        return sorted(self.masks.keys())


@dataclass
class CrossViewMatches:
    """Cross-view mask matching result."""
    matched_masks: MatchedMasks
    cam_ids: List[str]
    per_view: List[PerViewMasks]


@dataclass
class SegmentationResult:
    """Complete Phase 2 output for one frame.

    Attributes:
        per_view: Per-camera segmentation masks.
        cross_view: Cross-view matching (None if < 2 views with masks).
        best_masks: cluster_id -> best (H,W) bool mask for STEP encoding.
        skipped_clusters: cluster_ids with no valid mask in any view.
    """
    per_view: Dict[str, PerViewMasks] = field(default_factory=dict)
    cross_view: Optional[CrossViewMatches] = None
    best_masks: Dict[int, np.ndarray] = field(default_factory=dict)
    skipped_clusters: List[int] = field(default_factory=list)


# =====================================================================
# Core functions
# =====================================================================

def project_prompts_to_view(
    clustered_frame: ClusteredFrame,
    cam_id: str,
    camera: CameraModel,
    min_valid_points: int = 1,
) -> Dict[int, PromptProjection]:
    """Project all cluster prompts to a single camera view.

    Args:
        clustered_frame: Phase 1 output with per-cluster prompts.
        cam_id: Camera identifier.
        camera: CameraModel with world-to-camera extrinsics.
        min_valid_points: Minimum valid projected points to include.

    Returns:
        Dict mapping cluster_id -> PromptProjection.  Clusters where
        fewer than *min_valid_points* project validly are omitted.
    """
    projections: Dict[int, PromptProjection] = {}

    for label, prompt in clustered_frame.prompts.items():
        proj = project_points(prompt.prompt_points_world, camera)
        valid_mask = proj.valid
        valid_count = int(valid_mask.sum())

        if valid_count < min_valid_points:
            continue

        projections[label] = PromptProjection(
            cluster_id=label,
            cam_id=cam_id,
            xy=proj.xy[valid_mask],
            valid_count=valid_count,
            all_valid=(valid_count == prompt.prompt_points_world.shape[0]),
        )

    return projections


def segment_view(
    image: np.ndarray,
    projections: Dict[int, PromptProjection],
    sam2: SAM2Wrapper,
    cam_id: str,
    min_mask_score: float = 0.5,
) -> PerViewMasks:
    """Run SAM2 on one camera image for all projected clusters.

    For each cluster, calls ``SAM2Wrapper.predict()`` with the valid
    projected prompt points.  SAM2 may return multiple mask candidates;
    we keep the highest-scoring one above *min_mask_score*.

    Args:
        image: (H, W, 3) uint8 RGB image.
        projections: cluster_id -> PromptProjection.
        sam2: Loaded SAM2Wrapper instance.
        cam_id: Camera identifier.
        min_mask_score: Minimum score to accept a mask.

    Returns:
        PerViewMasks with one ClusterMask per accepted cluster.
    """
    per_view = PerViewMasks(cam_id=cam_id)

    for label, proj in projections.items():
        point_prompts: List[Tuple[int, int]] = [
            (int(round(proj.xy[i, 0])), int(round(proj.xy[i, 1])))
            for i in range(proj.valid_count)
        ]
        labels_list = [1] * len(point_prompts)  # all foreground

        masks = sam2.predict(image, point_prompts, labels=labels_list)
        if not masks:
            continue

        best = max(masks, key=lambda m: m.score)
        if best.score < min_mask_score:
            continue

        per_view.masks[label] = ClusterMask(
            cluster_id=label,
            cam_id=cam_id,
            mask=best.mask,
            score=best.score,
            prompt_xy=proj.xy,
        )

    return per_view


def match_across_views(
    per_view_list: List[PerViewMasks],
    iou_threshold: float = 0.0,
) -> CrossViewMatches:
    """Match masks across camera views using Hungarian IoU.

    Args:
        per_view_list: List of PerViewMasks (one per camera with masks).
        iou_threshold: Minimum IoU to accept a match.

    Returns:
        CrossViewMatches with matched_masks and camera ordering.
    """
    masks_per_view: List[List[np.ndarray]] = [
        pv.mask_list for pv in per_view_list
    ]
    matched = match_masks_across_views(masks_per_view, iou_threshold=iou_threshold)

    return CrossViewMatches(
        matched_masks=matched,
        cam_ids=[pv.cam_id for pv in per_view_list],
        per_view=per_view_list,
    )


def select_best_masks(
    per_view: Dict[str, PerViewMasks],
    cross_view: Optional[CrossViewMatches] = None,
) -> Dict[int, np.ndarray]:
    """Select the best mask per cluster with cross-view consistency enforcement.

    Paper reference (Section 3.2):
        "Consistency between masks of the same physical object across
        multiple camera views is enforced via Hungarian matching."

    Strategy:
        1. Collect all candidate masks per cluster across views.
        2. If cross_view matching is available, verify that masks for the
           same cluster_id in different views are matched by Hungarian.
        3. Clusters with inconsistent cross-view matches are EXCLUDED.
        4. Select the highest-scoring mask from consistent candidates only.

    Args:
        per_view: cam_id -> PerViewMasks.
        cross_view: Optional cross-view matching result.

    Returns:
        cluster_id -> (H, W) bool mask with highest score.
        Only clusters that pass cross-view consistency are included.
    """
    # Collect all candidates per cluster
    candidates: Dict[int, List[ClusterMask]] = {}
    for pv in per_view.values():
        for label, cm in pv.masks.items():
            candidates.setdefault(label, []).append(cm)

    # If no cross-view matching, fall back to simple score-based selection
    if cross_view is None or len(cross_view.per_view) < 2:
        best: Dict[int, np.ndarray] = {}
        for label, cms in candidates.items():
            winner = max(cms, key=lambda c: c.score)
            best[label] = winner.mask
        return best

    # Build cluster_id mapping for each view in cross_view
    # view_idx -> cluster_ids list (same order as mask_list)
    view_cluster_ids: List[List[int]] = [
        pv.cluster_ids for pv in cross_view.per_view
    ]

    # Check consistency: for each cluster, verify that its masks in different
    # views are actually matched by Hungarian algorithm
    consistent_clusters: set = set()

    for label in candidates.keys():
        # Find which views have this cluster
        view_indices_with_label = []
        mask_indices_in_views = []
        for view_idx, cluster_ids in enumerate(view_cluster_ids):
            if label in cluster_ids:
                view_indices_with_label.append(view_idx)
                mask_indices_in_views.append(cluster_ids.index(label))

        if len(view_indices_with_label) < 2:
            # Only one view has this cluster, cannot verify consistency
            # Include it (single-view clusters are acceptable)
            consistent_clusters.add(label)
            continue

        # Check if masks for this cluster are matched across view pairs
        is_consistent = True
        for i in range(len(view_indices_with_label)):
            for j in range(i + 1, len(view_indices_with_label)):
                va, vb = view_indices_with_label[i], view_indices_with_label[j]
                ma, mb = mask_indices_in_views[i], mask_indices_in_views[j]

                # Look up match in cross_view result
                pair_key = (min(va, vb), max(va, vb))
                if pair_key not in cross_view.matched_masks.matches:
                    continue

                matches = cross_view.matched_masks.matches[pair_key]
                # Check if (ma, mb) or (mb, ma) is in matches
                if va < vb:
                    expected_match = (ma, mb)
                else:
                    expected_match = (mb, ma)

                if expected_match not in matches:
                    # The masks for this cluster in these two views are not matched
                    # This indicates potential inconsistency - EXCLUDE this cluster
                    is_consistent = False
                    break
            if not is_consistent:
                break

        if is_consistent:
            consistent_clusters.add(label)
        # Inconsistent clusters are NOT added → excluded from output

    # Select best masks ONLY from consistent clusters (strict paper alignment)
    best: Dict[int, np.ndarray] = {}
    for label in consistent_clusters:
        cms = candidates[label]
        winner = max(cms, key=lambda c: c.score)
        best[label] = winner.mask

    return best


def run_phase2(
    frame_data: FrameData,
    clustered_frame: ClusteredFrame,
    sam2: SAM2Wrapper,
    sam2_config: SAM2Config,
    min_valid_points: int = 1,
    iou_threshold: float = 0.0,
) -> SegmentationResult:
    """Run the complete Phase 2 segmentation pipeline for one frame.

    Steps:
        1. For each camera, project cluster prompts to pixel space.
        2. For each camera, load image and run SAM2.
        3. Match masks across views (if >= 2 cameras with masks).
        4. Select best mask per cluster.

    Args:
        frame_data: FrameData with cameras and images.
        clustered_frame: Phase 1 output.
        sam2: Pre-loaded SAM2Wrapper.
        sam2_config: SAM2Config (for min_mask_score).
        min_valid_points: Min valid projected points to attempt SAM2.
        iou_threshold: IoU threshold for cross-view matching.

    Returns:
        SegmentationResult with per-view masks, cross-view matches,
        best_masks for downstream STEP encoding, and skipped clusters.
    """
    if not frame_data.cameras:
        return SegmentationResult(
            skipped_clusters=sorted(clustered_frame.prompts.keys()),
        )

    per_view: Dict[str, PerViewMasks] = {}
    all_cluster_ids = set(clustered_frame.prompts.keys())
    seen_cluster_ids: set = set()

    for cam_id, camera in frame_data.cameras.items():
        # 1. Project prompts
        projections = project_prompts_to_view(
            clustered_frame, cam_id, camera,
            min_valid_points=min_valid_points,
        )

        if not projections:
            per_view[cam_id] = PerViewMasks(cam_id=cam_id)
            continue

        # 2. Load image (with protection for missing images)
        # Check if image is available before attempting to load
        has_image = (
            cam_id in frame_data.images or
            cam_id in frame_data.image_paths
        )
        if not has_image:
            # Skip this camera if no image is available
            per_view[cam_id] = PerViewMasks(cam_id=cam_id)
            continue

        try:
            image = frame_data.load_image(cam_id)
        except (KeyError, FileNotFoundError, IOError):
            # Image loading failed, skip this camera
            per_view[cam_id] = PerViewMasks(cam_id=cam_id)
            continue

        # 3. Run SAM2
        pv = segment_view(
            image=image,
            projections=projections,
            sam2=sam2,
            cam_id=cam_id,
            min_mask_score=sam2_config.min_mask_score,
        )
        per_view[cam_id] = pv
        seen_cluster_ids.update(pv.masks.keys())

    # 4. Cross-view matching
    views_with_masks = [pv for pv in per_view.values() if pv.masks]
    cross_view: Optional[CrossViewMatches] = None
    if len(views_with_masks) >= 2:
        cross_view = match_across_views(views_with_masks, iou_threshold=iou_threshold)

    # 5. Select best mask per cluster (with cross-view consistency enforcement)
    best_masks = select_best_masks(per_view, cross_view=cross_view)

    # 6. Skipped clusters
    skipped = sorted(all_cluster_ids - seen_cluster_ids)

    return SegmentationResult(
        per_view=per_view,
        cross_view=cross_view,
        best_masks=best_masks,
        skipped_clusters=skipped,
    )
