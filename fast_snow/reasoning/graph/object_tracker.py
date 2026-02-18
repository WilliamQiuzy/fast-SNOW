"""Cross-frame object association using semantic and geometric cues.

This module implements the temporal association described in SNOW paper Eq. 7:
    F_k = {S_{t-T}^k, ..., S_t^k}

Objects are associated across frames using:
1. Geometric similarity: centroid distance + shape token distance
2. Semantic similarity: patch token overlap (IoU)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from fast_snow.reasoning.tokens.step_encoding import STEPToken
from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass
class TrackerConfig:
    """Configuration for object tracker."""

    # Cost weights
    geometric_weight: float = 0.5
    semantic_weight: float = 0.5

    # Distance thresholds
    max_centroid_distance: float = 5.0  # meters
    max_association_cost: float = 2.0

    # Shape distance weight relative to centroid
    shape_distance_weight: float = 0.1

    # Maximum frames to keep unmatched tracks before deletion
    max_age: int = 5


@dataclass
class Track:
    """A tracked object across frames.

    Paper reference (Eq. 7):
        F_k = {S_{t-T}^k, ..., S_t^k}

    This track maintains the temporal sequence of STEP tokens for a single
    object across multiple frames, enabling track-level temporal tokens.

    Attributes:
        track_id: Unique identifier for this track.
        steps: Sequence of STEPTokens observed for this object.
        frame_indices: Corresponding frame indices for each step.
        age: Frames since last update (for track management).
    """

    track_id: int
    steps: List[STEPToken] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    age: int = 0  # frames since last update

    @property
    def last_step(self) -> Optional[STEPToken]:
        """Get the most recent STEP token."""
        return self.steps[-1] if self.steps else None

    @property
    def last_frame(self) -> Optional[int]:
        """Get the most recent frame index."""
        return self.frame_indices[-1] if self.frame_indices else None

    @property
    def t_start(self) -> Optional[int]:
        """Get the first frame where this object appeared.

        Returns:
            First frame index, or None if track is empty.

        Note:
            Used to update temporal tokens in Phase 6.
        """
        return min(self.frame_indices) if self.frame_indices else None

    @property
    def t_end(self) -> Optional[int]:
        """Get the last frame where this object was observed.

        Returns:
            Last frame index, or None if track is empty.

        Note:
            Used to update temporal tokens in Phase 6.
        """
        return max(self.frame_indices) if self.frame_indices else None

    @property
    def track_length(self) -> int:
        """Get the number of frames in this track."""
        return len(self.frame_indices)

    def update(self, step: STEPToken, frame_idx: int) -> None:
        """Add a new observation to this track.

        Args:
            step: STEP token for this object at the current frame.
            frame_idx: Current frame index.
        """
        self.steps.append(step)
        self.frame_indices.append(frame_idx)
        self.age = 0

    def increment_age(self) -> None:
        """Increment age when track is not matched in current frame.

        Tracks with age > max_age are removed to prevent stale tracks.
        """
        self.age += 1


@dataclass
class TrackerState:
    """State of the object tracker.

    Maintains both active tracks (for matching) and archived tracks (for 4DSG).
    This ensures all historical object trajectories are preserved for Phase 6,
    even if they were not visible in recent frames.
    """

    tracks: Dict[int, Track] = field(default_factory=dict)  # Active tracks
    archived_tracks: Dict[int, Track] = field(default_factory=dict)  # Dead tracks (age > max_age)
    next_track_id: int = 0

    def create_track(self, step: STEPToken, frame_idx: int) -> Track:
        """Create a new track with the given step."""
        track = Track(track_id=self.next_track_id)
        track.update(step, frame_idx)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def remove_track(self, track_id: int) -> None:
        """Remove a track from active tracks and archive it.

        Paper requirement (Phase 6): All tracks within temporal window T
        should be preserved for 4DSG construction, even if they became
        inactive (age > max_age).
        """
        if track_id in self.tracks:
            # Archive the track instead of deleting it
            self.archived_tracks[track_id] = self.tracks[track_id]
            del self.tracks[track_id]

    def get_active_tracks(self, max_age: int) -> List[Track]:
        """Get tracks that are still active (not too old)."""
        return [t for t in self.tracks.values() if t.age <= max_age]

    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks (active + archived).

        Paper requirement (Eq. 7): F_k = {S_{t-T}^k, ..., S_t^k}
        This includes all tracks within the temporal window T, regardless
        of whether they are currently active or have been archived.

        Returns:
            Dict mapping track_id to Track for all historical tracks.
        """
        all_tracks = {}
        all_tracks.update(self.tracks)  # Active tracks
        all_tracks.update(self.archived_tracks)  # Archived tracks
        return all_tracks


def _check_scipy_available() -> None:
    if linear_sum_assignment is None:
        raise ImportError(
            "scipy is required for object tracking. Install scipy to use this module."
        ) from _SCIPY_IMPORT_ERROR


def compute_centroid_distance(step1: STEPToken, step2: STEPToken) -> float:
    """Compute Euclidean distance between centroids."""
    c1 = np.array([step1.centroid.x, step1.centroid.y, step1.centroid.z])
    c2 = np.array([step2.centroid.x, step2.centroid.y, step2.centroid.z])
    return float(np.linalg.norm(c1 - c2))


def compute_shape_distance(step1: STEPToken, step2: STEPToken) -> float:
    """Compute distance between shape tokens.

    Shape token is (12,): [μ_x, σ_x, x_min, x_max, μ_y, σ_y, y_min, y_max, μ_z, σ_z, z_min, z_max]
    """
    s1 = step1.shape
    s2 = step2.shape

    vec1 = np.array([
        s1.x_mu, s1.x_sigma, s1.x_min, s1.x_max,
        s1.y_mu, s1.y_sigma, s1.y_min, s1.y_max,
        s1.z_mu, s1.z_sigma, s1.z_min, s1.z_max,
    ])
    vec2 = np.array([
        s2.x_mu, s2.x_sigma, s2.x_min, s2.x_max,
        s2.y_mu, s2.y_sigma, s2.y_min, s2.y_max,
        s2.z_mu, s2.z_sigma, s2.z_min, s2.z_max,
    ])

    return float(np.linalg.norm(vec1 - vec2))


def compute_patch_overlap(patches1: List[PatchToken], patches2: List[PatchToken]) -> float:
    """Compute IoU of patch token positions.

    Returns:
        IoU in [0, 1], higher means more similar.
    """
    if not patches1 or not patches2:
        return 0.0

    set1 = {(p.row, p.col) for p in patches1}
    set2 = {(p.row, p.col) for p in patches2}

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_association_cost(
    step_curr: STEPToken,
    step_prev: STEPToken,
    config: TrackerConfig,
) -> float:
    """Compute association cost between two STEP tokens.

    Lower cost means more likely to be the same object.

    Cost = geometric_weight * geometric_cost + semantic_weight * semantic_cost

    Where:
        geometric_cost = centroid_distance + shape_distance_weight * shape_distance
        semantic_cost = 1 - patch_overlap
    """
    # Geometric cost
    centroid_dist = compute_centroid_distance(step_curr, step_prev)
    shape_dist = compute_shape_distance(step_curr, step_prev)
    geometric_cost = centroid_dist + config.shape_distance_weight * shape_dist

    # Semantic cost (1 - similarity)
    patch_sim = compute_patch_overlap(step_curr.patch_tokens, step_prev.patch_tokens)
    semantic_cost = 1.0 - patch_sim

    # Combined cost
    total_cost = (
        config.geometric_weight * geometric_cost +
        config.semantic_weight * semantic_cost
    )

    return total_cost


def build_cost_matrix(
    current_steps: List[STEPToken],
    tracks: List[Track],
    config: TrackerConfig,
) -> np.ndarray:
    """Build cost matrix for Hungarian matching.

    Args:
        current_steps: STEP tokens detected in current frame
        tracks: Active tracks from previous frames
        config: Tracker configuration

    Returns:
        Cost matrix of shape (n_current, n_tracks)
    """
    n_curr = len(current_steps)
    n_tracks = len(tracks)

    cost_matrix = np.full((n_curr, n_tracks), fill_value=1e6, dtype=float)

    for i, step_curr in enumerate(current_steps):
        for j, track in enumerate(tracks):
            step_prev = track.last_step
            if step_prev is None:
                continue

            # Check if centroid is within maximum distance
            centroid_dist = compute_centroid_distance(step_curr, step_prev)
            if centroid_dist > config.max_centroid_distance:
                continue

            cost = compute_association_cost(step_curr, step_prev, config)
            cost_matrix[i, j] = cost

    return cost_matrix


def associate_detections_to_tracks(
    current_steps: Dict[int, STEPToken],
    state: TrackerState,
    frame_idx: int,
    config: TrackerConfig,
) -> Tuple[Dict[int, int], Set[int], Set[int]]:
    """Associate current frame detections to existing tracks.

    Args:
        current_steps: Dict mapping detection_id -> STEPToken
        state: Current tracker state
        frame_idx: Current frame index
        config: Tracker configuration

    Returns:
        Tuple of:
            - matches: Dict mapping detection_id -> track_id
            - unmatched_detections: Set of detection_ids with no match
            - unmatched_tracks: Set of track_ids with no match
    """
    _check_scipy_available()

    detection_ids = list(current_steps.keys())
    detection_list = [current_steps[d] for d in detection_ids]

    active_tracks = state.get_active_tracks(config.max_age)
    track_ids = [t.track_id for t in active_tracks]

    if not detection_list or not active_tracks:
        return {}, set(detection_ids), set(track_ids)

    # Build cost matrix
    cost_matrix = build_cost_matrix(detection_list, active_tracks, config)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter matches by cost threshold
    matches: Dict[int, int] = {}
    matched_det_indices: Set[int] = set()
    matched_track_indices: Set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < config.max_association_cost:
            det_id = detection_ids[r]
            trk_id = track_ids[c]
            matches[det_id] = trk_id
            matched_det_indices.add(r)
            matched_track_indices.add(c)

    # Find unmatched
    unmatched_detections = {
        detection_ids[i] for i in range(len(detection_ids))
        if i not in matched_det_indices
    }
    unmatched_tracks = {
        track_ids[i] for i in range(len(track_ids))
        if i not in matched_track_indices
    }

    return matches, unmatched_detections, unmatched_tracks


class ObjectTracker:
    """Multi-object tracker using STEP tokens.

    Implements cross-frame association using Hungarian matching
    with geometric and semantic similarity.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.state = TrackerState()

    def update(
        self,
        detections: Dict[int, STEPToken],
        frame_idx: int,
    ) -> Dict[int, Track]:
        """Update tracker with new detections.

        Args:
            detections: Dict mapping detection_id -> STEPToken
            frame_idx: Current frame index

        Returns:
            Dict mapping track_id -> Track for all active tracks
        """
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            detections, self.state, frame_idx, self.config
        )

        # Update matched tracks
        for det_id, track_id in matches.items():
            track = self.state.tracks[track_id]
            track.update(detections[det_id], frame_idx)

        # Create new tracks for unmatched detections
        for det_id in unmatched_dets:
            self.state.create_track(detections[det_id], frame_idx)

        # Increment age for unmatched tracks
        for track_id in unmatched_tracks:
            if track_id in self.state.tracks:
                self.state.tracks[track_id].increment_age()

        # Remove dead tracks
        dead_tracks = [
            tid for tid, t in self.state.tracks.items()
            if t.age > self.config.max_age
        ]
        for tid in dead_tracks:
            self.state.remove_track(tid)

        return dict(self.state.tracks)

    def get_tracks(self) -> Dict[int, Track]:
        """Get all current active tracks.

        DEPRECATED for 4DSG construction: Use get_all_tracks() instead to
        include archived tracks within the temporal window.

        Returns:
            Dict of active tracks only.
        """
        return dict(self.state.tracks)

    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks (active + archived) for 4DSG construction.

        Paper requirement (Phase 6, Eq. 7): The 4DSG should include all
        object tracks within the temporal window T, not just currently
        active tracks. This ensures objects that disappeared earlier are
        still represented in the scene graph.

        Returns:
            Dict mapping track_id to Track for all historical tracks.

        Note:
            Use this method for 4DSG construction instead of get_tracks().
        """
        return self.state.get_all_tracks()

    def reset(self) -> None:
        """Reset tracker state."""
        self.state = TrackerState()


def track_objects_across_frames(
    frame_steps: List[Dict[int, STEPToken]],
    config: Optional[TrackerConfig] = None,
) -> Dict[int, Track]:
    """Track objects across multiple frames.

    This is the main entry point for offline tracking.

    Args:
        frame_steps: List of per-frame STEP token dicts
        config: Tracker configuration

    Returns:
        Dict mapping track_id -> Track with full trajectories (active + archived).

    Note:
        This function returns ALL tracks (including archived ones) to ensure
        complete temporal coverage. Objects that disappeared early are still
        included in the returned trajectories.
    """
    tracker = ObjectTracker(config)

    for frame_idx, detections in enumerate(frame_steps):
        tracker.update(detections, frame_idx)

    # Return ALL tracks (active + archived) for complete trajectories
    # Paper (Eq. 7): Should include all tracks within temporal window
    return tracker.get_all_tracks()
