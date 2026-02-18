"""Temporal linking of STEP tokens across frames.

Paper reference (Section 3.3, Eq. 7):
    F_k = {S_{t-T}^k, ..., S_t^k}

This module constructs temporal tracks from cross-frame object associations,
where each track maintains a sequence of STEP tokens for a single object
across multiple frames within the temporal window T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from fast_snow.reasoning.tokens.step_encoding import STEPToken, update_temporal_token


@dataclass(frozen=True)
class TemporalTrack:
    """A temporal track representing an object's STEP tokens across frames.

    Paper reference (Eq. 7): F_k = {S_{t-T}^k, ..., S_t^k}

    Attributes:
        track_id: Unique identifier for this track.
        steps: Sequence of STEPTokens with updated temporal tokens (t_start, t_end).
        frame_indices: Frame indices for each step (same length as steps).
                       CRITICAL for Phase 7 serialization - preserves real frame numbers.
    """
    track_id: int
    steps: List[STEPToken]
    frame_indices: List[int] = None  # Added for Phase 7 serialization


@dataclass(frozen=True)
class TemporalWindow:
    """Temporal window containing all object tracks.

    Paper reference: Temporal window T=10 frames at ~1 Hz.

    Attributes:
        tracks: Dict mapping track_id to TemporalTrack.
    """
    tracks: Dict[int, TemporalTrack]


def build_temporal_window_from_tracker(
    tracks: Dict[int, "Track"],  # From graph.object_tracker.Track
) -> TemporalWindow:
    """Build TemporalWindow from ObjectTracker results.

    This is the main entry point for Phase 6. It takes tracks from
    ObjectTracker (which uses Hungarian matching for cross-frame association)
    and constructs TemporalTracks with properly updated temporal tokens.

    Paper workflow (Phase 6):
    1. ObjectTracker associates STEP tokens across frames → Track objects
    2. For each track, update all STEP temporal tokens to [t_start, t_end]
    3. Construct TemporalWindow with updated tracks

    Args:
        tracks: Dict mapping track_id to Track (from ObjectTracker).

    Returns:
        TemporalWindow with updated temporal tokens.

    Note:
        This function ensures all STEP tokens in a track have consistent
        temporal tokens (t_start, t_end) reflecting the track's full span.
    """
    temporal_tracks: Dict[int, TemporalTrack] = {}

    for track_id, track in tracks.items():
        if not track.steps or not track.frame_indices:
            # Skip empty tracks
            continue

        # Get track-level temporal span
        # Paper (Eq. 7): Track spans from first to last appearance
        t_start = min(track.frame_indices)
        t_end = max(track.frame_indices)

        # Update all STEP tokens in this track to have consistent temporal tokens
        # Paper requirement: All STEP tokens in a track should reflect the
        # track's full temporal span, not just their individual frame
        updated_steps = [
            update_temporal_token(step, t_start=t_start, t_end=t_end)
            for step in track.steps
        ]

        # CRITICAL (Phase 7): Preserve frame_indices for serialization
        # Phase 6 updates temporal tokens to track-level [t_start, t_end],
        # but we need original frame numbers for JSON serialization
        temporal_tracks[track_id] = TemporalTrack(
            track_id=track_id,
            steps=updated_steps,
            frame_indices=track.frame_indices.copy(),  # Preserve real frame numbers
        )

    return TemporalWindow(tracks=temporal_tracks)


# Deprecated: Kept for backward compatibility only
def link_temporal_tokens(
    frame_steps: List[Dict[int, STEPToken]],
) -> TemporalWindow:
    """DEPRECATED: Use build_temporal_window_from_tracker() instead.

    This function assumes node_ids are consistent across frames, which is
    not guaranteed. Use ObjectTracker for proper cross-frame association.
    """
    tracks: Dict[int, List[STEPToken]] = {}
    for steps in frame_steps:
        for node_id, step in steps.items():
            tracks.setdefault(node_id, []).append(step)

    track_objs = {
        track_id: TemporalTrack(track_id=track_id, steps=steps)
        for track_id, steps in tracks.items()
    }
    return TemporalWindow(tracks=track_objs)
