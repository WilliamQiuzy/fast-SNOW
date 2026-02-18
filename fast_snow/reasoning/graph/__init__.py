"""Graph module for SNOW 4D Scene Graph construction."""

from fast_snow.reasoning.graph.scene_graph import SceneGraph, SceneNode, SceneEdge, build_scene_graph
from fast_snow.reasoning.graph.temporal_linking import (
    TemporalTrack,
    TemporalWindow,
    build_temporal_window_from_tracker,
    link_temporal_tokens,  # Deprecated: use build_temporal_window_from_tracker
)
from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph, build_four_d_scene_graph
from fast_snow.reasoning.graph.object_tracker import (
    ObjectTracker,
    TrackerConfig,
    Track,
    track_objects_across_frames,
)

__all__ = [
    # Scene Graph
    "SceneGraph",
    "SceneNode",
    "SceneEdge",
    "build_scene_graph",
    # Temporal Linking
    "TemporalTrack",
    "TemporalWindow",
    "build_temporal_window_from_tracker",  # NEW: Phase 6 entry point
    "link_temporal_tokens",  # DEPRECATED: use build_temporal_window_from_tracker
    # 4D Scene Graph
    "FourDSceneGraph",
    "build_four_d_scene_graph",
    # Object Tracker
    "ObjectTracker",
    "TrackerConfig",
    "Track",
    "track_objects_across_frames",
]
