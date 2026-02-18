"""4D Scene Graph (4DSG) representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fast_snow.reasoning.graph.scene_graph import SceneGraph
from fast_snow.reasoning.graph.temporal_linking import TemporalWindow


@dataclass(frozen=True)
class FourDSceneGraph:
    spatial_graphs: List[SceneGraph]
    temporal_window: TemporalWindow
    ego_poses: Dict[int, List[float]]
    track_metadata: Optional[Dict[int, Dict[str, Any]]] = None  # SAGE: label/motion per track


def build_four_d_scene_graph(
    spatial_graphs: List[SceneGraph],
    temporal_window: TemporalWindow,
    ego_poses: Dict[int, List[float]],
    track_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> FourDSceneGraph:
    """Assemble the 4DSG for a window."""
    return FourDSceneGraph(
        spatial_graphs=spatial_graphs,
        temporal_window=temporal_window,
        ego_poses=ego_poses,
        track_metadata=track_metadata,
    )
