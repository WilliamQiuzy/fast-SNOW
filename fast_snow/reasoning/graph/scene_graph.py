"""Spatial scene graph for a single frame.

This module implements the scene graph component described in SNOW paper,
where nodes represent objects (STEP tokens) and edges represent spatial
relations between objects.

The spatial relations include:
- Distance (metric distance in meters)
- Direction (unit vector from source to destination)
- Semantic relation (left/right, front/back, above/below)
- Ego-relative position (for egocentric reasoning)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fast_snow.reasoning.tokens.step_encoding import STEPToken


class HorizontalRelation(Enum):
    """Horizontal spatial relation."""
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"


class VerticalRelation(Enum):
    """Vertical spatial relation."""
    ABOVE = "above"
    BELOW = "below"
    LEVEL = "level"


class DistanceCategory(Enum):
    """Distance category for qualitative reasoning."""
    TOUCHING = "touching"  # < 0.5m
    NEAR = "near"          # 0.5-3m
    MEDIUM = "medium"      # 3-8m
    FAR = "far"            # 8-15m
    VERY_FAR = "very_far"  # > 15m


@dataclass(frozen=True)
class SpatialRelation:
    """Semantic spatial relation between two objects.

    Attributes:
        horizontal: Horizontal relation (left/right/front/back).
        vertical: Vertical relation (above/below/level).
        distance_category: Qualitative distance.
        is_adjacent: Whether objects are touching/overlapping.
    """
    horizontal: HorizontalRelation
    vertical: VerticalRelation
    distance_category: DistanceCategory
    is_adjacent: bool = False

    def __str__(self) -> str:
        """Return human-readable relation string."""
        if self.is_adjacent:
            return f"adjacent_{self.horizontal.value}_{self.vertical.value}"
        return f"{self.horizontal.value}_{self.vertical.value}_{self.distance_category.value}"

    @classmethod
    def from_direction_distance(
        cls,
        direction: np.ndarray,
        distance: float,
        angle_threshold: float = 0.3827,  # cos(67.5°) - for 8-way directions
    ) -> "SpatialRelation":
        """Compute spatial relation from direction vector and distance.

        Args:
            direction: Unit vector from source to destination (3,).
            distance: Distance in meters.
            angle_threshold: Threshold for diagonal detection.

        Returns:
            SpatialRelation instance.
        """
        dx, dy, dz = direction

        # Horizontal relation with 8-way direction
        if abs(dx) < angle_threshold and abs(dy) < angle_threshold:
            # Very close, use dominant direction
            horizontal = HorizontalRelation.FRONT if dy >= 0 else HorizontalRelation.BACK
        elif abs(dx) > angle_threshold and abs(dy) > angle_threshold:
            # Diagonal
            if dx > 0:
                horizontal = HorizontalRelation.FRONT_RIGHT if dy > 0 else HorizontalRelation.BACK_RIGHT
            else:
                horizontal = HorizontalRelation.FRONT_LEFT if dy > 0 else HorizontalRelation.BACK_LEFT
        elif abs(dx) > abs(dy):
            horizontal = HorizontalRelation.RIGHT if dx > 0 else HorizontalRelation.LEFT
        else:
            horizontal = HorizontalRelation.FRONT if dy > 0 else HorizontalRelation.BACK

        # Vertical relation
        if dz > 0.3:
            vertical = VerticalRelation.ABOVE
        elif dz < -0.3:
            vertical = VerticalRelation.BELOW
        else:
            vertical = VerticalRelation.LEVEL

        # Distance category
        if distance < 0.5:
            dist_cat = DistanceCategory.TOUCHING
            is_adjacent = True
        elif distance < 3.0:
            dist_cat = DistanceCategory.NEAR
            is_adjacent = False
        elif distance < 8.0:
            dist_cat = DistanceCategory.MEDIUM
            is_adjacent = False
        elif distance < 15.0:
            dist_cat = DistanceCategory.FAR
            is_adjacent = False
        else:
            dist_cat = DistanceCategory.VERY_FAR
            is_adjacent = False

        return cls(
            horizontal=horizontal,
            vertical=vertical,
            distance_category=dist_cat,
            is_adjacent=is_adjacent,
        )


@dataclass(frozen=True)
class SceneNode:
    """Node in the scene graph representing an object.

    Attributes:
        node_id: Unique identifier for this node.
        step: STEP token encoding for this object.
        position: 3D centroid position (x, y, z).
        extent: Bounding box extent (dx, dy, dz).
        ego_distance: Distance from ego/sensor origin.
        ego_angle: Angle from ego forward direction (radians).
    """
    node_id: int
    step: STEPToken
    position: np.ndarray  # (3,)
    extent: Optional[np.ndarray] = None  # (3,) - dx, dy, dz
    ego_distance: Optional[float] = None
    ego_angle: Optional[float] = None  # radians, 0 = forward

    # SAGE semantic fields (backward compatible, all Optional)
    label: Optional[str] = None              # e.g., "car", "person"
    label_confidence: Optional[float] = None # detection confidence [0, 1]
    motion_state: Optional[str] = None       # "moving" | "stationary"
    velocity: Optional[float] = None         # speed in m/s


@dataclass(frozen=True)
class SceneEdge:
    """Edge in the scene graph representing a spatial relation.

    Attributes:
        src: Source node ID.
        dst: Destination node ID.
        distance: Euclidean distance in meters.
        direction: Unit vector from source to destination.
        relation: Semantic spatial relation.
    """
    src: int
    dst: int
    distance: float
    direction: np.ndarray  # unit vector from src to dst
    relation: Optional[SpatialRelation] = None


@dataclass(frozen=True)
class SceneGraph:
    """Spatial scene graph for a single frame.

    Attributes:
        nodes: Dict mapping node_id to SceneNode.
        edges: List of SceneEdge representing spatial relations.
        frame_idx: Frame index this graph corresponds to.
        ego_pose: Optional ego pose (4x4 transform matrix).
    """
    nodes: Dict[int, SceneNode]
    edges: List[SceneEdge]
    frame_idx: int = 0
    ego_pose: Optional[np.ndarray] = None  # (4, 4)

    def get_node(self, node_id: int) -> Optional[SceneNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, SceneEdge]]:
        """Get all neighbors of a node with their edges."""
        neighbors = []
        for edge in self.edges:
            if edge.src == node_id:
                neighbors.append((edge.dst, edge))
            elif edge.dst == node_id:
                # Reverse the edge direction
                reverse_dir = -edge.direction if edge.direction is not None else None
                reverse_relation = None
                if edge.relation is not None:
                    # Flip the horizontal relation
                    flipped_horiz = _flip_horizontal(edge.relation.horizontal)
                    reverse_relation = SpatialRelation(
                        horizontal=flipped_horiz,
                        vertical=edge.relation.vertical,
                        distance_category=edge.relation.distance_category,
                        is_adjacent=edge.relation.is_adjacent,
                    )
                reverse_edge = SceneEdge(
                    src=edge.dst,
                    dst=edge.src,
                    distance=edge.distance,
                    direction=reverse_dir,
                    relation=reverse_relation,
                )
                neighbors.append((edge.src, reverse_edge))
        return neighbors

    def get_edge(self, src: int, dst: int) -> Optional[SceneEdge]:
        """Get edge between two nodes."""
        for edge in self.edges:
            if (edge.src == src and edge.dst == dst) or (edge.src == dst and edge.dst == src):
                return edge
        return None

    def get_objects_in_direction(
        self,
        reference_node_id: int,
        direction: HorizontalRelation,
        max_distance: Optional[float] = None,
    ) -> List[int]:
        """Get all objects in a specific direction from a reference object."""
        result = []
        for neighbor_id, edge in self.get_neighbors(reference_node_id):
            if edge.relation is None:
                continue
            if edge.relation.horizontal == direction:
                if max_distance is None or edge.distance <= max_distance:
                    result.append(neighbor_id)
        return result


def _flip_horizontal(h: HorizontalRelation) -> HorizontalRelation:
    """Flip horizontal relation to opposite direction."""
    flip_map = {
        HorizontalRelation.LEFT: HorizontalRelation.RIGHT,
        HorizontalRelation.RIGHT: HorizontalRelation.LEFT,
        HorizontalRelation.FRONT: HorizontalRelation.BACK,
        HorizontalRelation.BACK: HorizontalRelation.FRONT,
        HorizontalRelation.FRONT_LEFT: HorizontalRelation.BACK_RIGHT,
        HorizontalRelation.FRONT_RIGHT: HorizontalRelation.BACK_LEFT,
        HorizontalRelation.BACK_LEFT: HorizontalRelation.FRONT_RIGHT,
        HorizontalRelation.BACK_RIGHT: HorizontalRelation.FRONT_LEFT,
    }
    return flip_map.get(h, h)


@dataclass
class SceneGraphConfig:
    """Configuration for scene graph construction."""
    # Maximum distance to include edges (meters)
    max_edge_distance: float = 50.0

    # Whether to include semantic relations
    include_relations: bool = True

    # Whether to compute ego-relative positions
    include_ego_relative: bool = True

    # Minimum distance for valid edges (filter noise)
    min_edge_distance: float = 0.1


def build_scene_graph(
    steps: Dict[int, STEPToken],
    frame_idx: int = 0,
    ego_pose: Optional[np.ndarray] = None,
    config: Optional[SceneGraphConfig] = None,
    node_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> SceneGraph:
    """Build a spatial scene graph from STEP tokens.

    Args:
        steps: Dict mapping node_id to STEPToken.
        frame_idx: Frame index for this graph.
        ego_pose: Optional 4x4 ego pose matrix.
        config: Graph construction configuration.
        node_metadata: Optional per-node SAGE metadata (label, motion, etc.).

    Returns:
        SceneGraph with nodes and edges.
    """
    if config is None:
        config = SceneGraphConfig()

    nodes: Dict[int, SceneNode] = {}

    for node_id, step in steps.items():
        pos = np.array([step.centroid.x, step.centroid.y, step.centroid.z], dtype=float)

        # Compute extent from shape
        extent = np.array([
            step.shape.x_max - step.shape.x_min,
            step.shape.y_max - step.shape.y_min,
            step.shape.z_max - step.shape.z_min,
        ], dtype=float)

        # Compute ego-relative position if requested
        ego_distance = None
        ego_angle = None
        if config.include_ego_relative:
            if ego_pose is not None:
                # Transform position to ego frame
                ego_inv = np.linalg.inv(ego_pose)
                pos_homo = np.append(pos, 1.0)
                pos_ego = (ego_inv @ pos_homo)[:3]
                ego_distance = float(np.linalg.norm(pos_ego[:2]))  # 2D distance
                ego_angle = float(np.arctan2(pos_ego[0], pos_ego[1]))  # angle from forward
            else:
                # Use world coordinates directly (assume ego at origin looking +Y)
                ego_distance = float(np.linalg.norm(pos[:2]))
                ego_angle = float(np.arctan2(pos[0], pos[1]))

        # Extract SAGE metadata if available
        meta = {}
        if node_metadata is not None and node_id in node_metadata:
            meta = node_metadata[node_id]

        nodes[node_id] = SceneNode(
            node_id=node_id,
            step=step,
            position=pos,
            extent=extent,
            ego_distance=ego_distance,
            ego_angle=ego_angle,
            label=meta.get("label"),
            label_confidence=meta.get("label_confidence"),
            motion_state=meta.get("motion_state"),
            velocity=meta.get("velocity"),
        )

    # Build edges
    edges: List[SceneEdge] = []
    ids = sorted(nodes.keys())

    for i, src_id in enumerate(ids):
        for dst_id in ids[i + 1:]:
            src = nodes[src_id]
            dst = nodes[dst_id]

            delta = dst.position - src.position
            dist = float(np.linalg.norm(delta))

            # Filter by distance
            if dist < config.min_edge_distance or dist > config.max_edge_distance:
                continue

            if dist == 0:
                direction = np.zeros(3, dtype=float)
                relation = None
            else:
                direction = delta / dist
                relation = None
                if config.include_relations:
                    relation = SpatialRelation.from_direction_distance(direction, dist)

            edges.append(SceneEdge(
                src=src_id,
                dst=dst_id,
                distance=dist,
                direction=direction,
                relation=relation,
            ))

    return SceneGraph(
        nodes=nodes,
        edges=edges,
        frame_idx=frame_idx,
        ego_pose=ego_pose,
    )
