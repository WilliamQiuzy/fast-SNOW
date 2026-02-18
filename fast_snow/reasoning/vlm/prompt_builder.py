"""Prompt construction for VLM inference.

This module implements the 4DSG serialization for VLM input as described
in SNOW paper Section 3.4, Eq. 9:
    ŷ = VLM(q | M_t)

The 4DSG is serialized into a structured text format that includes:
- Ego agent information (poses over time)
- Object tracks with full temporal history
- Spatial relations between objects
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph
from fast_snow.reasoning.graph.scene_graph import SceneGraph, SceneEdge
from fast_snow.reasoning.graph.temporal_linking import TemporalTrack


class SerializationFormat(Enum):
    """Output format for 4DSG serialization."""
    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt building."""

    # Maximum number of tracks to include
    max_tracks: int = 50

    # Maximum frames to include per track
    max_frames_per_track: int = 10

    # Include ego pose information
    include_ego: bool = True

    # Include spatial relations
    include_relations: bool = True

    # Maximum relations per frame
    max_relations_per_frame: int = 20

    # Distance threshold for including relations (meters)
    relation_distance_threshold: float = 15.0

    # Output format
    format: SerializationFormat = SerializationFormat.TEXT

    # Decimal precision for coordinates
    precision: int = 2


@dataclass(frozen=True)
class Phase7SerializationConfig:
    """Phase 7 strict serialization config (paper reproduction).

    These parameters are FIXED for Phase 7 to ensure reproducibility.
    All values match the paper specifications and docs/roadmap/SEMANTIC_SG_AND_SAM2_TEMPORAL_PLAN.md.
    """
    precision: int = 2
    max_tracks: int = 50
    max_frames_per_track: int = 10
    max_relations_per_frame: int = 20
    relation_distance_threshold: float = 15.0


def _compute_spatial_relation(direction: np.ndarray, distance: float) -> str:
    """Compute discrete spatial relation from direction vector.

    Args:
        direction: Unit vector from source to destination (3,)
        distance: Distance in meters

    Returns:
        Spatial relation string like "right_level", "front_above", etc.
    """
    dx, dy, dz = direction

    # Horizontal relation
    if abs(dx) > abs(dy):
        horizontal = "right" if dx > 0 else "left"
    else:
        horizontal = "front" if dy > 0 else "back"

    # Vertical relation
    if dz > 0.5:
        vertical = "above"
    elif dz < -0.5:
        vertical = "below"
    else:
        vertical = "level"

    # Distance qualifier
    if distance < 3.0:
        dist_qual = "near"
    elif distance < 8.0:
        dist_qual = "medium"
    else:
        dist_qual = "far"

    return f"{horizontal}_{vertical}_{dist_qual}"


def serialize_ego_poses(
    ego_poses: Dict[int, List[float]],
    precision: int = 2,
) -> List[str]:
    """Serialize ego poses to text lines.

    Args:
        ego_poses: Dict mapping frame_idx -> [x, y, z, qw, qx, qy, qz] or [x, y, z]
        precision: Decimal precision

    Returns:
        List of formatted strings
    """
    lines = []
    for frame_idx in sorted(ego_poses.keys()):
        pose = ego_poses[frame_idx]
        if len(pose) >= 3:
            x, y, z = pose[:3]
            lines.append(
                f"  Frame {frame_idx}: position=({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f})"
            )
    return lines


def serialize_track(
    track: TemporalTrack,
    max_frames: int,
    precision: int = 2,
) -> List[str]:
    """Serialize a single track to text lines.

    Args:
        track: Track to serialize
        max_frames: Maximum frames to include
        precision: Decimal precision

    Returns:
        List of formatted strings
    """
    lines = []
    lines.append(f"Object ID {track.track_id}:")

    steps = track.steps[-max_frames:]  # Take most recent frames

    for i, step in enumerate(steps):
        c = step.centroid
        s = step.shape
        t = step.temporal

        # Position
        pos_str = f"({c.x:.{precision}f}, {c.y:.{precision}f}, {c.z:.{precision}f})"

        # Size (extent)
        extent_x = s.x_max - s.x_min
        extent_y = s.y_max - s.y_min
        extent_z = s.z_max - s.z_min
        size_str = f"({extent_x:.{precision}f}, {extent_y:.{precision}f}, {extent_z:.{precision}f})"

        # Temporal info
        time_str = f"visible=[{t.t_start}, {t.t_end}]"

        lines.append(f"  Step {i}: pos={pos_str} size={size_str} {time_str}")

    return lines


def serialize_spatial_relations(
    graph: SceneGraph,
    config: PromptConfig,
) -> List[str]:
    """Serialize spatial relations for a single frame.

    Args:
        graph: Scene graph for the frame
        config: Prompt configuration

    Returns:
        List of formatted strings
    """
    lines = []

    # Filter and sort edges by distance
    edges = [e for e in graph.edges if e.distance < config.relation_distance_threshold]
    edges = sorted(edges, key=lambda e: e.distance)[:config.max_relations_per_frame]

    for edge in edges:
        relation = _compute_spatial_relation(edge.direction, edge.distance)
        lines.append(
            f"  Object {edge.src} is {relation} Object {edge.dst} "
            f"(distance: {edge.distance:.{config.precision}f}m)"
        )

    return lines


def serialize_4dsg_text(
    graph: FourDSceneGraph,
    config: PromptConfig,
) -> str:
    """Serialize 4DSG to structured text format.

    Args:
        graph: 4D Scene Graph
        config: Prompt configuration

    Returns:
        Formatted text representation
    """
    sections = []
    precision = config.precision

    # Section 1: Ego Information
    if config.include_ego and graph.ego_poses:
        sections.append("=== Ego Agent Information ===")
        sections.extend(serialize_ego_poses(graph.ego_poses, precision))
        sections.append("")

    # Section 2: Object Tracks
    sections.append("=== Objects in Scene ===")
    tracks = list(graph.temporal_window.tracks.values())[:config.max_tracks]

    if tracks:
        for track in tracks:
            sections.extend(serialize_track(track, config.max_frames_per_track, precision))
            sections.append("")
    else:
        sections.append("  No objects detected")
        sections.append("")

    # Section 3: Spatial Relations
    if config.include_relations and graph.spatial_graphs:
        sections.append("=== Spatial Relations ===")

        # Use most recent frame for relations
        recent_graph = graph.spatial_graphs[-1]
        relations = serialize_spatial_relations(recent_graph, config)

        if relations:
            sections.append(f"Frame {len(graph.spatial_graphs) - 1}:")
            sections.extend(relations)
        else:
            sections.append("  No spatial relations")

    return "\n".join(sections)


def serialize_4dsg_json(
    graph: FourDSceneGraph,
    config: PromptConfig,
) -> Dict[str, Any]:
    """Serialize 4DSG to JSON format.

    Args:
        graph: 4D Scene Graph
        config: Prompt configuration

    Returns:
        JSON-serializable dictionary
    """
    precision = config.precision

    result: Dict[str, Any] = {
        "num_frames": len(graph.spatial_graphs),
        "num_tracks": len(graph.temporal_window.tracks),
    }

    # Ego poses
    if config.include_ego and graph.ego_poses:
        result["ego_poses"] = {
            str(k): [round(v, precision) for v in pose]
            for k, pose in graph.ego_poses.items()
        }

    # Object tracks
    tracks_data = {}
    for track_id, track in list(graph.temporal_window.tracks.items())[:config.max_tracks]:
        steps_data = []
        for step in track.steps[-config.max_frames_per_track:]:
            c = step.centroid
            s = step.shape
            t = step.temporal

            steps_data.append({
                "centroid": [round(c.x, precision), round(c.y, precision), round(c.z, precision)],
                "extent": [
                    round(s.x_max - s.x_min, precision),
                    round(s.y_max - s.y_min, precision),
                    round(s.z_max - s.z_min, precision),
                ],
                "temporal": [t.t_start, t.t_end],
                "num_patches": len(step.patch_tokens),
            })

        tracks_data[str(track_id)] = steps_data

    result["objects"] = tracks_data

    # Spatial relations
    if config.include_relations and graph.spatial_graphs:
        recent_graph = graph.spatial_graphs[-1]
        edges = [e for e in recent_graph.edges if e.distance < config.relation_distance_threshold]
        edges = sorted(edges, key=lambda e: e.distance)[:config.max_relations_per_frame]

        relations = []
        for edge in edges:
            relations.append({
                "source": edge.src,
                "target": edge.dst,
                "relation": _compute_spatial_relation(edge.direction, edge.distance),
                "distance": round(edge.distance, precision),
            })

        result["spatial_relations"] = relations

    return result


def serialize_4dsg_json_strict(
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> Dict[str, Any]:
    """Serialize 4DSG to JSON format with strict determinism (Phase 7).

    This is the STRICT version for Phase 7 paper reproduction. It ensures:
    1. Deterministic output: same input always produces identical JSON
    2. Fixed field order: metadata → ego_agent → objects → spatial_relations
    3. Explicit sorting: all lists sorted by specified keys
    4. Unified precision: all floats rounded to 2 decimal places

    See docs/phases/PHASE7_SERIALIZATION.md for full specification.

    Args:
        graph: 4D Scene Graph
        config: Phase 7 serialization config (uses defaults if None)

    Returns:
        JSON-serializable dictionary with strict field ordering
    """
    if config is None:
        config = Phase7SerializationConfig()

    precision = config.precision

    # === 1. Metadata (fixed order) ===
    result: Dict[str, Any] = {}
    result["metadata"] = {
        "num_frames": len(graph.spatial_graphs),
        "num_objects": len(graph.temporal_window.tracks),
        "temporal_window": 10,  # FIXED: Paper specification T=10 frames
    }

    # === 2. Ego Agent Trajectory (sorted by frame) ===
    ego_trajectory = []
    if graph.ego_poses:
        # CRITICAL: Sort by frame index for determinism
        for frame_idx in sorted(graph.ego_poses.keys()):
            pose = graph.ego_poses[frame_idx]
            entry = {"frame": frame_idx}

            if len(pose) >= 3:
                # Position (always present)
                entry["position"] = [
                    round(pose[0], precision),
                    round(pose[1], precision),
                    round(pose[2], precision),
                ]

            if len(pose) >= 7:
                # Rotation quaternion (optional: qx, qy, qz, qw)
                entry["rotation"] = [
                    round(pose[3], precision),
                    round(pose[4], precision),
                    round(pose[5], precision),
                    round(pose[6], precision),
                ]

            ego_trajectory.append(entry)

    result["ego_agent"] = {"trajectory": ego_trajectory}

    # === 3. Objects (sorted by object_id, steps sorted by frame) ===
    objects_list = []

    # CRITICAL: Sort tracks by track_id for determinism
    sorted_tracks = sorted(
        graph.temporal_window.tracks.items(),
        key=lambda x: x[0]  # Sort by track_id
    )[:config.max_tracks]  # Apply limit AFTER sorting

    for track_id, temporal_track in sorted_tracks:
        # Build track steps
        steps_data = []

        # CRITICAL (Phase 7 fix): Use frame_indices from TemporalTrack
        # Phase 6 updates temporal tokens to track-level [t_start, t_end],
        # so we can't use t.t_end as frame - all steps would have same value
        # Instead, use the preserved frame_indices list
        if temporal_track.frame_indices is None or len(temporal_track.frame_indices) != len(temporal_track.steps):
            # Fallback: use temporal span if frame_indices not available
            # This should not happen in Phase 7, but handle gracefully
            frame_indices = list(range(len(temporal_track.steps)))
        else:
            frame_indices = temporal_track.frame_indices

        # Take most recent frames (up to max_frames_per_track)
        num_steps = len(temporal_track.steps)
        start_idx = max(0, num_steps - config.max_frames_per_track)
        recent_steps = temporal_track.steps[start_idx:]
        recent_frames = frame_indices[start_idx:]

        # Build steps with real frame numbers
        for step, frame_idx in zip(recent_steps, recent_frames):
            c = step.centroid
            s = step.shape
            t = step.temporal

            step_entry = {
                "frame": frame_idx,  # FIXED: Use real frame index from tracker
                "centroid": [
                    round(c.x, precision),
                    round(c.y, precision),
                    round(c.z, precision),
                ],
                "extent": [
                    round(s.x_max - s.x_min, precision),
                    round(s.y_max - s.y_min, precision),
                    round(s.z_max - s.z_min, precision),
                ],
                "temporal_span": [t.t_start, t.t_end],
                "num_patches": len(step.patch_tokens),
            }
            steps_data.append(step_entry)

        # CRITICAL: Sort steps by frame for determinism
        steps_data.sort(key=lambda s: s["frame"])

        objects_list.append({
            "object_id": track_id,
            "track": steps_data,
        })

    result["objects"] = objects_list

    # === 4. Spatial Relations (sorted by distance) ===
    relations_list = []

    if graph.spatial_graphs:
        # Use most recent frame for spatial relations
        recent_graph = graph.spatial_graphs[-1]

        # Filter by distance threshold
        edges = [
            e for e in recent_graph.edges
            if e.distance < config.relation_distance_threshold
        ]

        # CRITICAL: Sort by distance for determinism
        edges = sorted(edges, key=lambda e: e.distance)[:config.max_relations_per_frame]

        for edge in edges:
            relations_list.append({
                "source_id": edge.src,
                "target_id": edge.dst,
                "relation": _compute_spatial_relation(edge.direction, edge.distance),
                "distance": round(edge.distance, precision),
            })

    result["spatial_relations"] = relations_list

    return result


def serialize_4dsg(
    graph: FourDSceneGraph,
    config: Optional[PromptConfig] = None,
) -> str:
    """Serialize 4DSG to the configured format.

    Args:
        graph: 4D Scene Graph
        config: Prompt configuration

    Returns:
        Serialized string (text or JSON)
    """
    if config is None:
        config = PromptConfig()

    if config.format == SerializationFormat.JSON:
        return json.dumps(serialize_4dsg_json(graph, config), indent=2)
    else:
        return serialize_4dsg_text(graph, config)


def serialize_4dsg_strict(
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> str:
    """Serialize 4DSG using strict Phase 7 JSON format.

    This is the recommended entry point for Phase 7 VLM inference.
    Uses serialize_4dsg_json_strict() with deterministic formatting.

    Args:
        graph: 4D Scene Graph
        config: Phase 7 config (uses defaults if None)

    Returns:
        JSON string with strict formatting (2-space indent, sorted keys)
    """
    if config is None:
        config = Phase7SerializationConfig()

    json_dict = serialize_4dsg_json_strict(graph, config)

    # Use sort_keys=True for additional determinism in JSON string output
    return json.dumps(json_dict, indent=2, sort_keys=False)  # Keep our field order


def build_messages(
    query: str,
    graph: FourDSceneGraph,
    cfg: Optional[PromptConfig] = None,
) -> List[Dict[str, object]]:
    """Build chat messages for VLM inference.

    This creates the input format for Gemma3 and similar models.

    Args:
        query: User question
        graph: 4D Scene Graph
        cfg: Prompt configuration

    Returns:
        List of message dicts with 'role' and 'content'
    """
    if cfg is None:
        cfg = PromptConfig()

    # Serialize 4DSG
    context = serialize_4dsg(graph, cfg)

    # System message
    system = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": (
                "You are a spatial reasoning assistant analyzing a 4D scene. "
                "The scene contains objects tracked over time with their 3D positions, "
                "sizes, and spatial relationships. Answer questions precisely based on "
                "the provided scene information."
            )
        }],
    }

    # User message with context and query
    num_frames = len(graph.spatial_graphs)
    num_tracks = len(graph.temporal_window.tracks)

    user_text = (
        f"Scene Summary:\n"
        f"- Total frames: {num_frames}\n"
        f"- Total tracked objects: {num_tracks}\n\n"
        f"Scene Details:\n{context}\n\n"
        f"Question: {query}"
    )

    user = {
        "role": "user",
        "content": [{"type": "text", "text": user_text}]
    }

    return [system, user]


def build_simple_prompt(
    query: str,
    graph: FourDSceneGraph,
    cfg: Optional[PromptConfig] = None,
) -> str:
    """Build a simple text prompt without chat format.

    Useful for models that don't use chat format.

    Args:
        query: User question
        graph: 4D Scene Graph
        cfg: Prompt configuration

    Returns:
        Complete prompt string
    """
    if cfg is None:
        cfg = PromptConfig()

    context = serialize_4dsg(graph, cfg)

    prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

{context}

Based on the scene information above, please answer the following question:
{query}

Answer:"""

    return prompt


def build_messages_phase7(
    query: str,
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> List[Dict[str, Any]]:
    """Build chat messages for VLM inference (Phase 7 strict version).

    This is the RECOMMENDED interface for Phase 7 VLM inference.
    Uses serialize_4dsg_json_strict() to ensure deterministic output.

    Args:
        query: User question
        graph: 4D Scene Graph
        config: Phase 7 serialization config

    Returns:
        List of message dicts with 'role' and 'content'
    """
    if config is None:
        config = Phase7SerializationConfig()

    # Serialize 4DSG using strict format
    context_json = serialize_4dsg_json_strict(graph, config)

    # System message
    system = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": (
                "You are a spatial reasoning assistant analyzing a 4D scene. "
                "The scene contains objects tracked over time with their 3D positions, "
                "sizes, and spatial relationships. Answer questions precisely based on "
                "the provided scene information."
            )
        }],
    }

    # User message with JSON context and query
    # Format: Structured JSON data + question
    user_text = (
        f"Scene Data (JSON):\n"
        f"{json.dumps(context_json, indent=2)}\n\n"
        f"Question: {query}\n\n"
        f"Answer with just the letter choice (A, B, C, or D):"
    )

    user = {
        "role": "user",
        "content": [{"type": "text", "text": user_text}]
    }

    return [system, user]


def build_prompt_phase7(
    query: str,
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> str:
    """Build a simple text prompt for Phase 7 (strict JSON format).

    This is for models that don't use chat format.

    Args:
        query: User question
        graph: 4D Scene Graph
        config: Phase 7 serialization config

    Returns:
        Complete prompt string
    """
    if config is None:
        config = Phase7SerializationConfig()

    # Serialize 4DSG using strict JSON format
    context_json = serialize_4dsg_json_strict(graph, config)

    prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

Scene Data (JSON):
{json.dumps(context_json, indent=2)}

Based on the scene information above, please answer the following question:
{query}

Answer with just the letter choice (A, B, C, or D):"""

    return prompt


# ---------------------------------------------------------------------------
# SAGE serialization functions
# ---------------------------------------------------------------------------


def serialize_4dsg_sage(
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> Dict[str, Any]:
    """Serialize 4DSG to JSON format enriched with SAGE semantic metadata.

    Extends the Phase 7 strict serializer with per-object semantic labels,
    motion states, velocities, and label confidences drawn from
    ``graph.track_metadata``.

    Determinism guarantees are identical to :func:`serialize_4dsg_json_strict`:
    fixed field order, explicit sorting of all lists, and unified float
    precision.

    Args:
        graph: 4D Scene Graph (with optional ``track_metadata``).
        config: Phase 7 serialization config (uses defaults if None).

    Returns:
        JSON-serializable dictionary with SAGE-enriched object entries.
    """
    if config is None:
        config = Phase7SerializationConfig()

    precision = config.precision
    track_meta = graph.track_metadata  # may be None

    # === 1. Metadata (fixed order) ===
    result: Dict[str, Any] = {}
    result["metadata"] = {
        "num_frames": len(graph.spatial_graphs),
        "num_objects": len(graph.temporal_window.tracks),
        "temporal_window": 10,  # FIXED: Paper specification T=10 frames
    }

    # === 2. Ego Agent Trajectory (sorted by frame) ===
    ego_trajectory: List[Dict[str, Any]] = []
    if graph.ego_poses:
        for frame_idx in sorted(graph.ego_poses.keys()):
            pose = graph.ego_poses[frame_idx]
            entry: Dict[str, Any] = {"frame": frame_idx}

            if len(pose) >= 3:
                entry["position"] = [
                    round(pose[0], precision),
                    round(pose[1], precision),
                    round(pose[2], precision),
                ]

            if len(pose) >= 7:
                entry["rotation"] = [
                    round(pose[3], precision),
                    round(pose[4], precision),
                    round(pose[5], precision),
                    round(pose[6], precision),
                ]

            ego_trajectory.append(entry)

    result["ego_agent"] = {"trajectory": ego_trajectory}

    # === 3. Objects (sorted by object_id, steps sorted by frame) ===
    objects_list: List[Dict[str, Any]] = []

    sorted_tracks = sorted(
        graph.temporal_window.tracks.items(),
        key=lambda x: x[0],
    )[:config.max_tracks]

    for track_id, temporal_track in sorted_tracks:
        # --- SAGE metadata for this track ---
        if track_meta is not None and track_id in track_meta:
            meta = track_meta[track_id]
            obj_label = meta.get("label", None)
            obj_label_confidence = meta.get("label_confidence", None)
            obj_motion_state = meta.get("motion_state", None)
            obj_velocity = meta.get("velocity", None)
        else:
            obj_label = None
            obj_label_confidence = None
            obj_motion_state = None
            obj_velocity = None

        # Round numeric SAGE fields to the configured precision
        if obj_label_confidence is not None:
            obj_label_confidence = round(float(obj_label_confidence), precision)
        if obj_velocity is not None:
            obj_velocity = round(float(obj_velocity), precision)

        # --- Build track steps ---
        steps_data: List[Dict[str, Any]] = []

        if temporal_track.frame_indices is None or len(temporal_track.frame_indices) != len(temporal_track.steps):
            frame_indices = list(range(len(temporal_track.steps)))
        else:
            frame_indices = temporal_track.frame_indices

        num_steps = len(temporal_track.steps)
        start_idx = max(0, num_steps - config.max_frames_per_track)
        recent_steps = temporal_track.steps[start_idx:]
        recent_frames = frame_indices[start_idx:]

        for step, fidx in zip(recent_steps, recent_frames):
            c = step.centroid
            s = step.shape

            step_entry: Dict[str, Any] = {
                "frame": fidx,
                "centroid": [
                    round(c.x, precision),
                    round(c.y, precision),
                    round(c.z, precision),
                ],
                "extent": [
                    round(s.x_max - s.x_min, precision),
                    round(s.y_max - s.y_min, precision),
                    round(s.z_max - s.z_min, precision),
                ],
            }
            steps_data.append(step_entry)

        # Sort steps by frame for determinism
        steps_data.sort(key=lambda s: s["frame"])

        objects_list.append({
            "object_id": track_id,
            "label": obj_label,
            "label_confidence": obj_label_confidence,
            "motion_state": obj_motion_state,
            "velocity": obj_velocity,
            "track": steps_data,
        })

    result["objects"] = objects_list

    # === 4. Spatial Relations (sorted by distance) ===
    relations_list: List[Dict[str, Any]] = []

    if graph.spatial_graphs:
        recent_graph = graph.spatial_graphs[-1]

        edges = [
            e for e in recent_graph.edges
            if e.distance < config.relation_distance_threshold
        ]

        edges = sorted(edges, key=lambda e: e.distance)[:config.max_relations_per_frame]

        for edge in edges:
            relations_list.append({
                "source_id": edge.src,
                "target_id": edge.dst,
                "relation": _compute_spatial_relation(edge.direction, edge.distance),
                "distance": round(edge.distance, precision),
            })

    result["spatial_relations"] = relations_list

    return result


def build_messages_sage(
    query: str,
    graph: FourDSceneGraph,
    key_frame_images: Optional[List[np.ndarray]] = None,
    config: Optional[Phase7SerializationConfig] = None,
) -> List[Dict[str, Any]]:
    """Build multimodal chat messages for VLM inference with SAGE metadata.

    Produces a ``[system, user]`` message list suitable for chat-based VLMs.
    The system prompt explicitly mentions that objects carry semantic labels
    and motion states.  If *key_frame_images* is supplied, each image is
    embedded in the user content list as an ``{"type": "image", "image": img}``
    entry so that multimodal models can attend to them.

    Args:
        query: User question (typically a multiple-choice question).
        graph: 4D Scene Graph (with optional ``track_metadata``).
        key_frame_images: Optional list of key-frame images (numpy arrays).
        config: Phase 7 serialization config (uses defaults if None).

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    if config is None:
        config = Phase7SerializationConfig()

    # Serialize 4DSG using SAGE format
    context_json = serialize_4dsg_sage(graph, config)

    # System message (enhanced for SAGE)
    system: Dict[str, Any] = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": (
                "You are a spatial reasoning assistant analyzing a 4D scene. "
                "The scene contains objects tracked over time with their 3D positions, "
                "sizes, and spatial relationships. Objects are labeled with semantic "
                "categories and motion states. Answer questions precisely based on "
                "the provided scene information."
            ),
        }],
    }

    # User message: optional images + JSON context + question
    user_content: List[Dict[str, Any]] = []

    if key_frame_images is not None:
        for img in key_frame_images:
            user_content.append({"type": "image", "image": img})

    user_text = (
        f"Scene Data (JSON):\n"
        f"{json.dumps(context_json, indent=2)}\n\n"
        f"Question: {query}\n\n"
        f"Answer with just the letter choice (A, B, C, or D):"
    )

    user_content.append({"type": "text", "text": user_text})

    user: Dict[str, Any] = {
        "role": "user",
        "content": user_content,
    }

    return [system, user]


def build_prompt_sage(
    query: str,
    graph: FourDSceneGraph,
    config: Optional[Phase7SerializationConfig] = None,
) -> str:
    """Build a simple text prompt for SAGE (no chat format).

    Same structure as :func:`build_prompt_phase7` but serializes the scene
    with :func:`serialize_4dsg_sage` so that semantic labels, motion states,
    velocities, and label confidences are included.

    Args:
        query: User question.
        graph: 4D Scene Graph (with optional ``track_metadata``).
        config: Phase 7 serialization config (uses defaults if None).

    Returns:
        Complete prompt string.
    """
    if config is None:
        config = Phase7SerializationConfig()

    context_json = serialize_4dsg_sage(graph, config)

    prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

Scene Data (JSON):
{json.dumps(context_json, indent=2)}

Based on the scene information above, please answer the following question:
{query}

Answer with just the letter choice (A, B, C, or D):"""

    return prompt