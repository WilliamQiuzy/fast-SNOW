"""SNOW pipeline configuration dataclasses.

This module defines all configuration dataclasses for the SNOW pipeline
with default values matching the paper specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints
import yaml


# =============================================================================
# Perception Configs
# =============================================================================

@dataclass
class HDBSCANConfig:
    """HDBSCAN clustering configuration.

    Paper reference: Section 3.1 - "HDBSCAN is used for density-based
    clustering on 3D points"
    """
    min_cluster_size: int = 30   # paper: ~30 for driving scenes (pedestrian-level)
    min_samples: int = 5         # paper: core-point neighbourhood
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # "eom" or "leaf"
    allow_single_cluster: bool = False


@dataclass
class SAM2Config:
    """SAM2 segmentation model configuration."""
    model_name: str = "hiera_large"  # hiera_tiny, hiera_small, hiera_base_plus, hiera_large
    device: str = "cuda"
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    # Multi-mask output settings
    multimask_output: bool = True
    # Score thresholds
    min_mask_score: float = 0.5


@dataclass
class HHopConfig:
    """H-hop geometric validation configuration.

    Paper reference: Section 3.2 - "H-hop validation filters geometrically
    implausible detections"
    """
    # Spatial extent limits
    max_extent: float = 50.0  # meters
    max_sigma: float = 10.0  # standard deviation

    # Aspect ratio limit
    max_aspect_ratio: float = 20.0

    # Velocity limits (for temporal consistency)
    max_velocity: float = 30.0  # m/s (about 108 km/h)

    # Size consistency
    max_size_change_ratio: float = 3.0

    # Minimum points
    min_points: int = 10


@dataclass
class RefinementConfig:
    """Iterative refinement configuration.

    Paper reference: Algorithm 1 - Iterative refinement loop.
    Paper parameters: N_iter=1, H_hop=1
    """
    n_iter: int = 1  # Number of iterations (Paper: N_iter=1)
    h_hop: int = 1   # Number of H-hop validation rounds per iteration (Paper: H_hop=1)

    # HDBSCAN settings
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)

    # H-hop validation
    hhop: HHopConfig = field(default_factory=HHopConfig)

    # Minimum unmapped points to continue iteration
    min_unmapped_points: int = 10  # Paper: continue until U_t is small

    # Whether to use SAM for mask refinement
    use_sam_refinement: bool = True


@dataclass
class SLAMConfig:
    """KISS-SLAM configuration for ego-motion estimation."""
    max_range: float = 100.0
    min_range: float = 1.0
    voxel_size: float = 0.5
    max_points_per_voxel: int = 20
    initial_threshold: float = 2.0
    min_motion_threshold: float = 0.1
    deskew: bool = True
    max_num_iterations: int = 50
    convergence_threshold: float = 0.0001
    use_adaptive_threshold: bool = True


@dataclass
class PerceptionConfig:
    """Combined perception module configuration."""
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)


# =============================================================================
# Graph Configs
# =============================================================================

@dataclass
class TrackerConfig:
    """Cross-frame object tracker configuration.

    Paper reference: Eq. 7 - Temporal object tracks.
    """
    # Cost function weights
    geometric_weight: float = 0.5
    semantic_weight: float = 0.5
    shape_distance_weight: float = 0.1

    # Association thresholds
    max_association_cost: float = 10.0
    max_centroid_distance: float = 5.0  # meters

    # Track management
    max_age: int = 10  # frames without match before track ends
    min_hits: int = 3  # minimum detections to confirm track

    # Temporal window (Eq. 7)
    temporal_window: int = 10  # T in the paper


@dataclass
class SceneGraphConfig:
    """Scene graph construction configuration."""
    max_edge_distance: float = 50.0
    include_relations: bool = True
    include_ego_relative: bool = True
    min_edge_distance: float = 0.1


@dataclass
class GraphConfig:
    """Combined graph module configuration."""
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)


# =============================================================================
# VLM Configs
# =============================================================================

@dataclass
class VLMConfig:
    """VLM inference configuration.

    Paper reference: Section 3.4 - Uses Gemma3-4B-IT for reasoning.
    """
    model_name: str = "Gemma3-4B-IT"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.0  # Deterministic
    do_sample: bool = False

    # Prompt configuration
    max_tracks: int = 50
    max_frames_per_track: int = 10
    include_ego: bool = True
    include_relations: bool = True
    max_relations_per_frame: int = 20
    relation_distance_threshold: float = 15.0
    precision: int = 2


# =============================================================================
# Evaluation Configs
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # VLM4D settings
    vlm4d_json_path: str = ""
    vlm4d_video_root: str = ""

    # NuScenes settings
    nuscenes_root: str = ""
    nuscenes_version: str = "v1.0-trainval"

    # Batch settings
    batch_size: int = 1
    num_workers: int = 4

    # Output
    output_dir: str = "outputs/eval"
    save_predictions: bool = True


# =============================================================================
# Data Config
# =============================================================================

@dataclass
class DataConfig:
    """Data loading and alignment configuration.

    Paper reference: Phase 0 -- sensors are temporally aligned and
    geometrically calibrated; T=10 frames at ~1 Hz.
    """
    # Source selection
    source: str = "vlm4d"  # "nuscenes" or "vlm4d"

    # Temporal sampling (paper: T=10 at ~1 Hz)
    num_frames: int = 10
    target_fps: float = 1.0

    # NuScenes-specific
    nuscenes_root: str = ""
    nuscenes_version: str = "v1.0-trainval"
    nuscenes_image_width: int = 1600
    nuscenes_image_height: int = 900

    # VLM4D-specific
    vlm4d_json_path: str = ""
    vlm4d_video_root: str = ""

    # MapAnything (used by VLM4D path)
    mapanything_device: str = "cuda"
    mapanything_use_amp: bool = True

    # Quality thresholds
    min_points_per_frame: int = 100
    min_point_depth: float = 0.1
    max_point_depth: float = 100.0


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class SNOWConfig:
    """Main SNOW pipeline configuration.

    This contains all sub-configurations for the complete pipeline.

    Paper: "SNOW: Spatio-Temporal Scene Understanding with World Knowledge
    for Open-World Embodied Reasoning"
    """
    # Sub-configurations
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Global settings
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False

    # Pipeline behavior toggles
    use_phase7_strict: bool = True

    # STEP token parameters
    step_grid_size: int = 16
    step_iou_threshold: float = 0.5
    min_cluster_points: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


# =============================================================================
# YAML Loading/Saving
# =============================================================================

def _get_dataclass_type(field_type) -> Optional[type]:
    """Extract dataclass type from field type annotation."""
    # Direct dataclass
    if hasattr(field_type, "__dataclass_fields__"):
        return field_type

    # Handle Optional, Union types
    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = getattr(field_type, "__args__", ())
        for arg in args:
            if arg is not type(None) and hasattr(arg, "__dataclass_fields__"):
                return arg

    return None


def _nested_dataclass_from_dict(cls, data: Dict[str, Any]) -> Any:
    """Recursively convert dict to nested dataclasses."""
    if data is None:
        return cls()

    if not hasattr(cls, "__dataclass_fields__"):
        return data

    # Get resolved field types using get_type_hints to handle forward references
    try:
        # Get the module where the class is defined for proper type resolution
        import sys
        module = sys.modules.get(cls.__module__, None)
        globalns = getattr(module, "__dict__", {}) if module else {}
        field_types = get_type_hints(cls, globalns=globalns, localns=None)
    except Exception:
        # Fallback to raw annotations if get_type_hints fails
        field_types = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            field_types[field_name] = field_info.type

    # Convert nested dicts
    converted = {}
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            dc_type = _get_dataclass_type(field_type)

            # Check if it's a dataclass and value is dict
            if dc_type is not None and isinstance(value, dict):
                converted[key] = _nested_dataclass_from_dict(dc_type, value)
            else:
                converted[key] = value
        else:
            # Skip unknown keys
            pass

    # Create instance with converted values, letting defaults fill in missing
    return cls(**converted)


def load_config(path: Union[str, Path]) -> SNOWConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        SNOWConfig instance.

    Example:
        config = load_config("configs/default.yaml")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return SNOWConfig()

    return _nested_dataclass_from_dict(SNOWConfig, data)


def save_config(config: SNOWConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: SNOWConfig instance to save.
        path: Output path for YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> SNOWConfig:
    """Get default configuration.

    Returns:
        SNOWConfig with default values matching paper specifications.
    """
    return SNOWConfig()
