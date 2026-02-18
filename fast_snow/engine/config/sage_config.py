"""SAGE pipeline configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Default vocabulary for open-world scene detection
DEFAULT_CLASS_NAMES: Tuple[str, ...] = (
    "car", "truck", "bus", "van", "motorcycle", "bicycle",
    "person", "pedestrian", "child",
    "traffic light", "traffic sign", "stop sign",
    "fire hydrant", "bench", "trash can",
    "dog", "cat", "bird",
    "tree", "pole", "building", "fence", "wall",
    "road", "sidewalk", "crosswalk",
    "cone", "barrier", "bollard",
)


@dataclass
class DA3Config:
    """DepthAnything v3 configuration."""
    model_name: str = "da3-large"
    device: str = "cuda"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"


@dataclass
class DetectionConfig:
    """YOLO-World detection configuration."""
    model_name: str = "yolov8x-worldv2"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda"
    max_detections: int = 50
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES


@dataclass
class MotionConfig:
    """Motion state classification from track centroids."""
    velocity_threshold: float = 0.5  # m/s: below = stationary
    min_track_length: int = 2        # need >= 2 frames for velocity


@dataclass
class KeyFrameConfig:
    """Key frame selection for multimodal VLM input."""
    num_key_frames: int = 3
    strategy: str = "uniform"  # "uniform" | "first_middle_last"
    max_image_size: Tuple[int, int] = (512, 512)


@dataclass
class SAGEPerceptionConfig:
    """Perception stack configuration for SAGE."""
    da3: DA3Config = field(default_factory=DA3Config)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    # H-hop filter params (reused from SNOW)
    hhop_max_extent: float = 50.0
    hhop_max_velocity: float = 30.0
    min_detection_points: int = 10  # min 3D points per detection after backprojection


@dataclass
class SAGETrackerConfig:
    """Tracker configuration for SAGE."""
    geometric_weight: float = 0.5
    semantic_weight: float = 0.5
    max_centroid_distance: float = 5.0
    max_association_cost: float = 10.0
    temporal_window: int = 10
    label_mismatch_penalty: float = 2.0  # extra cost when labels differ


@dataclass
class SAGEGraphConfig:
    """Scene graph configuration for SAGE."""
    tracker: SAGETrackerConfig = field(default_factory=SAGETrackerConfig)
    max_edge_distance: float = 50.0
    include_relations: bool = True
    include_ego_relative: bool = True
    min_edge_distance: float = 0.1


@dataclass
class SAGEVLMConfig:
    """VLM configuration for SAGE (supports multimodal)."""
    model_name: str = "gemma-3-4b-it"
    backend: str = "google_ai"
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.0
    multimodal: bool = True
    key_frames: KeyFrameConfig = field(default_factory=KeyFrameConfig)


@dataclass
class SAGEEvalConfig:
    """Evaluation configuration."""
    vlm4d_json: Optional[str] = None
    output_dir: str = "results"
    batch_size: int = 1
    max_samples: Optional[int] = None


@dataclass
class SAGEDataConfig:
    """Data source configuration."""
    source: str = "vlm4d"
    num_frames: int = 10
    target_fps: float = 1.0
    video_dir: Optional[str] = None


@dataclass
class SAGEConfig:
    """Top-level SAGE pipeline configuration."""
    perception: SAGEPerceptionConfig = field(default_factory=SAGEPerceptionConfig)
    graph: SAGEGraphConfig = field(default_factory=SAGEGraphConfig)
    vlm: SAGEVLMConfig = field(default_factory=SAGEVLMConfig)
    eval: SAGEEvalConfig = field(default_factory=SAGEEvalConfig)
    data: SAGEDataConfig = field(default_factory=SAGEDataConfig)

    device: str = "cuda"
    seed: int = 42
    verbose: bool = False

    # STEP token params
    step_grid_size: int = 16
    step_iou_threshold: float = 0.5
