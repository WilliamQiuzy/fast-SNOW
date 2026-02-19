"""Fast-SNOW pipeline configuration dataclasses.

All hyperparameters are defined in the implementation spec
(docs/roadmap/Fast-SNOW_IMPLEMENTATION.md, Section 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# =============================================================================
# Vision Model Configs
# =============================================================================

@dataclass
class SAM3Config:
    """SAM3 segmentation / tracking configuration (Step 3, spec Section 6.1).

    model_path: local directory containing sam3.pt, or direct path to the
        checkpoint file.

    Note: SAM3 requires CUDA (Sam3VideoPredictor hardcodes .cuda()).
    SAM3 with text prompts always enables automatic instance detection
    (allow_new_detections is implicit in text-prompt mode).
    """
    score_threshold_detection: float = 0.3
    model_path: str = "fast_snow/models/sam3"


@dataclass
class DA3Config:
    """DA3 monocular depth estimation configuration (Step 2).

    model_path: local directory or HF hub ID passed to
        ``DepthAnything3.from_pretrained()``.  Must point to a **metric**
        model (e.g. da3nested-giant-large) so that depth is in metres.
    """
    device: str = "cuda"
    model_path: str = "fast_snow/models/da3"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"


@dataclass
class RAMPlusConfig:
    """RAM++ image tagging configuration (Step 1)."""
    device: str = "cuda"
    model_path: str = "fast_snow/models/ram_plus"
    checkpoint_path: Optional[str] = None  # Explicit .pth path; auto-detect if None
    normalize_lowercase: bool = True
    deduplicate_tags: bool = True


@dataclass
class SamplingConfig:
    """Frame sampling and scheduling configuration (Step 0)."""
    target_fps: float = 10.0
    max_frames: Optional[int] = None


# =============================================================================
# Pipeline Step Configs
# =============================================================================

@dataclass
class DepthFilterConfig:
    """Depth & 3D filtering configuration (Step 4, spec Section 6.2)."""
    conf_thresh: float = 0.5
    min_points: int = 50
    max_extent: float = 30.0  # meters


@dataclass
class FusionConfig:
    """Global ID fusion configuration (Step 5, spec Section 6.3)."""
    cross_run_iou_thresh: float = 0.5
    merge_centroid_dist_m: float = 2.0
    merge_temporal_gap: int = 2
    lost_patience: int = 5   # frames
    archive_patience: int = 30  # frames


@dataclass
class STEPConfig:
    """STEP token configuration (Step 6, spec Section 6.4)."""
    grid_size: int = 16
    iou_threshold: float = 0.5


@dataclass
class EdgeConfig:
    """Scene graph edge configuration (Step 7, spec Section 6.5)."""
    elev_thresh: float = 0.5   # meters
    motion_thresh: float = 0.3  # m/frame
    lateral_thresh: float = 0.3  # m/frame
    knn_k: int = 3
    motion_window: int = 3  # frames — minimum history for motion inference


@dataclass
class SerializationConfig:
    """Serialization & VLM configuration (Step 8, spec Section 6.6)."""
    max_obj_relations: int = 20


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class FastSNOWConfig:
    """Main Fast-SNOW pipeline configuration.

    All default values match the implementation spec
    (docs/roadmap/Fast-SNOW_IMPLEMENTATION.md, Section 6).
    """
    # Vision models
    sam3: SAM3Config = field(default_factory=SAM3Config)
    da3: DA3Config = field(default_factory=DA3Config)
    ram_plus: RAMPlusConfig = field(default_factory=RAMPlusConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Pipeline steps
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    step: STEPConfig = field(default_factory=STEPConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)

    # Global settings
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


# =============================================================================
# YAML Loading / Saving
# =============================================================================

def _build_from_dict(cls, raw: Dict[str, Any]):
    """Recursively construct dataclass from a dict."""
    if not isinstance(raw, dict):
        return cls()
    kwargs = {}
    for name, field_info in cls.__dataclass_fields__.items():
        if name not in raw:
            continue
        val = raw[name]
        ft = field_info.type
        # Resolve string annotations
        if isinstance(ft, str):
            import sys
            module = sys.modules.get(cls.__module__)
            ft = getattr(module, ft, ft) if module else ft
        if hasattr(ft, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[name] = _build_from_dict(ft, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_fast_snow_config(path: Union[str, Path]) -> FastSNOWConfig:
    """Load FastSNOWConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _build_from_dict(FastSNOWConfig, data)


def save_fast_snow_config(config: FastSNOWConfig, path: Union[str, Path]) -> None:
    """Save FastSNOWConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
