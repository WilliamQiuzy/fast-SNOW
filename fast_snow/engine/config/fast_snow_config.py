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
    trim_past_non_cond_mem_for_eval: bool = True
    offload_state_to_cpu: bool = True
    offload_video_to_cpu: bool = True


@dataclass
class DA3Config:
    """DA3 monocular depth estimation configuration (Step 2).

    model_path: local directory or HF hub ID passed to
        ``DepthAnything3.from_pretrained()``.

    Supported model variants (smallest → largest):
      - ``da3-small``   (~34M)  — relative depth + K + T_wc
      - ``da3-base``    (~120M) — relative depth + K + T_wc
      - ``da3-large``   (~350M) — relative depth + K + T_wc
      - ``da3nested-giant-large`` (~1.4B) — metric depth + K + T_wc

    When ``require_metric`` is False (default), relative-depth models are
    accepted and 3D coordinates use an arbitrary but consistent scale.
    Set ``require_metric=True`` to enforce metric depth in metres.

    Chunked inference (see docs/bugs/DA3_BATCH_OOM.md):
      When ``chunk_size > 0`` and the frame count exceeds ``chunk_size``,
      ``infer_batch`` automatically splits into overlapping chunks and
      aligns them via SIM3 point-cloud matching.  Set ``chunk_size=0``
      to disable chunking (always full-batch).
    """
    device: str = "cuda"
    model_path: str = "fast_snow/models/da3-small"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    require_metric: bool = False
    chunk_size: int = 0    # Max frames per DA3 batch chunk. 0 = no chunking.
    chunk_overlap: int = 5  # Overlap frames between adjacent chunks for SIM3 alignment.


@dataclass
class RAMPlusConfig:
    """RAM++ image tagging configuration (Step 1)."""
    device: str = "cuda"
    model_path: str = "fast_snow/models/ram_plus"
    checkpoint_path: Optional[str] = None  # Explicit .pth path; auto-detect if None
    normalize_lowercase: bool = True
    deduplicate_tags: bool = True


@dataclass
class YOLOConfig:
    """YOLO bbox detection configuration (legacy, used by asset scripts)."""
    device: str = "cuda"
    model_path: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    imgsz: int = 640
    max_det: int = 200


@dataclass
class FastSAMConfig:
    """FastSAM class-agnostic segmentation configuration (Step 1).

    Replaces YOLO for initial object discovery.  FastSAM produces
    instance masks without class labels, enabling open-world detection.
    """
    device: str = "cuda"
    model_path: str = "fast_snow/models/fastsam/FastSAM-s.pt"
    conf_threshold: float = 0.55
    iou_threshold: float = 0.9
    imgsz: int = 640
    max_det: int = 200
    discovery_iou_thresh: float = 0.3  # Below this IoU = new object in two-pass discovery


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
    mask_outside_pixels: bool = True  # Zero out non-mask pixels in patch crops (paper default)
    patch_crop_size: int = 64  # Resize each cell crop to 64x64 for uniform visual tokens
    temporal_window: int = 10  # F_k sliding window: keep last T frames per track (SNOW §4.2, T=10)
    max_tau_per_step: int = 0  # Top-k patches per STEP token, sorted by IoU desc. 0=unlimited.


@dataclass
class VLMConfig:
    """VLM inference configuration (Step 9).

    Supports two providers:
    - "openai": Uses OpenAI API (default, for GPT-5.2 etc.)
    - "google": Uses Google genai API (for Gemini, Gemma etc.)

    The API key is read from the environment variable specified by api_key_env.
    """
    provider: str = "openai"          # "openai" | "google"
    model: str = "gpt-5.2"            # Model name sent to API
    max_output_tokens: int = 1024
    temperature: float = 1.0
    api_key_env: str = "OPENAI_API_KEY"  # Env var name for the API key
    base_url: Optional[str] = None       # Optional base URL override (e.g. for Gemini via OpenAI-compat)


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
    fastsam: FastSAMConfig = field(default_factory=FastSAMConfig)
    ram_plus: RAMPlusConfig = field(default_factory=RAMPlusConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Pipeline steps
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    step: STEPConfig = field(default_factory=STEPConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

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
