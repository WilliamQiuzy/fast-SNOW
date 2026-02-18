"""Segmentation module using SAM2.

Provides both single-image and video prediction interfaces.
"""

from fast_snow.vision.perception.segmentation.sam2_wrapper import (
    SAM2Config,
    SAM2Mask,
    VideoMaskResult,
    VideoPropagationResult,
    SAM2Wrapper,
    SAM2VideoPredictor,
    segment_cluster_sam2,
)

__all__ = [
    "SAM2Config",
    "SAM2Mask",
    "VideoMaskResult",
    "VideoPropagationResult",
    "SAM2Wrapper",
    "SAM2VideoPredictor",
    "segment_cluster_sam2",
]
