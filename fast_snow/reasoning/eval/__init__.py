"""Evaluation scripts for SNOW benchmarks.

This module provides evaluation utilities for:
- VLM4D: 4D video understanding benchmark
- NuScenes-QA: Driving scene question answering
- RoboSpatial: Indoor robot spatial reasoning
- LiDAR Segmentation: Point cloud semantic segmentation
"""

from fast_snow.reasoning.eval.vlm4d import (
    evaluate_vlm4d,
    VLM4DMetrics,
    VLM4D_QUESTION_TYPES,
)
from fast_snow.reasoning.eval.nuscenes_qa import (
    evaluate_nuscenes_qa,
    NuScenesQAMetrics,
    NUSCENES_QA_CATEGORIES,
)
from fast_snow.reasoning.eval.robospatial import (
    evaluate_robospatial,
    RoboSpatialMetrics,
    ROBOSPATIAL_CATEGORIES,
)
from fast_snow.reasoning.eval.lidar_seg import (
    evaluate_lidar_seg,
    LiDARSegMetrics,
    NUSCENES_LIDAR_CLASSES,
)

__all__ = [
    # VLM4D
    "evaluate_vlm4d",
    "VLM4DMetrics",
    "VLM4D_QUESTION_TYPES",
    # NuScenes-QA
    "evaluate_nuscenes_qa",
    "NuScenesQAMetrics",
    "NUSCENES_QA_CATEGORIES",
    # RoboSpatial
    "evaluate_robospatial",
    "RoboSpatialMetrics",
    "ROBOSPATIAL_CATEGORIES",
    # LiDAR Segmentation
    "evaluate_lidar_seg",
    "LiDARSegMetrics",
    "NUSCENES_LIDAR_CLASSES",
]
