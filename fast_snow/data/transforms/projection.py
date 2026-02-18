"""Project 3D points into camera image coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import CameraModel


@dataclass(frozen=True)
class ProjectionResult:
    xy: np.ndarray  # (N, 2)
    depth: np.ndarray  # (N,)
    valid: np.ndarray  # (N,) bool


def project_points(
    points_xyz: np.ndarray,
    camera: CameraModel,
    clip_to_image: bool = True,
    min_depth: float = 1e-6,
) -> ProjectionResult:
    """Project world points into image space.

    Args:
        points_xyz: (N, 3) world coordinates.
        camera: camera model (intrinsics + world-to-camera extrinsics).
        clip_to_image: mark points outside image as invalid.
        min_depth: minimum depth to consider a point valid.

    Returns:
        ProjectionResult with pixel coords, depth, and validity mask.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    # World -> camera
    R = camera.extrinsics.rotation
    t = camera.extrinsics.translation
    cam_xyz = (R @ points_xyz.T).T + t
    depth = cam_xyz[:, 2]

    valid = depth > min_depth
    if not np.any(valid):
        xy = np.zeros((points_xyz.shape[0], 2), dtype=float)
        return ProjectionResult(xy=xy, depth=depth, valid=valid)

    # Perspective projection
    K = camera.intrinsics.matrix()
    homog = (K @ cam_xyz.T).T
    xy = homog[:, :2] / homog[:, 2:3]

    if clip_to_image:
        width, height = camera.image_size
        in_x = (xy[:, 0] >= 0.0) & (xy[:, 0] < width)
        in_y = (xy[:, 1] >= 0.0) & (xy[:, 1] < height)
        valid = valid & in_x & in_y

    return ProjectionResult(xy=xy, depth=depth, valid=valid)
