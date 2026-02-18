"""Camera intrinsics/extrinsics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation: np.ndarray  # (3, 3) world-to-camera
    translation: np.ndarray  # (3,) world-to-camera

    def matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat


@dataclass(frozen=True)
class CameraModel:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    image_size: Tuple[int, int]  # (width, height)
