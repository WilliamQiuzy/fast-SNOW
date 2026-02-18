"""Unified data schema for SNOW pipeline.

Every downstream module (clustering, projection, tracking, 4DSG) consumes
FrameData.  Two adapters (NuScenes, VLM4D) produce it.

Coordinate conventions
----------------------
- ``points_world``  : (N, 3) in the unified world frame.
- ``cameras[cam_id].extrinsics`` : world-to-camera transform.
- ``ego_pose``      : (4, 4) ego-to-world SE(3) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fast_snow.data.calibration.camera_model import CameraModel


@dataclass
class FrameData:
    """Complete sensor data for a single timestamp."""

    frame_idx: int
    timestamp: float  # seconds (epoch or scene-relative)

    # Images -----------------------------------------------------------------
    image_paths: Dict[str, str] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict)

    # 3-D geometry -----------------------------------------------------------
    points_world: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    point_intensities: Optional[np.ndarray] = None  # (N,) – NuScenes only

    # Calibrated cameras -----------------------------------------------------
    cameras: Dict[str, CameraModel] = field(default_factory=dict)

    # Ego pose ---------------------------------------------------------------
    ego_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    # -- helpers -------------------------------------------------------------

    def load_image(self, camera_id: str) -> np.ndarray:
        """Load (and cache) an image for *camera_id*."""
        if camera_id in self.images:
            return self.images[camera_id]
        from PIL import Image

        img = np.array(Image.open(self.image_paths[camera_id]))
        self.images[camera_id] = img
        return img

    def num_points(self) -> int:
        return self.points_world.shape[0]

    def num_cameras(self) -> int:
        return len(self.cameras)


@dataclass
class SampleManifest:
    """An ordered sequence of :class:`FrameData` plus sample-level metadata."""

    sample_id: str
    dataset: str  # "nuscenes" | "vlm4d"
    frames: List[FrameData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def timestamps(self) -> List[float]:
        return [f.timestamp for f in self.frames]

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp
