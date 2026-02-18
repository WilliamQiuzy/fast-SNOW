"""NuScenes -> FrameData adapter.

Coordinate transform chain
--------------------------
**Points  (LiDAR sensor -> world):**

    p_ego   = R_s2e @ p_lidar + t_s2e      # sensor-to-ego
    p_world = R_e2w @ p_ego   + t_e2w      # ego-to-world

**Camera extrinsics  (world -> camera):**

    R_w2e = R_e2w^T            t_w2e = -R_e2w^T @ t_e2w
    R_e2s = R_s2e^T            t_e2s = -R_s2e^T @ t_s2e
    R_w2c = R_e2s @ R_w2e     t_w2c = R_e2s @ t_w2e + t_e2s

These match the convention in ``data.calibration.camera_model.CameraExtrinsics``
(rotation / translation are **world-to-camera**) and are consumed directly by
``data.transforms.projection.project_points``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraModel,
)
from fast_snow.data.loaders.nuscenes import (
    NUSCENES_CAMERAS,
    NuScenesCalibration,
    NuScenesLoader,
    NuScenesSample,
    NuScenesScene,
    quaternion_to_matrix,
)
from fast_snow.data.schema import FrameData, SampleManifest

logger = logging.getLogger(__name__)


class NuScenesAdapter:
    """Converts NuScenes data to unified :class:`FrameData`."""

    def __init__(
        self,
        loader: NuScenesLoader,
        image_size: Tuple[int, int] = (1600, 900),
        load_points: bool = True,
    ):
        """
        Args:
            loader: An initialised :class:`NuScenesLoader`.
            image_size: ``(width, height)`` of camera images.
            load_points: Whether to load LiDAR points.
        """
        self.loader = loader
        self.image_size = image_size
        self.load_points = load_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_to_frame(
        self,
        sample: NuScenesSample,
        frame_idx: int,
    ) -> FrameData:
        """Convert a single :class:`NuScenesSample` to :class:`FrameData`."""

        # 1. ego-to-world  (4x4)
        ego_pose = self._build_ego_pose(sample)

        # 2. LiDAR points -> world
        points_world = np.empty((0, 3), dtype=np.float64)
        point_intensities: Optional[np.ndarray] = None
        if self.load_points and sample.lidar_path:
            raw = self.loader.load_lidar_points(sample.lidar_path)
            points_lidar = raw[:, :3].astype(np.float64)
            if raw.shape[1] > 3:
                point_intensities = raw[:, 3]
            points_world = self._lidar_to_world(
                points_lidar, sample.lidar_calibration, ego_pose
            )

        # 3. Camera models  (world-to-camera extrinsics)
        cameras = {}
        image_paths = {}
        for cam_id in NUSCENES_CAMERAS:
            if cam_id not in sample.camera_calibrations:
                continue
            cal = sample.camera_calibrations[cam_id]
            if cal.camera_intrinsic is None:
                continue
            cameras[cam_id] = self._build_world2cam_model(cal, ego_pose)
            if cam_id in sample.camera_paths:
                image_paths[cam_id] = str(
                    self.loader.dataroot / sample.camera_paths[cam_id]
                )

        # 4. Timestamp  (NuScenes stores microseconds)
        timestamp = sample.timestamp / 1e6

        return FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            image_paths=image_paths,
            points_world=points_world,
            point_intensities=point_intensities,
            cameras=cameras,
            ego_pose=ego_pose,
        )

    def scene_to_manifest(
        self,
        scene: NuScenesScene,
        max_frames: int = 10,
        target_fps: float = 1.0,
    ) -> SampleManifest:
        """Convert a NuScenes scene to a :class:`SampleManifest`.

        NuScenes keyframes are at ~2 Hz.  ``target_fps=1.0`` selects
        roughly every other keyframe.

        Args:
            scene: Scene metadata.
            max_frames: Maximum *T* frames (paper: 10).
            target_fps: Target sampling rate in Hz.
        """
        all_samples = self.loader.get_scene_samples(scene.scene_token)
        if not all_samples:
            return SampleManifest(
                sample_id=scene.scene_token,
                dataset="nuscenes",
                metadata={"scene_name": scene.name},
            )

        selected = self._select_samples_by_fps(all_samples, target_fps, max_frames)
        frames = [
            self.sample_to_frame(s, frame_idx=idx)
            for idx, s in enumerate(selected)
        ]

        return SampleManifest(
            sample_id=scene.scene_token,
            dataset="nuscenes",
            frames=frames,
            metadata={
                "scene_name": scene.name,
                "description": scene.description,
                "total_keyframes": len(all_samples),
                "selected_keyframes": len(selected),
            },
        )

    # ------------------------------------------------------------------
    # Coordinate transforms (static so they can be unit-tested directly)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ego_pose(sample: NuScenesSample) -> np.ndarray:
        """Build 4x4 ego-to-world SE(3) matrix."""
        T = np.eye(4, dtype=np.float64)
        if sample.ego_rotation is not None:
            T[:3, :3] = quaternion_to_matrix(sample.ego_rotation)
        if sample.ego_translation is not None:
            T[:3, 3] = sample.ego_translation
        return T

    @staticmethod
    def _lidar_to_world(
        points_lidar: np.ndarray,
        lidar_cal: Optional[NuScenesCalibration],
        ego_pose: np.ndarray,
    ) -> np.ndarray:
        """LiDAR sensor frame -> world frame.

        p_ego   = R_s2e @ p + t_s2e
        p_world = R_e2w @ p_ego + t_e2w
        """
        if lidar_cal is not None:
            R_s2e = lidar_cal.rotation.astype(np.float64)
            t_s2e = lidar_cal.translation.astype(np.float64)
            points_ego = (R_s2e @ points_lidar.T).T + t_s2e
        else:
            points_ego = points_lidar

        R_e2w = ego_pose[:3, :3]
        t_e2w = ego_pose[:3, 3]
        return (R_e2w @ points_ego.T).T + t_e2w

    @staticmethod
    def _build_world2cam_model(
        cam_cal: NuScenesCalibration,
        ego_pose: np.ndarray,
    ) -> CameraModel:
        """Build a :class:`CameraModel` whose extrinsics are world-to-camera.

        Composition::

            world  ->  ego  (inverse of ego-to-world)
            ego    ->  sensor  (inverse of sensor-to-ego)
        """
        R_e2w = ego_pose[:3, :3]
        t_e2w = ego_pose[:3, 3]
        R_s2e = cam_cal.rotation.astype(np.float64)
        t_s2e = cam_cal.translation.astype(np.float64)

        # world -> ego
        R_w2e = R_e2w.T
        t_w2e = -R_e2w.T @ t_e2w

        # ego -> sensor (camera)
        R_e2s = R_s2e.T
        t_e2s = -R_s2e.T @ t_s2e

        # compose: world -> camera
        R_w2c = R_e2s @ R_w2e
        t_w2c = R_e2s @ t_w2e + t_e2s

        K = cam_cal.camera_intrinsic
        intrinsics = CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )
        extrinsics = CameraExtrinsics(rotation=R_w2c, translation=t_w2c)

        # image_size from calibration if available, else default
        w, h = cam_cal.image_size if cam_cal.image_size else (1600, 900)
        return CameraModel(intrinsics=intrinsics, extrinsics=extrinsics,
                           image_size=(w, h))

    @staticmethod
    def _select_samples_by_fps(
        samples: List[NuScenesSample],
        target_fps: float,
        max_frames: int,
    ) -> List[NuScenesSample]:
        """Sub-sample keyframes to approximate *target_fps*.

        Uses real timestamps (microseconds) with 80 % tolerance on the
        inter-frame interval so that small jitter doesn't skip frames.
        """
        if len(samples) <= max_frames and target_fps <= 0:
            return samples[:max_frames]

        target_interval_us = 1e6 / target_fps  # microseconds
        selected = [samples[0]]
        last_ts = samples[0].timestamp

        for s in samples[1:]:
            if len(selected) >= max_frames:
                break
            if (s.timestamp - last_ts) >= target_interval_us * 0.8:
                selected.append(s)
                last_ts = s.timestamp

        return selected
