"""NuScenes dataset loader for SNOW.

This module provides utilities for loading NuScenes data including:
- Point clouds (LiDAR)
- Camera images
- Calibration data
- Annotations

Compatible with NuScenes v1.0 format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import CameraIntrinsics, CameraExtrinsics, CameraModel

logger = logging.getLogger(__name__)


# NuScenes camera names
NUSCENES_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class NuScenesCalibration:
    """Calibration data for a single sensor."""

    sensor_token: str
    sensor_type: str  # "camera" or "lidar"
    channel: str  # e.g., "CAM_FRONT", "LIDAR_TOP"

    # Extrinsics (sensor to ego frame)
    translation: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3) or quaternion (4,)

    # Camera intrinsics (only for cameras)
    camera_intrinsic: Optional[np.ndarray] = None  # (3, 3)
    image_size: Optional[Tuple[int, int]] = None  # (width, height)


@dataclass
class NuScenesSample:
    """A single synchronized sample from NuScenes."""

    sample_token: str
    scene_token: str
    timestamp: int

    # Sensor data paths
    lidar_path: Optional[str] = None
    camera_paths: Dict[str, str] = field(default_factory=dict)

    # Calibration data
    lidar_calibration: Optional[NuScenesCalibration] = None
    camera_calibrations: Dict[str, NuScenesCalibration] = field(default_factory=dict)

    # Ego pose at this timestamp
    ego_translation: Optional[np.ndarray] = None  # (3,)
    ego_rotation: Optional[np.ndarray] = None  # (4,) quaternion


@dataclass
class NuScenesScene:
    """A scene (sequence) from NuScenes."""

    scene_token: str
    name: str
    description: str
    samples: List[NuScenesSample] = field(default_factory=list)

    # Scene-level metadata
    nbr_samples: int = 0
    first_sample_token: str = ""
    last_sample_token: str = ""


class NuScenesLoader:
    """Loader for NuScenes dataset.

    This provides a simplified interface for loading NuScenes data
    without requiring the full nuscenes-devkit.
    """

    def __init__(
        self,
        dataroot: Path,
        version: str = "v1.0-trainval",
        verbose: bool = False,
    ):
        """Initialize NuScenes loader.

        Args:
            dataroot: Path to NuScenes dataset root
            version: Dataset version (e.g., "v1.0-trainval", "v1.0-mini")
            verbose: Whether to print loading progress
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.verbose = verbose

        # Data tables
        self.scene: Dict[str, Dict] = {}
        self.sample: Dict[str, Dict] = {}
        self.sample_data: Dict[str, Dict] = {}
        self.calibrated_sensor: Dict[str, Dict] = {}
        self.ego_pose: Dict[str, Dict] = {}
        self.sensor: Dict[str, Dict] = {}

        # Load tables
        self._load_tables()

    def _load_tables(self) -> None:
        """Load NuScenes JSON tables."""
        tables_dir = self.dataroot / self.version

        table_names = [
            "scene", "sample", "sample_data",
            "calibrated_sensor", "ego_pose", "sensor",
        ]

        for table_name in table_names:
            table_path = tables_dir / f"{table_name}.json"
            if table_path.exists():
                with open(table_path, "r") as f:
                    records = json.load(f)
                # Index by token
                table = {r["token"]: r for r in records}
                setattr(self, table_name, table)
                if self.verbose:
                    logger.info(f"Loaded {len(table)} {table_name} records")
            else:
                logger.warning(f"Table not found: {table_path}")

    def get_scenes(self) -> List[NuScenesScene]:
        """Get all scenes."""
        scenes = []
        for token, record in self.scene.items():
            scene = NuScenesScene(
                scene_token=token,
                name=record.get("name", ""),
                description=record.get("description", ""),
                nbr_samples=record.get("nbr_samples", 0),
                first_sample_token=record.get("first_sample_token", ""),
                last_sample_token=record.get("last_sample_token", ""),
            )
            scenes.append(scene)
        return scenes

    def get_scene_samples(self, scene_token: str) -> List[NuScenesSample]:
        """Get all samples for a scene in temporal order."""
        scene_record = self.scene.get(scene_token)
        if scene_record is None:
            return []

        samples = []
        sample_token = scene_record.get("first_sample_token", "")

        while sample_token:
            sample_record = self.sample.get(sample_token)
            if sample_record is None:
                break

            sample = self._build_sample(sample_record)
            samples.append(sample)

            sample_token = sample_record.get("next", "")

        return samples

    def _build_sample(self, sample_record: Dict) -> NuScenesSample:
        """Build NuScenesSample from record."""
        sample = NuScenesSample(
            sample_token=sample_record["token"],
            scene_token=sample_record["scene_token"],
            timestamp=sample_record["timestamp"],
        )

        # Get sensor data for this sample
        for data_token in sample_record.get("data", {}).values():
            sd_record = self.sample_data.get(data_token)
            if sd_record is None:
                continue

            channel = sd_record.get("channel", "")
            filename = sd_record.get("filename", "")

            # Get calibration
            cs_token = sd_record.get("calibrated_sensor_token", "")
            cs_record = self.calibrated_sensor.get(cs_token, {})

            # Get ego pose
            ep_token = sd_record.get("ego_pose_token", "")
            ep_record = self.ego_pose.get(ep_token, {})

            if channel == "LIDAR_TOP":
                sample.lidar_path = filename
                sample.lidar_calibration = self._build_calibration(cs_record, channel)
                sample.ego_translation = np.array(ep_record.get("translation", [0, 0, 0]))
                sample.ego_rotation = np.array(ep_record.get("rotation", [1, 0, 0, 0]))

            elif channel in NUSCENES_CAMERAS:
                sample.camera_paths[channel] = filename
                sample.camera_calibrations[channel] = self._build_calibration(cs_record, channel)

        return sample

    def _build_calibration(self, cs_record: Dict, channel: str) -> NuScenesCalibration:
        """Build calibration from calibrated_sensor record."""
        sensor_token = cs_record.get("sensor_token", "")
        sensor_record = self.sensor.get(sensor_token, {})

        # Parse translation and rotation
        translation = np.array(cs_record.get("translation", [0, 0, 0]))
        rotation_quat = np.array(cs_record.get("rotation", [1, 0, 0, 0]))
        rotation_matrix = quaternion_to_matrix(rotation_quat)

        # Camera intrinsics
        camera_intrinsic = None
        if "camera_intrinsic" in cs_record:
            camera_intrinsic = np.array(cs_record["camera_intrinsic"])

        return NuScenesCalibration(
            sensor_token=sensor_token,
            sensor_type=sensor_record.get("modality", "unknown"),
            channel=channel,
            translation=translation,
            rotation=rotation_matrix,
            camera_intrinsic=camera_intrinsic,
        )

    def load_lidar_points(self, lidar_path: str) -> np.ndarray:
        """Load LiDAR point cloud.

        Args:
            lidar_path: Relative path to .bin file

        Returns:
            Point cloud array of shape (N, 4) with [x, y, z, intensity]
        """
        full_path = self.dataroot / lidar_path
        points = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, 5)
        return points[:, :4]  # x, y, z, intensity

    def load_image(self, image_path: str) -> np.ndarray:
        """Load camera image.

        Args:
            image_path: Relative path to image file

        Returns:
            Image array of shape (H, W, 3)
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required for loading images")

        full_path = self.dataroot / image_path
        img = Image.open(full_path)
        return np.array(img)

    def get_camera_model(
        self,
        calibration: NuScenesCalibration,
        image_size: Tuple[int, int] = (1600, 900),
    ) -> CameraModel:
        """Convert NuScenes calibration to CameraModel.

        Args:
            calibration: NuScenes calibration data
            image_size: (width, height) of camera images

        Returns:
            CameraModel for projection
        """
        if calibration.camera_intrinsic is None:
            raise ValueError("No camera intrinsic available")

        K = calibration.camera_intrinsic
        intrinsics = CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )

        extrinsics = CameraExtrinsics(
            rotation=calibration.rotation,
            translation=calibration.translation,
        )

        return CameraModel(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
        )

    def iter_samples(
        self,
        scene_tokens: Optional[List[str]] = None,
    ) -> Iterator[NuScenesSample]:
        """Iterate over samples.

        Args:
            scene_tokens: Optional list of scene tokens to filter

        Yields:
            NuScenesSample objects
        """
        scenes = self.get_scenes()

        if scene_tokens is not None:
            scenes = [s for s in scenes if s.scene_token in scene_tokens]

        for scene in scenes:
            for sample in self.get_scene_samples(scene.scene_token):
                yield sample


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Rotation matrix (3, 3)
    """
    w, x, y, z = q

    # Normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.

    Args:
        R: Rotation matrix (3, 3)

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def transform_points_to_camera(
    points: np.ndarray,
    calibration: NuScenesCalibration,
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
) -> np.ndarray:
    """Transform points from global frame to camera frame.

    Args:
        points: Points in global frame (N, 3)
        calibration: Camera calibration
        ego_translation: Ego position in global frame (3,)
        ego_rotation: Ego rotation quaternion (4,)

    Returns:
        Points in camera frame (N, 3)
    """
    # Global to ego frame
    # ego_R = R_e2w; inverse in row-vector form: (p - t) @ R_e2w
    ego_R = quaternion_to_matrix(ego_rotation)
    points_ego = (points - ego_translation) @ ego_R

    # Ego to sensor frame
    # sensor_R = R_s2e; inverse in row-vector form: (p - t) @ R_s2e
    sensor_R = calibration.rotation
    sensor_t = calibration.translation
    points_sensor = (points_ego - sensor_t) @ sensor_R

    return points_sensor


def project_to_image(
    points: np.ndarray,
    calibration: NuScenesCalibration,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to image plane.

    Args:
        points: Points in camera frame (N, 3)
        calibration: Camera calibration with intrinsics
        image_size: (width, height)

    Returns:
        Tuple of:
            - Image coordinates (N, 2)
            - Valid mask (N,) indicating points in front of camera and in image
    """
    if calibration.camera_intrinsic is None:
        raise ValueError("No camera intrinsic available")

    K = calibration.camera_intrinsic
    width, height = image_size

    # Project
    z = points[:, 2:3]
    valid_depth = z[:, 0] > 0.1  # At least 10cm in front

    # Avoid division by zero
    z_safe = np.where(z > 0.1, z, 0.1)
    uv_homo = (K @ points.T).T
    uv = uv_homo[:, :2] / z_safe

    # Check bounds
    valid_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )

    valid = valid_depth & valid_bounds

    return uv, valid
