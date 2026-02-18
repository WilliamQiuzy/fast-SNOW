"""KISS-SLAM wrapper for LiDAR-based odometry and mapping.

KISS-SLAM is a lightweight LiDAR SLAM system that provides:
- Scan-to-map registration using point-to-plane ICP
- Local map management with voxel-based subsampling
- Loop closure detection (optional)

Paper: Simple-Yet-Effective Approach to LiDAR-based Odometry

In SNOW, KISS-SLAM is used to estimate ego poses when GT poses are unavailable.
This enables training-free scene understanding on novel datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np


@dataclass(frozen=True)
class KISSSLAMConfig:
    """Configuration for KISS-SLAM.

    Attributes:
        max_range: Maximum range for points to consider (meters).
        min_range: Minimum range to filter out ego-vehicle points.
        voxel_size: Voxel size for local map downsampling.
        max_points_per_voxel: Maximum points stored per voxel.
        initial_threshold: Initial distance threshold for ICP.
        min_motion_threshold: Minimum motion to register a new frame.
        deskew: Whether to apply motion compensation/deskewing.
        max_num_iterations: Maximum ICP iterations per frame.
        convergence_threshold: ICP convergence criterion.
        use_adaptive_threshold: Adaptively adjust ICP threshold based on residuals.
    """
    max_range: float = 100.0
    min_range: float = 1.0
    voxel_size: float = 0.5
    max_points_per_voxel: int = 20
    initial_threshold: float = 2.0
    min_motion_threshold: float = 0.1
    deskew: bool = True
    max_num_iterations: int = 50
    convergence_threshold: float = 0.0001
    use_adaptive_threshold: bool = True


@dataclass
class SLAMFrame:
    """A single LiDAR frame for SLAM processing.

    Attributes:
        frame_idx: Frame index in the sequence.
        points: Point cloud (N, 3) in sensor frame.
        timestamps: Optional per-point timestamps for deskewing.
    """
    frame_idx: int
    points: np.ndarray  # (N, 3)
    timestamps: Optional[np.ndarray] = None  # (N,) relative timestamps

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got {self.points.shape}")


@dataclass
class SLAMResult:
    """Result of SLAM processing.

    Attributes:
        poses: Dict mapping frame_idx to 4x4 transformation matrix (sensor to world).
        local_map: Accumulated point cloud map.
        frame_indices: List of processed frame indices.
        success: Whether SLAM completed successfully.
    """
    poses: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> (4, 4)
    local_map: Optional[np.ndarray] = None  # (M, 3) accumulated map
    frame_indices: List[int] = field(default_factory=list)
    success: bool = True

    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as (N, 3) positions."""
        positions = []
        for idx in sorted(self.poses.keys()):
            pose = self.poses[idx]
            positions.append(pose[:3, 3])
        return np.array(positions)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get pose for a specific frame."""
        return self.poses.get(frame_idx)


class KISSSLAMWrapper:
    """Wrapper around KISS-ICP/KISS-SLAM for ego-motion estimation.

    This provides a consistent interface for SNOW regardless of whether
    kiss-icp or kiss-slam is installed, with fallback to simple ICP.

    Usage:
        wrapper = KISSSLAMWrapper(config)
        wrapper.load()

        for frame in lidar_frames:
            pose = wrapper.process_frame(frame)

        result = wrapper.get_result()
    """

    def __init__(self, config: Optional[KISSSLAMConfig] = None) -> None:
        self.config = config or KISSSLAMConfig()
        self._pipeline = None
        self._backend: str = "none"
        self._poses: Dict[int, np.ndarray] = {}
        self._current_pose: np.ndarray = np.eye(4)
        self._local_map_points: List[np.ndarray] = []
        self._frame_indices: List[int] = []

    def load(self) -> None:
        """Load KISS-ICP backend."""
        if self._pipeline is not None:
            return

        # Try kiss-icp first (more common installation)
        try:
            from kiss_icp.config import KISSConfig
            from kiss_icp.kiss_icp import KissICP

            kiss_config = KISSConfig()
            kiss_config.mapping.voxel_size = self.config.voxel_size
            kiss_config.mapping.max_points_per_voxel = self.config.max_points_per_voxel
            kiss_config.adaptive_threshold.initial_threshold = self.config.initial_threshold
            kiss_config.data.max_range = self.config.max_range
            kiss_config.data.min_range = self.config.min_range
            kiss_config.data.deskew = self.config.deskew

            self._pipeline = KissICP(config=kiss_config)
            self._backend = "kiss_icp"
            return
        except ImportError:
            pass

        # Try kiss-slam (if specifically installed)
        try:
            from kiss_slam import KISSSlam
            self._pipeline = KISSSlam(
                voxel_size=self.config.voxel_size,
                max_range=self.config.max_range,
                min_range=self.config.min_range,
            )
            self._backend = "kiss_slam"
            return
        except ImportError:
            pass

        # Fallback to simple ICP (no external dependency)
        self._backend = "simple_icp"
        self._pipeline = SimpleICPOdometry(self.config)

    def reset(self) -> None:
        """Reset SLAM state for a new sequence."""
        self._poses = {}
        self._current_pose = np.eye(4)
        self._local_map_points = []
        self._frame_indices = []

        if self._backend == "kiss_icp":
            from kiss_icp.kiss_icp import KissICP
            from kiss_icp.config import KISSConfig
            kiss_config = KISSConfig()
            kiss_config.mapping.voxel_size = self.config.voxel_size
            self._pipeline = KissICP(config=kiss_config)
        elif self._backend == "simple_icp":
            self._pipeline = SimpleICPOdometry(self.config)

    def process_frame(
        self,
        frame: SLAMFrame,
    ) -> np.ndarray:
        """Process a single LiDAR frame and return the estimated pose.

        Args:
            frame: SLAMFrame with point cloud data.

        Returns:
            4x4 pose matrix (sensor to world frame).
        """
        self.load()

        points = frame.points.copy()

        # Filter by range
        ranges = np.linalg.norm(points, axis=1)
        valid_mask = (ranges >= self.config.min_range) & (ranges <= self.config.max_range)
        points = points[valid_mask]

        if len(points) < 100:
            # Not enough points, return current pose
            self._poses[frame.frame_idx] = self._current_pose.copy()
            self._frame_indices.append(frame.frame_idx)
            return self._current_pose

        if self._backend == "kiss_icp":
            # Process with kiss-icp
            timestamps = None
            if frame.timestamps is not None and self.config.deskew:
                timestamps = frame.timestamps[valid_mask]

            source, keypoints = self._pipeline.voxelize(points)
            sigma = self._pipeline.get_adaptive_threshold()

            if len(self._pipeline.local_map) > 0:
                pose = self._pipeline.register_frame(source, sigma)
            else:
                pose = np.eye(4)
                self._pipeline.local_map.update(source, np.eye(4))

            self._current_pose = pose

        elif self._backend == "simple_icp":
            self._current_pose = self._pipeline.process(points)

        self._poses[frame.frame_idx] = self._current_pose.copy()
        self._frame_indices.append(frame.frame_idx)

        return self._current_pose

    def get_result(self) -> SLAMResult:
        """Get the complete SLAM result."""
        # Build local map from stored points
        local_map = None
        if self._local_map_points:
            local_map = np.vstack(self._local_map_points)
        elif self._backend == "kiss_icp" and self._pipeline is not None:
            try:
                local_map = np.asarray(self._pipeline.local_map.point_cloud())
            except:
                pass

        return SLAMResult(
            poses=self._poses,
            local_map=local_map,
            frame_indices=self._frame_indices,
            success=len(self._poses) > 0,
        )


class SimpleICPOdometry:
    """Simple point-to-point ICP odometry as fallback.

    This is a basic implementation for when kiss-icp is not available.
    It uses iterative closest point with voxel downsampling.
    """

    def __init__(self, config: KISSSLAMConfig) -> None:
        self.config = config
        self._local_map: Optional[np.ndarray] = None
        self._current_pose: np.ndarray = np.eye(4)

    def process(self, points: np.ndarray) -> np.ndarray:
        """Process a point cloud and return accumulated pose."""
        # Downsample input
        source = self._voxel_downsample(points, self.config.voxel_size)

        if self._local_map is None:
            # First frame - initialize map
            self._local_map = source
            return self._current_pose

        # Run ICP to get relative transform
        delta_pose = self._icp(source, self._local_map)

        # Update accumulated pose
        self._current_pose = self._current_pose @ delta_pose

        # Transform source to world and add to map
        source_world = self._transform_points(source, self._current_pose)
        self._update_local_map(source_world)

        return self._current_pose

    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Downsample points using voxel grid."""
        if len(points) == 0:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Use dictionary to keep one point per voxel
        voxel_dict = {}
        for i, vidx in enumerate(voxel_indices):
            key = tuple(vidx)
            if key not in voxel_dict:
                voxel_dict[key] = points[i]

        return np.array(list(voxel_dict.values()))

    def _icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Simple point-to-point ICP.

        Returns 4x4 transformation matrix from source to target.
        """
        pose = np.eye(4)

        for iteration in range(self.config.max_num_iterations):
            # Transform source by current estimate
            source_transformed = self._transform_points(source, pose)

            # Find correspondences (nearest neighbors)
            correspondences = self._find_correspondences(source_transformed, target)

            if len(correspondences) < 10:
                break

            src_pts = source_transformed[correspondences[:, 0]]
            tgt_pts = target[correspondences[:, 1]]

            # Compute optimal rigid transform
            delta = self._compute_rigid_transform(src_pts, tgt_pts)

            # Update pose
            pose = delta @ pose

            # Check convergence
            translation = np.linalg.norm(delta[:3, 3])
            rotation = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))

            if translation < self.config.convergence_threshold and rotation < 0.001:
                break

        return pose

    def _find_correspondences(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Find nearest neighbor correspondences."""
        correspondences = []
        threshold = self.config.initial_threshold

        for i, src_pt in enumerate(source):
            # Brute force nearest neighbor (simple but slow)
            distances = np.linalg.norm(target - src_pt, axis=1)
            nearest_idx = np.argmin(distances)
            min_dist = distances[nearest_idx]

            if min_dist < threshold:
                correspondences.append([i, nearest_idx])

        return np.array(correspondences) if correspondences else np.zeros((0, 2), dtype=int)

    def _compute_rigid_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal rigid transform using SVD."""
        # Center the points
        src_centroid = np.mean(source, axis=0)
        tgt_centroid = np.mean(target, axis=0)

        src_centered = source - src_centroid
        tgt_centered = target - tgt_centroid

        # Compute rotation using SVD
        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = tgt_centroid - R @ src_centroid

        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _transform_points(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Apply 4x4 transform to points."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        return (R @ points.T).T + t

    def _update_local_map(self, new_points: np.ndarray) -> None:
        """Update local map with new points."""
        if self._local_map is None:
            self._local_map = new_points
        else:
            # Combine and downsample
            combined = np.vstack([self._local_map, new_points])
            self._local_map = self._voxel_downsample(combined, self.config.voxel_size)

            # Keep map size manageable
            max_map_points = 100000
            if len(self._local_map) > max_map_points:
                indices = np.random.choice(len(self._local_map), max_map_points, replace=False)
                self._local_map = self._local_map[indices]


def estimate_poses_kiss_slam(
    frames: Sequence[SLAMFrame],
    config: Optional[KISSSLAMConfig] = None,
) -> SLAMResult:
    """Convenience function to estimate poses for a sequence of LiDAR frames.

    Args:
        frames: Sequence of SLAMFrame objects.
        config: Optional KISS-SLAM configuration.

    Returns:
        SLAMResult with estimated poses.
    """
    wrapper = KISSSLAMWrapper(config)

    for frame in frames:
        wrapper.process_frame(frame)

    return wrapper.get_result()
