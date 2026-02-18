"""End-to-end SNOW pipeline.

This module implements the complete SNOW pipeline as described in the paper:
1. Video frames -> MapAnything -> 3D point clouds
2. Point clouds -> HDBSCAN clustering -> Object segmentation
3. Objects -> Cross-frame tracking -> Temporal tracks
4. Tracks -> 4D Scene Graph -> Text serialization
5. Text 4DSG -> VLM -> Answer

Usage:
    config = SNOWConfig()
    pipeline = SNOWPipeline(config)
    result = pipeline.process_video(video_path, question)
    print(result.answer)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

from fast_snow.engine.config.snow_config import SNOWConfig, SAM2Config
from fast_snow.data.schema import FrameData, SampleManifest
from fast_snow.vision.perception.clustering.hdbscan import HDBSCANConfig, ClusterResult
from fast_snow.vision.perception.clustering.cluster_frame import ClusteredFrame, cluster_frame
from fast_snow.vision.perception.segmentation.phase2_segment import SegmentationResult, run_phase2
from fast_snow.vision.perception.segmentation.sam2_wrapper import SAM2Wrapper
from fast_snow.vision.perception.association.phase3_points import Phase3Result, run_phase3
from fast_snow.vision.perception.refinement.hhop_filter import HHopConfig, filter_implausible
from fast_snow.vision.perception.slam.kiss_slam_wrapper import KISSSLAMWrapper, SLAMFrame
from fast_snow.reasoning.tokens.step_encoding import STEPToken, build_step_token
from fast_snow.reasoning.graph.scene_graph import SceneGraph, SceneGraphConfig, build_scene_graph
from fast_snow.reasoning.graph.object_tracker import ObjectTracker, TrackerConfig, Track
from fast_snow.reasoning.graph.temporal_linking import TemporalTrack, TemporalWindow, build_temporal_window_from_tracker
from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph, build_four_d_scene_graph
from fast_snow.reasoning.vlm.prompt_builder import (
    serialize_4dsg,
    serialize_4dsg_strict,
    PromptConfig,
    Phase7SerializationConfig,
    SerializationFormat,
)


@dataclass
class PointCloudQuality:
    """Quality metrics for a point cloud from MapAnything."""
    num_points: int
    min_depth: float
    max_depth: float
    mean_depth: float
    std_depth: float
    num_filtered: int  # Points removed by quality filter
    is_valid: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_idx: int
    points_xyz: np.ndarray
    cluster_result: ClusterResult
    clustered_frame: Optional[ClusteredFrame]
    step_tokens: Dict[int, STEPToken]
    scene_graph: SceneGraph
    segmentation_result: Optional[SegmentationResult] = None
    phase3_result: Optional[Phase3Result] = None
    point_cloud_quality: Optional[PointCloudQuality] = None


@dataclass
class SNOWResult:
    """Result from the SNOW pipeline."""
    answer: str
    four_dsg: FourDSceneGraph
    scene_text: str
    tracks: Dict[int, Track]
    frame_results: List[FrameResult] = field(default_factory=list)


class SNOWPipeline:
    """End-to-end SNOW pipeline.

    This class orchestrates all components of SNOW:
    - MapAnything for image-to-3D reconstruction
    - HDBSCAN for point cloud clustering
    - STEP token encoding
    - H-hop geometric validation
    - Object tracking across frames
    - 4DSG construction
    - VLM inference
    """

    def __init__(self, config: Optional[SNOWConfig] = None):
        self.config = config or SNOWConfig()
        self._vlm_client = None
        self._mapanything_model = None
        self._sam2_wrapper: Optional[SAM2Wrapper] = None
        self._slam_wrapper: Optional[KISSSLAMWrapper] = None

    def _get_hdbscan_config(self) -> HDBSCANConfig:
        cfg = self.config.perception.hdbscan
        return HDBSCANConfig(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            cluster_selection_epsilon=cfg.cluster_selection_epsilon,
            metric=cfg.metric,
            cluster_selection_method=cfg.cluster_selection_method,
            allow_single_cluster=cfg.allow_single_cluster,
        )

    def _get_sam2_config(self) -> SAM2Config:
        cfg = self.config.perception.sam2
        return SAM2Config(
            model_name=cfg.model_name,
            device=cfg.device,
            offload_video_to_cpu=cfg.offload_video_to_cpu,
            offload_state_to_cpu=cfg.offload_state_to_cpu,
            multimask_output=cfg.multimask_output,
            min_mask_score=cfg.min_mask_score,
        )

    def _get_hhop_config(self) -> HHopConfig:
        cfg = self.config.perception.refinement.hhop
        return HHopConfig(
            max_extent=cfg.max_extent,
            max_sigma=cfg.max_sigma,
            max_aspect_ratio=cfg.max_aspect_ratio,
            max_velocity=cfg.max_velocity,
            max_size_change_ratio=cfg.max_size_change_ratio,
            min_points=cfg.min_points,
        )

    def _get_tracker_config(self) -> TrackerConfig:
        cfg = self.config.graph.tracker
        return TrackerConfig(
            geometric_weight=cfg.geometric_weight,
            semantic_weight=cfg.semantic_weight,
            max_centroid_distance=cfg.max_centroid_distance,
            max_association_cost=cfg.max_association_cost,
            shape_distance_weight=cfg.shape_distance_weight,
            max_age=cfg.max_age,
        )

    def _get_scene_graph_config(self) -> SceneGraphConfig:
        cfg = self.config.graph.scene_graph
        return SceneGraphConfig(
            max_edge_distance=cfg.max_edge_distance,
            include_relations=cfg.include_relations,
            include_ego_relative=cfg.include_ego_relative,
            min_edge_distance=cfg.min_edge_distance,
        )

    def _get_prompt_config(self) -> PromptConfig:
        cfg = self.config.vlm
        fmt = SerializationFormat.JSON if self.config.use_phase7_strict else SerializationFormat.TEXT
        return PromptConfig(
            max_tracks=cfg.max_tracks,
            max_frames_per_track=cfg.max_frames_per_track,
            include_ego=cfg.include_ego,
            include_relations=cfg.include_relations,
            max_relations_per_frame=cfg.max_relations_per_frame,
            relation_distance_threshold=cfg.relation_distance_threshold,
            format=fmt,
            precision=cfg.precision,
        )

    def _lazy_load_vlm(self):
        """Lazy load VLM client."""
        if self._vlm_client is not None:
            return

        import os
        try:
            from google import genai
            api_key = os.environ.get("GOOGLE_AI_API_KEY")
            if api_key:
                self._vlm_client = genai.Client(api_key=api_key)
            else:
                raise ValueError("GOOGLE_AI_API_KEY environment variable required")
        except ImportError:
            raise ImportError("google-genai is required for VLM inference")

    @staticmethod
    def _resolve_vlm_model_name(model_name: str) -> str:
        """Normalize model aliases for google-genai backend."""
        aliases = {
            "gemma3-4b-it": "gemma-3-4b-it",
            "Gemma3-4B-IT": "gemma-3-4b-it",
            "google/gemma-3-4b-it": "gemma-3-4b-it",
        }
        return aliases.get(model_name, model_name)

    def _lazy_load_mapanything(self):
        """Lazy load MapAnything model."""
        if self._mapanything_model is not None:
            return

        try:
            from fast_snow.vision.slam.map_anything import MapAnythingConfig
            self._mapanything_config = MapAnythingConfig(
                device=self.config.data.mapanything_device,
                use_amp=self.config.data.mapanything_use_amp,
            )
        except ImportError:
            raise ImportError("MapAnything is required for image-to-3D reconstruction")

    def _lazy_load_sam2(self) -> SAM2Wrapper:
        """Lazy-load SAM2 wrapper (loads model on first call)."""
        if self._sam2_wrapper is None:
            from fast_snow.vision.perception.segmentation.sam2_wrapper import (
                SAM2Config as WrapperSAM2Config,
            )
            wrapper_cfg = WrapperSAM2Config(
                model_name=self.config.perception.sam2.model_name,
                device=self.config.perception.sam2.device,
            )
            self._sam2_wrapper = SAM2Wrapper(wrapper_cfg)
            self._sam2_wrapper.load()
        return self._sam2_wrapper

    def _lazy_load_slam(self) -> KISSSLAMWrapper:
        """Lazy-load SLAM wrapper for ego-motion estimation."""
        if self._slam_wrapper is None:
            slam_config_obj = self.config.perception.slam

            from fast_snow.vision.perception.slam.kiss_slam_wrapper import KISSSLAMConfig
            kiss_config = KISSSLAMConfig(
                max_range=slam_config_obj.max_range,
                min_range=slam_config_obj.min_range,
                voxel_size=slam_config_obj.voxel_size,
                max_points_per_voxel=slam_config_obj.max_points_per_voxel,
                initial_threshold=slam_config_obj.initial_threshold,
                min_motion_threshold=slam_config_obj.min_motion_threshold,
                deskew=slam_config_obj.deskew,
                max_num_iterations=slam_config_obj.max_num_iterations,
                convergence_threshold=slam_config_obj.convergence_threshold,
                use_adaptive_threshold=slam_config_obj.use_adaptive_threshold,
            )
            self._slam_wrapper = KISSSLAMWrapper(kiss_config)
            self._slam_wrapper.load()
        return self._slam_wrapper

    def _run_slam_on_point_clouds(
        self,
        point_clouds: List[np.ndarray],
        use_slam: bool = False,
    ) -> Optional[Dict[int, np.ndarray]]:
        """Run SLAM on point clouds to estimate ego poses.

        Args:
            point_clouds: List of point clouds (N, 3).
            use_slam: Whether to run SLAM. If False, returns None.

        Returns:
            Dict mapping frame_idx to 4x4 pose matrix, or None if use_slam=False.

        Note:
            This is used by process_video and process_point_clouds when
            FrameData with ground truth poses is not available.
        """
        if not use_slam:
            return None

        slam = self._lazy_load_slam()
        slam.reset()

        slam_poses: Dict[int, np.ndarray] = {}
        for frame_idx, points in enumerate(point_clouds):
            if len(points) < 100:
                # Not enough points, use identity
                slam_poses[frame_idx] = np.eye(4)
                continue

            slam_frame = SLAMFrame(
                frame_idx=frame_idx,
                points=points,
            )
            pose = slam.process_frame(slam_frame)
            slam_poses[frame_idx] = pose

        return slam_poses

    def validate_point_cloud(
        self,
        points_xyz: np.ndarray,
        frame_idx: int = 0,
    ) -> tuple[np.ndarray, PointCloudQuality]:
        """Validate and filter point cloud from MapAnything.

        Checks:
        1. Minimum number of points
        2. Depth range (filters outliers)
        3. NaN/Inf values
        4. Reasonable spatial distribution

        Args:
            points_xyz: Raw point cloud (N, 3)
            frame_idx: Frame index for logging

        Returns:
            Tuple of (filtered_points, quality_metrics)
        """
        warnings = []
        original_count = len(points_xyz)

        # Check for empty point cloud
        if original_count == 0:
            return points_xyz, PointCloudQuality(
                num_points=0,
                min_depth=0.0,
                max_depth=0.0,
                mean_depth=0.0,
                std_depth=0.0,
                num_filtered=0,
                is_valid=False,
                warnings=["Empty point cloud from MapAnything"],
            )

        # Remove NaN/Inf values
        valid_mask = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[valid_mask]
        nan_removed = original_count - len(points_xyz)
        if nan_removed > 0:
            warnings.append(f"Removed {nan_removed} NaN/Inf points")

        # Compute depth (Z coordinate in camera frame, or distance from origin)
        depths = np.linalg.norm(points_xyz, axis=1)

        # Filter by depth range
        depth_mask = (
            (depths >= self.config.data.min_point_depth) &
            (depths <= self.config.data.max_point_depth)
        )
        filtered_points = points_xyz[depth_mask]
        depth_filtered = len(points_xyz) - len(filtered_points)
        if depth_filtered > 0:
            warnings.append(f"Filtered {depth_filtered} points outside depth range "
                          f"[{self.config.data.min_point_depth}, {self.config.data.max_point_depth}]m")

        # Compute quality metrics on filtered points
        if len(filtered_points) > 0:
            filtered_depths = np.linalg.norm(filtered_points, axis=1)
            min_depth = float(filtered_depths.min())
            max_depth = float(filtered_depths.max())
            mean_depth = float(filtered_depths.mean())
            std_depth = float(filtered_depths.std())
        else:
            min_depth = max_depth = mean_depth = std_depth = 0.0

        # Check minimum points
        is_valid = len(filtered_points) >= self.config.data.min_points_per_frame
        if not is_valid:
            warnings.append(f"Only {len(filtered_points)} points, need at least "
                          f"{self.config.data.min_points_per_frame}")

        # Check for degenerate cases
        if is_valid and std_depth < 0.1:
            warnings.append("Very low depth variance - possible planar scene")

        quality = PointCloudQuality(
            num_points=len(filtered_points),
            min_depth=min_depth,
            max_depth=max_depth,
            mean_depth=mean_depth,
            std_depth=std_depth,
            num_filtered=original_count - len(filtered_points),
            is_valid=is_valid,
            warnings=warnings,
        )

        if warnings:
            print(f"  Frame {frame_idx} point cloud warnings: {'; '.join(warnings)}")

        return filtered_points, quality

    def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
    ) -> List[Path]:
        """Extract frames from video and save as temporary files.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default from config)

        Returns:
            List of paths to extracted frame images
        """
        import cv2

        num_frames = num_frames or self.config.data.num_frames
        video_path = str(video_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        temp_dir = Path(tempfile.mkdtemp())
        frame_paths = []

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = temp_dir / f"frame_{i:04d}.jpg"
                cv2.imwrite(str(path), frame)
                frame_paths.append(path)

        cap.release()
        return frame_paths

    def run_mapanything(
        self,
        frame_paths: List[Path],
        validate: bool = True,
    ) -> tuple[List[np.ndarray], List[Optional[PointCloudQuality]]]:
        """Run MapAnything to get 3D point clouds from images.

        Args:
            frame_paths: List of paths to frame images
            validate: Whether to validate and filter point clouds

        Returns:
            Tuple of (point_clouds, quality_metrics) where each list has one entry per frame
        """
        self._lazy_load_mapanything()

        from fast_snow.vision.slam.map_anything import run_map_anything

        results = run_map_anything(frame_paths, self._mapanything_config)
        raw_clouds = [r.points_xyz for r in results]

        if not validate:
            # Return raw clouds with no quality info
            no_quality: List[Optional[PointCloudQuality]] = [None] * len(raw_clouds)
            return raw_clouds, no_quality

        # Validate each point cloud
        validated_clouds = []
        qualities = []
        for frame_idx, cloud in enumerate(raw_clouds):
            filtered_cloud, quality = self.validate_point_cloud(cloud, frame_idx)
            validated_clouds.append(filtered_cloud)
            qualities.append(quality)

            if not quality.is_valid:
                print(f"  WARNING: Frame {frame_idx} point cloud is invalid "
                      f"({quality.num_points} points)")

        # Summary statistics
        valid_count = sum(1 for q in qualities if q.is_valid)
        print(f"  MapAnything: {valid_count}/{len(qualities)} frames have valid point clouds")

        return validated_clouds, qualities

    def process_frame_data(
        self,
        frame: FrameData,
        seed: Optional[int] = None,
        prompts_per_cluster: int = 4,
        ego_pose: Optional[np.ndarray] = None,
    ) -> FrameResult:
        """Process a FrameData through Phase 1+ (cluster → STEP → graph).

        This is the preferred entry point when coming from Phase 0 adapters.
        It calls ``cluster_frame()`` internally so that downstream modules
        receive label-based cluster IDs and SAM2 prompt points.

        Args:
            frame: FrameData with ``points_world``.
            seed: Random seed for prompt sampling reproducibility.
            prompts_per_cluster: *m* – SAM2 prompt points per cluster (paper: 4).
            ego_pose: Optional 4×4 ego pose (overrides ``frame.ego_pose``).

        Returns:
            FrameResult with ClusteredFrame, STEP tokens, and scene graph.
        """
        ego = ego_pose if ego_pose is not None else frame.ego_pose
        return self._process_clustered(
            frame.points_world, frame.frame_idx,
            timestamp=frame.timestamp,
            seed=seed,
            prompts_per_cluster=prompts_per_cluster,
            ego_pose=ego,
            frame_data=frame,
        )

    def process_frame(
        self,
        points_xyz: np.ndarray,
        frame_idx: int,
        ego_pose: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> FrameResult:
        """Process a single frame's point cloud.

        Backward-compatible entry point that wraps raw arrays.  Internally
        delegates to ``_process_clustered`` (which uses ``cluster_frame``).

        Args:
            points_xyz: (N, 3) point cloud
            frame_idx: Frame index
            ego_pose: Optional 4x4 ego pose matrix
            seed: Random seed for prompt sampling reproducibility.

        Returns:
            FrameResult with clusters, STEP tokens, and scene graph
        """
        return self._process_clustered(
            points_xyz, frame_idx,
            seed=seed,
            ego_pose=ego_pose,
        )

    def _process_clustered(
        self,
        points_xyz: np.ndarray,
        frame_idx: int,
        *,
        timestamp: float = 0.0,
        seed: Optional[int] = None,
        prompts_per_cluster: int = 4,
        ego_pose: Optional[np.ndarray] = None,
        frame_data: Optional[FrameData] = None,
    ) -> FrameResult:
        """Shared implementation: cluster_frame → Phase 2 → STEP → H-hop → graph."""
        hdbscan_config = self._get_hdbscan_config()
        hhop_config = self._get_hhop_config()
        sg_config = self._get_scene_graph_config()

        # Step 1: Build FrameData (if not provided) and run Phase-1 clustering
        fd = frame_data if frame_data is not None else FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            points_world=points_xyz,
        )
        cf = cluster_frame(
            fd,
            config=hdbscan_config,
            prompts_per_cluster=prompts_per_cluster,
            seed=seed,
        )

        # Step 2: Phase 2 – SAM2 segmentation (when image data is available)
        seg_result: Optional[SegmentationResult] = None
        if frame_data is not None and frame_data.cameras:
            sam2 = self._lazy_load_sam2()
            sam2_config = self._get_sam2_config()
            seg_result = run_phase2(frame_data, cf, sam2, sam2_config)

        # Step 3: Phase 3 – 3D-2D Association (assign points to masks)
        # Paper: "Each 3D point is assigned to mask m_t^{k,c} if its projection
        # lies within the mask's support"
        phase3_result: Optional[Phase3Result] = None
        if seg_result is not None and frame_data is not None and frame_data.cameras:
            phase3_result = run_phase3(frame_data, seg_result)

        # Step 4: Create STEP tokens keyed by HDBSCAN label
        # Paper (Eq. 4): S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t, c_t^k, s_t^k, θ_t^k}
        # Only clusters with valid SAM2 masks AND valid point assignments
        # are included in 4DSG. Use Phase 3 object points (mask-assigned points)
        # instead of raw cluster points for strict paper alignment.
        frame_steps: Dict[int, STEPToken] = {}
        for label, prompt in cf.prompts.items():
            if len(prompt.point_indices) < self.config.min_cluster_points:
                continue

            # Skip clusters without valid SAM2 mask (strict paper alignment)
            if seg_result is None or label not in seg_result.best_masks:
                continue

            # Use Phase 3 object points if available (mask-assigned points)
            # Paper (Section 3.2): "Each 3D point is assigned to mask if its
            # projection lies within the mask's support"
            if phase3_result is not None and label in phase3_result.object_points:
                cluster_pts = phase3_result.object_points[label].points_xyz
            else:
                # Fallback to raw cluster points (for backwards compatibility)
                cluster_pts = points_xyz[prompt.point_indices]

            # Skip if no points after Phase 3 filtering
            if len(cluster_pts) < self.config.min_cluster_points:
                continue

            mask = seg_result.best_masks[label]

            # Build STEP token with single-frame temporal placeholder
            # Paper (Section 3.2): θ_t^k = (t_start, t_end) are TRACK-LEVEL
            # Phase 4: Initialize with current frame (placeholder)
            # Phase 6: Tracker will update temporal tokens with true track span
            step = build_step_token(
                mask=mask,
                points_xyz=cluster_pts,
                t_start=frame_idx,      # Placeholder: will be updated by tracker
                t_end=frame_idx,        # Placeholder: will be updated by tracker
                grid_size=self.config.step_grid_size,
                iou_threshold=self.config.step_iou_threshold,
            )
            frame_steps[label] = step

        # Step 4: H-hop geometric validation
        filtered_steps = filter_implausible(frame_steps, hhop_config)

        # Step 5: Build spatial scene graph
        scene_graph = build_scene_graph(
            filtered_steps,
            frame_idx=frame_idx,
            ego_pose=ego_pose,
            config=sg_config,
        )

        return FrameResult(
            frame_idx=frame_idx,
            points_xyz=points_xyz,
            cluster_result=cf.cluster_result,
            clustered_frame=cf,
            step_tokens=filtered_steps,
            scene_graph=scene_graph,
            segmentation_result=seg_result,
            phase3_result=phase3_result,
        )

    def build_4dsg(
        self,
        frame_results: List[FrameResult],
    ) -> tuple[FourDSceneGraph, Dict[int, Track]]:
        """Build 4D Scene Graph from frame results.

        Paper workflow (Phase 6):
        1. Track objects across frames using ObjectTracker (Hungarian matching)
        2. Update all STEP temporal tokens to reflect track-level [t_start, t_end]
        3. Extract ego poses from frame results (from SLAM or ground truth)
        4. Construct 4DSG with spatial graphs, temporal window, and ego poses

        Args:
            frame_results: List of per-frame results.

        Returns:
            Tuple of (4DSG, tracks dict).

        Note:
            This method implements the complete Phase 6 workflow:
            - Cross-frame association (Eq. 7: F_k = {S_{t-T}^k, ..., S_t^k})
            - Temporal token updates (t_start, t_end per track)
            - 4DSG construction with real ego poses
        """
        tracker_config = self._get_tracker_config()

        # Step 1: Track objects across frames using Hungarian matching
        # Paper (Section 3.3): Cross-frame association using geometric + semantic similarity
        tracker = ObjectTracker(tracker_config)
        spatial_graphs: List[SceneGraph] = []

        for fr in frame_results:
            tracker.update(fr.step_tokens, fr.frame_idx)
            spatial_graphs.append(fr.scene_graph)

        # Get ALL tracks (active + archived) for 4DSG construction
        # Paper (Eq. 7): F_k = {S_{t-T}^k, ..., S_t^k} - includes all tracks
        # within temporal window, not just currently active ones
        tracks = tracker.get_all_tracks()

        # Step 2: Build temporal window with updated STEP tokens
        # Paper requirement: All STEP tokens in a track should have consistent
        # temporal tokens (t_start, t_end) reflecting the track's full span
        temporal_window = build_temporal_window_from_tracker(tracks)

        # Step 2.5: Update original tracks with corrected temporal tokens
        # This ensures SNOWResult.tracks has updated STEP tokens, not placeholders
        updated_tracks = self._update_track_temporal_tokens(tracks, temporal_window)

        # Step 3: Extract ego poses from frame results
        # Paper (Phase 0): Ego poses come from SLAM (KISS-SLAM) or ground truth
        # Priority: frame_results ego_pose > scene_graph ego_pose > identity
        ego_poses = self._extract_ego_poses(frame_results)

        # Step 4: Construct 4DSG
        # Paper (Section 3.3): M_t = {G_t, F, E_t}
        # where G_t are spatial graphs, F are temporal tracks, E_t is ego trajectory
        four_dsg = build_four_d_scene_graph(
            spatial_graphs=spatial_graphs,
            temporal_window=temporal_window,
            ego_poses=ego_poses,
        )

        return four_dsg, updated_tracks

    def _extract_ego_poses(
        self,
        frame_results: List[FrameResult],
    ) -> Dict[int, List[float]]:
        """Extract ego poses from frame results.

        Extracts ego poses in the format expected by 4DSG: 7-element list
        [x, y, z, qx, qy, qz, qw] representing position + quaternion rotation.

        Priority:
        1. scene_graph.ego_pose (4x4 matrix from SLAM or ground truth)
        2. Identity pose if no pose available

        Args:
            frame_results: List of per-frame results.

        Returns:
            Dict mapping frame_idx to 7-element pose list.

        Note:
            If ego_pose is a 4x4 matrix, it's converted to [x,y,z,qx,qy,qz,qw].
            If no ego_pose is available, returns identity pose at origin.
        """
        ego_poses: Dict[int, List[float]] = {}

        for fr in frame_results:
            frame_idx = fr.frame_idx

            # Try to get ego pose from scene graph
            if fr.scene_graph.ego_pose is not None:
                pose_matrix = fr.scene_graph.ego_pose
                # Convert 4x4 matrix to 7-element list [x, y, z, qx, qy, qz, qw]
                ego_poses[frame_idx] = self._matrix_to_pose_list(pose_matrix)
            else:
                # Fallback: identity pose at origin
                # [x, y, z, qx, qy, qz, qw] where quaternion [0,0,0,1] is identity
                ego_poses[frame_idx] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        return ego_poses

    def _matrix_to_pose_list(self, pose_matrix: np.ndarray) -> List[float]:
        """Convert 4x4 transformation matrix to 7-element pose list.

        Args:
            pose_matrix: 4x4 transformation matrix (SE(3)).

        Returns:
            7-element list [x, y, z, qx, qy, qz, qw].

        Note:
            Uses scipy for rotation matrix to quaternion conversion.
        """
        # Extract translation
        x, y, z = pose_matrix[:3, 3]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = pose_matrix[:3, :3]

        # Convert rotation matrix to quaternion using scipy
        try:
            from scipy.spatial.transform import Rotation as R
            rotation = R.from_matrix(rotation_matrix)
            qx, qy, qz, qw = rotation.as_quat()  # Returns [x, y, z, w]
        except ImportError:
            # Fallback: identity quaternion if scipy not available
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

        return [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]

    def _update_track_temporal_tokens(
        self,
        tracks: Dict[int, Track],
        temporal_window: TemporalWindow,
    ) -> Dict[int, Track]:
        """Update original tracks with corrected temporal tokens from TemporalWindow.

        Problem: ObjectTracker.tracks contain STEP tokens with placeholder
        temporal tokens (t_start=t_end=frame_idx). Only TemporalWindow has
        updated STEP tokens with track-level [t_start, t_end].

        Solution: Replace STEP tokens in original tracks with updated ones
        from temporal_window.

        Args:
            tracks: Original tracks from ObjectTracker.
            temporal_window: TemporalWindow with updated STEP tokens.

        Returns:
            Updated tracks with corrected temporal tokens.

        Note:
            This ensures SNOWResult.tracks has correct temporal tokens,
            not placeholder ones.
        """
        from fast_snow.reasoning.tokens.step_encoding import update_temporal_token

        updated_tracks: Dict[int, Track] = {}

        for track_id, temporal_track in temporal_window.tracks.items():
            if track_id not in tracks:
                # Track exists in temporal_window but not in original tracks
                # This shouldn't happen, but handle it gracefully
                continue

            original_track = tracks[track_id]

            # Get track-level temporal span
            if not original_track.frame_indices:
                # Empty track, skip
                continue

            t_start = min(original_track.frame_indices)
            t_end = max(original_track.frame_indices)

            # Create new track with updated STEP tokens
            updated_track = Track(track_id=track_id)
            updated_track.frame_indices = original_track.frame_indices.copy()
            updated_track.age = original_track.age

            # Update all STEP tokens to have track-level temporal tokens
            updated_track.steps = [
                update_temporal_token(step, t_start=t_start, t_end=t_end)
                for step in original_track.steps
            ]

            updated_tracks[track_id] = updated_track

        return updated_tracks

    def serialize_4dsg(self, four_dsg: FourDSceneGraph) -> str:
        """Serialize 4DSG to text/JSON for VLM input.

        Uses Phase 7 strict serialization if use_phase7_strict=True in config.

        Args:
            four_dsg: The 4D scene graph

        Returns:
            Serialized text or JSON representation
        """
        if self.config.use_phase7_strict:
            # Phase 7 strict mode: use deterministic JSON serialization
            phase7_config = Phase7SerializationConfig()
            return serialize_4dsg_strict(four_dsg, phase7_config)
        else:
            # Legacy mode: use configurable text serialization
            prompt_config = self._get_prompt_config()
            return serialize_4dsg(four_dsg, prompt_config)

    def query_vlm(
        self,
        scene_text: str,
        question: str,
    ) -> str:
        """Query VLM with 4DSG scene and question.

        Args:
            scene_text: Serialized 4DSG (JSON or text)
            question: User question

        Returns:
            VLM answer

        Note:
            In Phase 7 strict mode, scene_text is JSON format.
            Legacy mode uses text format.
        """
        self._lazy_load_vlm()

        from google.genai import types

        if self.config.use_phase7_strict:
            # Phase 7 strict prompt format
            full_prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

Scene Data (JSON):
{scene_text}

Based on the scene information above, please answer the following question:
{question}

Answer with just the letter choice (A, B, C, or D):"""
        else:
            # Legacy prompt format
            full_prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

{scene_text}

Based on the scene information above, answer the following question:

{question}

Think step by step about the spatial relationships and provide your final answer."""

        # Phase 7 strict mode: Force fixed parameters per paper specification
        if self.config.use_phase7_strict:
            max_tokens = 256  # FIXED: Paper specification
            temperature = 0.0  # FIXED: Deterministic
        else:
            max_tokens = self.config.vlm.max_new_tokens
            temperature = self.config.vlm.temperature

        response = self._vlm_client.models.generate_content(
            model=self._resolve_vlm_model_name(self.config.vlm.model_name),
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

        return response.text

    def process_manifest(
        self,
        manifest: SampleManifest,
        question: Optional[str] = None,
    ) -> SNOWResult:
        """Process a SampleManifest (Phase 0 output) through the full pipeline.

        This is the canonical entry point when data comes from Phase 0
        adapters (NuScenes or VLM4D).  Each ``FrameData`` in the manifest
        is processed via ``process_frame_data`` which preserves camera
        calibration, ego pose, and uses deterministic per-frame seeds.

        Args:
            manifest: SampleManifest produced by an adapter.
            question: Question to answer.  If *None*, falls back to
                ``manifest.metadata["question"]``.

        Returns:
            SNOWResult with answer and intermediate results.
        """
        if question is None:
            question = manifest.metadata.get("question")
        if question is None:
            raise ValueError(
                "No question provided and manifest.metadata has no 'question' key"
            )

        frame_results: List[FrameResult] = []
        for frame in manifest.frames:
            if frame.num_points() == 0:
                continue
            frame_seed = self.config.seed + frame.frame_idx
            fr = self.process_frame_data(frame, seed=frame_seed)
            frame_results.append(fr)

        # Build 4DSG
        four_dsg, tracks = self.build_4dsg(frame_results)

        # Serialize 4DSG
        scene_text = self.serialize_4dsg(four_dsg)

        # Query VLM
        answer = self.query_vlm(scene_text, question)

        return SNOWResult(
            answer=answer,
            four_dsg=four_dsg,
            scene_text=scene_text,
            tracks=tracks,
            frame_results=frame_results,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        question: str,
        num_frames: Optional[int] = None,
        use_slam: bool = True,
    ) -> SNOWResult:
        """Process video through full SNOW pipeline.

        Args:
            video_path: Path to video file
            question: Question to answer
            num_frames: Number of frames to extract
            use_slam: Whether to use SLAM for ego-motion estimation (default True)

        Returns:
            SNOWResult with answer and intermediate results

        Note:
            When use_slam=True, KISS-SLAM is used to estimate ego poses from
            point clouds. This implements the paper's requirement for SLAM-based
            ego-motion estimation when ground truth poses are not available.
        """
        # Step 1: Extract frames
        frame_paths = self.extract_frames(video_path, num_frames)

        # Step 2: Get point clouds from MapAnything (with quality validation)
        point_clouds, qualities = self.run_mapanything(frame_paths)

        # Step 2.5: Run SLAM if requested (Phase 6 requirement)
        # Paper: KISS-SLAM provides ego poses when ground truth not available
        slam_poses = self._run_slam_on_point_clouds(point_clouds, use_slam=use_slam)

        # Step 3: Process each frame with SLAM-estimated ego poses
        frame_results = []
        for frame_idx, points_xyz in enumerate(point_clouds):
            quality = qualities[frame_idx] if qualities else None

            # Skip frames with invalid point clouds
            if quality is not None and not quality.is_valid:
                print(f"  Skipping frame {frame_idx} due to invalid point cloud")
                continue

            # Get ego pose from SLAM if available
            ego_pose = slam_poses.get(frame_idx) if slam_poses else None

            frame_seed = self.config.seed + frame_idx
            fr = self.process_frame(points_xyz, frame_idx, ego_pose=ego_pose, seed=frame_seed)
            fr.point_cloud_quality = quality
            frame_results.append(fr)

        # Step 4: Build 4DSG
        four_dsg, tracks = self.build_4dsg(frame_results)

        # Step 5: Serialize 4DSG
        scene_text = self.serialize_4dsg(four_dsg)

        # Step 6: Query VLM
        answer = self.query_vlm(scene_text, question)

        # Cleanup temp files
        for fp in frame_paths:
            fp.unlink(missing_ok=True)

        return SNOWResult(
            answer=answer,
            four_dsg=four_dsg,
            scene_text=scene_text,
            tracks=tracks,
            frame_results=frame_results,
        )

    def build_4dsg_from_video(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
        use_slam: bool = True,
    ) -> Tuple[FourDSceneGraph, Dict[int, Track]]:
        """Process video to 4DSG without VLM inference.

        This is the preferred method for Phase 8 evaluation to avoid
        double VLM calls. Use this when you want to control VLM inference
        separately with Phase 7 strict parameters.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default from config)
            use_slam: Whether to use SLAM for ego-motion estimation

        Returns:
            Tuple of (FourDSceneGraph, tracks dict)

        Example:
            # Phase 8 evaluation pattern
            four_dsg, tracks = pipeline.build_4dsg_from_video(video_path)
            scene_json = serialize_4dsg_strict(four_dsg, Phase7SerializationConfig())
            answer = vlm.infer_text(scene_json + question)
        """
        # Step 1: Extract frames
        frame_paths = self.extract_frames(video_path, num_frames)

        # Step 2: Get point clouds from MapAnything
        point_clouds, qualities = self.run_mapanything(frame_paths)

        # Step 2.5: Run SLAM if requested
        slam_poses = self._run_slam_on_point_clouds(point_clouds, use_slam=use_slam)

        # Step 3: Process each frame
        frame_results = []
        for frame_idx, points_xyz in enumerate(point_clouds):
            quality = qualities[frame_idx] if qualities else None

            # Skip frames with invalid point clouds
            if quality is not None and not quality.is_valid:
                logger.warning(
                    f"Skipping frame {frame_idx} due to invalid point cloud quality"
                )
                continue

            # Use SLAM pose if available, otherwise use identity
            ego_pose = (
                slam_poses[frame_idx] if slam_poses and frame_idx < len(slam_poses)
                else None
            )

            # Use frame-specific seed for deterministic HDBSCAN sampling
            frame_seed = self.config.seed + frame_idx
            frame_result = self.process_frame(
                points_xyz,
                frame_idx,
                ego_pose=ego_pose,
                seed=frame_seed,
            )
            frame_results.append(frame_result)

        # Step 4: Build 4DSG (stops here, no VLM)
        four_dsg, tracks = self.build_4dsg(frame_results)

        # Cleanup temp files
        for fp in frame_paths:
            fp.unlink(missing_ok=True)

        return four_dsg, tracks

    def process_point_clouds(
        self,
        point_clouds: List[np.ndarray],
        question: str,
        validate: bool = True,
        use_slam: bool = True,
    ) -> SNOWResult:
        """Process pre-computed point clouds (skip MapAnything).

        Args:
            point_clouds: List of (N, 3) point clouds
            question: Question to answer
            validate: Whether to validate point cloud quality
            use_slam: Whether to use SLAM for ego-motion estimation (default True)

        Returns:
            SNOWResult with answer and intermediate results

        Note:
            When use_slam=True, KISS-SLAM is used to estimate ego poses from
            point clouds. This implements the paper's requirement for SLAM-based
            ego-motion estimation when ground truth poses are not available.
        """
        # Step 1: Run SLAM if requested (Phase 6 requirement)
        slam_poses = self._run_slam_on_point_clouds(point_clouds, use_slam=use_slam)

        # Step 2: Process each frame with SLAM-estimated ego poses
        frame_results = []
        for frame_idx, points_xyz in enumerate(point_clouds):
            # Optionally validate point cloud
            quality = None
            if validate:
                points_xyz, quality = self.validate_point_cloud(points_xyz, frame_idx)
                if not quality.is_valid:
                    print(f"  Skipping frame {frame_idx} due to invalid point cloud")
                    continue

            # Get ego pose from SLAM if available
            ego_pose = slam_poses.get(frame_idx) if slam_poses else None

            frame_seed = self.config.seed + frame_idx
            fr = self.process_frame(points_xyz, frame_idx, ego_pose=ego_pose, seed=frame_seed)
            fr.point_cloud_quality = quality
            frame_results.append(fr)

        # Step 3: Build 4DSG
        four_dsg, tracks = self.build_4dsg(frame_results)

        # Step 4: Serialize 4DSG
        scene_text = self.serialize_4dsg(four_dsg)

        # Step 5: Query VLM
        answer = self.query_vlm(scene_text, question)

        return SNOWResult(
            answer=answer,
            four_dsg=four_dsg,
            scene_text=scene_text,
            tracks=tracks,
            frame_results=frame_results,
        )

    def get_quality_report(self, result: SNOWResult) -> Dict:
        """Generate a quality report from SNOW result.

        Args:
            result: SNOWResult from process_video or process_point_clouds

        Returns:
            Dict with quality statistics
        """
        qualities = [
            fr.point_cloud_quality
            for fr in result.frame_results
            if fr.point_cloud_quality is not None
        ]

        if not qualities:
            return {"status": "no_quality_data", "frames_processed": len(result.frame_results)}

        valid_count = sum(1 for q in qualities if q.is_valid)
        total_points = sum(q.num_points for q in qualities)
        total_filtered = sum(q.num_filtered for q in qualities)
        all_warnings = [w for q in qualities for w in q.warnings]

        depths = [q.mean_depth for q in qualities if q.num_points > 0]
        mean_depth = np.mean(depths) if depths else 0.0

        return {
            "status": "ok" if valid_count == len(qualities) else "partial",
            "frames_total": len(result.frame_results),
            "frames_with_quality": len(qualities),
            "frames_valid": valid_count,
            "frames_invalid": len(qualities) - valid_count,
            "total_points": total_points,
            "total_filtered": total_filtered,
            "filter_rate": total_filtered / (total_points + total_filtered) if (total_points + total_filtered) > 0 else 0.0,
            "mean_depth": float(mean_depth),
            "warnings": all_warnings,
        }
