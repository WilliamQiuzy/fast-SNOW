"""VLM4D -> FrameData adapter.

For VLM4D (image-only path) the pipeline is:

1. Extract *T* frames from the video at ~1 Hz (real timestamps).
2. Run MapAnything on all frames in a single batch so that they share one
   world coordinate system.
3. For monocular video, camera ≡ ego, so ``ego_pose = cam2world = inv(world2cam)``.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import CameraModel
from fast_snow.data.loaders.vlm4d import VLM4DSample
from fast_snow.data.schema import FrameData, SampleManifest

logger = logging.getLogger(__name__)


@dataclass
class VLM4DAdapterConfig:
    """Configuration for VLM4D frame extraction and reconstruction."""

    num_frames: int = 10  # T in paper
    target_fps: float = 1.0  # ~1 Hz sampling

    # MapAnything settings (forwarded to MapAnythingConfig)
    mapanything_device: str = "cuda"
    mapanything_use_amp: bool = True

    # Quality thresholds
    min_points: int = 100
    max_depth: float = 100.0


class VLM4DAdapter:
    """Converts VLM4D video samples to unified :class:`FrameData`."""

    def __init__(self, config: Optional[VLM4DAdapterConfig] = None):
        self.config = config or VLM4DAdapterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_to_manifest(
        self,
        sample: VLM4DSample,
        video_path: Optional[str] = None,
    ) -> SampleManifest:
        """Full pipeline: video -> frames -> MapAnything -> SampleManifest.

        Args:
            sample: VLM4D sample with QA metadata.
            video_path: Override local video path (else ``sample.video``).
        """
        from fast_snow.vision.slam.map_anything import MapAnythingConfig, run_map_anything

        vpath = video_path or sample.video

        # 1. Extract frames with real timestamps
        frame_paths, timestamps = self.extract_frames_with_timestamps(vpath)

        # 2. Run MapAnything on the full batch (shared world frame)
        ma_config = MapAnythingConfig(
            device=self.config.mapanything_device,
            use_amp=self.config.mapanything_use_amp,
        )
        ma_results = run_map_anything(frame_paths, ma_config)

        # 3. Convert to FrameData
        frames: List[FrameData] = []
        for idx, (fpath, ts, result) in enumerate(
            zip(frame_paths, timestamps, ma_results)
        ):
            fd = self._mapanything_to_frame(
                camera=result.camera,
                points_xyz=result.points_xyz,
                image_path=str(fpath),
                frame_idx=idx,
                timestamp=ts,
            )
            frames.append(fd)

        return SampleManifest(
            sample_id=sample.sample_id,
            dataset="vlm4d",
            frames=frames,
            metadata={
                "question": sample.question,
                "question_type": sample.question_type,
                "choices": sample.choices,
                "answer": sample.answer,
                "video": vpath,
            },
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_frames_with_timestamps(
        self,
        video_path: str,
    ) -> Tuple[List[Path], List[float]]:
        """Extract frames at ~``target_fps`` using real video timestamps.

        Unlike the current pipeline's ``np.linspace`` uniform-index sampling,
        this computes target timestamps first, then seeks to the nearest frame.

        Returns:
            ``(frame_paths, timestamps_in_seconds)``
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(
                f"Invalid video metadata: fps={fps}, total_frames={total_frames}"
            )

        duration = total_frames / fps

        # Target timestamps at ~1 Hz, capped at num_frames
        interval = 1.0 / self.config.target_fps
        num = min(self.config.num_frames, max(1, int(duration / interval) + 1))
        target_times = [i * interval for i in range(num)]

        temp_dir = Path(tempfile.mkdtemp(prefix="snow_vlm4d_"))
        frame_paths: List[Path] = []
        timestamps: List[float] = []

        for i, target_t in enumerate(target_times):
            fidx = min(int(target_t * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d (t=%.2fs)", fidx, target_t)
                continue

            actual_t = fidx / fps
            path = temp_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_paths.append(path)
            timestamps.append(actual_t)

        cap.release()
        return frame_paths, timestamps

    # ------------------------------------------------------------------
    # MapAnything result -> FrameData
    # ------------------------------------------------------------------

    @staticmethod
    def _mapanything_to_frame(
        camera: CameraModel,
        points_xyz: np.ndarray,
        image_path: str,
        frame_idx: int,
        timestamp: float,
    ) -> FrameData:
        """Convert a single MapAnything prediction to :class:`FrameData`.

        For monocular video ``camera ≡ ego``, so::

            ego_pose = cam2world = inv(world2cam)
        """
        R_w2c = camera.extrinsics.rotation.astype(np.float64)
        t_w2c = camera.extrinsics.translation.astype(np.float64)

        # Invert world2cam -> cam2world (= ego_pose)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        ego_pose = np.eye(4, dtype=np.float64)
        ego_pose[:3, :3] = R_c2w
        ego_pose[:3, 3] = t_c2w

        return FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            image_paths={"cam0": image_path},
            points_world=points_xyz.astype(np.float64),
            cameras={"cam0": camera},
            ego_pose=ego_pose,
        )
