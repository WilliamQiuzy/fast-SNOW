"""SAM2 segmentation wrapper.

This module provides interfaces for both single-image and video prediction
using SAM2. The video predictor supports temporal mask propagation.

Paper reference: SAM2 uses a streaming memory architecture where:
- init_state() initializes the video inference state
- add_new_points_or_box() adds prompts at specific frames
- propagate_in_video() propagates masks to other frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np


@dataclass(frozen=True)
class SAM2Config:
    """Configuration for SAM2 models."""
    model_name: str = "hiera_large"
    device: str = "cuda"
    # Video predictor specific
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True


@dataclass(frozen=True)
class SAM2Mask:
    """Single mask prediction result."""
    mask: np.ndarray  # (H, W) bool
    score: float


@dataclass
class VideoMaskResult:
    """Mask propagation result for video."""
    frame_idx: int
    object_id: int
    mask: np.ndarray  # (H, W) bool
    score: float


@dataclass
class VideoPropagationResult:
    """Complete video propagation result."""
    # frame_idx -> object_id -> mask
    masks: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    scores: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def get_mask(self, frame_idx: int, object_id: int) -> Optional[np.ndarray]:
        """Get mask for specific frame and object."""
        if frame_idx in self.masks and object_id in self.masks[frame_idx]:
            return self.masks[frame_idx][object_id]
        return None

    def get_all_masks_for_frame(self, frame_idx: int) -> Dict[int, np.ndarray]:
        """Get all object masks for a specific frame."""
        return self.masks.get(frame_idx, {})


class SAM2Wrapper:
    """Wrapper around SAM2 for point-prompted segmentation (single image)."""

    def __init__(self, config: SAM2Config) -> None:
        self.config = config
        self._sam2 = None

    def load(self) -> None:
        """Lazy-load SAM2 backend if available."""
        if self._sam2 is not None:
            return
        try:
            from fast_snow.vision.sam2.sam2_image_predictor import SAM2ImagePredictor
            from fast_snow.vision.sam2.build_sam import build_sam2
        except ImportError:
            try:
                from fast_snow.vision.sam2 import Sam2Predictor  # type: ignore
                self._sam2 = Sam2Predictor(
                    model_name=self.config.model_name,
                    device=self.config.device
                )
                return
            except Exception as exc:
                raise ImportError(
                    "SAM2 is not installed. Install from https://github.com/facebookresearch/sam2"
                ) from exc

        # Use official SAM2 API
        checkpoint_map = {
            "hiera_tiny": "sam2.1_hiera_tiny.pt",
            "hiera_small": "sam2.1_hiera_small.pt",
            "hiera_base_plus": "sam2.1_hiera_base_plus.pt",
            "hiera_large": "sam2.1_hiera_large.pt",
        }
        config_map = {
            "hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        checkpoint = checkpoint_map.get(self.config.model_name, "sam2.1_hiera_large.pt")
        model_cfg = config_map.get(self.config.model_name, "configs/sam2.1/sam2.1_hiera_l.yaml")

        sam2_model = build_sam2(model_cfg, checkpoint, device=self.config.device)
        self._sam2 = SAM2ImagePredictor(sam2_model)

    def predict(
        self,
        image: np.ndarray,
        point_prompts: Sequence[Tuple[int, int]],
        labels: Optional[Sequence[int]] = None,
    ) -> List[SAM2Mask]:
        """Predict masks for a single image given point prompts.

        Args:
            image: (H, W, 3) uint8 RGB image.
            point_prompts: list of (x, y) points in pixel coordinates.
            labels: optional list of labels (1 for foreground, 0 for background).

        Returns:
            List of SAM2Mask with predicted masks and scores.
        """
        self.load()
        if labels is None:
            labels = [1] * len(point_prompts)
        if len(point_prompts) == 0:
            return []

        points = np.array(point_prompts, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        # Set image first for official API
        if hasattr(self._sam2, 'set_image'):
            self._sam2.set_image(image)
            masks, scores, _ = self._sam2.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=True,
            )
        else:
            # Fallback for alternative API
            masks, scores, _ = self._sam2.predict(
                image=image,
                point_coords=points,
                point_labels=point_labels,
            )
        return [SAM2Mask(mask=m.astype(bool), score=float(s)) for m, s in zip(masks, scores)]


class SAM2VideoPredictor:
    """SAM2 Video Predictor for temporal mask propagation.

    This implements the streaming memory architecture from SAM2 paper:
    1. Initialize video state with init_state()
    2. Add prompts at keyframes with add_new_points() or add_new_box()
    3. Propagate masks through video with propagate_in_video()

    The predictor maintains a memory bank of object tokens that enables
    consistent tracking across frames.
    """

    def __init__(self, config: SAM2Config) -> None:
        self.config = config
        self._predictor = None
        self._inference_state: Optional[Any] = None
        self._frame_names: List[str] = []
        self._video_dir: Optional[str] = None

    def load(self) -> None:
        """Lazy-load SAM2 video predictor."""
        if self._predictor is not None:
            return
        try:
            from fast_snow.vision.sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 video predictor not available. "
                "Install from https://github.com/facebookresearch/sam2"
            ) from exc

        checkpoint_map = {
            "hiera_tiny": "sam2.1_hiera_tiny.pt",
            "hiera_small": "sam2.1_hiera_small.pt",
            "hiera_base_plus": "sam2.1_hiera_base_plus.pt",
            "hiera_large": "sam2.1_hiera_large.pt",
        }
        config_map = {
            "hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        checkpoint = checkpoint_map.get(self.config.model_name, "sam2.1_hiera_large.pt")
        model_cfg = config_map.get(self.config.model_name, "configs/sam2.1/sam2.1_hiera_l.yaml")

        self._predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint,
            device=self.config.device,
        )

    def init_state(
        self,
        video_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        frame_names: Optional[List[str]] = None,
        offload_video_to_cpu: Optional[bool] = None,
        offload_state_to_cpu: Optional[bool] = None,
    ) -> None:
        """Initialize video inference state.

        Args:
            video_path: Path to video directory containing JPEG frames.
            frames: Alternative: list of (H, W, 3) uint8 RGB frames.
            frame_names: Optional names for frames (used for video_path mode).
            offload_video_to_cpu: Offload video tensors to CPU to save GPU memory.
            offload_state_to_cpu: Offload state tensors to CPU.
        """
        self.load()

        offload_video = offload_video_to_cpu if offload_video_to_cpu is not None else self.config.offload_video_to_cpu
        offload_state = offload_state_to_cpu if offload_state_to_cpu is not None else self.config.offload_state_to_cpu

        if video_path is not None:
            # Load from directory of JPEG frames
            self._video_dir = video_path
            self._inference_state = self._predictor.init_state(
                video_path=video_path,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
            )
            if frame_names:
                self._frame_names = frame_names
        elif frames is not None:
            # Load from numpy arrays - need to handle this specially
            # SAM2 expects video_path, so we may need to write temp files
            # or use the internal _load_video_frames method
            import tempfile
            import os
            from PIL import Image

            self._video_dir = tempfile.mkdtemp(prefix="sam2_video_")
            self._frame_names = []
            for i, frame in enumerate(frames):
                fname = f"{i:06d}.jpg"
                self._frame_names.append(fname)
                img = Image.fromarray(frame)
                img.save(os.path.join(self._video_dir, fname))

            self._inference_state = self._predictor.init_state(
                video_path=self._video_dir,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
            )
        else:
            raise ValueError("Must provide either video_path or frames")

    def reset_state(self) -> None:
        """Reset the inference state to start fresh."""
        if self._inference_state is not None:
            self._predictor.reset_state(self._inference_state)

    def add_new_points(
        self,
        frame_idx: int,
        object_id: int,
        points: Sequence[Tuple[int, int]],
        labels: Sequence[int],
        clear_old_points: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Add point prompts for an object at a specific frame.

        Args:
            frame_idx: Frame index to add prompts.
            object_id: Unique ID for this object (will be tracked across frames).
            points: List of (x, y) point coordinates.
            labels: List of labels (1=foreground, 0=background).
            clear_old_points: Whether to clear previous points for this object.

        Returns:
            Tuple of (mask, score) for the prompted frame.
        """
        if self._inference_state is None:
            raise RuntimeError("Must call init_state() before adding points")

        points_arr = np.array(points, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points_arr,
            labels=labels_arr,
            clear_old_points=clear_old_points,
        )

        # Get mask for the current object
        obj_idx = list(out_obj_ids).index(object_id)
        mask_logits = out_mask_logits[obj_idx]
        mask = (mask_logits > 0.0).cpu().numpy().squeeze()
        score = float(mask_logits.sigmoid().mean().cpu())

        return mask, score

    def add_new_box(
        self,
        frame_idx: int,
        object_id: int,
        box: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, float]:
        """Add box prompt for an object at a specific frame.

        Args:
            frame_idx: Frame index to add prompt.
            object_id: Unique ID for this object.
            box: Bounding box (x1, y1, x2, y2).

        Returns:
            Tuple of (mask, score) for the prompted frame.
        """
        if self._inference_state is None:
            raise RuntimeError("Must call init_state() before adding box")

        box_arr = np.array(box, dtype=np.float32)

        _, out_obj_ids, out_mask_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            box=box_arr,
        )

        obj_idx = list(out_obj_ids).index(object_id)
        mask_logits = out_mask_logits[obj_idx]
        mask = (mask_logits > 0.0).cpu().numpy().squeeze()
        score = float(mask_logits.sigmoid().mean().cpu())

        return mask, score

    def propagate_in_video(
        self,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ) -> VideoPropagationResult:
        """Propagate masks through the video.

        Args:
            start_frame_idx: Starting frame for propagation. If None, starts from
                the frame with prompts.
            max_frame_num_to_track: Maximum number of frames to propagate.
            reverse: If True, propagate backwards in time.

        Returns:
            VideoPropagationResult containing masks for all frames and objects.
        """
        if self._inference_state is None:
            raise RuntimeError("Must call init_state() before propagating")

        result = VideoPropagationResult()

        propagate_kwargs = {"inference_state": self._inference_state}
        if start_frame_idx is not None:
            propagate_kwargs["start_frame_idx"] = start_frame_idx
        if max_frame_num_to_track is not None:
            propagate_kwargs["max_frame_num_to_track"] = max_frame_num_to_track
        if reverse:
            propagate_kwargs["reverse"] = reverse

        for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
            **propagate_kwargs
        ):
            result.masks[frame_idx] = {}
            result.scores[frame_idx] = {}

            for obj_idx, obj_id in enumerate(obj_ids):
                logits = mask_logits[obj_idx]
                mask = (logits > 0.0).cpu().numpy().squeeze()
                score = float(logits.sigmoid().mean().cpu())

                result.masks[frame_idx][obj_id] = mask
                result.scores[frame_idx][obj_id] = score

        return result

    def propagate_bidirectional(
        self,
        prompt_frame_idx: int,
    ) -> VideoPropagationResult:
        """Propagate masks both forward and backward from prompt frame.

        Args:
            prompt_frame_idx: The frame where prompts were added.

        Returns:
            Combined VideoPropagationResult with masks for all frames.
        """
        # Forward propagation
        forward_result = self.propagate_in_video(
            start_frame_idx=prompt_frame_idx,
            reverse=False,
        )

        # Backward propagation
        backward_result = self.propagate_in_video(
            start_frame_idx=prompt_frame_idx,
            reverse=True,
        )

        # Merge results (backward takes precedence for overlapping frames)
        combined = VideoPropagationResult()
        combined.masks = {**forward_result.masks, **backward_result.masks}
        combined.scores = {**forward_result.scores, **backward_result.scores}

        return combined


def segment_cluster_sam2(
    image: np.ndarray,
    point_prompts: List[Tuple[int, int]],
    config: Optional[SAM2Config] = None,
) -> Optional[SAM2Mask]:
    """Convenience function to segment a cluster using SAM2.

    Args:
        image: (H, W, 3) uint8 RGB image.
        point_prompts: List of (x, y) points from projected cluster centroids.
        config: Optional SAM2 configuration.

    Returns:
        Best SAM2Mask or None if no valid mask found.
    """
    if config is None:
        config = SAM2Config()

    wrapper = SAM2Wrapper(config)
    masks = wrapper.predict(image, point_prompts)

    if not masks:
        return None

    # Return highest scoring mask
    return max(masks, key=lambda m: m.score)
