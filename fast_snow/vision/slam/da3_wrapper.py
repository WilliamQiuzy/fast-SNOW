"""DepthAnything v3 wrapper for depth estimation and camera pose recovery."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from fast_snow.data.calibration.camera_model import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraModel,
)

# Add DA3 source to path
_DA3_SRC = Path(__file__).resolve().parents[1] / "da3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))


@dataclass(frozen=True)
class DA3Config:
    """Configuration for DA3 depth estimation."""
    model_name: str = "da3-large"
    device: str = "cuda"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"


@dataclass(frozen=True)
class DA3FrameResult:
    """Per-frame result from DA3 inference."""
    depth_map: np.ndarray           # (H_proc, W_proc) metric depth in meters
    intrinsics_3x3: np.ndarray      # (3, 3) camera intrinsics at processed resolution
    extrinsics_4x4: np.ndarray      # (4, 4) world-to-camera transform
    confidence: Optional[np.ndarray] # (H_proc, W_proc) confidence map
    processed_image: Optional[np.ndarray]  # (H_proc, W_proc, 3) uint8
    original_image_size: Tuple[int, int]   # (orig_W, orig_H)
    processed_image_size: Tuple[int, int]  # (proc_W, proc_H)
    camera: CameraModel             # for backward compatibility with MapAnythingResult


def _build_camera_model(
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> CameraModel:
    """Build CameraModel from DA3 intrinsics/extrinsics.

    Args:
        intrinsics: (3, 3) camera intrinsics matrix.
        extrinsics: (4, 4) or (3, 4) world-to-camera transform.
        image_size: (width, height).
    """
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    if extrinsics.shape == (3, 4):
        R_wc = extrinsics[:3, :3]
        t_wc = extrinsics[:3, 3]
    else:
        R_wc = extrinsics[:3, :3]
        t_wc = extrinsics[:3, 3]

    return CameraModel(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        extrinsics=CameraExtrinsics(rotation=R_wc, translation=t_wc),
        image_size=image_size,
    )


def _ensure_4x4(ext: np.ndarray) -> np.ndarray:
    """Ensure extrinsics is (4, 4)."""
    if ext.shape == (3, 4):
        mat = np.eye(4, dtype=ext.dtype)
        mat[:3, :] = ext
        return mat
    return ext


def run_da3(
    image_paths: List[Path],
    config: DA3Config,
) -> List[DA3FrameResult]:
    """Run DA3 inference on a list of images.

    Args:
        image_paths: Paths to input images.
        config: DA3 configuration.

    Returns:
        List of DA3FrameResult, one per input image.
    """
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "DepthAnything3 is not available. "
            "Install with: cd fast_snow/vision/da3 && pip install -e ."
        ) from exc
    from PIL import Image as PILImage

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = DepthAnything3(model_name=config.model_name).to(device)
    model.eval()

    # Get original image sizes before DA3 processing
    original_sizes = []
    for p in image_paths:
        img = PILImage.open(str(p))
        original_sizes.append((img.width, img.height))  # (W, H)

    prediction = model.inference(
        image=[str(p) for p in image_paths],
        process_res=config.process_res,
        process_res_method=config.process_res_method,
    )

    n_frames = prediction.depth.shape[0]
    proc_h, proc_w = prediction.depth.shape[1], prediction.depth.shape[2]

    results: List[DA3FrameResult] = []
    for i in range(n_frames):
        depth_i = prediction.depth[i]  # (H, W)

        # Intrinsics
        if prediction.intrinsics is not None:
            intr_i = prediction.intrinsics[i]  # (3, 3)
        else:
            # Fallback: estimate intrinsics from image size
            intr_i = np.array([
                [proc_w, 0.0, proc_w / 2.0],
                [0.0, proc_w, proc_h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        # Extrinsics (world-to-camera)
        if prediction.extrinsics is not None:
            ext_i = _ensure_4x4(prediction.extrinsics[i])
        else:
            ext_i = np.eye(4, dtype=np.float64)

        # Confidence
        conf_i = prediction.conf[i] if prediction.conf is not None else None

        # Processed image
        proc_img_i = prediction.processed_images[i] if prediction.processed_images is not None else None

        camera = _build_camera_model(intr_i, ext_i, (proc_w, proc_h))

        results.append(DA3FrameResult(
            depth_map=depth_i,
            intrinsics_3x3=intr_i,
            extrinsics_4x4=ext_i,
            confidence=conf_i,
            processed_image=proc_img_i,
            original_image_size=original_sizes[i],
            processed_image_size=(proc_w, proc_h),
            camera=camera,
        ))

    return results


def da3_depth_to_points(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> np.ndarray:
    """Backproject depth map to 3D world-frame points.

    Args:
        depth_map: (H, W) metric depth in meters.
        intrinsics: (3, 3) camera intrinsics.
        extrinsics: (4, 4) world-to-camera transform.
        mask: Optional (H, W) boolean mask. Only backproject pixels where True.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        (N, 3) array of 3D points in world frame.
    """
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    # Depth validity mask
    valid = (depth_map > min_depth) & (depth_map < max_depth) & np.isfinite(depth_map)
    if mask is not None:
        valid = valid & mask

    z = depth_map[valid]
    u = uu[valid]
    v = vv[valid]

    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Backproject to camera frame
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    points_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)  # (N, 4)

    # Transform to world frame: world = inv(extrinsics) @ cam
    ext_4x4 = _ensure_4x4(extrinsics)
    cam_to_world = np.linalg.inv(ext_4x4)
    points_world = (cam_to_world @ points_cam.T).T[:, :3]  # (N, 3)

    return points_world
