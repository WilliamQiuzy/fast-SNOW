"""MapAnything wrapper for image-only reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import numpy as np

from fast_snow.data.calibration.camera_model import CameraExtrinsics, CameraIntrinsics, CameraModel

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPANYTHING_ROOT = REPO_ROOT / "data" / "MapAnything"
if MAPANYTHING_ROOT.exists() and str(MAPANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPANYTHING_ROOT))


@dataclass(frozen=True)
class MapAnythingConfig:
    model_id: str = "facebook/map-anything"
    device: str = "cuda"
    memory_efficient_inference: bool = False
    use_amp: bool = True
    amp_dtype: str = "bf16"
    apply_mask: bool = True
    mask_edges: bool = True
    apply_confidence_mask: bool = False
    confidence_percentile: int = 10


@dataclass(frozen=True)
class MapAnythingResult:
    points_xyz: np.ndarray
    camera: CameraModel


def _to_world_points(pred: dict) -> np.ndarray:
    pts3d = pred["pts3d"]
    mask = pred.get("mask")
    if isinstance(pts3d, np.ndarray):
        pts = pts3d
    else:
        pts = pts3d.detach().cpu().numpy()
    if pts.ndim == 4:
        pts = pts[0]
    if mask is None:
        return pts.reshape(-1, 3)
    if not isinstance(mask, np.ndarray):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 4:
        mask = mask[0]
    mask = mask.squeeze(-1).astype(bool)
    return pts[mask]


def _camera_from_pred(pred: dict, image_size: Tuple[int, int]) -> CameraModel:
    intr = pred["intrinsics"]
    pose = pred["camera_poses"]
    if not isinstance(intr, np.ndarray):
        intr = intr.detach().cpu().numpy()
    if not isinstance(pose, np.ndarray):
        pose = pose.detach().cpu().numpy()
    if intr.ndim == 3:
        intr = intr[0]
    if pose.ndim == 3:
        pose = pose[0]

    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    # camera_poses are cam2world, convert to world2cam
    R_cw = pose[:3, :3]
    t_cw = pose[:3, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    return CameraModel(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        extrinsics=CameraExtrinsics(rotation=R_wc, translation=t_wc),
        image_size=(image_size[1], image_size[0]),
    )


def run_map_anything(
    image_paths: List[Path],
    config: MapAnythingConfig,
) -> List[MapAnythingResult]:
    """Run MapAnything inference on a list of image paths."""
    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MapAnything is not available. Install it in a Python 3.10+ env."
        ) from exc

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = MapAnything.from_pretrained(config.model_id).to(device)
    views = load_images([str(p) for p in image_paths])

    predictions = model.infer(
        views,
        memory_efficient_inference=config.memory_efficient_inference,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        apply_mask=config.apply_mask,
        mask_edges=config.mask_edges,
        apply_confidence_mask=config.apply_confidence_mask,
        confidence_percentile=config.confidence_percentile,
    )

    results: List[MapAnythingResult] = []
    for pred in predictions:
        points_xyz = _to_world_points(pred)
        img = pred["img_no_norm"]
        if not isinstance(img, np.ndarray):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        camera = _camera_from_pred(pred, image_size=img.shape[:2])
        results.append(MapAnythingResult(points_xyz=points_xyz, camera=camera))

    return results
