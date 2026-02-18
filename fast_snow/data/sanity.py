"""Sanity-check utilities for verifying data alignment.

Three levels of verification:

1. **Per-frame / per-camera projection statistics** -- point counts, depth
   ranges, valid-projection ratios.
2. **Projection overlay** -- render projected point cloud on camera images
   (depth-coloured or uniform) for visual inspection.
3. **Manifest report** -- aggregated human-readable summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fast_snow.data.schema import FrameData, SampleManifest
from fast_snow.data.transforms.projection import project_points

logger = logging.getLogger(__name__)


# ======================================================================
# Statistics dataclasses
# ======================================================================

@dataclass
class CameraProjectionStats:
    """Projection statistics for a single camera in a single frame."""

    camera_id: str
    num_projected: int  # points that fall inside the image
    num_total: int
    valid_ratio: float
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_std: float


@dataclass
class FrameStats:
    """Aggregated statistics for one :class:`FrameData`."""

    frame_idx: int
    timestamp: float
    num_points: int
    num_cameras: int
    camera_stats: Dict[str, CameraProjectionStats] = field(default_factory=dict)


# ======================================================================
# Core validation functions
# ======================================================================

def compute_frame_stats(frame: FrameData) -> FrameStats:
    """Compute projection statistics for every camera in *frame*."""
    cam_stats: Dict[str, CameraProjectionStats] = {}

    for cam_id, camera in frame.cameras.items():
        if frame.num_points() == 0:
            cam_stats[cam_id] = CameraProjectionStats(
                camera_id=cam_id,
                num_projected=0,
                num_total=0,
                valid_ratio=0.0,
                depth_min=0.0,
                depth_max=0.0,
                depth_mean=0.0,
                depth_std=0.0,
            )
            continue

        result = project_points(frame.points_world, camera)
        valid_count = int(result.valid.sum())
        valid_depths = result.depth[result.valid]

        cam_stats[cam_id] = CameraProjectionStats(
            camera_id=cam_id,
            num_projected=valid_count,
            num_total=frame.num_points(),
            valid_ratio=(
                valid_count / frame.num_points() if frame.num_points() > 0 else 0.0
            ),
            depth_min=float(valid_depths.min()) if valid_count > 0 else 0.0,
            depth_max=float(valid_depths.max()) if valid_count > 0 else 0.0,
            depth_mean=float(valid_depths.mean()) if valid_count > 0 else 0.0,
            depth_std=float(valid_depths.std()) if valid_count > 0 else 0.0,
        )

    return FrameStats(
        frame_idx=frame.frame_idx,
        timestamp=frame.timestamp,
        num_points=frame.num_points(),
        num_cameras=frame.num_cameras(),
        camera_stats=cam_stats,
    )


def validate_manifest(manifest: SampleManifest) -> List[FrameStats]:
    """Validate all frames in a manifest.  Returns per-frame stats."""
    return [compute_frame_stats(f) for f in manifest.frames]


# ======================================================================
# Projection overlay (requires OpenCV)
# ======================================================================

def render_projection_overlay(
    frame: FrameData,
    camera_id: str,
    point_size: int = 2,
    colormap: str = "depth",
) -> np.ndarray:
    """Render point cloud projected onto a camera image.

    Args:
        frame: Frame with ``points_world`` and ``cameras``.
        camera_id: Camera to project onto.
        point_size: Circle radius in pixels.
        colormap: ``"depth"`` (jet by depth) or ``"uniform"`` (red).

    Returns:
        ``(H, W, 3)`` uint8 BGR image with overlay.
    """
    import cv2

    img = frame.load_image(camera_id).copy()
    # Convert RGB -> BGR for OpenCV drawing if needed
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    camera = frame.cameras[camera_id]
    if frame.num_points() == 0:
        return img

    result = project_points(frame.points_world, camera)
    valid_xy = result.xy[result.valid].astype(int)
    valid_depth = result.depth[result.valid]

    if len(valid_xy) == 0:
        return img

    if colormap == "depth":
        d_min, d_max = float(valid_depth.min()), float(valid_depth.max())
        if d_max > d_min:
            normed = ((valid_depth - d_min) / (d_max - d_min) * 255).astype(
                np.uint8
            )
        else:
            normed = np.full(len(valid_depth), 128, dtype=np.uint8)
        colors_bgr = cv2.applyColorMap(
            normed.reshape(-1, 1), cv2.COLORMAP_JET
        ).reshape(-1, 3)
    else:
        colors_bgr = np.full((len(valid_xy), 3), [0, 0, 255], dtype=np.uint8)

    for (x, y), color in zip(valid_xy, colors_bgr):
        cv2.circle(img, (int(x), int(y)), point_size, color.tolist(), -1)

    return img


def save_projection_overlay(
    frame: FrameData,
    camera_id: str,
    output_path: str,
    point_size: int = 2,
    colormap: str = "depth",
) -> None:
    """Render and save projection overlay to disk."""
    import cv2

    img = render_projection_overlay(frame, camera_id, point_size, colormap)
    cv2.imwrite(output_path, img)
    logger.info("Saved overlay: %s", output_path)


# ======================================================================
# Human-readable report
# ======================================================================

def print_manifest_report(manifest: SampleManifest) -> str:
    """Print and return a human-readable alignment report."""
    stats_list = validate_manifest(manifest)
    lines = [
        f"=== Manifest: {manifest.sample_id} ({manifest.dataset}) ===",
        f"Frames: {manifest.num_frames}  Duration: {manifest.duration:.1f}s",
    ]
    empty_frames = 0
    for fs in stats_list:
        lines.append(
            f"  Frame {fs.frame_idx} (t={fs.timestamp:.3f}s): "
            f"{fs.num_points} pts, {fs.num_cameras} cams"
        )
        if fs.num_points == 0:
            empty_frames += 1
        for cam_id in sorted(fs.camera_stats):
            cs = fs.camera_stats[cam_id]
            lines.append(
                f"    {cam_id}: {cs.num_projected}/{cs.num_total} valid "
                f"({cs.valid_ratio:.1%}), "
                f"depth [{cs.depth_min:.1f}, {cs.depth_max:.1f}]m"
            )

    if empty_frames > 0:
        lines.append(f"  WARNING: {empty_frames} frame(s) have 0 points")

    report = "\n".join(lines)
    print(report)
    return report
