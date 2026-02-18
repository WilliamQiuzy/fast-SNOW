"""Minimal end-to-end pipeline sanity check.

This script demonstrates the data flow using a single video frame and synthetic
point cloud. It is meant for input validation, not for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import sys
import tempfile

import numpy as np
import imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_snow.data.calibration.camera_model import CameraExtrinsics, CameraIntrinsics, CameraModel
from fast_snow.data.transforms.projection import project_points
from fast_snow.vision.perception.segmentation.sam2_wrapper import SAM2Config, SAM2Mask, SAM2Wrapper
from fast_snow.reasoning.tokens.step_encoding import build_step_token
from fast_snow.reasoning.graph.scene_graph import build_scene_graph
from fast_snow.reasoning.graph.temporal_linking import link_temporal_tokens
from fast_snow.reasoning.graph.four_d_sg import build_four_d_scene_graph
from fast_snow.reasoning.vlm.inference import VLMConfig, VLMInterface
from fast_snow.vision.slam.map_anything import MapAnythingConfig, run_map_anything


@dataclass(frozen=True)
class PipelineConfig:
    dataset_json: Path
    index: int = 0
    num_frames: int = 10
    num_points: int = 200
    use_mapanything: bool = True


def _read_frames(video_path: Path, num_frames: int) -> List[np.ndarray]:
    reader = imageio.get_reader(video_path)
    frames: List[np.ndarray] = []
    try:
        for i in range(num_frames):
            frames.append(reader.get_data(i))
    finally:
        reader.close()
    return frames


def _dummy_point_cloud(num_points: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(low=-2.0, high=2.0, size=(num_points, 3))


def _dummy_camera(image_shape: tuple) -> CameraModel:
    h, w = image_shape[:2]
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=w / 2.0, cy=h / 2.0)
    extr = CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(3))
    return CameraModel(intrinsics=intr, extrinsics=extr, image_size=(w, h))


def _fallback_mask(image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    mask[y0:y1, x0:x1] = True
    return mask


def _predict_masks(frame: np.ndarray) -> List[SAM2Mask]:
    try:
        sam2 = SAM2Wrapper(SAM2Config())
        return sam2.predict(frame, point_prompts=[(frame.shape[1] // 2, frame.shape[0] // 2)])
    except Exception:
        return [SAM2Mask(mask=_fallback_mask(frame.shape), score=1.0)]


def _write_frames_to_dir(frames: List[np.ndarray], root: Path) -> List[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx, frame in enumerate(frames):
        path = root / f"frame_{idx:04d}.png"
        imageio.imwrite(path, frame)
        paths.append(path)
    return paths


def run_pipeline(cfg: PipelineConfig) -> None:
    from fast_snow.data.loaders.vlm4d import load_vlm4d_json, resolve_video_path

    samples = load_vlm4d_json(cfg.dataset_json)
    if cfg.index >= len(samples):
        raise ValueError("index out of range for dataset")
    sample = samples[cfg.index]
    local_root = Path("fast_snow/data/vlm4d_videos")
    video_path = Path(
        resolve_video_path(
            sample.video, local_video_root=local_root, allow_download=True
        )
    )
    if not video_path.exists():
        raise SystemExit("Video not found or download failed.")

    frames = _read_frames(video_path, cfg.num_frames)

    points_per_frame: List[np.ndarray] = []
    cameras_per_frame: List[CameraModel] = []
    if cfg.use_mapanything:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = _write_frames_to_dir(frames, Path(tmpdir))
            map_results = run_map_anything(frame_paths, MapAnythingConfig())
        for res in map_results:
            points_per_frame.append(res.points_xyz)
            cameras_per_frame.append(res.camera)
    else:
        points = _dummy_point_cloud(cfg.num_points)
        camera = _dummy_camera(frames[0].shape)
        for _ in frames:
            points_per_frame.append(points)
            cameras_per_frame.append(camera)

    spatial_graphs = []
    frame_steps = []
    for t, (frame, points, camera) in enumerate(
        zip(frames, points_per_frame, cameras_per_frame)
    ):
        proj = project_points(points, camera)
        masks = _predict_masks(frame)
        if not masks:
            raise RuntimeError("No masks produced")
        best_mask = max(masks, key=lambda m: m.score).mask
        step = build_step_token(best_mask, points, t_start=0, t_end=t)
        steps = {0: step}
        spatial_graphs.append(build_scene_graph(steps))
        frame_steps.append(steps)

    window = link_temporal_tokens(frame_steps)
    graph = build_four_d_scene_graph(spatial_graphs, window, ego_poses={0: [0.0, 0.0, 0.0]})

    vlm = VLMInterface(VLMConfig())
    out = vlm.infer(sample.question, graph)
    print("sample_id", sample.sample_id)
    print("question", sample.question)
    print("num_frames", len(spatial_graphs))
    print("num_tracks", len(window.tracks))
    print("vlm_out", out)


if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(
            dataset_json=Path("fast_snow/data/VLM4D-main/data/real_mc.json"),
            index=0,
            num_frames=10,
        )
    )
