from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd


def test_fastsam_config_defaults():
    cfg = FastSNOWConfig()
    assert cfg.fastsam.model_path == "fast_snow/models/fastsam/FastSAM-s.pt"
    assert cfg.fastsam.conf_threshold == 0.55
    assert cfg.fastsam.iou_threshold == 0.9
    assert cfg.fastsam.imgsz == 640
    assert cfg.fastsam.max_det == 200
    assert cfg.fastsam.discovery_iou_thresh == 0.3


def test_step0_sampling_target_fps_and_max_frames():
    cv2 = pytest.importorskip("cv2")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / "tiny.mp4"

        # Build a tiny 6-frame video.
        h, w = 64, 96
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (w, h),
        )
        for i in range(6):
            frame = np.full((h, w, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        cfg = FastSNOWConfig()
        cfg.sampling.target_fps = 2.5
        cfg.sampling.max_frames = 2
        e2e = FastSNOWEndToEnd(cfg)

        frames, frame_dir, source_indices, keyframe_paths, timestamps_s = e2e._extract_frames(video_path)
        try:
            assert len(frames) == 2
            assert source_indices == [0, 2]
            assert (frame_dir / "000000.jpg").exists()
            assert (frame_dir / "000001.jpg").exists()

            # keyframe_paths must match source_indices and point to existing JPEGs
            assert len(keyframe_paths) == 2
            assert keyframe_paths[0][0] == 0
            assert keyframe_paths[1][0] == 2
            assert keyframe_paths[0][1].exists()
            assert keyframe_paths[1][1].exists()

            # timestamps_s must be provided
            assert len(timestamps_s) == 2
            assert timestamps_s[0] == pytest.approx(0.0, abs=0.01)
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
