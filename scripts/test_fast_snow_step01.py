from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd


def test_step1_tag_normalization_and_new_tag_logic():
    cfg = FastSNOWConfig()
    cfg.ram_plus.normalize_lowercase = True
    cfg.ram_plus.deduplicate_tags = True

    e2e = FastSNOWEndToEnd(cfg)

    tags = e2e._normalize_tags([" Car ", "tree", "car", "", " TREE ", "person"])
    assert tags == ["car", "tree", "person"]

    global_tag_set = {"car"}
    new_tags = [t for t in tags if t not in global_tag_set]
    global_tag_set.update(new_tags)

    assert new_tags == ["tree", "person"]
    assert global_tag_set == {"car", "tree", "person"}


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

        frames, frame_dir, source_indices, keyframe_paths = e2e._extract_frames(video_path)
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
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
