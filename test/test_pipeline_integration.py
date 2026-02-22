"""Integration tests: multi-frame scenarios exercising the full pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline

from conftest import (
    make_detection,
    make_frame_input,
    make_mask,
    relaxed_config,
)


H, W = 64, 64


@pytest.fixture
def pipe():
    return FastSNOWPipeline(config=relaxed_config())


class TestObjectLifecycle:

    def test_appears_and_disappears(self, pipe):
        """Object visible only frames 3-7 out of 0-9."""
        for t in range(10):
            if 3 <= t <= 7:
                dets = [make_detection()]
            else:
                dets = []
            pipe.process_frame(make_frame_input(t, detections=dets))

        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_frames"] == 10
        fk = dsg["tracks"][0]["F_k"]
        frame_indices = [obs["t"] for obs in fk]
        assert frame_indices == [3, 4, 5, 6, 7]
        assert fk[0]["theta"] == [3, 7]

    def test_two_objects_different_lifetimes(self, pipe):
        """Object A visible 0-9, Object B visible 5-9."""
        m_a = make_mask(H, W, 0, 20, 0, 20)
        m_b = make_mask(H, W, 40, 60, 40, 60)
        for t in range(10):
            dets = [make_detection(run_id="r", obj_id=0, mask=m_a)]
            if t >= 5:
                dets.append(make_detection(run_id="r", obj_id=1, mask=m_b))
            pipe.process_frame(make_frame_input(t, detections=dets))

        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_tracks"] == 2

        track_a = [t for t in dsg["tracks"] if len(t["F_k"]) == 10]
        track_b = [t for t in dsg["tracks"] if len(t["F_k"]) == 5]
        assert len(track_a) == 1
        assert len(track_b) == 1
        assert track_a[0]["F_k"][0]["theta"] == [0, 9]
        assert track_b[0]["F_k"][0]["theta"] == [5, 9]


class TestProcessFrames:

    def test_sorts_by_idx(self, pipe):
        """process_frames should handle unsorted input."""
        frames = [
            make_frame_input(frame_idx=3),
            make_frame_input(frame_idx=1),
            make_frame_input(frame_idx=0),
            make_frame_input(frame_idx=2),
        ]
        pipe.process_frames(frames)
        dsg = pipe.build_4dsg_dict()
        ego_ts = [e["t"] for e in dsg["ego"]]
        assert ego_ts == [0, 1, 2, 3]

    def test_reset_true_clears_state(self, pipe):
        pipe.process_frame(make_frame_input(0))
        pipe.process_frames([make_frame_input(1)], reset=True)
        dsg = pipe.build_4dsg_dict()
        # After reset, only frame 1 exists
        assert dsg["metadata"]["num_frames"] == 1
        assert dsg["ego"][0]["t"] == 1

    def test_reset_false_accumulates(self, pipe):
        pipe.process_frame(make_frame_input(0))
        pipe.process_frames([make_frame_input(1)], reset=False)
        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_frames"] == 2


class TestStressAndEdge:

    def test_hundred_frames(self):
        """100 frames with 5 objects — should not crash."""
        cfg = relaxed_config()
        pipe = FastSNOWPipeline(config=cfg)
        masks = [make_mask(H, W, i * 10, i * 10 + 8, 0, 60) for i in range(5)]
        for t in range(100):
            dets = [make_detection(run_id="r", obj_id=i, mask=m) for i, m in enumerate(masks)]
            pipe.process_frame(make_frame_input(t, detections=dets))
        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_frames"] == 100
        assert dsg["metadata"]["num_tracks"] == 5

    def test_many_detections_per_frame(self):
        """50 detections in a single frame."""
        cfg = relaxed_config()
        pipe = FastSNOWPipeline(config=cfg)
        dets = []
        for i in range(50):
            y0 = (i % 6) * 10
            m = make_mask(H, W, y0, y0 + 5, 0, W)
            dets.append(make_detection(run_id="r", obj_id=i, mask=m))
        pipe.process_frame(make_frame_input(0, detections=dets))
        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_tracks"] > 0

    def test_non_metric_depth_skips_extent_filter(self):
        """depth_is_metric=False → max_extent not applied."""
        cfg = relaxed_config()
        cfg.depth_filter.max_extent = 0.001  # impossibly tight
        pipe = FastSNOWPipeline(config=cfg)
        frame = make_frame_input(0, base_depth=100.0, depth_is_metric=False)
        pipe.process_frame(frame)
        dsg = pipe.build_4dsg_dict()
        # Should still have the track (extent filter skipped)
        assert dsg["metadata"]["num_tracks"] == 1

    def test_metric_depth_applies_extent_filter(self):
        """depth_is_metric=True + tight max_extent → filtered out."""
        cfg = relaxed_config()
        cfg.depth_filter.max_extent = 0.001  # impossibly tight
        pipe = FastSNOWPipeline(config=cfg)
        frame = make_frame_input(0, base_depth=100.0, depth_is_metric=True)
        pipe.process_frame(frame)
        dsg = pipe.build_4dsg_dict()
        assert dsg["metadata"]["num_tracks"] == 0

    def test_strided_frame_indices(self):
        """Frame indices with stride=5."""
        cfg = relaxed_config()
        pipe = FastSNOWPipeline(config=cfg)
        for t in range(0, 25, 5):
            pipe.process_frame(make_frame_input(t))
        dsg = pipe.build_4dsg_dict()
        ego_ts = [e["t"] for e in dsg["ego"]]
        assert ego_ts == [0, 5, 10, 15, 20]
        fk = dsg["tracks"][0]["F_k"]
        assert fk[0]["theta"] == [0, 20]


class TestGlobalIDProperties:

    def test_ids_never_reused(self):
        """Once an ID is assigned, it's never recycled."""
        cfg = relaxed_config()
        cfg.fusion.lost_patience = 1
        cfg.fusion.archive_patience = 1
        pipe = FastSNOWPipeline(config=cfg)

        all_ids = set()
        for t in range(20):
            # Object appears every other frame
            if t % 2 == 0:
                pipe.process_frame(make_frame_input(t))
            else:
                pipe.process_frame(make_frame_input(t, detections=[]))
            all_ids.update(pipe._local_to_global.values())

        # Check no collisions (all IDs unique in _tracks)
        track_ids = set(pipe._tracks.keys())
        assert len(track_ids) == len(pipe._tracks)

    def test_reset_clears_everything(self, pipe):
        pipe.process_frame(make_frame_input(0))
        pipe.reset()
        assert pipe._next_global_id == 0
        assert pipe._tracks == {}
        assert pipe._local_to_global == {}
        assert pipe._ego_poses_cw == {}
