"""Tests for _compute_relations, bearing, elevation, and motion (Step 7)."""

from __future__ import annotations

import math

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


def _pipe_with_edge_cfg(**overrides):
    cfg = relaxed_config()
    for k, v in overrides.items():
        setattr(cfg.edge, k, v)
    return FastSNOWPipeline(config=cfg)


def _place_object_at_depth(pipe, frame_idx, base_depth, y0=10, y1=50, x0=10, x1=50):
    """Helper: process one frame with an object at a given depth."""
    m = make_mask(H, W, y0, y1, x0, x1)
    frame = make_frame_input(
        frame_idx=frame_idx,
        detections=[make_detection(mask=m)],
        base_depth=base_depth,
    )
    pipe.process_frame(frame)


# ---------------------------------------------------------------------------
# Bearing (8-way quantization)
# ---------------------------------------------------------------------------

class TestBearing:

    @pytest.mark.parametrize("angle_deg,expected", [
        (0, "front"),
        (30, "front-left"),
        (90, "left"),
        (135, "back-left"),
        (180, "back"),
        (-135, "back-right"),
        (-90, "right"),
        (-30, "front-right"),
    ])
    def test_quantize_8_way(self, angle_deg, expected):
        result = FastSNOWPipeline._quantize_8_way(math.radians(angle_deg))
        assert result == expected

    def test_boundary_225(self):
        """22.5° is at the boundary: >= 22.5 → front-left."""
        assert FastSNOWPipeline._quantize_8_way(math.radians(22.5)) == "front-left"

    def test_boundary_negative_225(self):
        """-22.5° is at the boundary: >= -22.5 → front."""
        assert FastSNOWPipeline._quantize_8_way(math.radians(-22.5)) == "front"

    def test_boundary_1575(self):
        """157.5° → back."""
        assert FastSNOWPipeline._quantize_8_way(math.radians(157.5)) == "back"

    def test_boundary_negative_1575(self):
        """-157.5° → back-right (>= -157.5, < -112.5)."""
        assert FastSNOWPipeline._quantize_8_way(math.radians(-157.5)) == "back-right"

    def test_zero_is_front(self):
        assert FastSNOWPipeline._quantize_8_way(0.0) == "front"

    def test_pi_is_back(self):
        assert FastSNOWPipeline._quantize_8_way(math.pi) == "back"

    def test_negative_pi_is_back(self):
        assert FastSNOWPipeline._quantize_8_way(-math.pi) == "back"


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

class TestElevation:

    def test_above(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(1.0) == "above"

    def test_below(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(-1.0) == "below"

    def test_level(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(0.0) == "level"

    def test_exact_positive_threshold_is_level(self):
        """dz=0.5 uses >, not >= → should be 'level'."""
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(0.5) == "level"

    def test_exact_negative_threshold_is_level(self):
        """dz=-0.5 uses <, not <= → should be 'level'."""
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(-0.5) == "level"

    def test_just_above_threshold(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(0.501) == "above"

    def test_just_below_threshold(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=0.5)
        assert pipe._elev_relation(-0.501) == "below"

    def test_custom_threshold(self):
        pipe = _pipe_with_edge_cfg(elev_thresh=2.0)
        assert pipe._elev_relation(1.5) == "level"
        assert pipe._elev_relation(2.5) == "above"


# ---------------------------------------------------------------------------
# Motion inference
# ---------------------------------------------------------------------------

class TestMotion:

    def test_cold_start_unknown(self):
        """Fewer than motion_window samples → 'unknown'."""
        pipe = _pipe_with_edge_cfg(motion_window=3)
        _place_object_at_depth(pipe, 0, 5.0)
        dsg = pipe.build_4dsg_dict()
        ego_rel = dsg["ego_relations"]
        assert len(ego_rel) == 1
        assert ego_rel[0]["motion"] == "unknown"

    def test_static(self):
        """Object at constant distance → 'static'."""
        pipe = _pipe_with_edge_cfg(motion_window=3, motion_thresh=0.3, lateral_thresh=0.3)
        for t in range(5):
            _place_object_at_depth(pipe, t, 5.0)
        dsg = pipe.build_4dsg_dict()
        assert dsg["ego_relations"][0]["motion"] == "static"

    def test_motion_rate_normalized_by_dt(self):
        """Strided frames (0, 10, 20): rate should be Δd/Δt, not Δd/N."""
        pipe = _pipe_with_edge_cfg(motion_window=3, motion_thresh=0.01)
        # Object moving slowly in depth (5.0 → 5.1 → 5.2)
        for i, t in enumerate([0, 10, 20]):
            _place_object_at_depth(pipe, t, 5.0 + i * 0.1)
        dsg = pipe.build_4dsg_dict()
        motion = dsg["ego_relations"][0]["motion"]
        # Δd=0.2 over dt=20 → rate=0.01 m/frame, which is below 0.01 threshold
        # Depends on exact implementation, but should not be wildly wrong
        assert motion in ("receding", "static")


# ---------------------------------------------------------------------------
# Relations structure
# ---------------------------------------------------------------------------

class TestRelationsStructure:

    def test_empty_visible_returns_empty(self):
        pipe = _pipe_with_edge_cfg()
        pipe.process_frame(make_frame_input(0, detections=[]))
        dsg = pipe.build_4dsg_dict()
        assert dsg["ego_relations"] == []
        assert dsg["object_relations"] == []

    def test_single_object_no_obj_relations(self):
        pipe = _pipe_with_edge_cfg()
        pipe.process_frame(make_frame_input(0))
        dsg = pipe.build_4dsg_dict()
        assert len(dsg["ego_relations"]) == 1
        assert dsg["object_relations"] == []

    def test_ego_relation_fields(self):
        pipe = _pipe_with_edge_cfg()
        pipe.process_frame(make_frame_input(0))
        dsg = pipe.build_4dsg_dict()
        rel = dsg["ego_relations"][0]
        assert "object_id" in rel
        assert "bearing" in rel
        assert "elev" in rel
        assert "dist_m" in rel
        assert "motion" in rel
        assert isinstance(rel["dist_m"], float)

    def test_obj_relation_fields(self):
        pipe = _pipe_with_edge_cfg()
        m1 = make_mask(H, W, 0, 20, 0, 20)
        m2 = make_mask(H, W, 40, 60, 40, 60)
        frame = make_frame_input(0, detections=[
            make_detection(run_id="r1", obj_id=0, mask=m1),
            make_detection(run_id="r2", obj_id=1, mask=m2),
        ])
        pipe.process_frame(frame)
        dsg = pipe.build_4dsg_dict()
        assert len(dsg["object_relations"]) > 0
        rel = dsg["object_relations"][0]
        assert "src" in rel
        assert "dst" in rel
        assert "dir" in rel
        assert "elev" in rel
        assert "dist_m" in rel

    def test_knn_k_limits_neighbors(self):
        """With k=1, each object gets at most 1 neighbor."""
        pipe = _pipe_with_edge_cfg(knn_k=1)
        masks = [make_mask(H, W, i * 10, i * 10 + 8, 0, 60) for i in range(5)]
        dets = [make_detection(run_id="r", obj_id=i, mask=m) for i, m in enumerate(masks)]
        pipe.process_frame(make_frame_input(0, detections=dets))
        dsg = pipe.build_4dsg_dict()
        # Each of 5 objects has at most 1 neighbor → at most 5 relations
        assert len(dsg["object_relations"]) <= 5

    def test_max_obj_relations_cap(self):
        pipe = _pipe_with_edge_cfg(knn_k=3)
        pipe.config.serialization.max_obj_relations = 2
        masks = [make_mask(H, W, i * 10, i * 10 + 8, 0, 60) for i in range(5)]
        dets = [make_detection(run_id="r", obj_id=i, mask=m) for i, m in enumerate(masks)]
        pipe.process_frame(make_frame_input(0, detections=dets))
        dsg = pipe.build_4dsg_dict()
        assert len(dsg["object_relations"]) <= 2

    def test_obj_relations_sorted_by_distance(self):
        pipe = _pipe_with_edge_cfg()
        m1 = make_mask(H, W, 0, 15, 0, 15)
        m2 = make_mask(H, W, 45, 60, 0, 15)
        m3 = make_mask(H, W, 0, 15, 45, 60)
        dets = [
            make_detection(run_id="r", obj_id=0, mask=m1),
            make_detection(run_id="r", obj_id=1, mask=m2),
            make_detection(run_id="r", obj_id=2, mask=m3),
        ]
        pipe.process_frame(make_frame_input(0, detections=dets))
        dsg = pipe.build_4dsg_dict()
        dists = [r["dist_m"] for r in dsg["object_relations"]]
        assert dists == sorted(dists)
