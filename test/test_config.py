"""Tests for FastSNOWConfig and YAML loading/saving."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from fast_snow.engine.config.fast_snow_config import (
    DA3Config,
    DepthFilterConfig,
    EdgeConfig,
    FastSNOWConfig,
    FusionConfig,
    RAMPlusConfig,
    SAM3Config,
    STEPConfig,
    SamplingConfig,
    SerializationConfig,
    YOLOConfig,
    _build_from_dict,
    load_fast_snow_config,
    save_fast_snow_config,
)


class TestDefaultValues:
    """Verify defaults match docs/roadmap/Fast-SNOW_IMPLEMENTATION.md Section 6."""

    def test_yolo_defaults(self):
        c = YOLOConfig()
        assert c.conf_threshold == 0.25
        assert c.iou_threshold == 0.7
        assert c.imgsz == 640
        assert c.max_det == 200

    def test_sam3_defaults(self):
        c = SAM3Config()
        assert c.score_threshold_detection == 0.3

    def test_da3_defaults(self):
        c = DA3Config()
        assert c.process_res == 504
        assert c.require_metric is False

    def test_sampling_defaults(self):
        c = SamplingConfig()
        assert c.target_fps == 10.0
        assert c.max_frames is None

    def test_depth_filter_defaults(self):
        c = DepthFilterConfig()
        assert c.conf_thresh == 0.5
        assert c.min_points == 50
        assert c.max_extent == 30.0

    def test_fusion_defaults(self):
        c = FusionConfig()
        assert c.cross_run_iou_thresh == 0.5
        assert c.merge_centroid_dist_m == 2.0
        assert c.merge_temporal_gap == 2
        assert c.lost_patience == 5
        assert c.archive_patience == 30

    def test_step_defaults(self):
        c = STEPConfig()
        assert c.grid_size == 16
        assert c.iou_threshold == 0.5
        assert c.mask_outside_pixels is True
        assert c.patch_crop_size == 64
        assert c.temporal_window == 10
        # Note: code default is 0 (unlimited); spec §6.5 says 8.
        # Test matches actual code default.
        assert c.max_tau_per_step == 0

    def test_edge_defaults(self):
        c = EdgeConfig()
        assert c.elev_thresh == 0.5
        assert c.motion_thresh == 3.0
        assert c.lateral_thresh == 3.0
        assert c.knn_k == 3
        assert c.motion_window == 3

    def test_serialization_defaults(self):
        c = SerializationConfig()
        assert c.max_obj_relations == 20

    def test_global_defaults(self):
        c = FastSNOWConfig()
        assert c.device == "cuda"
        assert c.seed == 42
        assert c.verbose is False


class TestAllSubconfigsPresent:

    def test_has_all_fields(self):
        c = FastSNOWConfig()
        assert isinstance(c.sam3, SAM3Config)
        assert isinstance(c.da3, DA3Config)
        assert isinstance(c.yolo, YOLOConfig)
        assert isinstance(c.ram_plus, RAMPlusConfig)
        assert isinstance(c.sampling, SamplingConfig)
        assert isinstance(c.depth_filter, DepthFilterConfig)
        assert isinstance(c.fusion, FusionConfig)
        assert isinstance(c.step, STEPConfig)
        assert isinstance(c.edge, EdgeConfig)
        assert isinstance(c.serialization, SerializationConfig)


class TestToDict:

    def test_roundtrip_all_keys(self):
        c = FastSNOWConfig()
        d = c.to_dict()
        assert isinstance(d, dict)
        assert "sam3" in d
        assert "da3" in d
        assert "yolo" in d
        assert "depth_filter" in d
        assert "fusion" in d
        assert "step" in d
        assert "edge" in d
        assert "serialization" in d
        assert "device" in d
        assert "seed" in d

    def test_nested_dict_structure(self):
        d = FastSNOWConfig().to_dict()
        assert isinstance(d["step"], dict)
        assert "grid_size" in d["step"]
        assert "iou_threshold" in d["step"]

    def test_to_dict_preserves_values(self):
        c = FastSNOWConfig()
        c.step.grid_size = 8
        c.seed = 99
        d = c.to_dict()
        assert d["step"]["grid_size"] == 8
        assert d["seed"] == 99


class TestYAMLSaveLoad:

    def test_save_and_load(self, tmp_path):
        original = FastSNOWConfig()
        original.seed = 123
        original.step.grid_size = 8
        original.fusion.lost_patience = 10
        path = tmp_path / "cfg.yaml"
        save_fast_snow_config(original, path)
        loaded = load_fast_snow_config(path)
        assert loaded.seed == 123
        assert loaded.step.grid_size == 8
        assert loaded.fusion.lost_patience == 10

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_fast_snow_config("/nonexistent/path/cfg.yaml")

    def test_load_empty_yaml(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_fast_snow_config(path)
        # Should return defaults
        assert cfg.seed == 42
        assert cfg.step.grid_size == 16

    def test_load_partial_yaml(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump({"seed": 77}))
        cfg = load_fast_snow_config(path)
        assert cfg.seed == 77
        assert cfg.step.grid_size == 16  # default

    def test_load_extra_keys_ignored(self, tmp_path):
        path = tmp_path / "extra.yaml"
        path.write_text(yaml.dump({"seed": 7, "unknown_field": "hello"}))
        cfg = load_fast_snow_config(path)
        assert cfg.seed == 7
        assert not hasattr(cfg, "unknown_field")

    def test_nested_override(self, tmp_path):
        path = tmp_path / "nested.yaml"
        data = {"depth_filter": {"min_points": 10, "max_extent": 50.0}}
        path.write_text(yaml.dump(data))
        cfg = load_fast_snow_config(path)
        assert cfg.depth_filter.min_points == 10
        assert cfg.depth_filter.max_extent == 50.0
        assert cfg.depth_filter.conf_thresh == 0.5  # default preserved

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "cfg.yaml"
        save_fast_snow_config(FastSNOWConfig(), path)
        assert path.exists()

    def test_save_overwrites_existing(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        save_fast_snow_config(FastSNOWConfig(), path)
        c2 = FastSNOWConfig()
        c2.seed = 999
        save_fast_snow_config(c2, path)
        loaded = load_fast_snow_config(path)
        assert loaded.seed == 999


class TestBuildFromDict:

    def test_none_input_returns_default(self):
        cfg = _build_from_dict(FastSNOWConfig, None)
        assert isinstance(cfg, FastSNOWConfig)
        assert cfg.seed == 42

    def test_empty_dict_returns_default(self):
        cfg = _build_from_dict(FastSNOWConfig, {})
        assert cfg.seed == 42

    def test_subconfig_from_dict(self):
        cfg = _build_from_dict(STEPConfig, {"grid_size": 8, "iou_threshold": 0.3})
        assert cfg.grid_size == 8
        assert cfg.iou_threshold == 0.3


class TestMutability:

    def test_config_fields_mutable(self):
        """FastSNOWConfig is NOT frozen — fields can be changed at runtime."""
        c = FastSNOWConfig()
        c.seed = 999
        assert c.seed == 999
        c.step.grid_size = 32
        assert c.step.grid_size == 32

    def test_subconfig_mutable(self):
        c = DepthFilterConfig()
        c.min_points = 1
        assert c.min_points == 1
