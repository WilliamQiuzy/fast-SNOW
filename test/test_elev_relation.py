"""Tests for _elev_relation elevation classification."""

from __future__ import annotations

import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline

from conftest import relaxed_config


def _pipe(elev_thresh: float = 0.5):
    cfg = relaxed_config()
    cfg.edge.elev_thresh = elev_thresh
    return FastSNOWPipeline(config=cfg)


class TestElevRelation:

    def test_above(self):
        assert _pipe()._elev_relation(1.0) == "above"

    def test_below(self):
        assert _pipe()._elev_relation(-1.0) == "below"

    def test_level_zero(self):
        assert _pipe()._elev_relation(0.0) == "level"

    def test_exact_positive_threshold_is_level(self):
        """dz = thresh exactly → 'level' (uses > not >=)."""
        assert _pipe(0.5)._elev_relation(0.5) == "level"

    def test_exact_negative_threshold_is_level(self):
        """dz = -thresh exactly → 'level' (uses < not <=)."""
        assert _pipe(0.5)._elev_relation(-0.5) == "level"

    def test_just_above_threshold(self):
        assert _pipe(0.5)._elev_relation(0.500001) == "above"

    def test_just_below_threshold(self):
        assert _pipe(0.5)._elev_relation(-0.500001) == "below"

    def test_custom_threshold_2m(self):
        p = _pipe(2.0)
        assert p._elev_relation(1.5) == "level"
        assert p._elev_relation(2.5) == "above"
        assert p._elev_relation(-2.5) == "below"

    def test_very_large_value(self):
        assert _pipe()._elev_relation(1000.0) == "above"

    def test_very_large_negative(self):
        assert _pipe()._elev_relation(-1000.0) == "below"

    def test_tiny_threshold(self):
        p = _pipe(0.001)
        assert p._elev_relation(0.01) == "above"
        assert p._elev_relation(-0.01) == "below"
        assert p._elev_relation(0.0005) == "level"
