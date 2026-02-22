"""Tests for _quantize_8_way direction quantization."""

from __future__ import annotations

import math

import pytest

from fast_snow.engine.pipeline.fast_snow_pipeline import FastSNOWPipeline


Q = FastSNOWPipeline._quantize_8_way


class TestAllSectors:
    """Each sector centre should map to the correct direction."""

    @pytest.mark.parametrize("deg,expected", [
        (0, "front"),
        (45, "front-left"),
        (90, "left"),
        (135, "back-left"),
        (180, "back"),
        (-180, "back"),
        (-135, "back-right"),
        (-90, "right"),
        (-45, "front-right"),
    ])
    def test_sector_centres(self, deg, expected):
        assert Q(math.radians(deg)) == expected


class TestBoundaries:
    """Boundary angles: check >= / < transitions."""

    def test_22_5_is_front_left(self):
        assert Q(math.radians(22.5)) == "front-left"

    def test_just_below_22_5_is_front(self):
        assert Q(math.radians(22.49)) == "front"

    def test_neg_22_5_is_front(self):
        assert Q(math.radians(-22.5)) == "front"

    def test_67_5_is_left(self):
        assert Q(math.radians(67.5)) == "left"

    def test_just_below_67_5_is_front_left(self):
        assert Q(math.radians(67.49)) == "front-left"

    def test_112_5_is_back_left(self):
        assert Q(math.radians(112.5)) == "back-left"

    def test_157_5_is_back(self):
        assert Q(math.radians(157.5)) == "back"

    def test_neg_157_5_is_back_right(self):
        assert Q(math.radians(-157.5)) == "back-right"

    def test_neg_112_5_is_right(self):
        assert Q(math.radians(-112.5)) == "right"

    def test_neg_67_5_is_front_right(self):
        """-67.5° is NOT in [-112.5, -67.5) range → falls to 'front-right'."""
        assert Q(math.radians(-67.5)) == "front-right"

    def test_just_above_neg_67_5_is_front_right(self):
        assert Q(math.radians(-67.49)) == "front-right"


class TestSpecialValues:

    def test_zero(self):
        assert Q(0.0) == "front"

    def test_pi(self):
        assert Q(math.pi) == "back"

    def test_negative_pi(self):
        assert Q(-math.pi) == "back"

    def test_two_pi_equivalent_to_zero(self):
        # atan2 range is [-π, π], so 2π should not be produced,
        # but if passed, check it doesn't crash
        result = Q(2 * math.pi)
        assert isinstance(result, str)

    def test_very_small_positive(self):
        assert Q(1e-10) == "front"

    def test_very_small_negative(self):
        assert Q(-1e-10) == "front"
