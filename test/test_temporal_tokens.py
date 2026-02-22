"""Tests for TemporalToken."""

from __future__ import annotations

import pytest

from fast_snow.reasoning.tokens.temporal_tokens import TemporalToken


class TestTemporalToken:

    def test_basic_creation(self):
        t = TemporalToken(t_start=0, t_end=10)
        assert t.t_start == 0
        assert t.t_end == 10

    def test_single_frame(self):
        t = TemporalToken(t_start=5, t_end=5)
        assert t.t_start == t.t_end == 5

    def test_frozen_cannot_modify(self):
        t = TemporalToken(t_start=0, t_end=10)
        with pytest.raises(AttributeError):
            t.t_start = 99  # type: ignore[misc]
        with pytest.raises(AttributeError):
            t.t_end = 99  # type: ignore[misc]

    def test_inverted_range_stored_as_is(self):
        """No validation prevents t_start > t_end."""
        t = TemporalToken(t_start=10, t_end=3)
        assert t.t_start == 10
        assert t.t_end == 3

    def test_negative_frames(self):
        t = TemporalToken(t_start=-5, t_end=-1)
        assert t.t_start == -5
        assert t.t_end == -1

    def test_zero_both(self):
        t = TemporalToken(t_start=0, t_end=0)
        assert t.t_start == 0
        assert t.t_end == 0

    def test_large_values(self):
        t = TemporalToken(t_start=0, t_end=1_000_000)
        assert t.t_end == 1_000_000

    def test_equality(self):
        a = TemporalToken(t_start=1, t_end=5)
        b = TemporalToken(t_start=1, t_end=5)
        assert a == b

    def test_inequality(self):
        a = TemporalToken(t_start=1, t_end=5)
        b = TemporalToken(t_start=1, t_end=6)
        assert a != b

    def test_hashable(self):
        t1 = TemporalToken(t_start=0, t_end=10)
        t2 = TemporalToken(t_start=0, t_end=10)
        s = {t1, t2}
        assert len(s) == 1

    def test_repr_contains_fields(self):
        t = TemporalToken(t_start=3, t_end=7)
        r = repr(t)
        assert "t_start=3" in r
        assert "t_end=7" in r
