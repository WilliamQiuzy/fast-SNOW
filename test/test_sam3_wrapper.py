"""Tests for SAM3 dataclasses (no model loading, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.vision.perception.sam3_wrapper import SAM3Mask, SAM3Run


class TestSAM3Mask:

    def test_creation(self):
        m = np.ones((64, 64), dtype=bool)
        mask = SAM3Mask(run_id=0, obj_id_local=1, mask=m, score=0.95)
        assert mask.run_id == 0
        assert mask.obj_id_local == 1
        assert mask.score == 0.95
        assert mask.mask.shape == (64, 64)
        assert mask.mask.dtype == bool

    def test_mask_values(self):
        m = np.zeros((32, 32), dtype=bool)
        m[10:20, 10:20] = True
        mask = SAM3Mask(run_id=0, obj_id_local=0, mask=m, score=0.8)
        assert mask.mask.sum() == 100

    def test_zero_score(self):
        m = np.ones((8, 8), dtype=bool)
        mask = SAM3Mask(run_id=0, obj_id_local=0, mask=m, score=0.0)
        assert mask.score == 0.0


class TestSAM3Run:

    def test_creation_defaults(self):
        run = SAM3Run(run_id=0, tag="person", start_frame=5)
        assert run.run_id == 0
        assert run.tag == "person"
        assert run.start_frame == 5
        assert run.session_id is None
        assert run.status == "created"
        assert run.last_propagated_frame == -1

    def test_status_lifecycle(self):
        run = SAM3Run(run_id=0, tag="car", start_frame=0)
        assert run.status == "created"
        run.status = "active"
        assert run.status == "active"
        run.status = "ended"
        assert run.status == "ended"

    def test_session_id_assignment(self):
        run = SAM3Run(run_id=1, tag="dog", start_frame=10)
        run.session_id = "sess_abc123"
        assert run.session_id == "sess_abc123"

    def test_propagation_tracking(self):
        run = SAM3Run(run_id=0, tag="horse", start_frame=0)
        run.last_propagated_frame = 5
        assert run.last_propagated_frame == 5
        run.last_propagated_frame = 10
        assert run.last_propagated_frame == 10

    def test_multiple_runs_independent(self):
        r1 = SAM3Run(run_id=0, tag="person", start_frame=0)
        r2 = SAM3Run(run_id=1, tag="car", start_frame=3)
        r1.status = "active"
        r2.status = "ended"
        assert r1.status == "active"
        assert r2.status == "ended"
        assert r1.run_id != r2.run_id
