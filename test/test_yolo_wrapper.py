"""Tests for YoloBBoxDetector with mocked YOLO model (no GPU needed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fast_snow.engine.config.fast_snow_config import YOLOConfig
from fast_snow.vision.perception.yolo_wrapper import YoloBBoxDetector, YoloDetection


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_result(
    xyxy: np.ndarray,
    confs: np.ndarray,
    cls: np.ndarray,
    names: Dict[int, str],
):
    """Create a mock ultralytics Result with boxes."""
    import torch

    boxes = MagicMock()
    boxes.xyxy = torch.tensor(xyxy, dtype=torch.float32) if xyxy.size > 0 else None
    boxes.conf = torch.tensor(confs, dtype=torch.float32) if confs.size > 0 else None
    boxes.cls = torch.tensor(cls, dtype=torch.float32) if cls.size > 0 else None

    result = MagicMock()
    result.boxes = boxes
    result.names = names
    return result


def _make_detector_with_mock(xyxy, confs, cls, names, h=100, w=200):
    """Create a detector with pre-injected mock model."""
    cfg = YOLOConfig(device="cpu", model_path="fake.pt")
    detector = YoloBBoxDetector(config=cfg)

    mock_result = _make_mock_result(xyxy, confs, cls, names)
    mock_model = MagicMock(return_value=[mock_result])
    detector._model = mock_model

    return detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectReturns:

    def test_returns_list(self):
        xyxy = np.array([[10, 20, 50, 80], [60, 10, 190, 90], [5, 5, 15, 15]], dtype=float)
        confs = np.array([0.9, 0.7, 0.5])
        cls = np.array([0, 1, 2])
        names = {0: "person", 1: "car", 2: "dog"}
        det = _make_detector_with_mock(xyxy, confs, cls, names)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_coordinates_normalized_0_1(self):
        xyxy = np.array([[10, 20, 50, 80]], dtype=float)
        confs = np.array([0.9])
        cls = np.array([0])
        names = {0: "cat"}
        det = _make_detector_with_mock(xyxy, confs, cls, names, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        d = result[0]
        assert 0 <= d.bbox_xyxy[0] <= 1
        assert 0 <= d.bbox_xyxy[1] <= 1
        assert 0 <= d.bbox_xyxy[2] <= 1
        assert 0 <= d.bbox_xyxy[3] <= 1

    def test_xywh_conversion_correct(self):
        """Verify xyxy→xywh math: xywh = (x1, y1, x2-x1, y2-y1) normalized."""
        xyxy = np.array([[20, 10, 80, 50]], dtype=float)
        confs = np.array([0.8])
        cls = np.array([0])
        names = {0: "box"}
        det = _make_detector_with_mock(xyxy, confs, cls, names, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        d = result[0]
        x1n, y1n, x2n, y2n = d.bbox_xyxy
        xn, yn, wn, hn = d.bbox_xywh_norm
        assert xn == pytest.approx(x1n)
        assert yn == pytest.approx(y1n)
        assert wn == pytest.approx(x2n - x1n)
        assert hn == pytest.approx(y2n - y1n)


class TestFiltering:

    def test_zero_width_filtered(self):
        """Box with x1==x2 → zero width → excluded."""
        xyxy = np.array([[50, 10, 50, 80]], dtype=float)
        confs = np.array([0.9])
        cls = np.array([0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "x"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert len(result) == 0

    def test_zero_height_filtered(self):
        xyxy = np.array([[10, 50, 80, 50]], dtype=float)
        confs = np.array([0.9])
        cls = np.array([0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "x"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert len(result) == 0

    def test_inverted_box_filtered(self):
        """x2 < x1 → negative width → excluded."""
        xyxy = np.array([[80, 10, 20, 90]], dtype=float)
        confs = np.array([0.9])
        cls = np.array([0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "x"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert len(result) == 0

    def test_no_detections_empty(self):
        xyxy = np.empty((0, 4), dtype=float)
        confs = np.empty((0,))
        cls = np.empty((0,))
        det = _make_detector_with_mock(xyxy, confs, cls, {})
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert result == []

    def test_no_results_at_all(self):
        """Model returns empty list."""
        cfg = YOLOConfig(device="cpu", model_path="fake.pt")
        detector = YoloBBoxDetector(config=cfg)
        detector._model = MagicMock(return_value=[])
        result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert result == []

    def test_boxes_none(self):
        """Result.boxes is None."""
        cfg = YOLOConfig(device="cpu", model_path="fake.pt")
        detector = YoloBBoxDetector(config=cfg)
        mock_result = MagicMock()
        mock_result.boxes = None
        detector._model = MagicMock(return_value=[mock_result])
        result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert result == []


class TestSorting:

    def test_sorted_by_score_descending(self):
        xyxy = np.array([[0, 0, 50, 50], [60, 60, 100, 100], [0, 0, 30, 30]], dtype=float)
        confs = np.array([0.3, 0.9, 0.6])
        cls = np.array([0, 0, 0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "a"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        scores = [d.score for d in result]
        assert scores == sorted(scores, reverse=True)


class TestMetadata:

    def test_class_name_from_names_dict(self):
        xyxy = np.array([[10, 10, 50, 50]], dtype=float)
        confs = np.array([0.8])
        cls = np.array([2])
        names = {0: "cat", 1: "dog", 2: "horse"}
        det = _make_detector_with_mock(xyxy, confs, cls, names, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        assert result[0].class_name == "horse"

    def test_frame_idx_propagated(self):
        xyxy = np.array([[10, 10, 50, 50]], dtype=float)
        confs = np.array([0.8])
        cls = np.array([0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "x"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=42)
        assert result[0].frame_idx == 42

    def test_out_of_bounds_clamped(self):
        """Coordinates exceeding image bounds are clamped to [0,1]."""
        xyxy = np.array([[-10, -20, 300, 150]], dtype=float)
        confs = np.array([0.9])
        cls = np.array([0])
        det = _make_detector_with_mock(xyxy, confs, cls, {0: "x"}, h=100, w=200)
        result = det.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_idx=0)
        d = result[0]
        assert d.bbox_xyxy[0] == 0.0
        assert d.bbox_xyxy[1] == 0.0
        assert d.bbox_xyxy[2] == 1.0
        assert d.bbox_xyxy[3] == 1.0


class TestYoloDetectionDataclass:

    def test_fields_populated(self):
        d = YoloDetection(
            class_name="person",
            score=0.95,
            frame_idx=7,
            bbox_xyxy=(0.1, 0.2, 0.3, 0.4),
            bbox_xywh_norm=(0.1, 0.2, 0.2, 0.2),
        )
        assert d.class_name == "person"
        assert d.score == 0.95
        assert d.frame_idx == 7
