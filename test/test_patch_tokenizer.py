"""Tests for PatchToken and mask_to_patch_tokens."""

from __future__ import annotations

import numpy as np
import pytest

from fast_snow.reasoning.tokens.patch_tokenizer import PatchToken, mask_to_patch_tokens


H, W = 64, 64  # default test dimensions


class TestMaskToPatchTokensBasic:

    def test_full_mask_all_cells_retained(self):
        mask = np.ones((256, 256), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=16, iou_threshold=0.5)
        assert len(tokens) == 16 * 16

    def test_empty_mask_no_tokens(self):
        mask = np.zeros((H, W), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.1)
        assert tokens == []

    def test_half_mask_top(self):
        """Top half of image is True → roughly half the cells should pass."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :] = True
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.5)
        # Top 2 rows of 4x4 grid → 8 cells should have 100% coverage
        top_rows = [t for t in tokens if t.row < 2]
        assert len(top_rows) == 8

    def test_threshold_boundary_exact_05_excluded(self):
        """A cell with exactly 50% coverage has iou=0.5, which is NOT > 0.5,
        so it must be excluded."""
        # 4x4 grid on 16x16 mask → each cell is 4x4=16 pixels
        mask = np.zeros((16, 16), dtype=bool)
        # Fill exactly half (8 of 16 pixels) in cell (0,0)
        mask[0:2, 0:4] = True  # 2 rows × 4 cols = 8 pixels = 50%
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.5)
        cell_00 = [t for t in tokens if t.row == 0 and t.col == 0]
        assert len(cell_00) == 0, "Exactly 50% should be excluded (> not >=)"

    def test_threshold_boundary_just_above(self):
        """A cell with >50% coverage should be included."""
        mask = np.zeros((16, 16), dtype=bool)
        # 9 of 16 pixels → 56.25%
        mask[0:3, 0:3] = True  # 9 pixels in cell (0,0)
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.5)
        cell_00 = [t for t in tokens if t.row == 0 and t.col == 0]
        assert len(cell_00) == 1

    def test_custom_grid_size_4(self):
        mask = np.ones((H, W), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.0)
        assert len(tokens) == 4 * 4

    def test_grid_size_1(self):
        """Entire image is one cell."""
        mask = np.ones((H, W), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=1, iou_threshold=0.0)
        assert len(tokens) == 1
        assert tokens[0].row == 0 and tokens[0].col == 0
        assert tokens[0].iou == pytest.approx(1.0)

    def test_non_divisible_dimensions(self):
        """480x640 with grid_size=16: last cells absorb remainder pixels."""
        mask = np.ones((480, 640), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=16, iou_threshold=0.0)
        assert len(tokens) == 256
        # Last cell (row=15) should cover rows 450..480 (30 rows, not 30)
        last_row_tokens = [t for t in tokens if t.row == 15]
        assert len(last_row_tokens) == 16

    def test_iou_values_correct(self):
        """Known mask pattern with predictable IoU.
        Note: uses iou > threshold, so threshold=-1 to include zero-coverage cells.
        """
        # 8x8 mask, grid_size=2 → each cell is 4x4=16 pixels
        mask = np.zeros((8, 8), dtype=bool)
        mask[0:4, 0:4] = True  # cell (0,0) fully covered → iou=1.0
        mask[0:2, 4:8] = True  # cell (0,1) half covered → iou=0.5
        # Use threshold < 0 so ALL cells are returned (including iou=0)
        tokens = mask_to_patch_tokens(mask, grid_size=2, iou_threshold=-1.0)
        token_map = {(t.row, t.col): t.iou for t in tokens}
        assert token_map[(0, 0)] == pytest.approx(1.0)
        assert token_map[(0, 1)] == pytest.approx(0.5)
        assert token_map[(1, 0)] == pytest.approx(0.0)
        assert token_map[(1, 1)] == pytest.approx(0.0)

    def test_row_col_indices_match_grid(self):
        mask = np.ones((H, W), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.0)
        rows = sorted(set(t.row for t in tokens))
        cols = sorted(set(t.col for t in tokens))
        assert rows == [0, 1, 2, 3]
        assert cols == [0, 1, 2, 3]

    def test_diagonal_thin_line_few_cells(self):
        """Thin diagonal line — only cells touching the line should pass."""
        mask = np.zeros((64, 64), dtype=bool)
        for i in range(64):
            mask[i, i] = True  # 1-pixel diagonal
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.01)
        # Each cell is 16x16=256 pixels; diagonal touches ~16 pixels per cell
        # on the diagonal → iou ≈ 16/256 ≈ 0.0625
        assert len(tokens) <= 4  # only diagonal cells


# ── Validation errors ──────────────────────────────────────────────────────

class TestValidation:

    def test_mask_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            mask_to_patch_tokens(np.ones((4, 4, 3), dtype=bool))

    def test_mask_too_small_raises(self):
        with pytest.raises(ValueError, match="grid_size"):
            mask_to_patch_tokens(np.ones((8, 8), dtype=bool), grid_size=16)

    def test_image_shape_mismatch_raises(self):
        mask = np.ones((H, W), dtype=bool)
        img = np.zeros((H + 1, W, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="mismatch"):
            mask_to_patch_tokens(mask, image=img)

    def test_image_not_3_channel_raises(self):
        mask = np.ones((H, W), dtype=bool)
        img = np.zeros((H, W, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            mask_to_patch_tokens(mask, image=img)

    def test_image_2d_raises(self):
        mask = np.ones((H, W), dtype=bool)
        img = np.zeros((H, W), dtype=np.uint8)
        with pytest.raises(ValueError):
            mask_to_patch_tokens(mask, image=img)


# ── Image crop behaviour ──────────────────────────────────────────────────

class TestImageCrops:

    def test_no_image_crop_is_none(self):
        mask = np.ones((H, W), dtype=bool)
        tokens = mask_to_patch_tokens(mask, grid_size=4, iou_threshold=0.0)
        assert all(t.image_crop is None for t in tokens)

    def test_with_image_crop_shape(self):
        mask = np.ones((H, W), dtype=bool)
        img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        tokens = mask_to_patch_tokens(
            mask, grid_size=4, iou_threshold=0.0, image=img
        )
        for t in tokens:
            assert t.image_crop is not None
            assert t.image_crop.ndim == 3
            assert t.image_crop.shape[2] == 3

    def test_mask_outside_true_zeros_non_mask(self):
        """When mask_outside=True, pixels outside mask should be 0."""
        mask = np.zeros((16, 16), dtype=bool)
        mask[0:4, 0:4] = True  # only top-left cell fully covered
        # Image with all pixels = 128
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        # Use threshold < 0 to get ALL cells including those with iou=0
        tokens = mask_to_patch_tokens(
            mask, grid_size=4, iou_threshold=-1.0, image=img, mask_outside=True
        )
        # Cell (0,0) should be all 128 (fully masked)
        t00 = [t for t in tokens if t.row == 0 and t.col == 0][0]
        assert t00.image_crop is not None
        assert np.all(t00.image_crop == 128)
        # Cell (0,1) has no mask → all zeros (mask_outside zeroed them)
        t01 = [t for t in tokens if t.row == 0 and t.col == 1][0]
        assert t01.image_crop is not None
        assert np.all(t01.image_crop == 0)

    def test_mask_outside_false_preserves_original(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[0:2, 0:2] = True
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        # Use threshold < 0 to get ALL cells including those with iou=0
        tokens = mask_to_patch_tokens(
            mask, grid_size=4, iou_threshold=-1.0, image=img, mask_outside=False
        )
        # Even unmasked cells should have original pixel values
        t11 = [t for t in tokens if t.row == 1 and t.col == 1][0]
        assert t11.image_crop is not None
        assert np.all(t11.image_crop == 200)

    def test_crop_resize(self):
        mask = np.ones((H, W), dtype=bool)
        img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        tokens = mask_to_patch_tokens(
            mask, grid_size=4, iou_threshold=0.0, image=img, crop_size=32
        )
        for t in tokens:
            assert t.image_crop is not None
            assert t.image_crop.shape == (32, 32, 3)

    def test_crop_size_none_native_resolution(self):
        """crop_size=None → keep native cell dimensions."""
        mask = np.ones((H, W), dtype=bool)
        img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        tokens = mask_to_patch_tokens(
            mask, grid_size=4, iou_threshold=0.0, image=img, crop_size=None
        )
        cell_h = H // 4
        cell_w = W // 4
        for t in tokens:
            assert t.image_crop is not None
            # Non-last cells should match exact cell size
            if t.row < 3 and t.col < 3:
                assert t.image_crop.shape[0] == cell_h
                assert t.image_crop.shape[1] == cell_w


# ── PatchToken dataclass ──────────────────────────────────────────────────

class TestPatchTokenDataclass:

    def test_image_crop_excluded_from_eq(self):
        """image_crop has compare=False → not used in equality."""
        a = PatchToken(row=1, col=2, iou=0.8, image_crop=np.zeros((4, 4, 3)))
        b = PatchToken(row=1, col=2, iou=0.8, image_crop=np.ones((4, 4, 3)))
        assert a == b

    def test_default_image_crop_none(self):
        t = PatchToken(row=0, col=0, iou=1.0)
        assert t.image_crop is None
