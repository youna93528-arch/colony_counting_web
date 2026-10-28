"""Tests for Text-to-Visual duplicate filtering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from services.visual_duplicate_filter import (
    calculate_mask_iou,
    filter_visual_duplicates_with_text,
)


class VisualDuplicateFilterTests(unittest.TestCase):
    def test_overlapping_visual_mask_is_removed(self):
        text = np.zeros((10, 10), dtype=bool)
        text[2:8, 2:8] = True

        visual = text.copy()

        result = filter_visual_duplicates_with_text(
            visual_masks=[visual],
            visual_scores=[0.9],
            text_masks=[text],
            iou_threshold=0.5,
        )

        self.assertEqual(result["raw_count"], 1)
        self.assertEqual(result["unique_count"], 0)
        self.assertEqual(result["duplicate_count"], 1)
        self.assertEqual(result["masks"], [])
        self.assertEqual(result["scores"], [])

    def test_separate_visual_mask_is_kept(self):
        text = np.zeros((10, 10), dtype=bool)
        text[1:4, 1:4] = True

        visual = np.zeros((10, 10), dtype=bool)
        visual[6:9, 6:9] = True

        result = filter_visual_duplicates_with_text(
            visual_masks=[visual],
            visual_scores=[0.8],
            text_masks=[text],
        )

        self.assertEqual(result["unique_count"], 1)
        self.assertEqual(result["duplicate_count"], 0)
        self.assertEqual(result["scores"], [0.8])
        np.testing.assert_array_equal(
            result["masks"][0],
            visual,
        )

    def test_scores_follow_kept_mask_indices(self):
        text = np.zeros((8, 8), dtype=bool)
        text[0:4, 0:4] = True

        duplicate = text.copy()

        unique = np.zeros((8, 8), dtype=bool)
        unique[5:8, 5:8] = True

        result = filter_visual_duplicates_with_text(
            visual_masks=[duplicate, unique],
            visual_scores=[0.91, 0.72],
            text_masks=[text],
        )

        self.assertEqual(result["kept_indices"], [1])
        self.assertEqual(result["scores"], [0.72])
        self.assertEqual(result["unique_count"], 1)

    def test_iou_threshold_boundary_is_removed(self):
        text = np.zeros((2, 4), dtype=bool)
        text[:, 0:2] = True

        visual = np.ones((2, 4), dtype=bool)

        self.assertEqual(
            calculate_mask_iou(visual, text),
            0.5,
        )

        result = filter_visual_duplicates_with_text(
            visual_masks=[visual],
            visual_scores=[0.5],
            text_masks=[text],
            iou_threshold=0.5,
        )

        self.assertEqual(result["unique_count"], 0)

    def test_empty_text_masks_keep_all_visual_masks(self):
        visual = np.ones((4, 4), dtype=bool)

        result = filter_visual_duplicates_with_text(
            visual_masks=[visual],
            visual_scores=[0.6],
            text_masks=[],
        )

        self.assertEqual(result["unique_count"], 1)
        self.assertEqual(result["scores"], [0.6])

    def test_inputs_are_not_modified(self):
        text = np.zeros((5, 5), dtype=bool)
        text[1:4, 1:4] = True

        visual = text.copy()
        text_before = text.copy()
        visual_before = visual.copy()

        filter_visual_duplicates_with_text(
            visual_masks=[visual],
            visual_scores=[0.9],
            text_masks=[text],
        )

        np.testing.assert_array_equal(text, text_before)
        np.testing.assert_array_equal(visual, visual_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
