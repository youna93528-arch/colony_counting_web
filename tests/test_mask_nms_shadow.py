"""Tests for observation-only Mask-NMS."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from services.mask_nms_shadow import (
    analyze_mask_nms_shadow,
    calculate_mask_iou,
)

class MaskNmsShadowTests(unittest.TestCase):
    def test_identical_masks_are_duplicates(self):
        first = np.zeros((10, 10), dtype=bool)
        first[2:6, 2:6] = True

        second = first.copy()

        result = analyze_mask_nms_shadow(
            text_masks=[first],
            text_scores=[0.95],
            visual_masks=[second],
            visual_scores=[0.80],
            iou_threshold=0.5,
        )

        self.assertEqual(result["raw_count"], 2)
        self.assertEqual(
            result["shadow_unique_count"],
            1,
        )
        self.assertEqual(
            result["shadow_duplicate_count"],
            1,
        )
        self.assertFalse(
            result["applied_to_result"]
        )

    def test_separated_masks_are_preserved(self):
        first = np.zeros((10, 10), dtype=bool)
        first[1:4, 1:4] = True

        second = np.zeros((10, 10), dtype=bool)
        second[6:9, 6:9] = True

        result = analyze_mask_nms_shadow(
            text_masks=[first],
            text_scores=[0.90],
            visual_masks=[second],
            visual_scores=[0.85],
            iou_threshold=0.5,
        )

        self.assertEqual(result["raw_count"], 2)
        self.assertEqual(
            result["shadow_unique_count"],
            2,
        )
        self.assertEqual(
            result["shadow_duplicate_count"],
            0,
        )

    def test_highest_score_candidate_is_kept(self):
        first = np.zeros((10, 10), dtype=bool)
        first[2:7, 2:7] = True

        second = first.copy()

        result = analyze_mask_nms_shadow(
            text_masks=[first],
            text_scores=[0.60],
            visual_masks=[second],
            visual_scores=[0.95],
            iou_threshold=0.5,
        )

        self.assertEqual(
            result["kept_candidates"][0][
                "candidate_id"
            ],
            "visual-1",
        )

        self.assertEqual(
            result["suppressed_candidates"][0][
                "duplicate_of"
            ],
            "visual-1",
        )

    def test_empty_masks_have_zero_iou(self):
        first = np.zeros((10, 10), dtype=bool)
        second = np.zeros((10, 10), dtype=bool)

        self.assertEqual(
            calculate_mask_iou(first, second),
            0.0,
        )

    def test_input_masks_are_not_modified(self):
        first = np.zeros(
            (10, 10),
            dtype=np.float32,
        )
        first[2:6, 2:6] = 0.9

        original = first.copy()

        analyze_mask_nms_shadow(
            text_masks=[first],
            text_scores=[0.90],
            visual_masks=[],
            visual_scores=[],
            iou_threshold=0.5,
            mask_threshold=0.5,
        )

        np.testing.assert_array_equal(
            first,
            original,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)