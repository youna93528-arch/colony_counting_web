"""Tests for GT-based detection evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

for path in (PROJECT_ROOT, BACKEND_ROOT):
    value = str(path)

    if value not in sys.path:
        sys.path.insert(0, value)

from backend.services.detection_evaluation import (
    evaluate_masks,
    load_ground_truth_masks,
)


def square(top, left, bottom, right):
    mask = np.zeros((10, 10), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class DetectionEvaluationTests(unittest.TestCase):
    def test_perfect_prediction(self):
        target = square(1, 1, 4, 4)
        result = evaluate_masks([target], [target], stage="text")
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1_score"], 1.0)

    def test_extra_prediction_reduces_precision(self):
        first = square(1, 1, 4, 4)
        extra = square(6, 6, 9, 9)
        result = evaluate_masks([first, extra], [first], stage="final")
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1_score"], 2.0 / 3.0, places=5)

    def test_missed_gt_reduces_recall(self):
        first = square(1, 1, 4, 4)
        second = square(6, 6, 9, 9)
        result = evaluate_masks([first], [first, second], stage="text")
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)

    def test_matching_is_one_to_one(self):
        target = square(1, 1, 5, 5)
        result = evaluate_masks([target, target], [target], stage="final")
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)

    def test_known_coco_file_decodes_ground_truth(self):
        result = load_ground_truth_masks(
            "train_6-C-a-10-6-Na-3-_png.rf.9e171e28513accdd1549e2d5964bb05e.jpg",
            1008,
            1008,
        )
        self.assertTrue(result["available"], result["message"])
        self.assertGreater(len(result["masks"]), 0)
        self.assertEqual(result["masks"][0].shape, (1008, 1008))

    def test_unknown_filename_has_no_gt(self):
        result = load_ground_truth_masks(
            "__missing_gt_image__.jpg",
            1008,
            1008,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["masks"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
