"""Unit tests for dynamic SAM3 mask threshold conversion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

BACKEND_ROOT = (
    PROJECT_ROOT
    / "backend"
)

SAM3_ROOT = (
    PROJECT_ROOT
    / "sam3"
)

for path in (
    PROJECT_ROOT,
    BACKEND_ROOT,
    SAM3_ROOT,
):
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(
            0,
            path_text,
        )


from backend.services.sam3_service import (  # noqa: E402
    masks_to_numpy,
)


class DynamicMaskThresholdTests(
    unittest.TestCase
):
    def setUp(self):
        self.probabilities = np.array(
            [
                [
                    [
                        [0.0, 0.25, 0.5],
                        [0.51, 0.75, 1.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )

    def test_default_threshold_is_point_five(self):
        result = masks_to_numpy(
            self.probabilities
        )

        expected = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result[0],
            expected,
        )

    def test_custom_low_threshold(self):
        result = masks_to_numpy(
            self.probabilities,
            threshold=0.25,
        )

        expected = np.array(
            [
                [0, 0, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result[0],
            expected,
        )

    def test_custom_high_threshold(self):
        result = masks_to_numpy(
            self.probabilities,
            threshold=0.75,
        )

        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result[0],
            expected,
        )

    def test_zero_threshold_is_allowed(self):
        result = masks_to_numpy(
            self.probabilities,
            threshold=0.0,
        )

        expected = np.array(
            [
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result[0],
            expected,
        )

    def test_one_threshold_is_allowed(self):
        result = masks_to_numpy(
            self.probabilities,
            threshold=1.0,
        )

        expected = np.zeros(
            (2, 3),
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result[0],
            expected,
        )

    def test_invalid_negative_threshold(self):
        with self.assertRaises(
            ValueError
        ):
            masks_to_numpy(
                self.probabilities,
                threshold=-0.01,
            )

    def test_invalid_threshold_above_one(self):
        with self.assertRaises(
            ValueError
        ):
            masks_to_numpy(
                self.probabilities,
                threshold=1.01,
            )

    def test_nan_threshold_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            masks_to_numpy(
                self.probabilities,
                threshold=float("nan"),
            )

    def test_input_probabilities_are_not_modified(self):
        before = (
            self.probabilities.copy()
        )

        masks_to_numpy(
            self.probabilities,
            threshold=0.3,
        )

        np.testing.assert_array_equal(
            self.probabilities,
            before,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )