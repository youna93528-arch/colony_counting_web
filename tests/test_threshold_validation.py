"""Unit tests for configurable inference threshold validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.services.threshold_validation import (  # noqa: E402
    ThresholdValidationError,
    validate_inference_thresholds,
    validate_threshold,
)


class ThresholdValidationTests(
    unittest.TestCase
):
    def test_zero_is_allowed(self):
        self.assertEqual(
            validate_threshold(
                0,
                "Confidence Threshold",
            ),
            0.0,
        )

    def test_one_is_allowed(self):
        self.assertEqual(
            validate_threshold(
                1,
                "Confidence Threshold",
            ),
            1.0,
        )

    def test_decimal_is_allowed(self):
        self.assertEqual(
            validate_threshold(
                0.37,
                "Mask Threshold",
            ),
            0.37,
        )

    def test_numeric_string_is_allowed(self):
        self.assertEqual(
            validate_threshold(
                "0.25",
                "Mask Threshold",
            ),
            0.25,
        )

    def test_both_thresholds_are_returned(self):
        confidence, mask = (
            validate_inference_thresholds(
                0.4,
                0.7,
            )
        )

        self.assertEqual(
            confidence,
            0.4,
        )
        self.assertEqual(
            mask,
            0.7,
        )

    def test_negative_value_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                -0.01,
                "Confidence Threshold",
            )

    def test_value_above_one_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                1.01,
                "Mask Threshold",
            )

    def test_nan_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                float("nan"),
                "Confidence Threshold",
            )

    def test_positive_infinity_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                float("inf"),
                "Confidence Threshold",
            )

    def test_negative_infinity_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                float("-inf"),
                "Confidence Threshold",
            )

    def test_boolean_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                True,
                "Mask Threshold",
            )

    def test_none_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                None,
                "Mask Threshold",
            )

    def test_empty_string_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                "",
                "Mask Threshold",
            )

    def test_non_numeric_string_is_rejected(self):
        with self.assertRaises(
            ThresholdValidationError
        ):
            validate_threshold(
                "abc",
                "Confidence Threshold",
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )