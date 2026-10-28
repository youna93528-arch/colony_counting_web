"""Unit tests for observation-only image quality checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.services.image_quality_shadow import (  # noqa: E402
    analyze_image_quality_shadow,
)


def warning_codes(
    result: dict,
) -> set[str]:
    return {
        warning["code"]
        for warning in result["warnings"]
    }


class ImageQualityShadowTests(
    unittest.TestCase
):
    def test_dark_image_is_reported(self):
        image = Image.new(
            "RGB",
            (1008, 1008),
            (15, 15, 15),
        )

        result = (
            analyze_image_quality_shadow(
                image
            )
        )

        self.assertEqual(
            result["mode"],
            "shadow",
        )
        self.assertFalse(
            result["applied_to_result"]
        )
        self.assertFalse(
            result["blocks_inference"]
        )
        self.assertIn(
            "TOO_DARK",
            warning_codes(result),
        )

    def test_bright_image_is_reported(self):
        image = Image.new(
            "RGB",
            (1008, 1008),
            (245, 245, 245),
        )

        result = (
            analyze_image_quality_shadow(
                image
            )
        )

        self.assertIn(
            "TOO_BRIGHT",
            warning_codes(result),
        )

    def test_small_image_is_reported(self):
        image = Image.new(
            "RGB",
            (320, 240),
            (120, 120, 120),
        )

        result = (
            analyze_image_quality_shadow(
                image
            )
        )

        self.assertIn(
            "LOW_RESOLUTION",
            warning_codes(result),
        )

    def test_detailed_noise_image_passes(self):
        generator = (
            np.random.default_rng(1234)
        )

        pixels = generator.integers(
            low=40,
            high=215,
            size=(768, 768, 3),
            dtype=np.uint8,
        )

        image = Image.fromarray(
            pixels,
            mode="RGB",
        )

        result = (
            analyze_image_quality_shadow(
                image
            )
        )

        codes = warning_codes(result)

        self.assertNotIn(
            "TOO_DARK",
            codes,
        )
        self.assertNotIn(
            "TOO_BRIGHT",
            codes,
        )
        self.assertNotIn(
            "LOW_CONTRAST",
            codes,
        )
        self.assertNotIn(
            "POSSIBLE_BLUR",
            codes,
        )

    def test_input_image_is_not_modified(self):
        generator = (
            np.random.default_rng(7)
        )

        pixels = generator.integers(
            low=0,
            high=256,
            size=(600, 800, 3),
            dtype=np.uint8,
        )

        image = Image.fromarray(
            pixels,
            mode="RGB",
        )

        before_size = image.size
        before_mode = image.mode
        before_pixels = np.asarray(
            image
        ).copy()

        analyze_image_quality_shadow(
            image
        )

        self.assertEqual(
            image.size,
            before_size,
        )
        self.assertEqual(
            image.mode,
            before_mode,
        )

        np.testing.assert_array_equal(
            np.asarray(image),
            before_pixels,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )