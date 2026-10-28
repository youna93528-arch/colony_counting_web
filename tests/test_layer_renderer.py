"""Tests for independent UI overlay layers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from services.layer_renderer import create_overlay_layers


class LayerRendererTests(unittest.TestCase):
    def setUp(self):
        self.base = Image.new("RGB", (32, 32), (120, 120, 120))
        self.text_mask = np.zeros((32, 32), dtype=bool)
        self.text_mask[4:12, 4:12] = True
        self.visual_mask = np.zeros((32, 32), dtype=bool)
        self.visual_mask[18:26, 18:26] = True

    def test_layers_have_expected_modes_and_sizes(self):
        layers = create_overlay_layers(
            base_image=self.base,
            text_masks=[self.text_mask],
            visual_masks=[self.visual_mask],
            boxes=[[2, 2, 14, 14]],
        )

        self.assertEqual(set(layers), {"base", "text", "visual", "boxes"})
        self.assertEqual(layers["base"].mode, "RGB")

        for name in ("text", "visual", "boxes"):
            self.assertEqual(layers[name].mode, "RGBA")
            self.assertEqual(layers[name].size, (32, 32))

    def test_transparent_layers_contain_visible_pixels(self):
        layers = create_overlay_layers(
            base_image=self.base,
            text_masks=[self.text_mask],
            visual_masks=[self.visual_mask],
            boxes=[[2, 2, 14, 14]],
        )

        for name in ("text", "visual", "boxes"):
            alpha = np.asarray(layers[name])[:, :, 3]
            self.assertGreater(int(alpha.max()), 0)

    def test_input_masks_are_not_modified(self):
        original_text = self.text_mask.copy()
        original_visual = self.visual_mask.copy()

        create_overlay_layers(
            base_image=self.base,
            text_masks=[self.text_mask],
            visual_masks=[self.visual_mask],
            boxes=[],
        )

        np.testing.assert_array_equal(self.text_mask, original_text)
        np.testing.assert_array_equal(self.visual_mask, original_visual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
