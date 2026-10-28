"""Unit tests for rendering one requested result layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    value = str(candidate)

    if value not in sys.path:
        sys.path.insert(0, value)

from services.layer_renderer import create_overlay_layer


class SingleLayerRendererTests(unittest.TestCase):
    def setUp(self):
        self.base = Image.new("RGB", (32, 24), (20, 30, 40))
        self.mask = np.zeros((24, 32), dtype=bool)
        self.mask[4:12, 5:15] = True

    def test_base_layer_is_rgb_copy(self):
        layer = create_overlay_layer(
            layer_name="base",
            base_image=self.base,
        )
        self.assertEqual(layer.mode, "RGB")
        self.assertEqual(layer.size, (32, 24))
        self.assertIsNot(layer, self.base)

    def test_text_layer_is_transparent_rgba(self):
        layer = create_overlay_layer(
            layer_name="text",
            base_image=self.base,
            text_masks=[self.mask],
        )
        self.assertEqual(layer.mode, "RGBA")
        self.assertEqual(layer.size, (32, 24))
        self.assertGreater(
            np.asarray(layer)[..., 3].sum(),
            0,
        )

    def test_visual_and_box_layers_render(self):
        visual = create_overlay_layer(
            layer_name="visual",
            base_image=self.base,
            visual_masks=[self.mask],
        )
        boxes = create_overlay_layer(
            layer_name="boxes",
            base_image=self.base,
            boxes=[[2, 3, 20, 18]],
        )
        self.assertGreater(np.asarray(visual)[..., 3].sum(), 0)
        self.assertGreater(np.asarray(boxes)[..., 3].sum(), 0)

    def test_unknown_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            create_overlay_layer(
                layer_name="unknown",
                base_image=self.base,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
