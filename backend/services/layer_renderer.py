"""Render observation-only PNG layers for the analysis UI.

The renderer never mutates SAM3 masks, scores, boxes, or counts. It only
converts the existing inference output into transparent presentation layers.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _mask_to_bool(
    mask: Any,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    value = mask

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.squeeze(np.asarray(value))

    if array.ndim != 2:
        raise ValueError(
            "레이어 렌더링에는 2차원 마스크가 필요합니다. "
            f"현재 shape: {array.shape}"
        )

    if array.shape != (target_height, target_width):
        mask_image = Image.fromarray(
            (array.astype(bool).astype(np.uint8) * 255)
        )
        mask_image = mask_image.resize(
            (target_width, target_height),
            Image.Resampling.NEAREST,
        )
        array = np.asarray(mask_image) > 0

    return array.astype(bool, copy=True)


def _create_mask_layer(
    *,
    width: int,
    height: int,
    masks: list[Any],
    fill_color: tuple[int, int, int],
    contour_color: tuple[int, int, int],
    alpha: float = 0.20,
    halo_width: int = 6,
    contour_width: int = 3,
) -> Image.Image:
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    fill_rgba = (*fill_color, int(round(255 * alpha)))

    for mask in masks:
        mask_bool = _mask_to_bool(mask, width, height)
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        layer[mask_bool] = fill_rgba

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            layer,
            contours,
            -1,
            (0, 0, 0, 255),
            halo_width,
            lineType=cv2.LINE_AA,
        )
        cv2.drawContours(
            layer,
            contours,
            -1,
            (*contour_color, 255),
            contour_width,
            lineType=cv2.LINE_AA,
        )

    return Image.fromarray(layer, mode="RGBA")


def _create_box_layer(
    *,
    width: int,
    height: int,
    boxes: list[list[float]],
) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle(
            [
                int(x1) - 2,
                int(y1) - 2,
                int(x2) + 2,
                int(y2) + 2,
            ],
            outline=(0, 0, 0, 255),
            width=7,
        )
        draw.rectangle(
            [int(x1), int(y1), int(x2), int(y2)],
            outline=(255, 255, 0, 255),
            width=4,
        )

    return layer


def create_overlay_layer(
    *,
    layer_name: str,
    base_image: Image.Image,
    text_masks: list[Any] | None = None,
    visual_masks: list[Any] | None = None,
    boxes: list[list[float]] | None = None,
) -> Image.Image:
    """Render exactly one cacheable presentation layer."""

    width, height = base_image.size

    if layer_name == "base":
        return base_image.convert("RGB").copy()

    if layer_name == "text":
        return _create_mask_layer(
            width=width,
            height=height,
            masks=list(text_masks or []),
            fill_color=(255, 40, 40),
            contour_color=(255, 20, 20),
        )

    if layer_name == "visual":
        return _create_mask_layer(
            width=width,
            height=height,
            masks=list(visual_masks or []),
            fill_color=(30, 255, 80),
            contour_color=(0, 255, 60),
        )

    if layer_name == "boxes":
        return _create_box_layer(
            width=width,
            height=height,
            boxes=list(boxes or []),
        )

    raise ValueError(
        f"지원하지 않는 결과 레이어입니다: {layer_name}"
    )


def create_overlay_layers(
    *,
    base_image: Image.Image,
    text_masks: list[Any] | None = None,
    visual_masks: list[Any] | None = None,
    boxes: list[list[float]] | None = None,
) -> dict[str, Image.Image]:
    """Return a base image and three independently composable layers."""

    base = base_image.convert("RGB").copy()
    width, height = base.size

    return {
        "base": base,
        "text": _create_mask_layer(
            width=width,
            height=height,
            masks=list(text_masks or []),
            fill_color=(255, 40, 40),
            contour_color=(255, 20, 20),
        ),
        "visual": _create_mask_layer(
            width=width,
            height=height,
            masks=list(visual_masks or []),
            fill_color=(30, 255, 80),
            contour_color=(0, 255, 60),
        ),
        "boxes": _create_box_layer(
            width=width,
            height=height,
            boxes=list(boxes or []),
        ),
    }
