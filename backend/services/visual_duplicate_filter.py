"""Remove Visual Prompt masks that duplicate Text Prompt detections."""

from __future__ import annotations

from typing import Any

import numpy as np


VISUAL_TEXT_IOU_THRESHOLD = 0.5


def _as_bool_mask(mask: Any) -> np.ndarray:
    """Return one copied two-dimensional boolean mask."""

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
            "Visual duplicate filtering requires 2D masks. "
            f"Current shape: {array.shape}"
        )

    return array.astype(bool, copy=True)


def _resize_nearest(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbor sampling."""

    target_height, target_width = target_shape

    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target mask size must be positive.")

    if mask.shape == target_shape:
        return mask.copy()

    source_height, source_width = mask.shape

    y_indices = np.minimum(
        (
            np.arange(target_height)
            * source_height
            / target_height
        ).astype(np.int64),
        source_height - 1,
    )

    x_indices = np.minimum(
        (
            np.arange(target_width)
            * source_width
            / target_width
        ).astype(np.int64),
        source_width - 1,
    )

    return mask[np.ix_(y_indices, x_indices)].copy()


def calculate_mask_iou(
    first_mask: Any,
    second_mask: Any,
) -> float:
    """Calculate mask IoU, resizing the first mask when needed."""

    first = _as_bool_mask(first_mask)
    second = _as_bool_mask(second_mask)

    if first.shape != second.shape:
        first = _resize_nearest(
            first,
            second.shape,
        )

    intersection = np.logical_and(
        first,
        second,
    ).sum(dtype=np.int64)

    union = np.logical_or(
        first,
        second,
    ).sum(dtype=np.int64)

    if union == 0:
        return 0.0

    return float(intersection / union)


def filter_visual_duplicates_with_text(
    *,
    visual_masks: list[Any],
    visual_scores: list[Any],
    text_masks: list[Any],
    iou_threshold: float = VISUAL_TEXT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """
    Keep only Visual masks whose maximum IoU with Text masks is below
    the threshold.

    Text masks always remain unchanged. Visual scores are filtered with
    the same indices so masks, scores, counts, and rendered layers agree.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "IoU threshold must be between 0.0 and 1.0."
        )

    text_mask_list = list(text_masks)
    visual_mask_list = list(visual_masks)
    visual_score_list = list(visual_scores)

    kept_masks: list[Any] = []
    kept_scores: list[Any] = []
    kept_indices: list[int] = []
    suppressed: list[dict[str, Any]] = []

    for visual_index, visual_mask in enumerate(
        visual_mask_list
    ):
        max_iou = 0.0
        duplicate_text_index = None

        for text_index, text_mask in enumerate(
            text_mask_list
        ):
            iou = calculate_mask_iou(
                visual_mask,
                text_mask,
            )

            if iou > max_iou:
                max_iou = iou
                duplicate_text_index = text_index

        if max_iou < iou_threshold:
            kept_indices.append(visual_index)
            kept_masks.append(visual_mask)

            if visual_index < len(visual_score_list):
                kept_scores.append(
                    visual_score_list[visual_index]
                )

            continue

        suppressed.append(
            {
                "visual_index": visual_index,
                "text_index": duplicate_text_index,
                "iou": round(max_iou, 6),
            }
        )

    return {
        "masks": kept_masks,
        "scores": kept_scores,
        "kept_indices": kept_indices,
        "suppressed": suppressed,
        "raw_count": len(visual_mask_list),
        "unique_count": len(kept_masks),
        "duplicate_count": len(suppressed),
        "iou_threshold": iou_threshold,
    }
