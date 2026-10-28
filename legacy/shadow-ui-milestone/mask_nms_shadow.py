"""Observation-only Mask-NMS analysis.

This module calculates possible duplicate masks without modifying the
original SAM3 masks, scores, counts, or rendered result.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy_mask(
    mask: Any,
    mask_threshold: float,
) -> np.ndarray:
    """Convert a SAM3 mask to a copied 2D boolean NumPy array."""

    value = mask

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "float"):
        value = value.float()

    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            "Mask-NMS Shadow는 2차원 마스크만 분석할 수 있습니다. "
            f"현재 shape: {array.shape}"
        )

    if array.dtype == np.bool_:
        return array.copy()

    return (array > mask_threshold).copy()


def _to_float_score(value: Any) -> float:
    """Convert tensors, NumPy values, or numbers to a Python float."""

    score = value

    if hasattr(score, "detach"):
        score = score.detach()

    if hasattr(score, "cpu"):
        score = score.cpu()

    if hasattr(score, "item"):
        try:
            return float(score.item())
        except (TypeError, ValueError):
            pass

    array = np.asarray(score).reshape(-1)

    if array.size == 0:
        return 0.0

    return float(array[0])


def calculate_mask_iou(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> float:
    """Calculate IoU between two boolean masks."""

    if first_mask.shape != second_mask.shape:
        raise ValueError(
            "비교할 마스크 크기가 서로 다릅니다: "
            f"{first_mask.shape} != {second_mask.shape}"
        )

    intersection = np.logical_and(
        first_mask,
        second_mask,
    ).sum(dtype=np.int64)

    union = np.logical_or(
        first_mask,
        second_mask,
    ).sum(dtype=np.int64)

    if union == 0:
        return 0.0

    return float(intersection / union)


def _build_candidates(
    source: str,
    masks: list[Any],
    scores: list[Any],
    mask_threshold: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for index, mask in enumerate(masks):
        score = (
            _to_float_score(scores[index])
            if index < len(scores)
            else 0.0
        )

        candidates.append(
            {
                "candidate_id": f"{source}-{index + 1}",
                "source": source,
                "source_index": index,
                "score": score,
                "mask": _to_numpy_mask(
                    mask,
                    mask_threshold,
                ),
            }
        )

    return candidates


def analyze_mask_nms_shadow(
    *,
    text_masks: list[Any],
    text_scores: list[Any],
    visual_masks: list[Any],
    visual_scores: list[Any],
    iou_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run score-ordered Mask-NMS as an observation-only calculation.

    Returned counts are Shadow results. The input lists and masks are never
    changed, and the returned Shadow result must not replace SAM3 counts.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "IoU Threshold는 0.0에서 1.0 사이여야 합니다."
        )

    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError(
            "Mask Threshold는 0.0에서 1.0 사이여야 합니다."
        )

    candidates = [
        *_build_candidates(
            "text",
            list(text_masks),
            list(text_scores),
            mask_threshold,
        ),
        *_build_candidates(
            "visual",
            list(visual_masks),
            list(visual_scores),
            mask_threshold,
        ),
    ]

    source_priority = {
        "text": 0,
        "visual": 1,
    }

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            source_priority[candidate["source"]],
            candidate["source_index"],
        )
    )

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    warnings: list[str] = []

    for candidate in candidates:
        duplicate_of = None
        duplicate_iou = 0.0

        for kept_candidate in kept:
            if (
                candidate["mask"].shape
                != kept_candidate["mask"].shape
            ):
                warnings.append(
                    "마스크 크기가 달라 비교를 건너뛰었습니다: "
                    f'{candidate["candidate_id"]} / '
                    f'{kept_candidate["candidate_id"]}'
                )
                continue

            iou = calculate_mask_iou(
                candidate["mask"],
                kept_candidate["mask"],
            )

            if iou >= iou_threshold:
                duplicate_of = kept_candidate
                duplicate_iou = iou
                break

        if duplicate_of is None:
            kept.append(candidate)
            continue

        suppressed.append(
            {
                "candidate_id":
                    candidate["candidate_id"],
                "source": candidate["source"],
                "score": round(
                    candidate["score"],
                    6,
                ),
                "duplicate_of":
                    duplicate_of["candidate_id"],
                "iou": round(
                    duplicate_iou,
                    6,
                ),
            }
        )

    kept_candidates = [
        {
            "candidate_id":
                candidate["candidate_id"],
            "source": candidate["source"],
            "score": round(
                candidate["score"],
                6,
            ),
        }
        for candidate in kept
    ]

    return {
        "mode": "shadow",
        "applied_to_result": False,
        "iou_threshold": iou_threshold,
        "mask_threshold": mask_threshold,
        "raw_count": len(candidates),
        "shadow_unique_count": len(kept),
        "shadow_duplicate_count": len(suppressed),
        "kept_candidates": kept_candidates,
        "suppressed_candidates": suppressed,
        "warnings": warnings,
    }