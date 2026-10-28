"""Ground-truth based Precision, Recall, and F1 evaluation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


EVALUATION_IOU_THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_PATH = (
    PROJECT_ROOT
    / "1. test 원본 이미지 (어노테이션 포함)"
    / "_annotations.coco.json"
)


def _boolean_mask(mask: Any) -> np.ndarray:
    value = np.asarray(mask)

    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]

    if value.ndim != 2:
        raise ValueError("마스크는 2차원 배열이어야 합니다.")

    return value.astype(bool, copy=False)


def _resize_nearest(
    mask: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    source = _boolean_mask(mask)

    if source.shape == (target_height, target_width):
        return source.copy()

    y_index = np.minimum(
        (
            np.arange(target_height, dtype=np.float64)
            * source.shape[0]
            / target_height
        ).astype(np.int64),
        source.shape[0] - 1,
    )
    x_index = np.minimum(
        (
            np.arange(target_width, dtype=np.float64)
            * source.shape[1]
            / target_width
        ).astype(np.int64),
        source.shape[1] - 1,
    )

    return source[np.ix_(y_index, x_index)].copy()


def mask_iou(first: Any, second: Any) -> float:
    left = _boolean_mask(first)
    right = _boolean_mask(second)

    if left.shape != right.shape:
        raise ValueError("비교할 마스크 크기가 서로 다릅니다.")

    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)

    if union == 0:
        return 0.0

    return float(intersection / union)


@lru_cache(maxsize=1)
def _load_coco_document() -> dict[str, Any]:
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def unavailable_evaluation(
    stage: str,
    message: str,
) -> dict[str, Any]:
    return {
        "available": False,
        "stage": stage,
        "iou_threshold": EVALUATION_IOU_THRESHOLD,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "prediction_count": None,
        "gt_count": None,
        "message": message,
    }


def load_ground_truth_masks(
    original_filename: str,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    if not ANNOTATION_PATH.is_file():
        return {
            "available": False,
            "masks": [],
            "message": "COCO 정답 어노테이션 파일을 찾을 수 없습니다.",
        }

    try:
        document = _load_coco_document()
        requested_name = Path(str(original_filename)).name.casefold()

        image_record = next(
            (
                item
                for item in document.get("images", [])
                if Path(str(item.get("file_name", ""))).name.casefold()
                == requested_name
            ),
            None,
        )

        if image_record is None:
            return {
                "available": False,
                "masks": [],
                "message": (
                    "업로드한 파일명과 일치하는 GT가 "
                    "어노테이션 목록에 없습니다."
                ),
            }

        # Imported lazily so metric-only tests do not require pycocotools.
        from pycocotools import mask as mask_utils

        masks: list[np.ndarray] = []

        for annotation in document.get("annotations", []):
            if annotation.get("image_id") != image_record.get("id"):
                continue

            segmentation = annotation.get("segmentation")

            if not isinstance(segmentation, dict):
                continue

            rle = dict(segmentation)

            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")

            decoded = np.asarray(mask_utils.decode(rle))

            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)

            masks.append(
                _resize_nearest(
                    decoded,
                    target_width,
                    target_height,
                )
            )

        return {
            "available": True,
            "masks": masks,
            "message": "",
        }
    except Exception as error:
        return {
            "available": False,
            "masks": [],
            "message": (
                "GT 어노테이션을 읽지 못했습니다: "
                f"{error}"
            ),
        }


def evaluate_masks(
    prediction_masks: list[Any],
    ground_truth_masks: list[Any],
    stage: str,
    iou_threshold: float = EVALUATION_IOU_THRESHOLD,
) -> dict[str, Any]:
    predictions = [_boolean_mask(mask) for mask in prediction_masks]
    ground_truth = [_boolean_mask(mask) for mask in ground_truth_masks]
    pairs: list[tuple[float, int, int]] = []

    for prediction_index, prediction in enumerate(predictions):
        for gt_index, gt_mask in enumerate(ground_truth):
            iou = mask_iou(prediction, gt_mask)

            if iou >= iou_threshold:
                pairs.append((iou, prediction_index, gt_index))

    pairs.sort(key=lambda item: item[0], reverse=True)
    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()

    for _, prediction_index, gt_index in pairs:
        if (
            prediction_index in matched_predictions
            or gt_index in matched_ground_truth
        ):
            continue

        matched_predictions.add(prediction_index)
        matched_ground_truth.add(gt_index)

    true_positive = len(matched_predictions)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "available": True,
        "stage": stage,
        "iou_threshold": float(iou_threshold),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1_score), 6),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "prediction_count": len(predictions),
        "gt_count": len(ground_truth),
        "message": "",
    }
