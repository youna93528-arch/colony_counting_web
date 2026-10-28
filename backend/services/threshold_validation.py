"""Validation helpers for user-configurable inference thresholds."""

from __future__ import annotations

import math
from typing import Any


MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5


class ThresholdValidationError(ValueError):
    """Raised when an inference threshold is invalid."""


def validate_threshold(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate one threshold and return it as a float.

    Valid values include both boundary values 0 and 1.
    Boolean values are rejected even though bool is a subclass
    of int in Python.
    """

    if isinstance(value, bool):
        raise ThresholdValidationError(
            f"{field_name} 값은 0에서 1 사이의 숫자여야 합니다."
        )

    try:
        threshold = float(value)

    except (TypeError, ValueError) as error:
        raise ThresholdValidationError(
            f"{field_name} 값은 0에서 1 사이의 숫자여야 합니다."
        ) from error

    if not math.isfinite(threshold):
        raise ThresholdValidationError(
            f"{field_name} 값은 유한한 숫자여야 합니다."
        )

    if (
        threshold < MIN_THRESHOLD
        or threshold > MAX_THRESHOLD
    ):
        raise ThresholdValidationError(
            f"{field_name} 값은 0 이상 1 이하여야 합니다."
        )

    return threshold


def validate_inference_thresholds(
    confidence_threshold: Any,
    mask_threshold: Any,
) -> tuple[float, float]:
    """
    Validate Confidence and Mask Threshold together.

    The values are returned without rounding so the backend uses
    exactly the validated numeric values supplied by the user.
    """

    validated_confidence = validate_threshold(
        confidence_threshold,
        "Confidence Threshold",
    )

    validated_mask = validate_threshold(
        mask_threshold,
        "Mask Threshold",
    )

    return (
        validated_confidence,
        validated_mask,
    )