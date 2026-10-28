"""Observation-only image quality checks for colony images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


# These values are provisional warning thresholds.
# They do not affect SAM3 inference or detection results.
MIN_RECOMMENDED_WIDTH = 512
MIN_RECOMMENDED_HEIGHT = 512

DARK_BRIGHTNESS_THRESHOLD = 45.0
BRIGHT_BRIGHTNESS_THRESHOLD = 220.0
LOW_CONTRAST_THRESHOLD = 20.0
BLUR_SHARPNESS_THRESHOLD = 35.0

QUALITY_SAMPLE_LONG_SIDE = 1024


def _resize_for_quality_check(
    image: Image.Image,
) -> Image.Image:
    """
    Create a bounded copy for quality measurement.

    The original image is not modified.
    """

    sample = image.convert("RGB").copy()
    width, height = sample.size
    long_side = max(width, height)

    if long_side <= QUALITY_SAMPLE_LONG_SIDE:
        return sample

    scale = (
        QUALITY_SAMPLE_LONG_SIDE
        / float(long_side)
    )

    resized_width = max(
        1,
        int(round(width * scale)),
    )

    resized_height = max(
        1,
        int(round(height * scale)),
    )

    return sample.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )


def _warning(
    code: str,
    message: str,
    severity: str = "warning",
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
    }


def analyze_image_quality_shadow(
    image: Image.Image,
) -> dict[str, Any]:
    """
    Measure basic image quality without changing inference.

    This is a Shadow-only diagnostic. It does not reject images,
    change thresholds, alter pixels passed to SAM3, or modify
    detection counts.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "PIL Image 형식이 필요합니다."
        )

    original_width, original_height = (
        image.size
    )

    sample = _resize_for_quality_check(
        image
    )

    grayscale = np.asarray(
        sample.convert("L"),
        dtype=np.uint8,
    )

    if grayscale.size == 0:
        raise ValueError(
            "이미지 픽셀을 확인할 수 없습니다."
        )

    brightness_mean = float(
        np.mean(grayscale)
    )

    contrast_std = float(
        np.std(grayscale)
    )

    sharpness_variance = float(
        cv2.Laplacian(
            grayscale,
            cv2.CV_64F,
        ).var()
    )

    warnings: list[dict[str, str]] = []

    if (
        original_width
        < MIN_RECOMMENDED_WIDTH
        or original_height
        < MIN_RECOMMENDED_HEIGHT
    ):
        warnings.append(
            _warning(
                code="LOW_RESOLUTION",
                message=(
                    "권장 해상도보다 작은 이미지입니다. "
                    "가로와 세로 모두 512픽셀 이상을 "
                    "권장합니다."
                ),
            )
        )

    if (
        brightness_mean
        < DARK_BRIGHTNESS_THRESHOLD
    ):
        warnings.append(
            _warning(
                code="TOO_DARK",
                message=(
                    "이미지가 지나치게 어두울 수 있습니다. "
                    "조명과 노출을 확인하세요."
                ),
            )
        )

    if (
        brightness_mean
        > BRIGHT_BRIGHTNESS_THRESHOLD
    ):
        warnings.append(
            _warning(
                code="TOO_BRIGHT",
                message=(
                    "이미지가 지나치게 밝거나 "
                    "과노출되었을 수 있습니다."
                ),
            )
        )

    if (
        contrast_std
        < LOW_CONTRAST_THRESHOLD
    ):
        warnings.append(
            _warning(
                code="LOW_CONTRAST",
                message=(
                    "이미지 대비가 낮아 콜로니와 "
                    "배경의 구분이 어려울 수 있습니다."
                ),
            )
        )

    if (
        sharpness_variance
        < BLUR_SHARPNESS_THRESHOLD
    ):
        warnings.append(
            _warning(
                code="POSSIBLE_BLUR",
                message=(
                    "이미지가 흐리거나 초점이 맞지 "
                    "않았을 가능성이 있습니다."
                ),
            )
        )

    return {
        "schema_version": 1,
        "mode": "shadow",
        "status": (
            "warning"
            if warnings
            else "passed"
        ),
        "applied_to_result": False,
        "blocks_inference": False,
        "sample_width": sample.size[0],
        "sample_height": sample.size[1],
        "metrics": {
            "original_width": original_width,
            "original_height": original_height,
            "brightness_mean": round(
                brightness_mean,
                2,
            ),
            "contrast_std": round(
                contrast_std,
                2,
            ),
            "sharpness_variance": round(
                sharpness_variance,
                2,
            ),
        },
        "thresholds": {
            "minimum_recommended_width":
                MIN_RECOMMENDED_WIDTH,
            "minimum_recommended_height":
                MIN_RECOMMENDED_HEIGHT,
            "dark_brightness":
                DARK_BRIGHTNESS_THRESHOLD,
            "bright_brightness":
                BRIGHT_BRIGHTNESS_THRESHOLD,
            "low_contrast":
                LOW_CONTRAST_THRESHOLD,
            "possible_blur":
                BLUR_SHARPNESS_THRESHOLD,
        },
        "warning_count": len(warnings),
        "warnings": warnings,
        "notes": [
            (
                "이 검사는 참고용 Shadow 분석이며 "
                "SAM3 입력 이미지나 검출 결과를 "
                "변경하지 않습니다."
            ),
            (
                "품질 경고가 없더라도 학습 범위 밖 "
                "이미지의 검출 정확도를 보장하지 않습니다."
            ),
        ],
    }