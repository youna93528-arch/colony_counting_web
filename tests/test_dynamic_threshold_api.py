"""Live API tests for dynamic Confidence and Mask Thresholds."""

from __future__ import annotations

import sys

from test_session_api import (
    BASE_URL,
    EXPECTED_TEXT_COUNT,
    assert_equal,
    request_json,
    upload_image,
)


def assert_float_equal(
    name,
    actual,
    expected,
):
    if abs(float(actual) - float(expected)) > 1e-9:
        raise AssertionError(
            f"{name}: 기준값={expected}, 현재값={actual}"
        )


def main():
    print("=" * 70)
    print("SAM3 Dynamic Threshold API 테스트")
    print("=" * 70)

    status, created = upload_image()

    assert_equal(
        "세션 생성 HTTP",
        status,
        201,
    )

    session_id = created["session_id"]

    print(
        "[PASS] 이미지 업로드 및 세션 생성: "
        f"{session_id[:8]}..."
    )

    try:
        # 1. 기존 기준 Threshold
        status, baseline = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={
                "prompt": "A",
                "confidence_threshold": 0.5,
                "mask_threshold": 0.5,
            },
        )

        assert_equal(
            "기준 Threshold HTTP",
            status,
            200,
        )

        assert_equal(
            "기준 TextCount",
            baseline["text_count"],
            EXPECTED_TEXT_COUNT,
        )

        assert_float_equal(
            "기준 Confidence",
            baseline["confidence_threshold"],
            0.5,
        )

        assert_float_equal(
            "기준 Mask Threshold",
            baseline["mask_threshold"],
            0.5,
        )

        print(
            "[PASS] 기준값 적용: "
            f"Text={baseline['text_count']}, "
            f"Confidence={baseline['confidence_threshold']}, "
            f"Mask={baseline['mask_threshold']}"
        )

        # 2. Confidence 1.0
        status, maximum_confidence = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={
                "prompt": "A",
                "confidence_threshold": 1.0,
                "mask_threshold": 0.5,
            },
        )

        assert_equal(
            "Confidence 1.0 HTTP",
            status,
            200,
        )

        assert_equal(
            "Confidence 1.0 TextCount",
            maximum_confidence["text_count"],
            0,
        )

        assert_float_equal(
            "Confidence 1.0 응답",
            maximum_confidence[
                "confidence_threshold"
            ],
            1.0,
        )

        print(
            "[PASS] Confidence 1.0 실제 적용: "
            f"Text={maximum_confidence['text_count']}"
        )

        # 3. Confidence 복원 및 Mask Threshold 변경
        status, restored = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={
                "prompt": "A",
                "confidence_threshold": 0.5,
                "mask_threshold": 0.75,
            },
        )

        assert_equal(
            "Threshold 복원 HTTP",
            status,
            200,
        )

        assert_equal(
            "Confidence 복원 TextCount",
            restored["text_count"],
            EXPECTED_TEXT_COUNT,
        )

        assert_float_equal(
            "복원 Confidence",
            restored["confidence_threshold"],
            0.5,
        )

        assert_float_equal(
            "변경 Mask Threshold",
            restored["mask_threshold"],
            0.75,
        )

        print(
            "[PASS] Threshold 재설정: "
            f"Text={restored['text_count']}, "
            f"Confidence={restored['confidence_threshold']}, "
            f"Mask={restored['mask_threshold']}"
        )

        # 4. 음수 Confidence 차단
        status, invalid_negative = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={
                "prompt": "A",
                "confidence_threshold": -0.01,
                "mask_threshold": 0.5,
            },
        )

        assert_equal(
            "음수 Confidence HTTP",
            status,
            400,
        )

        if invalid_negative.get("success") is not False:
            raise AssertionError(
                "음수 Confidence 요청이 차단되지 않았습니다."
            )

        print(
            "[PASS] 음수 Confidence HTTP 400 차단"
        )

        # 5. 1 초과 Mask Threshold 차단
        status, invalid_mask = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={
                "prompt": "A",
                "confidence_threshold": 0.5,
                "mask_threshold": 1.01,
            },
        )

        assert_equal(
            "Mask Threshold 1 초과 HTTP",
            status,
            400,
        )

        if invalid_mask.get("success") is not False:
            raise AssertionError(
                "1 초과 Mask Threshold 요청이 "
                "차단되지 않았습니다."
            )

        print(
            "[PASS] 1 초과 Mask Threshold HTTP 400 차단"
        )

        # 6. 잘못된 요청 이후 세션 설정 유지
        status, metadata = request_json(
            f"{BASE_URL}/{session_id}"
        )

        assert_equal(
            "세션 조회 HTTP",
            status,
            200,
        )

        assert_float_equal(
            "세션 Confidence 유지",
            metadata["confidence_threshold"],
            0.5,
        )

        assert_float_equal(
            "세션 Mask Threshold 유지",
            metadata["mask_threshold"],
            0.75,
        )

        assert_equal(
            "세션 TextCount 유지",
            metadata["text_count"],
            EXPECTED_TEXT_COUNT,
        )

        print(
            "[PASS] 잘못된 요청 이후 기존 세션 설정 유지"
        )

    finally:
        status, deleted = request_json(
            f"{BASE_URL}/{session_id}",
            method="DELETE",
        )

        assert_equal(
            "세션 삭제 HTTP",
            status,
            200,
        )

        if not deleted.get("success"):
            raise AssertionError(
                "세션 삭제에 실패했습니다."
            )

        print("[PASS] 테스트 세션 안전 삭제")

    print()
    print("=" * 70)
    print("모든 Dynamic Threshold API 테스트가 통과했습니다.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"[FAIL] {error}")
        sys.exit(1)