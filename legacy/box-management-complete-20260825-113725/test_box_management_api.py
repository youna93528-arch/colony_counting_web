"""Live API test for undoing and clearing Visual Prompt boxes."""

from __future__ import annotations

import sys

from test_session_api import (
    BASE_URL,
    EXPECTED_TEXT_COUNT,
    EXPECTED_VISUAL_COUNT,
    TEST_BOX,
    assert_equal,
    request_json,
    upload_image,
)


def main():
    print("=" * 70)
    print("SAM3 Box 관리 API 테스트")
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
        # 1. Text Prompt 분석
        status, text_result = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={"prompt": "A"},
        )

        assert_equal(
            "Text HTTP",
            status,
            200,
        )
        assert_equal(
            "TextCount",
            text_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "Text 단계 BoxCount",
            text_result["box_count"],
            0,
        )

        print(
            "[PASS] Text Prompt: "
            f"Text={text_result['text_count']}, "
            f"Visual={text_result['visual_count']}, "
            f"Box={text_result['box_count']}"
        )

        # 2. 기준 Box 추가
        status, box_result = request_json(
            f"{BASE_URL}/{session_id}/box-infer",
            method="POST",
            payload=TEST_BOX,
        )

        assert_equal(
            "Box 추가 HTTP",
            status,
            200,
        )
        assert_equal(
            "Box 추가 TextCount",
            box_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "Box 추가 VisualCount",
            box_result["visual_count"],
            EXPECTED_VISUAL_COUNT,
        )
        assert_equal(
            "Box 추가 BoxCount",
            box_result["box_count"],
            1,
        )

        print(
            "[PASS] Box 추가: "
            f"Text={box_result['text_count']}, "
            f"Visual={box_result['visual_count']}, "
            f"Box={box_result['box_count']}"
        )

        # 3. 마지막 Box 취소
        status, undo_result = request_json(
            (
                f"{BASE_URL}/{session_id}"
                "/boxes/undo"
            ),
            method="POST",
        )

        assert_equal(
            "Box 취소 HTTP",
            status,
            200,
        )
        assert_equal(
            "Box 취소 Action",
            undo_result.get("box_action"),
            "undo",
        )
        assert_equal(
            "Box 취소 TextCount",
            undo_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "Box 취소 VisualCount",
            undo_result["visual_count"],
            0,
        )
        assert_equal(
            "Box 취소 BoxCount",
            undo_result["box_count"],
            0,
        )

        print(
            "[PASS] 마지막 Box 취소: "
            f"Text={undo_result['text_count']}, "
            f"Visual={undo_result['visual_count']}, "
            f"Box={undo_result['box_count']}"
        )

        # 4. 취소할 Box가 없는 상태에서 한 번 더 요청
        status, empty_undo_result = request_json(
            (
                f"{BASE_URL}/{session_id}"
                "/boxes/undo"
            ),
            method="POST",
        )

        assert_equal(
            "빈 Box 취소 HTTP",
            status,
            200,
        )
        assert_equal(
            "빈 Box 취소 TextCount",
            empty_undo_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "빈 Box 취소 VisualCount",
            empty_undo_result["visual_count"],
            0,
        )
        assert_equal(
            "빈 Box 취소 BoxCount",
            empty_undo_result["box_count"],
            0,
        )

        print(
            "[PASS] 취소할 Box가 없는 경우도 안전하게 처리"
        )

        # 5. 기준 Box 다시 추가
        status, second_box_result = request_json(
            f"{BASE_URL}/{session_id}/box-infer",
            method="POST",
            payload=TEST_BOX,
        )

        assert_equal(
            "두 번째 Box 추가 HTTP",
            status,
            200,
        )
        assert_equal(
            "두 번째 Box VisualCount",
            second_box_result["visual_count"],
            EXPECTED_VISUAL_COUNT,
        )
        assert_equal(
            "두 번째 Box BoxCount",
            second_box_result["box_count"],
            1,
        )

        print(
            "[PASS] 전체 삭제 시험용 Box 재등록"
        )

        # 6. 모든 Box 삭제
        status, clear_result = request_json(
            f"{BASE_URL}/{session_id}/boxes",
            method="DELETE",
        )

        assert_equal(
            "모든 Box 삭제 HTTP",
            status,
            200,
        )
        assert_equal(
            "모든 Box 삭제 Action",
            clear_result.get("box_action"),
            "clear",
        )
        assert_equal(
            "모든 Box 삭제 TextCount",
            clear_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "모든 Box 삭제 VisualCount",
            clear_result["visual_count"],
            0,
        )
        assert_equal(
            "모든 Box 삭제 BoxCount",
            clear_result["box_count"],
            0,
        )

        print(
            "[PASS] 모든 Box 삭제: "
            f"Text={clear_result['text_count']}, "
            f"Visual={clear_result['visual_count']}, "
            f"Box={clear_result['box_count']}"
        )

        # 7. 실제 세션 상태 확인
        status, session_result = request_json(
            f"{BASE_URL}/{session_id}"
        )

        assert_equal(
            "세션 조회 HTTP",
            status,
            200,
        )
        assert_equal(
            "세션 TextCount 유지",
            session_result["text_count"],
            EXPECTED_TEXT_COUNT,
        )
        assert_equal(
            "세션 VisualCount 초기화",
            session_result["visual_count"],
            0,
        )
        assert_equal(
            "세션 BoxCount 초기화",
            session_result["box_count"],
            0,
        )

        print(
            "[PASS] 세션 상태 확인: "
            "Text 결과 유지, Visual·Box 초기화"
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
    print("모든 Box 관리 API 테스트가 통과했습니다.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"[FAIL] {error}")
        sys.exit(1)