import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:7860"

EXPECTED_TEXT_COUNT = 58
EXPECTED_VISUAL_COUNT = 1
EXPECTED_BOX_COUNT = 1

TEST_BOX = {
    "x1": 188.33,
    "y1": 555.07,
    "x2": 258.33,
    "y2": 614.07,
}

REPEAT_COUNT = 3
REQUEST_TIMEOUT_SECONDS = 300


def post_json(path, payload=None):
    url = f"{BASE_URL}{path}"

    if payload is None:
        request_data = b""
    else:
        request_data = json.dumps(
            payload
        ).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=request_data,
        method="POST",
        headers={
            "Content-Type": "application/json"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            response_text = (
                response
                .read()
                .decode("utf-8")
            )

            return json.loads(response_text)

    except urllib.error.HTTPError as error:
        error_body = (
            error
            .read()
            .decode(
                "utf-8",
                errors="replace"
            )
        )

        raise RuntimeError(
            f"HTTP {error.code}: {error_body}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            "SAM3 서버에 연결할 수 없습니다. "
            "backend/legacy_server.py가 실행 중인지 확인하세요."
        ) from error


def check_equal(
    test_number,
    field_name,
    actual,
    expected
):
    if actual != expected:
        raise AssertionError(
            f"시험 {test_number}: "
            f"{field_name} 값이 변경되었습니다. "
            f"기준값={expected}, 현재값={actual}"
        )


def run_single_test(test_number):
    reset_result = post_json(
        "/reset"
    )

    if not reset_result.get("success"):
        raise AssertionError(
            f"시험 {test_number}: Reset 실패"
        )

    text_result = post_json(
        "/text_infer",
        {
            "prompt": "A"
        }
    )

    if not text_result.get("success"):
        raise AssertionError(
            f"시험 {test_number}: "
            f"Text 추론 실패: "
            f"{text_result.get('error')}"
        )

    box_result = post_json(
        "/box_infer",
        TEST_BOX
    )

    if not box_result.get("success"):
        raise AssertionError(
            f"시험 {test_number}: "
            f"Visual 추론 실패: "
            f"{box_result.get('error')}"
        )

    text_count = text_result.get(
        "text_count"
    )

    visual_count = box_result.get(
        "visual_count"
    )

    box_count = box_result.get(
        "box_count"
    )

    check_equal(
        test_number,
        "TextCount",
        text_count,
        EXPECTED_TEXT_COUNT
    )

    check_equal(
        test_number,
        "VisualCount",
        visual_count,
        EXPECTED_VISUAL_COUNT
    )

    check_equal(
        test_number,
        "BoxCount",
        box_count,
        EXPECTED_BOX_COUNT
    )

    print(
        f"[PASS {test_number}/{REPEAT_COUNT}] "
        f"Text={text_count}, "
        f"Visual={visual_count}, "
        f"Box={box_count}"
    )

    return {
        "test": test_number,
        "text_count": text_count,
        "visual_count": visual_count,
        "box_count": box_count,
    }


def main():
    print("=" * 70)
    print("SAM3 API 회귀 테스트")
    print("=" * 70)

    print(
        f"기준값: "
        f"Text={EXPECTED_TEXT_COUNT}, "
        f"Visual={EXPECTED_VISUAL_COUNT}, "
        f"Box={EXPECTED_BOX_COUNT}"
    )

    print(
        "기준 Box:",
        TEST_BOX
    )

    print()

    results = []

    for test_number in range(
        1,
        REPEAT_COUNT + 1
    ):
        result = run_single_test(
            test_number
        )

        results.append(result)

    print()
    print("=" * 70)
    print("회귀 테스트 완료")
    print("=" * 70)
    print(
        "모든 시험에서 기존 기준값과 "
        "동일한 결과가 확인되었습니다."
    )

    output_path = (
        "legacy/"
        "python-regression-results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            {
                "expected": {
                    "text_count":
                        EXPECTED_TEXT_COUNT,

                    "visual_count":
                        EXPECTED_VISUAL_COUNT,

                    "box_count":
                        EXPECTED_BOX_COUNT,

                    "box":
                        TEST_BOX,
                },

                "results":
                    results,
            },
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"결과 저장: {output_path}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 70)
        print("회귀 테스트 실패")
        print("=" * 70)
        print(error)

        sys.exit(1)