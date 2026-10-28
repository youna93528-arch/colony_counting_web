"""Live regression test for secure session upload and isolated inference APIs."""

from __future__ import annotations

import json
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:7860/api/sessions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = (
    PROJECT_ROOT
    / "train_1-P-g-1-1-3-_png.rf.df8ed643465e2c21048b5deb4c074575.jpg"
)

EXPECTED_TEXT_COUNT = 58
EXPECTED_VISUAL_COUNT = 1
EXPECTED_BOX_COUNT = 1
TEST_BOX = {"x1": 188.33, "y1": 555.07, "x2": 258.33, "y2": 614.07}
TIMEOUT_SECONDS = 300


def request_json(url, method="GET", payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"success": False, "error": body}
        return error.code, parsed


def upload_image():
    boundary = f"----sam3-{secrets.token_hex(16)}"
    image_bytes = IMAGE_PATH.read_bytes()
    filename = IMAGE_PATH.name

    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    request = urllib.request.Request(
        url=BASE_URL,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}: 기준값={expected}, 현재값={actual}")


def main():
    print("=" * 70)
    print("SAM3 보안 세션 API 테스트")
    print("=" * 70)

    status, created = upload_image()
    assert_equal("세션 생성 HTTP", status, 201)
    session_id = created["session_id"]
    print(f"[PASS] 이미지 업로드 및 세션 생성: {session_id[:8]}...")

    try:
        status, text_result = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={"prompt": "A"},
        )
        assert_equal("Text HTTP", status, 200)
        assert_equal("TextCount", text_result["text_count"], EXPECTED_TEXT_COUNT)
        print(f"[PASS] Text Prompt: {text_result['text_count']}")

        status, box_result = request_json(
            f"{BASE_URL}/{session_id}/box-infer",
            method="POST",
            payload=TEST_BOX,
        )
        assert_equal("Box HTTP", status, 200)
        assert_equal("TextCount", box_result["text_count"], EXPECTED_TEXT_COUNT)
        assert_equal("VisualCount", box_result["visual_count"], EXPECTED_VISUAL_COUNT)
        assert_equal("BoxCount", box_result["box_count"], EXPECTED_BOX_COUNT)
        print(
            "[PASS] Visual Prompt: "
            f"Text={box_result['text_count']}, "
            f"Visual={box_result['visual_count']}, "
            f"Box={box_result['box_count']}"
        )

        status, metadata = request_json(f"{BASE_URL}/{session_id}")
        assert_equal("세션 조회 HTTP", status, 200)
        assert_equal("세션 조회 TextCount", metadata["text_count"], EXPECTED_TEXT_COUNT)
        print("[PASS] 사용자별 세션 조회")

    finally:
        status, deleted = request_json(
            f"{BASE_URL}/{session_id}",
            method="DELETE",
        )
        assert_equal("세션 삭제 HTTP", status, 200)
        if not deleted.get("success"):
            raise AssertionError("세션 삭제 실패")
        print("[PASS] 세션 안전 삭제")

    print()
    print("모든 보안 세션 API 테스트가 통과했습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}")
        sys.exit(1)
