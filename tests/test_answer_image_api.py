"""Live API test for session-scoped answer-image lookup."""

from __future__ import annotations

import json
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:7860/api/sessions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "1. test 원본 이미지 (어노테이션 포함)"
ANSWER_DIR = PROJECT_ROOT / "3. test 정답 이미지"
UNMATCHED_IMAGE = (
    PROJECT_ROOT
    / "train_1-P-g-1-1-3-_png.rf.df8ed643465e2c21048b5deb4c074575.jpg"
)
TIMEOUT_SECONDS = 60


def request_json(url, method="GET"):
    request = urllib.request.Request(url=url, method=method)
    try:
        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT_SECONDS,
        ) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"success": False, "error": body}
        return error.code, parsed


def upload_image(image_path):
    boundary = f"----answer-image-{secrets.token_hex(16)}"
    image_bytes = image_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
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
        headers={
            "Content-Type": (
                f"multipart/form-data; boundary={boundary}"
            )
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        return (
            response.status,
            json.loads(response.read().decode("utf-8")),
        )


def matching_image_path():
    answer_names = {
        path.name.casefold()
        for path in ANSWER_DIR.glob("*.jpg")
    }
    for original in ORIGINAL_DIR.glob("*.jpg"):
        if original.name.casefold() in answer_names:
            return original
    raise AssertionError("원본·정답 파일명이 같은 시험 이미지를 찾을 수 없습니다.")


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(
            f"{name}: 기준값={expected}, 현재값={actual}"
        )


def delete_session(session_id):
    status, result = request_json(
        f"{BASE_URL}/{session_id}",
        method="DELETE",
    )
    assert_equal("세션 삭제 HTTP", status, 200)
    if not result.get("success"):
        raise AssertionError("세션 삭제 실패")


def main():
    print("=" * 72)
    print("정답 이미지 API 테스트")
    print("=" * 72)

    matched = matching_image_path()
    status, created = upload_image(matched)
    assert_equal("일치 이미지 세션 생성 HTTP", status, 201)
    session_id = created["session_id"]
    print(f"[PASS] 일치 파일 업로드: {matched.name}")

    try:
        status, answer = request_json(
            f"{BASE_URL}/{session_id}/answer-image"
        )
        assert_equal("정답 이미지 HTTP", status, 200)
        assert_equal("정답 파일명", answer["answer_filename"], matched.name)
        if not answer.get("image", "").startswith("data:image/png;base64,"):
            raise AssertionError("정답 이미지가 PNG Data URL로 반환되지 않았습니다.")
        if answer.get("width", 0) <= 0 or answer.get("height", 0) <= 0:
            raise AssertionError("정답 이미지 크기가 올바르지 않습니다.")
        print(
            "[PASS] 정답 이미지 조회: "
            f"{answer['width']} x {answer['height']}"
        )
    finally:
        delete_session(session_id)
        print("[PASS] 일치 파일 세션 삭제")

    status, created = upload_image(UNMATCHED_IMAGE)
    assert_equal("불일치 이미지 세션 생성 HTTP", status, 201)
    unmatched_session_id = created["session_id"]

    try:
        status, missing = request_json(
            f"{BASE_URL}/{unmatched_session_id}/answer-image"
        )
        assert_equal("정답 없음 HTTP", status, 404)
        if missing.get("success"):
            raise AssertionError("정답이 없는 파일이 성공 처리되었습니다.")
        print("[PASS] 정답이 없는 파일은 HTTP 404로 안전하게 처리")
    finally:
        delete_session(unmatched_session_id)
        print("[PASS] 불일치 파일 세션 삭제")

    print()
    print("모든 정답 이미지 API 테스트가 통과했습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}")
        sys.exit(1)
