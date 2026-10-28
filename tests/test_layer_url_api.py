"""Live API test for cacheable binary result layers."""

from __future__ import annotations

import json
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path


ORIGIN = "http://127.0.0.1:7860"
BASE_URL = f"{ORIGIN}/api/sessions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = (
    PROJECT_ROOT
    / "train_1-P-g-1-1-3-_png.rf.df8ed643465e2c21048b5deb4c074575.jpg"
)
TEST_BOX = {
    "x1": 188.33,
    "y1": 555.07,
    "x2": 258.33,
    "y2": 614.07,
}
TIMEOUT_SECONDS = 300
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def request_json(url, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}

    if payload is not None:
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers=headers,
    )

    with urllib.request.urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        body = response.read()

        return (
            response.status,
            json.loads(body.decode("utf-8")),
            len(body),
        )


def upload_image():
    boundary = f"----sam3-{secrets.token_hex(16)}"
    image_bytes = IMAGE_PATH.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{IMAGE_PATH.name}"\r\n'
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
        response_body = response.read()

        return (
            response.status,
            json.loads(response_body.decode("utf-8")),
            len(response_body),
        )


def request_png(relative_url):
    request = urllib.request.Request(
        url=f"{ORIGIN}{relative_url}",
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        body = response.read()
        content_type = response.headers.get(
            "Content-Type",
            "",
        )
        cache_control = response.headers.get(
            "Cache-Control",
            "",
        )

        if response.status != 200:
            raise AssertionError(
                f"PNG HTTP: {response.status}"
            )

        if not content_type.startswith("image/png"):
            raise AssertionError(
                f"PNG Content-Type: {content_type}"
            )

        if not body.startswith(PNG_SIGNATURE):
            raise AssertionError(
                "응답이 실제 PNG 바이너리가 아닙니다."
            )

        if "private" not in cache_control:
            raise AssertionError(
                f"캐시 헤더 누락: {cache_control}"
            )

        return len(body)


def assert_lightweight_json(name, result, body_size):
    if "image" in result:
        raise AssertionError(
            f"{name}: Base64 image 필드가 남아 있습니다."
        )

    expected_layers = {
        "base",
        "text",
        "visual",
        "boxes",
    }

    if set(result.get("layers", {})) != expected_layers:
        raise AssertionError(
            f"{name}: 레이어 URL 구성이 올바르지 않습니다."
        )

    for value in result["layers"].values():
        if not value.startswith("/api/sessions/"):
            raise AssertionError(
                f"{name}: 잘못된 레이어 URL {value}"
            )

    if not result.get("result_image_url"):
        raise AssertionError(
            f"{name}: 결과 이미지 URL이 없습니다."
        )

    if body_size >= 200_000:
        raise AssertionError(
            f"{name}: JSON 응답이 너무 큽니다: {body_size} bytes"
        )


def main():
    print("=" * 74)
    print("SAM3 캐시 이미지 URL API 테스트")
    print("=" * 74)

    status, created, upload_size = upload_image()

    if status != 201:
        raise AssertionError(
            f"업로드 HTTP: {status}"
        )

    assert_lightweight_json(
        "업로드",
        created,
        upload_size,
    )
    session_id = created["session_id"]
    upload_layers = created["layers"]
    print(
        "[PASS] 업로드 JSON 경량화: "
        f"{upload_size:,} bytes"
    )

    try:
        base_bytes = request_png(
            upload_layers["base"]
        )
        print(
            "[PASS] Base PNG URL 및 캐시: "
            f"{base_bytes:,} bytes"
        )

        status, text_result, text_size = request_json(
            f"{BASE_URL}/{session_id}/text-infer",
            method="POST",
            payload={"prompt": "A"},
        )

        if status != 200:
            raise AssertionError(
                f"Text HTTP: {status}"
            )

        assert_lightweight_json(
            "1차 검출",
            text_result,
            text_size,
        )

        if text_result["text_count"] != 58:
            raise AssertionError(
                "TextCount 기준값이 변경되었습니다."
            )

        if (
            text_result["layers"]["base"]
            != upload_layers["base"]
        ):
            raise AssertionError(
                "변경되지 않은 Base URL이 바뀌었습니다."
            )

        if (
            text_result["layers"]["text"]
            == upload_layers["text"]
        ):
            raise AssertionError(
                "Text revision이 갱신되지 않았습니다."
            )

        request_png(text_result["layers"]["text"])
        print(
            "[PASS] 1차 검출 JSON 및 Text 레이어 갱신: "
            f"{text_size:,} bytes"
        )

        status, box_result, box_size = request_json(
            f"{BASE_URL}/{session_id}/box-infer",
            method="POST",
            payload=TEST_BOX,
        )

        if status != 200:
            raise AssertionError(
                f"Box HTTP: {status}"
            )

        assert_lightweight_json(
            "2차 검출",
            box_result,
            box_size,
        )

        if (
            box_result["text_count"],
            box_result["visual_count"],
            box_result["box_count"],
        ) != (58, 1, 1):
            raise AssertionError(
                "Text/Visual/Box 기준값이 변경되었습니다."
            )

        if (
            box_result["layers"]["base"]
            != text_result["layers"]["base"]
            or box_result["layers"]["text"]
            != text_result["layers"]["text"]
        ):
            raise AssertionError(
                "Box 처리 중 Base 또는 Text URL이 바뀌었습니다."
            )

        if (
            box_result["layers"]["visual"]
            == text_result["layers"]["visual"]
        ):
            raise AssertionError(
                "Visual revision이 갱신되지 않았습니다."
            )

        request_png(box_result["layers"]["visual"])
        request_png(box_result["layers"]["boxes"])
        result_bytes = request_png(
            box_result["result_image_url"]
        )
        print(
            "[PASS] 2차 검출 및 결과 PNG URL: "
            f"JSON {box_size:,} bytes, "
            f"PNG {result_bytes:,} bytes"
        )

    finally:
        request_json(
            f"{BASE_URL}/{session_id}",
            method="DELETE",
        )
        print("[PASS] 테스트 세션 안전 삭제")

    print()
    print("모든 캐시 이미지 URL API 테스트가 통과했습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}")
        sys.exit(1)
