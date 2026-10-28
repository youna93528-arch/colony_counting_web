"""Security validation for user-supplied colony images."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF"}


class ImageValidationError(ValueError):
    """Raised when uploaded bytes are not an allowed, safe image."""


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    original_filename: str
    detected_format: str
    original_width: int
    original_height: int
    file_size: int


def _safe_display_name(filename: str | None) -> str:
    if not filename:
        return "upload"
    return Path(filename.replace("\\", "/")).name[:255] or "upload"


def validate_image_upload(filename: str | None, payload: bytes) -> ValidatedImage:
    display_name = _safe_display_name(filename)
    extension = Path(display_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            "지원하지 않는 파일 확장자입니다. JPG, PNG, TIFF만 업로드할 수 있습니다."
        )

    if not payload:
        raise ImageValidationError("업로드된 파일이 비어 있습니다.")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("이미지 파일은 최대 30MB까지 업로드할 수 있습니다.")

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            detected_format = (probe.format or "").upper()
            original_width, original_height = probe.size
            probe.verify()

        if detected_format not in ALLOWED_FORMATS:
            raise ImageValidationError("파일 내용이 허용된 이미지 형식이 아닙니다.")

        pixel_count = original_width * original_height
        if original_width <= 0 or original_height <= 0:
            raise ImageValidationError("이미지 크기가 올바르지 않습니다.")
        if pixel_count > MAX_IMAGE_PIXELS:
            raise ImageValidationError("이미지 해상도가 너무 큽니다. 최대 2,500만 픽셀입니다.")

        # Reopen after verify(), fully decode, and copy into a clean RGB image.
        # This also drops EXIF and other untrusted metadata from inference input.
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
            clean_image = decoded.convert("RGB").copy()

    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageValidationError("손상되었거나 올바르지 않은 이미지 파일입니다.") from error
    except Image.DecompressionBombError as error:
        raise ImageValidationError("안전 제한을 초과한 고해상도 이미지입니다.") from error

    return ValidatedImage(
        image=clean_image,
        original_filename=display_name,
        detected_format=detected_format,
        original_width=original_width,
        original_height=original_height,
        file_size=len(payload),
    )
