"""SAM3 model loading and inference service.

This module intentionally preserves the inference behavior of the verified
legacy server. UI rendering and HTTP concerns stay outside this module.
"""

from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SAM3_DIR = PROJECT_ROOT / "sam3"

BPE_PATH = SAM3_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = (
    SAM3_DIR / "sam3" / "logs" / "checkpoints" / "checkpoint.pt"
)
DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "train_1-P-g-1-1-3-_png.rf.df8ed643465e2c21048b5deb4c074575.jpg"
)

CONFIDENCE_THRESHOLD = 0.5
IMAGE_LONG_SIDE = 1008
MASK_THRESHOLD = 0.5

PROMPT_MAP = {
    "A": "P.g",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(SAM3_DIR))

from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


class PromptRequiredError(ValueError):
    """Raised when a visual prompt is requested before text inference."""


class InvalidBoxError(ValueError):
    """Raised when a bounding box is invalid."""


def check_required_files(image_path: Path = DEFAULT_IMAGE_PATH) -> None:
    required_files = {
        "SAM3 폴더": SAM3_DIR,
        "BPE 파일": BPE_PATH,
        "체크포인트": CHECKPOINT_PATH,
        "테스트 이미지": image_path,
    }

    print()
    print("=" * 70)
    print("필수 파일 확인")
    print("=" * 70)

    missing_files = []
    for label, path in required_files.items():
        exists = path.exists()
        print(f"{label}: {path}")
        print(f"  존재 여부: {exists}")
        if not exists:
            missing_files.append(f"{label}: {path}")

    if missing_files:
        raise FileNotFoundError(
            "다음 파일 또는 폴더를 찾을 수 없습니다.\n"
            + "\n".join(missing_files)
        )


def resize_long_side(image: Image.Image, target_size: int = 1008) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)

    if long_side == target_size:
        return image

    scale = target_size / long_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def masks_to_numpy(masks: Any) -> list[np.ndarray]:
    if masks is None:
        return []

    if torch.is_tensor(masks):
        masks = masks.detach().float().cpu().numpy()

    converted_masks = []
    for mask in masks:
        mask = np.asarray(mask)
        while mask.ndim > 2:
            mask = mask[0]
        converted_masks.append((mask > MASK_THRESHOLD).astype(np.uint8))

    return converted_masks


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sam3_model():
    print()
    print("=" * 70)
    print("SAM3 모델 생성 중")
    print("=" * 70)
    print(f"사용 장치: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU 메모리: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("CUDA를 사용할 수 없어 CPU를 사용합니다.")

    sam_model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        device="cpu",
        load_from_HF=False,
    )

    print()
    print("체크포인트를 CPU 메모리로 불러오는 중...")
    print("파일이 약 4GB이므로 시간이 오래 걸릴 수 있습니다.")

    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        trained_weights = checkpoint["model"]
    else:
        trained_weights = checkpoint

    model_weights = sam_model.state_dict()
    compatible_weights = {
        key: value
        for key, value in trained_weights.items()
        if key in model_weights and value.shape == model_weights[key].shape
    }

    skipped_count = len(trained_weights) - len(compatible_weights)
    load_result = sam_model.load_state_dict(compatible_weights, strict=False)
    epoch = (
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print()
    print(f"체크포인트 Epoch: {epoch}")
    print(f"적용된 가중치 수: {len(compatible_weights)}")
    print(f"건너뛴 가중치 수: {skipped_count}")
    print(f"누락된 모델 키 수: {len(load_result.missing_keys)}")
    print(f"예상하지 못한 키 수: {len(load_result.unexpected_keys)}")

    del checkpoint
    del trained_weights
    del model_weights
    del compatible_weights
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print(f"모델을 {DEVICE} 장치로 이동하는 중...")
    sam_model.to(DEVICE)
    sam_model.eval()
    print("SAM3 모델 로딩 완료")
    return sam_model


class Sam3InferenceService:
    """Owns the shared model and the legacy single-session inference state."""

    def __init__(self, image_path: Path = DEFAULT_IMAGE_PATH) -> None:
        self.image_path = Path(image_path)
        check_required_files(self.image_path)

        self.model = load_sam3_model()
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=DEVICE,
        )

        with Image.open(self.image_path) as source_image:
            image = source_image.convert("RGB")

        self.original_image = resize_long_side(
            image,
            target_size=IMAGE_LONG_SIDE,
        )
        self.image_width, self.image_height = self.original_image.size

        print()
        print(f"테스트 이미지 크기: {self.image_width} × {self.image_height}")

        self.lock = threading.Lock()
        self.text_state = None
        self.visual_state = None
        self.text_masks: list[np.ndarray] = []
        self.visual_masks: list[np.ndarray] = []
        self.current_prompt = ""
        self.current_boxes: list[list[float]] = []

    def run_text_prompt(self, user_prompt: str) -> dict[str, Any]:
        user_prompt = str(user_prompt).strip().upper()
        if not user_prompt:
            raise ValueError("Text Prompt가 비어 있습니다.")

        model_prompt = PROMPT_MAP.get(user_prompt)
        if model_prompt is None:
            raise ValueError("지원하지 않는 Prompt입니다. A를 입력하세요.")

        with self.lock:
            print()
            print("=" * 70)
            print("[TEXT PROMPT]")
            print(f"화면 Prompt: {user_prompt}")
            print("Text 추론을 시작합니다.")

            with torch.inference_mode():
                self.text_state = self.processor.set_image(self.original_image)
                self.processor.reset_all_prompts(self.text_state)
                self.text_state = self.processor.set_text_prompt(
                    state=self.text_state,
                    prompt=model_prompt,
                )

            self.text_masks = masks_to_numpy(self.text_state.get("masks"))
            self.visual_state = None
            self.visual_masks = []
            self.current_boxes = []
            self.current_prompt = user_prompt

            print(f"Text 검출 수: {len(self.text_masks)}")

            return {
                "display_prompt": user_prompt,
                "text_masks": list(self.text_masks),
                "visual_masks": [],
                "boxes": [],
                "text_count": len(self.text_masks),
            }

    def run_box_prompt(self, x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
        if self.text_state is None:
            raise PromptRequiredError("먼저 Text Prompt 추론을 실행하세요.")

        x1 = max(0.0, min(float(x1), self.image_width - 1))
        y1 = max(0.0, min(float(y1), self.image_height - 1))
        x2 = max(0.0, min(float(x2), self.image_width - 1))
        y2 = max(0.0, min(float(y2), self.image_height - 1))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 5 or box_height <= 5:
            raise InvalidBoxError("Bounding Box 크기가 너무 작습니다.")

        normalized_box = [
            ((x1 + x2) / 2.0) / self.image_width,
            ((y1 + y2) / 2.0) / self.image_height,
            box_width / self.image_width,
            box_height / self.image_height,
        ]

        with self.lock:
            print()
            print("=" * 70)
            print("[VISUAL BOX PROMPT]")
            print("Pixel Box:", [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
            print(
                "Normalized Box [cx, cy, w, h]:",
                [round(value, 5) for value in normalized_box],
            )

            with torch.inference_mode():
                if self.visual_state is None:
                    self.visual_state = self.processor.set_image(self.original_image)
                    self.processor.reset_all_prompts(self.visual_state)

                self.visual_state = self.processor.add_geometric_prompt(
                    box=normalized_box,
                    label=True,
                    state=self.visual_state,
                )

            self.visual_masks = masks_to_numpy(self.visual_state.get("masks"))
            self.current_boxes.append([x1, y1, x2, y2])

            print(f"Text 검출 수: {len(self.text_masks)}")
            print(f"Visual 검출 수: {len(self.visual_masks)}")
            print(f"Box 수: {len(self.current_boxes)}")

            return {
                "text_masks": list(self.text_masks),
                "visual_masks": list(self.visual_masks),
                "boxes": [list(box) for box in self.current_boxes],
                "text_count": len(self.text_masks),
                "visual_count": len(self.visual_masks),
                "box_count": len(self.current_boxes),
            }

    def reset(self) -> None:
        with self.lock:
            self.text_state = None
            self.visual_state = None
            self.text_masks = []
            self.visual_masks = []
            self.current_prompt = ""
            self.current_boxes = []
            clear_cuda_cache()

        print()
        print("[RESET] 추론 상태가 초기화되었습니다.")
