"""Isolated, in-memory inference sessions for the new web UI."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from security.file_validation import ValidatedImage
from services.sam3_service import IMAGE_LONG_SIDE, masks_to_numpy, resize_long_side


SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 4

# Values are passed to SAM3 exactly as written. Do not normalize case.
ALLOWED_PROMPTS = {
    "A": "P.g",  # backward-compatible blind prompt
    "P.g": "P.g",
    "P.a": "P.a",
    "C.a": "C.a",
    "S.a": "S.a",
    "E.c": "E.c",
    "B.c": "B.c",
    "A.b": "A.b",
    "S.a_TNTC": "S.a_TNTC",
    "E.c_TNTC": "E.c_TNTC",
    "P.g_TNTC": "P.g_TNTC",
    "P.a_TNTC": "P.a_TNTC",
    "B.c_TNTC": "B.c_TNTC",
    "C.a_TNTC": "C.a_TNTC",
    "A.b_TNTC": "A.b_TNTC",
}


class SessionNotFoundError(LookupError):
    pass


class SessionLimitError(RuntimeError):
    pass


class SessionPromptRequiredError(ValueError):
    pass


class SessionInvalidBoxError(ValueError):
    pass


def _tensor_list(value: Any) -> list:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


@dataclass
class InferenceSession:
    session_id: str
    image: Image.Image
    original_filename: str
    detected_format: str
    original_width: int
    original_height: int
    file_size: int
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    prompt: str = ""
    model_prompt: str = ""
    text_done: bool = False
    text_masks: list[np.ndarray] = field(default_factory=list)
    visual_masks: list[np.ndarray] = field(default_factory=list)
    text_scores: list = field(default_factory=list)
    visual_scores: list = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]


class SessionInferenceManager:
    """Shares one model while keeping every user's image and results isolated."""

    def __init__(self, processor, gpu_lock: threading.Lock) -> None:
        self.processor = processor
        self.gpu_lock = gpu_lock
        self._sessions: dict[str, InferenceSession] = {}
        self._registry_lock = threading.Lock()

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def create(self, validated: ValidatedImage) -> InferenceSession:
        image = resize_long_side(validated.image, target_size=IMAGE_LONG_SIDE)

        with self._registry_lock:
            self._cleanup_expired_locked()
            if len(self._sessions) >= MAX_SESSIONS:
                raise SessionLimitError(
                    "동시에 사용할 수 있는 분석 세션 수를 초과했습니다. 기존 세션을 종료하세요."
                )

            session_id = secrets.token_urlsafe(24)
            session = InferenceSession(
                session_id=session_id,
                image=image,
                original_filename=validated.original_filename,
                detected_format=validated.detected_format,
                original_width=validated.original_width,
                original_height=validated.original_height,
                file_size=validated.file_size,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> InferenceSession:
        if not session_id or len(session_id) > 128:
            raise SessionNotFoundError("분석 세션을 찾을 수 없습니다.")

        with self._registry_lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError("분석 세션을 찾을 수 없습니다.")
            session.updated_at = time.time()
            return session

    def delete(self, session_id: str) -> None:
        with self._registry_lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionNotFoundError("분석 세션을 찾을 수 없습니다.")

    def reset(self, session_id: str) -> InferenceSession:
        session = self.get(session_id)
        with session.lock:
            session.prompt = ""
            session.model_prompt = ""
            session.text_done = False
            session.text_masks = []
            session.visual_masks = []
            session.text_scores = []
            session.visual_scores = []
            session.boxes = []
            session.updated_at = time.time()
        return session

    def run_text_prompt(self, session_id: str, prompt: str) -> dict[str, Any]:
        session = self.get(session_id)
        display_prompt = str(prompt).strip()
        model_prompt = ALLOWED_PROMPTS.get(display_prompt)
        if model_prompt is None:
            raise ValueError("지원하지 않는 미생물 Prompt입니다.")

        with session.lock, self.gpu_lock:
            with torch.inference_mode():
                state = self.processor.set_image(session.image)
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(
                    state=state,
                    prompt=model_prompt,
                )

            session.prompt = display_prompt
            session.model_prompt = model_prompt
            session.text_done = True
            session.text_masks = masks_to_numpy(state.get("masks"))
            session.text_scores = _tensor_list(state.get("scores"))
            session.visual_masks = []
            session.visual_scores = []
            session.boxes = []
            session.updated_at = time.time()

            return self.snapshot(session)

    def run_box_prompt(
        self,
        session_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> dict[str, Any]:
        session = self.get(session_id)

        x1 = max(0.0, min(float(x1), session.width - 1))
        y1 = max(0.0, min(float(y1), session.height - 1))
        x2 = max(0.0, min(float(x2), session.width - 1))
        y2 = max(0.0, min(float(y2), session.height - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if x2 - x1 <= 5 or y2 - y1 <= 5:
            raise SessionInvalidBoxError("Bounding Box 크기가 너무 작습니다.")

        # Rebuild and replay visual prompts for each request. This avoids keeping
        # multiple users' GPU states resident in 8GB VRAM while preserving order.
        with session.lock:
            if not session.text_done:
                raise SessionPromptRequiredError("먼저 Text Prompt 추론을 실행하세요.")

            requested_boxes = [*session.boxes, [x1, y1, x2, y2]]
            with self.gpu_lock:
                with torch.inference_mode():
                    state = self.processor.set_image(session.image)
                    self.processor.reset_all_prompts(state)

                    for pixel_box in requested_boxes:
                        bx1, by1, bx2, by2 = pixel_box
                        normalized_box = [
                            ((bx1 + bx2) / 2.0) / session.width,
                            ((by1 + by2) / 2.0) / session.height,
                            (bx2 - bx1) / session.width,
                            (by2 - by1) / session.height,
                        ]
                        state = self.processor.add_geometric_prompt(
                            box=normalized_box,
                            label=True,
                            state=state,
                        )

            session.boxes = requested_boxes
            session.visual_masks = masks_to_numpy(state.get("masks"))
            session.visual_scores = _tensor_list(state.get("scores"))
            session.updated_at = time.time()
            return self.snapshot(session)

    @staticmethod
    def snapshot(session: InferenceSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "prompt": session.prompt,
            "model_prompt": session.model_prompt,
            "text_done": session.text_done,
            "text_masks": list(session.text_masks),
            "visual_masks": list(session.visual_masks),
            "text_scores": list(session.text_scores),
            "visual_scores": list(session.visual_scores),
            "boxes": [list(box) for box in session.boxes],
            "text_count": len(session.text_masks),
            "visual_count": len(session.visual_masks),
            "box_count": len(session.boxes),
            "width": session.width,
            "height": session.height,
            "original_width": session.original_width,
            "original_height": session.original_height,
            "original_filename": session.original_filename,
            "detected_format": session.detected_format,
            "file_size": session.file_size,
        }
