"""Isolated, in-memory inference sessions for the new web UI."""

from __future__ import annotations

import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from security.file_validation import ValidatedImage
from services.sam3_service import (
    IMAGE_LONG_SIDE,
    masks_to_numpy,
    resize_long_side,
)

from services.threshold_validation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MASK_THRESHOLD,
    validate_inference_thresholds,
)

from services.visual_duplicate_filter import (
    filter_visual_duplicates_with_text,
)
from services.detection_evaluation import (
    evaluate_masks,
    load_ground_truth_masks,
    unavailable_evaluation,
)

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

def _masks_from_state(
    state: dict[str, Any],
    mask_threshold: float,
) -> list[np.ndarray]:
    """
    Convert the probability masks in masks_logits using
    the selected Mask Threshold.

    The boolean state["masks"] is used only as a fallback
    for compatibility with an unexpected processor state.
    """

    mask_source = state.get(
        "masks_logits"
    )

    if mask_source is None:
        mask_source = state.get(
            "masks"
        )

    return masks_to_numpy(
        mask_source,
        threshold=mask_threshold,
    )


@contextmanager
def _temporary_confidence_threshold(
    processor,
    confidence_threshold: float,
):
    """
    Apply a Confidence Threshold only while holding
    the shared GPU lock, then restore the previous value.
    """

    previous_threshold = float(
        processor.confidence_threshold
    )

    processor.set_confidence_threshold(
        confidence_threshold
    )

    try:
        yield

    finally:
        processor.set_confidence_threshold(
            previous_threshold
        )

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
    image_quality_shadow: dict[str, Any] | None = None
    ground_truth_masks: list[np.ndarray] = field(default_factory=list)
    ground_truth_available: bool = False
    ground_truth_message: str = ""
    text_evaluation: dict[str, Any] | None = None
    final_evaluation: dict[str, Any] | None = None
    confidence_threshold: float = (
        DEFAULT_CONFIDENCE_THRESHOLD
    )
    mask_threshold: float = (
        DEFAULT_MASK_THRESHOLD
    )
    lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]


class SessionInferenceManager:
    """Shares one model while keeping every user's image and results isolated."""

    def __init__(
        self,
        processor,
        gpu_lock: threading.Lock,
    ) -> None:
        self.processor = processor
        self.gpu_lock = gpu_lock
        self._sessions: dict[str, InferenceSession] = {}
        self._registry_lock = threading.Lock()

    def _cleanup_expired_locked(self) -> None:
        now = time.time()

        expired = [
            session_id
            for session_id, session
            in self._sessions.items()
            if (
                now - session.updated_at
                > SESSION_TTL_SECONDS
            )
        ]

        for session_id in expired:
            del self._sessions[session_id]

    def create(
        self,
        validated: ValidatedImage,
    ) -> InferenceSession:
        image = resize_long_side(
            validated.image,
            target_size=IMAGE_LONG_SIDE,
        )

        with self._registry_lock:
            self._cleanup_expired_locked()

            if len(self._sessions) >= MAX_SESSIONS:
                raise SessionLimitError(
                    "동시에 사용할 수 있는 분석 세션 수를 "
                    "초과했습니다. 기존 세션을 종료하세요."
                )

            session_id = secrets.token_urlsafe(24)

            session = InferenceSession(
                session_id=session_id,
                image=image,
                original_filename=(
                    validated.original_filename
                ),
                detected_format=(
                    validated.detected_format
                ),
                original_width=(
                    validated.original_width
                ),
                original_height=(
                    validated.original_height
                ),
                file_size=validated.file_size,
            )

            self._sessions[session_id] = session

            return session

    def get(
        self,
        session_id: str,
    ) -> InferenceSession:
        if (
            not session_id
            or len(session_id) > 128
        ):
            raise SessionNotFoundError(
                "분석 세션을 찾을 수 없습니다."
            )

        with self._registry_lock:
            self._cleanup_expired_locked()

            session = self._sessions.get(
                session_id
            )

            if session is None:
                raise SessionNotFoundError(
                    "분석 세션을 찾을 수 없습니다."
                )

            session.updated_at = time.time()

            return session

    def delete(
        self,
        session_id: str,
    ) -> None:
        with self._registry_lock:
            session = self._sessions.pop(
                session_id,
                None,
            )

        if session is None:
            raise SessionNotFoundError(
                "분석 세션을 찾을 수 없습니다."
            )

    def reset(
        self,
        session_id: str,
    ) -> InferenceSession:
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
            session.ground_truth_masks = []
            session.ground_truth_available = False
            session.ground_truth_message = ""
            session.text_evaluation = None
            session.final_evaluation = None
            session.updated_at = time.time()

        return session

    def run_text_prompt(
        self,
        session_id: str,
        prompt: str,
        confidence_threshold: float = (
            DEFAULT_CONFIDENCE_THRESHOLD
        ),
        mask_threshold: float = (
            DEFAULT_MASK_THRESHOLD
        ),
    ) -> dict[str, Any]:
        session = self.get(session_id)

        display_prompt = str(prompt).strip()

        model_prompt = ALLOWED_PROMPTS.get(
            display_prompt
        )

        if model_prompt is None:
            raise ValueError(
                "지원하지 않는 미생물 Prompt입니다."
            )

        (
            validated_confidence,
            validated_mask,
        ) = validate_inference_thresholds(
            confidence_threshold,
            mask_threshold,
        )

        with session.lock, self.gpu_lock:
            with _temporary_confidence_threshold(
                self.processor,
                validated_confidence,
            ):
                with torch.inference_mode():
                    state = (
                        self.processor.set_image(
                            session.image
                        )
                    )

                    self.processor.reset_all_prompts(
                        state
                    )

                    state = (
                        self.processor.set_text_prompt(
                            state=state,
                            prompt=model_prompt,
                        )
                    )

            session.prompt = display_prompt
            session.model_prompt = model_prompt
            session.text_done = True

            session.confidence_threshold = (
                validated_confidence
            )

            session.mask_threshold = (
                validated_mask
            )

            session.text_masks = (
                _masks_from_state(
                    state,
                    validated_mask,
                )
            )

            session.text_scores = (
                _tensor_list(
                    state.get("scores")
                )
            )

            session.visual_masks = []
            session.visual_scores = []
            session.boxes = []

            ground_truth = load_ground_truth_masks(
                original_filename=session.original_filename,
                target_width=session.width,
                target_height=session.height,
            )
            session.ground_truth_available = bool(
                ground_truth["available"]
            )
            session.ground_truth_masks = list(
                ground_truth["masks"]
            )
            session.ground_truth_message = str(
                ground_truth["message"]
            )

            if session.ground_truth_available:
                session.text_evaluation = evaluate_masks(
                    prediction_masks=session.text_masks,
                    ground_truth_masks=session.ground_truth_masks,
                    stage="text",
                )
            else:
                session.text_evaluation = unavailable_evaluation(
                    stage="text",
                    message=session.ground_truth_message,
                )

            session.final_evaluation = None
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

        x1 = max(
            0.0,
            min(float(x1), session.width - 1),
        )
        y1 = max(
            0.0,
            min(float(y1), session.height - 1),
        )
        x2 = max(
            0.0,
            min(float(x2), session.width - 1),
        )
        y2 = max(
            0.0,
            min(float(y2), session.height - 1),
        )

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        if (
            x2 - x1 <= 5
            or y2 - y1 <= 5
        ):
            raise SessionInvalidBoxError(
                "Bounding Box 크기가 너무 작습니다."
            )

        with session.lock:
            if not session.text_done:
                raise SessionPromptRequiredError(
                    "먼저 Text Prompt 추론을 실행하세요."
                )

            requested_boxes = [
                *session.boxes,
                [x1, y1, x2, y2],
            ]

            with self.gpu_lock:
                with _temporary_confidence_threshold(
                    self.processor,
                    session.confidence_threshold,
                ):
                    with torch.inference_mode():
                        state = (
                            self.processor.set_image(
                                session.image
                            )
                        )

                        self.processor.reset_all_prompts(
                            state
                        )

                        for pixel_box in requested_boxes:
                            (
                                bx1,
                                by1,
                                bx2,
                                by2,
                            ) = pixel_box

                            normalized_box = [
                                (
                                    (bx1 + bx2) / 2.0
                                ) / session.width,
                                (
                                    (by1 + by2) / 2.0
                                ) / session.height,
                                (
                                    bx2 - bx1
                                ) / session.width,
                                (
                                    by2 - by1
                                ) / session.height,
                            ]

                            state = (
                                self.processor
                                .add_geometric_prompt(
                                    box=normalized_box,
                                    label=True,
                                    state=state,
                                )
                            )

            session.boxes = requested_boxes

            raw_visual_masks = (
                _masks_from_state(
                    state,
                    session.mask_threshold,
                )
            )

            raw_visual_scores = (
                _tensor_list(
                    state.get("scores")
                )
            )

            filtered_visual = (
                filter_visual_duplicates_with_text(
                    visual_masks=raw_visual_masks,
                    visual_scores=raw_visual_scores,
                    text_masks=session.text_masks,
                    iou_threshold=0.5,
                )
            )

            session.visual_masks = (
                filtered_visual["masks"]
            )

            session.visual_scores = (
                filtered_visual["scores"]
            )

            if session.ground_truth_available:
                session.final_evaluation = evaluate_masks(
                    prediction_masks=[
                        *session.text_masks,
                        *session.visual_masks,
                    ],
                    ground_truth_masks=session.ground_truth_masks,
                    stage="final",
                )
            else:
                session.final_evaluation = unavailable_evaluation(
                    stage="final",
                    message=session.ground_truth_message,
                )

            session.updated_at = time.time()

            return self.snapshot(session)

    def undo_last_box(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Remove the last Visual Prompt Box and replay
        only the remaining boxes using the thresholds
        stored in the current analysis session.
        """

        session = self.get(session_id)

        with session.lock:
            if not session.text_done:
                raise SessionPromptRequiredError(
                    "먼저 Text Prompt 추론을 실행하세요."
                )

            if not session.boxes:
                return self.snapshot(session)

            remaining_boxes = [
                list(box)
                for box in session.boxes[:-1]
            ]

            if not remaining_boxes:
                session.boxes = []
                session.visual_masks = []
                session.visual_scores = []
                session.final_evaluation = None
                session.updated_at = time.time()

                return self.snapshot(session)

            with self.gpu_lock:
                with _temporary_confidence_threshold(
                    self.processor,
                    session.confidence_threshold,
                ):
                    with torch.inference_mode():
                        state = (
                            self.processor.set_image(
                                session.image
                            )
                        )

                        self.processor.reset_all_prompts(
                            state
                        )

                        for pixel_box in remaining_boxes:
                            (
                                bx1,
                                by1,
                                bx2,
                                by2,
                            ) = pixel_box

                            normalized_box = [
                                (
                                    (bx1 + bx2) / 2.0
                                ) / session.width,
                                (
                                    (by1 + by2) / 2.0
                                ) / session.height,
                                (
                                    bx2 - bx1
                                ) / session.width,
                                (
                                    by2 - by1
                                ) / session.height,
                            ]

                            state = (
                                self.processor
                                .add_geometric_prompt(
                                    box=normalized_box,
                                    label=True,
                                    state=state,
                                )
                            )

            session.boxes = remaining_boxes

            raw_visual_masks = (
                _masks_from_state(
                    state,
                    session.mask_threshold,
                )
            )

            raw_visual_scores = (
                _tensor_list(
                    state.get("scores")
                )
            )

            filtered_visual = (
                filter_visual_duplicates_with_text(
                    visual_masks=raw_visual_masks,
                    visual_scores=raw_visual_scores,
                    text_masks=session.text_masks,
                    iou_threshold=0.5,
                )
            )

            session.visual_masks = (
                filtered_visual["masks"]
            )

            session.visual_scores = (
                filtered_visual["scores"]
            )

            if session.ground_truth_available:
                session.final_evaluation = evaluate_masks(
                    prediction_masks=[
                        *session.text_masks,
                        *session.visual_masks,
                    ],
                    ground_truth_masks=session.ground_truth_masks,
                    stage="final",
                )
            else:
                session.final_evaluation = unavailable_evaluation(
                    stage="final",
                    message=session.ground_truth_message,
                )

            session.updated_at = time.time()

            return self.snapshot(session)

    def clear_boxes(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Remove all Visual Prompt Boxes and results.

        Text Prompt results are preserved. No model
        inference or threshold change is performed.
        """

        session = self.get(session_id)

        with session.lock:
            if not session.text_done:
                raise SessionPromptRequiredError(
                    "먼저 Text Prompt 추론을 실행하세요."
                )

            session.boxes = []
            session.visual_masks = []
            session.visual_scores = []
            session.final_evaluation = None
            session.updated_at = time.time()

            return self.snapshot(session)

    @staticmethod
    def snapshot(
        session: InferenceSession,
    ) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "prompt": session.prompt,
            "model_prompt": session.model_prompt,
            "text_done": session.text_done,
            "text_masks": list(
                session.text_masks
            ),
            "visual_masks": list(
                session.visual_masks
            ),
            "text_scores": list(
                session.text_scores
            ),
            "visual_scores": list(
                session.visual_scores
            ),
            "boxes": [
                list(box)
                for box in session.boxes
            ],
            "text_count": len(
                session.text_masks
            ),
            "visual_count": len(
                session.visual_masks
            ),
            "box_count": len(
                session.boxes
            ),
            "evaluation": (
                session.final_evaluation
                if session.boxes
                else session.text_evaluation
            ),
            "text_evaluation": session.text_evaluation,
            "final_evaluation": session.final_evaluation,
            "image_quality_shadow": (
                session.image_quality_shadow
            ),
            "confidence_threshold": (
                session.confidence_threshold
            ),
            "mask_threshold": (
                session.mask_threshold
            ),
            "width": session.width,
            "height": session.height,
            "original_width": (
                session.original_width
            ),
            "original_height": (
                session.original_height
            ),
            "original_filename": (
                session.original_filename
            ),
            "detected_format": (
                session.detected_format
            ),
            "file_size": session.file_size,
        }