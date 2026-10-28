"""Flask routes for secure, isolated SAM3 analysis sessions."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError

from security.file_validation import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    validate_image_upload,
)
from services.image_quality_shadow import (
    analyze_image_quality_shadow,
)
from services.layer_renderer import create_overlay_layers
from services.mask_nms_shadow import (
    analyze_mask_nms_shadow,
)
from services.session_manager import (
    ALLOWED_PROMPTS,
    SessionInvalidBoxError,
    SessionLimitError,
    SessionNotFoundError,
    SessionPromptRequiredError,
)
from services.sam3_service import (
    CONFIDENCE_THRESHOLD,
    MASK_THRESHOLD,
)


MASK_NMS_IOU_THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_IMAGE_DIRECTORY = (
    PROJECT_ROOT / "3. test 정답 이미지"
)
ANSWER_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


def find_answer_image(original_filename):
    """Find one trusted answer image by the sanitized upload filename."""

    safe_name = Path(original_filename).name

    if (
        not safe_name
        or safe_name != original_filename
        or Path(safe_name).suffix.lower()
        not in ANSWER_IMAGE_EXTENSIONS
        or not ANSWER_IMAGE_DIRECTORY.is_dir()
    ):
        return None

    expected_name = safe_name.casefold()

    for candidate in ANSWER_IMAGE_DIRECTORY.iterdir():
        if (
            candidate.is_file()
            and candidate.suffix.lower()
            in ANSWER_IMAGE_EXTENSIONS
            and candidate.name.casefold()
            == expected_name
        ):
            return candidate

    return None


def elapsed_ms(
    started_at,
    finished_at,
):
    """Convert a perf_counter interval to milliseconds."""

    return round(
        (finished_at - started_at) * 1000,
        2,
    )


def processing_result(
    stage,
    started_at,
    finished_at,
    steps=None,
    nms_mode="not_applied",
):
    """
    Create observation-only processing metadata.

    This metadata reports elapsed time only. It does not change thresholds,
    masks, scores, model weights, prompt conversion, or detection counts.
    """

    return {
        "schema_version": 1,
        "stage": stage,
        "duration_ms": elapsed_ms(
            started_at,
            finished_at,
        ),
        "steps": steps or {},
        "algorithm_preserved": True,
        "measurement_only": True,
        "nms_mode": nms_mode,
    }


def create_session_blueprint(
    manager,
    create_overlay_image,
    pil_to_base64,
):
    blueprint = Blueprint(
        "analysis_sessions",
        __name__,
        url_prefix="/api/sessions",
    )

    def encoded_layers(
        *,
        base_image,
        text_masks=None,
        visual_masks=None,
        boxes=None,
    ):
        layers = create_overlay_layers(
            base_image=base_image,
            text_masks=text_masks,
            visual_masks=visual_masks,
            boxes=boxes,
        )

        return {
            name: pil_to_base64(layer)
            for name, layer in layers.items()
        }

    def public_result(
        result,
        image_data=None,
        layers_data=None,
    ):
        payload = {
            "success": True,
            "session_id": result["session_id"],
            "prompt": result["prompt"],
            "text_done": result["text_done"],
            "text_count": result["text_count"],
            "visual_count": result["visual_count"],
            "box_count": result["box_count"],
            "boxes": result["boxes"],
            "text_scores": result["text_scores"],
            "visual_scores": result["visual_scores"],
            "width": result["width"],
            "height": result["height"],
            "original_width": (
                result["original_width"]
            ),
            "original_height": (
                result["original_height"]
            ),
            "original_filename": (
                result["original_filename"]
            ),
            "detected_format": (
                result["detected_format"]
            ),
            "file_size": result["file_size"],
            "confidence_threshold": (
                result["confidence_threshold"]
            ),
            "mask_threshold": (
                result["mask_threshold"]
            ),
            "evaluation": result.get("evaluation"),
            "text_evaluation": result.get(
                "text_evaluation"
            ),
            "final_evaluation": result.get(
                "final_evaluation"
            ),
        }

        image_quality_shadow = result.get(
            "image_quality_shadow"
        )

        if image_quality_shadow is not None:
            payload["image_quality_shadow"] = (
                image_quality_shadow
            )

        if image_data is not None:
            payload["image"] = image_data

        if layers_data is not None:
            payload["layers"] = layers_data

        return payload

    def create_shadow_result(result):
        """
        Calculate comparison-only Mask-NMS metadata.

        The returned Shadow result never replaces the SAM3 masks,
        scores, boxes, or detection counts.
        """

        try:
            shadow_result = analyze_mask_nms_shadow(
                text_masks=result["text_masks"],
                text_scores=result["text_scores"],
                visual_masks=result["visual_masks"],
                visual_scores=result["visual_scores"],
                iou_threshold=MASK_NMS_IOU_THRESHOLD,
                mask_threshold=MASK_THRESHOLD,
            )

            shadow_result["status"] = "completed"

            return shadow_result

        except Exception:
            return {
                "mode": "shadow",
                "status": "error",
                "applied_to_result": False,
                "iou_threshold": MASK_NMS_IOU_THRESHOLD,
                "mask_threshold": MASK_THRESHOLD,
                "raw_count": (
                    result["text_count"]
                    + result["visual_count"]
                ),
                "shadow_unique_count": None,
                "shadow_duplicate_count": None,
                "kept_candidates": [],
                "suppressed_candidates": [],
                "warnings": [
                    (
                        "Mask-NMS Shadow 분석에 실패했지만 "
                        "기존 SAM3 결과는 정상적으로 유지되었습니다."
                    )
                ],
            }

    @blueprint.post("")
    def create_session():
        request_started_at = perf_counter()

        if (
            request.content_length
            and request.content_length
            > MAX_UPLOAD_BYTES + 1024 * 1024
        ):
            return jsonify(
                success=False,
                error="업로드 요청 크기가 제한을 초과했습니다.",
            ), 413

        uploaded = request.files.get("image")

        if uploaded is None:
            return jsonify(
                success=False,
                error="image 파일이 필요합니다.",
            ), 400

        read_started_at = perf_counter()

        payload = uploaded.stream.read(
            MAX_UPLOAD_BYTES + 1
        )

        read_finished_at = perf_counter()

        if len(payload) > MAX_UPLOAD_BYTES:
            return jsonify(
                success=False,
                error=(
                    "이미지 파일은 최대 30MB까지 "
                    "업로드할 수 있습니다."
                ),
            ), 413

        try:
            validation_started_at = perf_counter()

            validated = validate_image_upload(
                uploaded.filename,
                payload,
            )

            validation_finished_at = perf_counter()

            quality_started_at = perf_counter()

            try:
                image_quality_shadow = (
                    analyze_image_quality_shadow(
                        validated.image
                    )
                )

            except Exception:
                # 품질검사는 Shadow 기능이므로 실패해도
                # 이미지 업로드와 SAM3 분석을 막지 않습니다.
                image_quality_shadow = {
                    "schema_version": 1,
                    "mode": "shadow",
                    "status": "error",
                    "applied_to_result": False,
                    "blocks_inference": False,
                    "warning_count": 0,
                    "warnings": [],
                    "notes": [
                        (
                            "이미지 품질 Shadow 분석에 "
                            "실패했지만 업로드와 SAM3 "
                            "추론은 정상적으로 유지됩니다."
                        )
                    ],
                }

            quality_finished_at = perf_counter()

            session_started_at = perf_counter()
           

            session = manager.create(validated)

            session.image_quality_shadow = (
                image_quality_shadow
            )

            result = manager.snapshot(session)

            session_finished_at = perf_counter()

            encoding_started_at = perf_counter()

            image_data = pil_to_base64(
                session.image
            )

            layers_data = encoded_layers(
                base_image=session.image,
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["allowed_prompts"] = list(
                ALLOWED_PROMPTS.keys()
            )

            response["image_quality_shadow"] = (
                image_quality_shadow
            )

            response["processing"] = processing_result(

                stage="upload",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                steps={
                    "file_read_ms": elapsed_ms(
                        read_started_at,
                        read_finished_at,
                    ),
                    "validation_ms": elapsed_ms(
                        validation_started_at,
                        validation_finished_at,
                    ),
                    "image_quality_shadow_ms": elapsed_ms(
                        quality_started_at,
                        quality_finished_at,
                    ),
                    "session_creation_ms": elapsed_ms(
                        session_started_at,
                        session_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response), 201

        except ImageValidationError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 400

        except SessionLimitError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 429

    @blueprint.get("/<session_id>")
    def get_session(session_id):
        try:
            session = manager.get(session_id)

            return jsonify(
                public_result(
                    manager.snapshot(session)
                )
            )

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

    @blueprint.post(
        "/<session_id>/text-infer"
    )
    def text_infer(session_id):
        request_started_at = perf_counter()

        data = request.get_json(
            silent=True
        ) or {}

        try:
            inference_started_at = perf_counter()

            result = manager.run_text_prompt(
                session_id=session_id,
                prompt=data.get(
                    "prompt",
                    "",
                ),
                confidence_threshold=data.get(
                    "confidence_threshold",
                    CONFIDENCE_THRESHOLD,
                ),
                mask_threshold=data.get(
                    "mask_threshold",
                    MASK_THRESHOLD,
                ),
            )

            inference_finished_at = perf_counter()

            session = manager.get(session_id)

            overlay_started_at = perf_counter()

            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=[],
                boxes=[],
            )

            overlay_finished_at = perf_counter()
            encoding_started_at = perf_counter()

            image_data = pil_to_base64(overlay)

            layers_data = encoded_layers(
                base_image=session.image,
                text_masks=result["text_masks"],
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["processing"] = processing_result(
                stage="text_prompt",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                steps={
                    "sam3_text_inference_ms": elapsed_ms(
                        inference_started_at,
                        inference_finished_at,
                    ),
                    "overlay_rendering_ms": elapsed_ms(
                        overlay_started_at,
                        overlay_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response)

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

        except ValueError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 400

    @blueprint.post(
        "/<session_id>/box-infer"
    )
    def box_infer(session_id):
        request_started_at = perf_counter()

        data = request.get_json(
            silent=True
        ) or {}

        try:
            required = (
                "x1",
                "y1",
                "x2",
                "y2",
            )

            if any(
                key not in data
                for key in required
            ):
                return jsonify(
                    success=False,
                    error=(
                        "Bounding Box 좌표가 필요합니다."
                    ),
                ), 400

            inference_started_at = perf_counter()

            result = manager.run_box_prompt(
                session_id=session_id,
                x1=data["x1"],
                y1=data["y1"],
                x2=data["x2"],
                y2=data["y2"],
            )

            inference_finished_at = perf_counter()

            shadow_started_at = perf_counter()

            shadow_result = create_shadow_result(
                result
            )

            shadow_finished_at = perf_counter()

            session = manager.get(session_id)

            overlay_started_at = perf_counter()

            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=result["visual_masks"],
                boxes=result["boxes"],
            )

            overlay_finished_at = perf_counter()
            encoding_started_at = perf_counter()

            image_data = pil_to_base64(overlay)

            layers_data = encoded_layers(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=result["visual_masks"],
                boxes=result["boxes"],
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["mask_nms_shadow"] = (
                shadow_result
            )

            response["processing"] = processing_result(
                stage="visual_prompt",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                nms_mode="shadow",
                steps={
                    "sam3_visual_inference_ms": elapsed_ms(
                        inference_started_at,
                        inference_finished_at,
                    ),
                    "mask_nms_shadow_ms": elapsed_ms(
                        shadow_started_at,
                        shadow_finished_at,
                    ),
                    "overlay_rendering_ms": elapsed_ms(
                        overlay_started_at,
                        overlay_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response)

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

        except (
            SessionPromptRequiredError,
            SessionInvalidBoxError,
        ) as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 400

        except (TypeError, ValueError):
            return jsonify(
                success=False,
                error=(
                    "Bounding Box 좌표는 숫자여야 합니다."
                ),
            ), 400

    @blueprint.post(
        "/<session_id>/boxes/undo"
    )
    def undo_last_box(session_id):
        """
        Remove the latest Visual Prompt Box.

        Text Prompt results are preserved. Remaining boxes
        are replayed using the existing SAM3 procedure.
        """

        request_started_at = perf_counter()

        try:
            inference_started_at = perf_counter()

            result = manager.undo_last_box(
                session_id
            )

            inference_finished_at = perf_counter()

            shadow_started_at = perf_counter()

            shadow_result = create_shadow_result(
                result
            )

            shadow_finished_at = perf_counter()

            session = manager.get(session_id)

            overlay_started_at = perf_counter()

            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=result["visual_masks"],
                boxes=result["boxes"],
            )

            overlay_finished_at = perf_counter()
            encoding_started_at = perf_counter()

            image_data = pil_to_base64(overlay)

            layers_data = encoded_layers(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=result["visual_masks"],
                boxes=result["boxes"],
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["box_action"] = "undo"
            response["mask_nms_shadow"] = (
                shadow_result
            )

            response["processing"] = processing_result(
                stage="box_undo",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                nms_mode="shadow",
                steps={
                    "sam3_visual_inference_ms": elapsed_ms(
                        inference_started_at,
                        inference_finished_at,
                    ),
                    "mask_nms_shadow_ms": elapsed_ms(
                        shadow_started_at,
                        shadow_finished_at,
                    ),
                    "overlay_rendering_ms": elapsed_ms(
                        overlay_started_at,
                        overlay_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response)

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

        except SessionPromptRequiredError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 400

    @blueprint.delete(
        "/<session_id>/boxes"
    )
    def clear_boxes(session_id):
        """
        Remove all Visual Prompt Boxes and Visual masks.

        Text Prompt masks and counts are preserved.
        """

        request_started_at = perf_counter()

        try:
            action_started_at = perf_counter()

            result = manager.clear_boxes(
                session_id
            )

            action_finished_at = perf_counter()

            shadow_started_at = perf_counter()

            shadow_result = create_shadow_result(
                result
            )

            shadow_finished_at = perf_counter()

            session = manager.get(session_id)

            overlay_started_at = perf_counter()

            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=[],
                boxes=[],
            )

            overlay_finished_at = perf_counter()
            encoding_started_at = perf_counter()

            image_data = pil_to_base64(overlay)

            layers_data = encoded_layers(
                base_image=session.image,
                text_masks=result["text_masks"],
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["box_action"] = "clear"
            response["mask_nms_shadow"] = (
                shadow_result
            )

            response["processing"] = processing_result(
                stage="box_clear",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                nms_mode="shadow",
                steps={
                    "sam3_visual_inference_ms": elapsed_ms(
                        action_started_at,
                        action_finished_at,
                    ),
                    "mask_nms_shadow_ms": elapsed_ms(
                        shadow_started_at,
                        shadow_finished_at,
                    ),
                    "overlay_rendering_ms": elapsed_ms(
                        overlay_started_at,
                        overlay_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response)

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

        except SessionPromptRequiredError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 400

    @blueprint.get("/<session_id>/answer-image")
    def get_answer_image(session_id):
        try:
            session = manager.get(session_id)
            answer_path = find_answer_image(
                session.original_filename
            )

            if answer_path is None:
                return jsonify(
                    success=False,
                    error=(
                        "업로드한 파일명과 일치하는 "
                        "정답 이미지를 찾을 수 없습니다."
                    ),
                ), 404

            if answer_path.stat().st_size > MAX_UPLOAD_BYTES:
                return jsonify(
                    success=False,
                    error="정답 이미지가 안전 용량 제한을 초과했습니다.",
                ), 413

            try:
                with Image.open(answer_path) as decoded:
                    decoded.load()
                    answer_image = (
                        decoded.convert("RGB").copy()
                    )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ):
                return jsonify(
                    success=False,
                    error="정답 이미지 파일을 읽을 수 없습니다.",
                ), 500

            return jsonify(
                success=True,
                session_id=session.session_id,
                original_filename=(
                    session.original_filename
                ),
                answer_filename=answer_path.name,
                width=answer_image.width,
                height=answer_image.height,
                image=pil_to_base64(answer_image),
            )

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404


    @blueprint.post(
        "/<session_id>/reset"
    )
    def reset_session(session_id):
        request_started_at = perf_counter()

        try:
            reset_started_at = perf_counter()

            session = manager.reset(session_id)
            result = manager.snapshot(session)

            reset_finished_at = perf_counter()
            encoding_started_at = perf_counter()

            image_data = pil_to_base64(
                session.image
            )

            layers_data = encoded_layers(
                base_image=session.image,
            )

            encoding_finished_at = perf_counter()

            response = public_result(
                result,
                image_data,
                layers_data,
            )

            response["processing"] = processing_result(
                stage="reset",
                started_at=request_started_at,
                finished_at=encoding_finished_at,
                steps={
                    "session_reset_ms": elapsed_ms(
                        reset_started_at,
                        reset_finished_at,
                    ),
                    "image_encoding_ms": elapsed_ms(
                        encoding_started_at,
                        encoding_finished_at,
                    ),
                },
            )

            return jsonify(response)

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

    @blueprint.delete("/<session_id>")
    def delete_session(session_id):
        try:
            manager.delete(session_id)

            return jsonify(
                success=True,
            )

        except SessionNotFoundError as error:
            return jsonify(
                success=False,
                error=str(error),
            ), 404

    return blueprint