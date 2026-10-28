"""Flask routes for secure, isolated SAM3 analysis sessions."""

from __future__ import annotations

from time import perf_counter

from flask import Blueprint, jsonify, request

from security.file_validation import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    validate_image_upload,
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
            "original_width":
                result["original_width"],
            "original_height":
                result["original_height"],
            "original_filename":
                result["original_filename"],
            "detected_format":
                result["detected_format"],
            "file_size": result["file_size"],
            "confidence_threshold":
                CONFIDENCE_THRESHOLD,
            "mask_threshold":
                MASK_THRESHOLD,
        }

        if image_data is not None:
            payload["image"] = image_data

        if layers_data is not None:
            payload["layers"] = layers_data

        return payload

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
                error=(
                    "업로드 요청 크기가 "
                    "제한을 초과했습니다."
                ),
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

            session_started_at = perf_counter()

            session = manager.create(validated)
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

            response["processing"] = (
                processing_result(
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
                        "session_creation_ms":
                            elapsed_ms(
                                session_started_at,
                                session_finished_at,
                            ),
                        "image_encoding_ms":
                            elapsed_ms(
                                encoding_started_at,
                                encoding_finished_at,
                            ),
                    },
                )
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
            inference_started_at = (
                perf_counter()
            )

            result = manager.run_text_prompt(
                session_id,
                data.get("prompt", ""),
            )

            inference_finished_at = (
                perf_counter()
            )

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

            response["processing"] = (
                processing_result(
                    stage="text_prompt",
                    started_at=request_started_at,
                    finished_at=encoding_finished_at,
                    steps={
                        "sam3_text_inference_ms":
                            elapsed_ms(
                                inference_started_at,
                                inference_finished_at,
                            ),
                        "overlay_rendering_ms":
                            elapsed_ms(
                                overlay_started_at,
                                overlay_finished_at,
                            ),
                        "image_encoding_ms":
                            elapsed_ms(
                                encoding_started_at,
                                encoding_finished_at,
                            ),
                    },
                )
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
                        "Bounding Box 좌표가 "
                        "필요합니다."
                    ),
                ), 400

            inference_started_at = (
                perf_counter()
            )

            result = manager.run_box_prompt(
                session_id=session_id,
                x1=data["x1"],
                y1=data["y1"],
                x2=data["x2"],
                y2=data["y2"],
            )

            inference_finished_at = (
                perf_counter()
            )

            shadow_started_at = perf_counter()

            try:
                shadow_result = (
                    analyze_mask_nms_shadow(
                        text_masks=(
                            result["text_masks"]
                        ),
                        text_scores=(
                            result["text_scores"]
                        ),
                        visual_masks=(
                            result["visual_masks"]
                        ),
                        visual_scores=(
                            result["visual_scores"]
                        ),
                        iou_threshold=(
                            MASK_NMS_IOU_THRESHOLD
                        ),
                        mask_threshold=(
                            MASK_THRESHOLD
                        ),
                    )
                )

                shadow_result["status"] = (
                    "completed"
                )

            except Exception:
                # Shadow 실패가 기존 SAM3 결과를
                # 막거나 변경하면 안 됩니다.
                shadow_result = {
                    "mode": "shadow",
                    "status": "error",
                    "applied_to_result": False,
                    "iou_threshold":
                        MASK_NMS_IOU_THRESHOLD,
                    "mask_threshold":
                        MASK_THRESHOLD,
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
                            "Mask-NMS Shadow 분석에 "
                            "실패했지만 기존 SAM3 "
                            "결과는 정상적으로 "
                            "유지되었습니다."
                        )
                    ],
                }

            shadow_finished_at = perf_counter()

            session = manager.get(session_id)

            overlay_started_at = perf_counter()

            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=(
                    result["visual_masks"]
                ),
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

            response["processing"] = (
                processing_result(
                    stage="visual_prompt",
                    started_at=request_started_at,
                    finished_at=encoding_finished_at,
                    nms_mode="shadow",
                    steps={
                        "sam3_visual_inference_ms":
                            elapsed_ms(
                                inference_started_at,
                                inference_finished_at,
                            ),
                        "mask_nms_shadow_ms":
                            elapsed_ms(
                                shadow_started_at,
                                shadow_finished_at,
                            ),
                        "overlay_rendering_ms":
                            elapsed_ms(
                                overlay_started_at,
                                overlay_finished_at,
                            ),
                        "image_encoding_ms":
                            elapsed_ms(
                                encoding_started_at,
                                encoding_finished_at,
                            ),
                    },
                )
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
                    "Bounding Box 좌표는 "
                    "숫자여야 합니다."
                ),
            ), 400

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

            response["processing"] = (
                processing_result(
                    stage="reset",
                    started_at=request_started_at,
                    finished_at=encoding_finished_at,
                    steps={
                        "session_reset_ms":
                            elapsed_ms(
                                reset_started_at,
                                reset_finished_at,
                            ),
                        "image_encoding_ms":
                            elapsed_ms(
                                encoding_started_at,
                                encoding_finished_at,
                            ),
                    },
                )
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
