"""Flask routes for secure, isolated SAM3 analysis sessions."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from security.file_validation import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    validate_image_upload,
)
from services.session_manager import (
    ALLOWED_PROMPTS,
    SessionInvalidBoxError,
    SessionLimitError,
    SessionNotFoundError,
    SessionPromptRequiredError,
)
from services.sam3_service import CONFIDENCE_THRESHOLD, MASK_THRESHOLD


def create_session_blueprint(manager, create_overlay_image, pil_to_base64):
    blueprint = Blueprint("analysis_sessions", __name__, url_prefix="/api/sessions")

    def public_result(result, image_data=None):
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
            "original_width": result["original_width"],
            "original_height": result["original_height"],
            "original_filename": result["original_filename"],
            "detected_format": result["detected_format"],
            "file_size": result["file_size"],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "mask_threshold": MASK_THRESHOLD,
        }
        if image_data is not None:
            payload["image"] = image_data
        return payload

    @blueprint.post("")
    def create_session():
        if request.content_length and request.content_length > MAX_UPLOAD_BYTES + 1024 * 1024:
            return jsonify(success=False, error="업로드 요청 크기가 제한을 초과했습니다."), 413

        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(success=False, error="image 파일이 필요합니다."), 400

        payload = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            return jsonify(success=False, error="이미지 파일은 최대 30MB까지 업로드할 수 있습니다."), 413

        try:
            validated = validate_image_upload(uploaded.filename, payload)
            session = manager.create(validated)
            result = manager.snapshot(session)
            response = public_result(result, pil_to_base64(session.image))
            response["allowed_prompts"] = list(ALLOWED_PROMPTS.keys())
            return jsonify(response), 201
        except ImageValidationError as error:
            return jsonify(success=False, error=str(error)), 400
        except SessionLimitError as error:
            return jsonify(success=False, error=str(error)), 429

    @blueprint.get("/<session_id>")
    def get_session(session_id):
        try:
            session = manager.get(session_id)
            return jsonify(public_result(manager.snapshot(session)))
        except SessionNotFoundError as error:
            return jsonify(success=False, error=str(error)), 404

    @blueprint.post("/<session_id>/text-infer")
    def text_infer(session_id):
        data = request.get_json(silent=True) or {}
        try:
            result = manager.run_text_prompt(session_id, data.get("prompt", ""))
            session = manager.get(session_id)
            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=[],
                boxes=[],
            )
            return jsonify(public_result(result, pil_to_base64(overlay)))
        except SessionNotFoundError as error:
            return jsonify(success=False, error=str(error)), 404
        except ValueError as error:
            return jsonify(success=False, error=str(error)), 400

    @blueprint.post("/<session_id>/box-infer")
    def box_infer(session_id):
        data = request.get_json(silent=True) or {}
        try:
            required = ("x1", "y1", "x2", "y2")
            if any(key not in data for key in required):
                return jsonify(success=False, error="Bounding Box 좌표가 필요합니다."), 400

            result = manager.run_box_prompt(
                session_id=session_id,
                x1=data["x1"],
                y1=data["y1"],
                x2=data["x2"],
                y2=data["y2"],
            )
            session = manager.get(session_id)
            overlay = create_overlay_image(
                base_image=session.image,
                text_masks=result["text_masks"],
                visual_masks=result["visual_masks"],
                boxes=result["boxes"],
            )
            return jsonify(public_result(result, pil_to_base64(overlay)))
        except SessionNotFoundError as error:
            return jsonify(success=False, error=str(error)), 404
        except (SessionPromptRequiredError, SessionInvalidBoxError) as error:
            return jsonify(success=False, error=str(error)), 400
        except (TypeError, ValueError):
            return jsonify(success=False, error="Bounding Box 좌표는 숫자여야 합니다."), 400

    @blueprint.post("/<session_id>/reset")
    def reset_session(session_id):
        try:
            session = manager.reset(session_id)
            result = manager.snapshot(session)
            return jsonify(public_result(result, pil_to_base64(session.image)))
        except SessionNotFoundError as error:
            return jsonify(success=False, error=str(error)), 404

    @blueprint.delete("/<session_id>")
    def delete_session(session_id):
        try:
            manager.delete(session_id)
            return jsonify(success=True)
        except SessionNotFoundError as error:
            return jsonify(success=False, error=str(error)), 404

    return blueprint
