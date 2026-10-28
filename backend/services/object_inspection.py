"""Read-only object geometry for the analysis UI; never runs or modifies inference."""
from __future__ import annotations
import math
import cv2
import numpy as np
from flask import jsonify, request
from services.layer_renderer import _mask_to_bool
from services.manual_review import inspection_review, review_state


def build_object_descriptors(width, height, text_masks, visual_masks, text_scores, visual_scores):
    objects = []
    for source, masks, scores in (("text", text_masks, text_scores), ("visual", visual_masks, visual_scores)):
        for index, raw_mask in enumerate(masks):
            mask = _mask_to_bool(raw_mask, width, height)
            ys, xs = np.nonzero(mask)
            score = None
            if index < len(scores):
                try:
                    candidate = np.asarray(scores[index]).reshape(-1)
                    if candidate.size == 1 and math.isfinite(float(candidate[0])):
                        score = float(candidate[0])
                except (ValueError, TypeError, OverflowError):
                    pass
            bbox, commands, runs = [0, 0, 0, 0], [], []
            if xs.size:
                x, y = int(xs.min()), int(ys.min())
                w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
                bbox = [x, y, w, h]
                crop = mask[y:y+h, x:x+w]
                flat = crop.reshape(-1).astype(np.int8)
                transitions = np.flatnonzero(np.diff(np.pad(flat, (1, 1))))
                for start, end in transitions.reshape(-1, 2):
                    runs.extend((int(start), int(end - start)))
                contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points = contour.reshape(-1, 2)
                    if len(points) == 1:
                        px, py = int(points[0][0]) + x, int(points[0][1]) + y
                        commands.append(f"M{px} {py}h1v1h-1Z")
                    elif len(points):
                        commands.append("M" + "L".join(f"{int(px)+x} {int(py)+y}" for px, py in points) + "Z")
            objects.append({"id": f"{source}:{index}", "source": source, "index": index, "score": score,
                            "area_px": int(xs.size), "bbox": bbox, "path": "".join(commands), "hit_runs": runs})
    return objects


def register_object_inspection_routes(blueprint, manager, not_found_error):
    @blueprint.get("/<session_id>/objects")
    def get_inspection_objects(session_id):
        text_revision = request.args.get("text", type=int)
        visual_revision = request.args.get("visual", type=int)
        if text_revision is None or visual_revision is None:
            return jsonify(success=False, error="분석 결과 버전이 필요합니다."), 400
        try:
            session = manager.get(session_id)
            with session.lock:
                if text_revision != session.text_layer_revision or visual_revision != session.visual_layer_revision:
                    return jsonify(success=False, error="분석 결과가 변경되었습니다. 현재 결과로 다시 확인해 주세요."), 409
                width, height = session.width, session.height
                text_masks, visual_masks = list(session.text_masks), list(session.visual_masks)
                text_scores, visual_scores = list(session.text_scores), list(session.visual_scores)
                state = review_state(session)
                requested_review = request.args.get("review", type=int)
                if requested_review is not None and requested_review != state["revision"]:
                    return jsonify(success=False, error="수동 보정이 변경되었습니다. 현재 결과를 다시 불러와 주세요."), 409
                objects = build_object_descriptors(width, height, text_masks, visual_masks, text_scores, visual_scores)
                review_revision = inspection_review(session, objects, True)
            response = jsonify(success=True, session_id=session_id, text_revision=text_revision,
                               visual_revision=visual_revision, review_revision=review_revision, width=width, height=height, objects=objects)
            response.headers["Cache-Control"] = "no-store"
            return response
        except not_found_error as error:
            return jsonify(success=False, error=str(error)), 404

# colony-manual-review-v1

# colony-selection-focus-v1
