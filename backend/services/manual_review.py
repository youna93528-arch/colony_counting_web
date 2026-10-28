"""Reversible manual count review. Never modifies model masks or runs inference."""
from __future__ import annotations
import hashlib
import math
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageDraw
from flask import jsonify, request, send_file
from services.layer_renderer import _mask_to_bool, create_overlay_layer

class ReviewConflict(ValueError): pass

def _keys(session):
    entries = []
    for source, masks in (('text', session.text_masks), ('visual', session.visual_masks)):
        duplicates = {}
        for index, raw in enumerate(masks):
            mask = _mask_to_bool(raw, session.width, session.height)
            digest = hashlib.sha256(np.packbits(mask.reshape(-1)).tobytes()).hexdigest()
            digest = str(session.width) + 'x' + str(session.height) + ':' + digest
            occurrence = duplicates.get(digest, 0); duplicates[digest] = occurrence + 1
            entries.append(dict(id=f'{source}:{index}', key=f'{source}:{digest}:{occurrence}',
                source=source, index=index, label=('T' if source == 'text' else 'V') + str(index+1).zfill(3)))
    return entries

def review_state(session, need_keys=False):
    state = getattr(session, 'manual_review_state', None)
    if state is None:
        state = dict(text=session.text_layer_revision, visual=session.visual_layer_revision,
            revision=0, keys=None, excluded=set(), history=[])
        session.manual_review_state = state
    changed_text = state['text'] != session.text_layer_revision
    changed_visual = state['visual'] != session.visual_layer_revision
    if changed_text or changed_visual:
        if changed_text:
            state['excluded'].clear()
        state.update(text=session.text_layer_revision, visual=session.visual_layer_revision,
            revision=state['revision']+1, keys=None, history=[])
        if state['excluded']:
            state['keys'] = _keys(session)
            state['excluded'].intersection_update(item['key'] for item in state['keys'])
    if need_keys and state['keys'] is None:
        state['keys'] = _keys(session)
    return state

def review_metadata(session):
    state = review_state(session)
    excluded = [item for item in state['keys'] or [] if item['key'] in state['excluded']]
    text_excluded = sum(item['source']=='text' for item in excluded)
    visual_excluded = len(excluded)-text_excluded
    automatic = len(session.text_masks)+len(session.visual_masks)
    return dict(revision=state['revision'], automatic_count=automatic, excluded_count=len(excluded),
        final_count=automatic-len(excluded), text_included_count=len(session.text_masks)-text_excluded,
        visual_included_count=len(session.visual_masks)-visual_excluded,
        excluded_ids=[item['id'] for item in excluded],
        object_keys={item['id']:item['key'] for item in state['keys'] or []},
        undo_available=bool(state['history']), evaluation_basis='automatic_detection')

def decorate_review_payload(payload, result):
    review = result.get('manual_review')
    if review is None: return payload
    payload['manual_review'] = review
    if review['excluded_count']:
        sid = result['session_id']
        version = f"text={result['text_layer_revision']}&visual={result['visual_layer_revision']}&review={review['revision']}"
        payload['layers'].update({name:f'/api/sessions/{sid}/review/layers/{name}.png?{version}' for name in ('text','visual','excluded')})
        payload['result_image_url'] = f'/api/sessions/{sid}/review/result.png?{version}'
    return payload

def _check_version(session, state, data):
    for key, expected in (('text',session.text_layer_revision),('visual',session.visual_layer_revision),('review',state['revision'])):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value,int) or value != expected:
            raise ReviewConflict('분석 결과나 수동 보정이 변경되었습니다. 현재 결과를 다시 불러온 뒤 선택해 주세요.')

def apply_review(session, data):
    state = review_state(session)
    _check_version(session,state,data)
    if not session.text_done: raise ValueError('1차 검출을 먼저 실행해 주세요.')
    action = data.get('action')
    if action == 'undo':
        if state['history']:
            state['excluded'] = state['history'].pop(); state['revision'] += 1
        return
    if action not in ('exclude','restore'): raise ValueError('지원하지 않는 수동 보정입니다.')
    state = review_state(session, True)
    key = data.get('object_key')
    if not isinstance(key,str) or key not in {item['key'] for item in state['keys']}:
        raise ValueError('현재 분석 결과에 없는 객체입니다.')
    changed = key not in state['excluded'] if action == 'exclude' else key in state['excluded']
    if changed:
        state['history'].append(set(state['excluded'])); state['history'] = state['history'][-100:]
        if action == 'exclude': state['excluded'].add(key)
        else: state['excluded'].discard(key)
        state['revision'] += 1

def inspection_review(session, objects, editing=False):
    state = review_state(session, editing)
    entries = {item['id']:item for item in state['keys'] or []}
    for item in objects:
        entry=entries.get(item['id'])
        item['object_key'] = entry['key'] if entry else None
        item['excluded'] = bool(entry and entry['key'] in state['excluded'])
    return state['revision']

def _active_masks(session, source):
    state=review_state(session)
    excluded = {item['index'] for item in state['keys'] or [] if item['source']==source and item['key'] in state['excluded']}
    return [mask for index,mask in enumerate(session.text_masks if source=='text' else session.visual_masks) if index not in excluded]

def _dashed(draw, points, fill, width):
    for start in range(0,len(points),10):
        segment=points[start:start+5]
        if len(segment)>1:draw.line(segment,fill=fill,width=width)
        elif segment:draw.point(segment[0],fill=fill)


def render_review_layer(session, name):
    if name in ('text','visual'):
        layer=create_overlay_layer(layer_name=name,base_image=session.image,
            text_masks=_active_masks(session,'text') if name=='text' else [],
            visual_masks=_active_masks(session,'visual') if name=='visual' else [],boxes=[])
        pixels=np.array(layer)
        colored=(pixels[:,:,3]>0)&(pixels[:,:,:3].max(axis=2)>35)
        pixels[colored,:3]=(239,68,68) if name=='text' else (251,146,60)
        return Image.fromarray(pixels,'RGBA')
    image=Image.new('RGBA',(session.width,session.height),(0,0,0,0)); draw=ImageDraw.Draw(image)
    state=review_state(session)
    for item in state['keys'] or []:
        if item['key'] not in state['excluded']:continue
        masks=session.text_masks if item['source']=='text' else session.visual_masks
        mask=_mask_to_bool(masks[item['index']],session.width,session.height)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            points=[tuple(map(int,point)) for point in contour.reshape(-1,2)]
            if len(points)>1:_dashed(draw,points,(148,163,184,255),2)
            elif points:draw.point(points[0],fill=(148,163,184,255))
        ys,xs=np.nonzero(mask)
        if xs.size:
            xy=(max(0,int(xs.min())),max(0,int(ys.min())-12))
            draw.text(xy,'X '+item['label'],fill=(203,213,225,255),stroke_width=1,stroke_fill=(15,23,42,255))
    return image

def _png(image):
    output=BytesIO();image.save(output,format='PNG',compress_level=3);output.seek(0)
    response=send_file(output,mimetype='image/png')
    response.headers['Cache-Control']='private, no-store'
    return response

def register_manual_review_routes(blueprint, manager, public_result, not_found_error):
    @blueprint.post('/<session_id>/review')
    def change_review(session_id):
        data=request.get_json(silent=True)
        if not isinstance(data,dict):return jsonify(success=False,error='올바른 보정 요청이 필요합니다.'),400
        try:
            session=manager.get(session_id)
            with session.lock:
                apply_review(session,data)
                response=public_result(manager.snapshot(session))
            return jsonify(response)
        except not_found_error as error:return jsonify(success=False,error=str(error)),404
        except ReviewConflict as error:return jsonify(success=False,error=str(error)),409
        except ValueError as error:return jsonify(success=False,error=str(error)),400

    def checked_session(session_id):
        return manager.get(session_id),{key:request.args.get(key,type=int) for key in ('text','visual','review')}

    @blueprint.get('/<session_id>/review/layers/<name>.png')
    def review_layer(session_id,name):
        if name not in ('text','visual','excluded'):return jsonify(success=False),404
        try:
            session,version=checked_session(session_id)
            with session.lock:
                _check_version(session,review_state(session),version)
                image=render_review_layer(session,name)
            return _png(image)
        except not_found_error as error:return jsonify(success=False,error=str(error)),404
        except ReviewConflict as error:return jsonify(success=False,error=str(error)),409

    @blueprint.get('/<session_id>/review/result.png')
    def review_result(session_id):
        try:
            session,version=checked_session(session_id)
            with session.lock:
                _check_version(session,review_state(session),version)
                image=session.image.convert('RGBA')
                for name in ('text','visual','boxes'):
                    if name=='boxes':layer=create_overlay_layer(layer_name='boxes',base_image=session.image,text_masks=[],visual_masks=[],boxes=session.boxes)
                    else:layer=render_review_layer(session,name)
                    image=Image.alpha_composite(image,layer)
                image=Image.alpha_composite(image,render_review_layer(session,'excluded'))
            response=_png(image)
            response.headers['Content-Disposition']='attachment; filename="CosMiNet-reviewed-result.png"'
            return response
        except not_found_error as error:return jsonify(success=False,error=str(error)),404
        except ReviewConflict as error:return jsonify(success=False,error=str(error)),409
