"""Request-local observation; no model calls, mask changes or cancellation."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import json
import csv
import logging
import queue
import re
import threading
from time import perf_counter
from uuid import uuid4

_current = ContextVar('analysis_runtime_request', default=None)
_lock = threading.Lock()
_records = {}
_MAX_RECORDS = 256
_PHASES = {'received', 'waiting_session', 'waiting_gpu', 'preparing', 'inference', 'postprocessing', 'completed', 'error'}


def set_runtime_phase(phase):
    record = _current.get()
    if record is None or phase not in _PHASES:
        return
    now = perf_counter()
    with _lock:
        previous = record['phase']
        elapsed = max(0.0, (now - record['_phase_started']) * 1000)
        times = record['phase_durations_ms']
        times[previous] = times.get(previous, 0.0) + elapsed
        record['phase'] = phase
        record['_phase_started'] = now


@contextmanager
def observed_lock(lock, waiting_phase):
    set_runtime_phase(waiting_phase)
    lock.acquire()
    try:
        set_runtime_phase('preparing')
        yield
    finally:
        lock.release()


def install_runtime_observability(app, project_root):
    from flask import g, jsonify, request
    log_directory = Path(project_root) / 'runtime' / 'performance'
    writes = queue.Queue(maxsize=1024)
    app.extensions['runtime_performance_queue'] = writes

    def write_records():
        while True:
            item = writes.get()
            try:
                log_directory.mkdir(parents=True, exist_ok=True)
                path = log_directory / (item['started_at_utc'][:10] + '.jsonl')
                with path.open('a', encoding='utf-8') as output:
                    output.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                csv_path = path.with_suffix('.csv')
                new_file = not csv_path.exists() or csv_path.stat().st_size == 0
                result = item.get('result', {})
                times = item['phase_durations_ms']
                stages = {'upload': '이미지 업로드', 'text_prompt': '1차 검출', 'visual_prompt': '2차 검출', 'box_undo': '마지막 Box 취소', 'box_clear': '모든 Box 지우기', 'reset': '초기화'}
                header = ['시작시각_UTC', '요청_ID', '작업', '성공', 'HTTP상태', '원본파일명', '원본너비', '원본높이', '분석너비', '분석높이', '미생물종류', '적용_Confidence', '적용_Mask_Threshold', '1차_검출수', '2차_검출수', 'Box_수', '요청전체_ms', '동일이미지_대기_ms', '다른검출_대기_ms', '추론구간_ms', '결과정리구간_ms']
                row = [item['started_at_utc'], item['request_id'], stages.get(item['stage'], item['stage']), item['success'], item['http_status']]
                row.extend(result.get(key, '') for key in ('original_filename', 'original_width', 'original_height', 'width', 'height', 'prompt', 'confidence_threshold', 'mask_threshold', 'text_count', 'visual_count', 'box_count'))
                row.extend([item['total_ms'], times.get('waiting_session', 0), times.get('waiting_gpu', 0), times.get('inference', 0), times.get('postprocessing', 0)])
                def safe_cell(value):
                    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                        return "'" + value
                    return value
                with csv_path.open('a', encoding='utf-8-sig' if new_file else 'utf-8', newline='') as output:
                    writer = csv.writer(output)
                    if new_file:
                        writer.writerow(header)
                    writer.writerow([safe_cell(value) for value in row])
            except Exception:
                logging.getLogger(__name__).warning('Performance record could not be written.', exc_info=True)
            finally:
                writes.task_done()

    threading.Thread(target=write_records, name='analysis-performance-log', daemon=True).start()

    def action():
        if request.method not in ('POST', 'DELETE'):
            return None
        if request.path.rstrip('/') == '/api/sessions' and request.method == 'POST':
            return 'upload'
        match = re.fullmatch(r'/api/sessions/[^/]+/(text-infer|box-infer|boxes/undo|boxes|reset)', request.path)
        return {'text-infer': 'text_prompt', 'box-infer': 'visual_prompt', 'boxes/undo': 'box_undo', 'boxes': 'box_clear', 'reset': 'reset'}.get(match.group(1)) if match else None

    @app.before_request
    def start_runtime_record():
        stage = action()
        if stage is None:
            return
        requested_id = request.headers.get('X-Analysis-Request-Id', '')
        request_id = requested_id if re.fullmatch(r'[A-Za-z0-9-]{16,80}', requested_id) else str(uuid4())
        now = perf_counter()
        record = {'request_id': request_id, 'stage': stage, 'phase': 'received', 'started_at_utc': datetime.now(timezone.utc).isoformat(), '_started': now, '_phase_started': now, 'phase_durations_ms': {}}
        with _lock:
            # Never let a reused identifier overwrite another active request.
            if request_id in _records:
                return jsonify(success=False, error='Duplicate analysis request identifier.'), 409
            if len(_records) >= _MAX_RECORDS:
                finished = [key for key, value in _records.items() if value['phase'] in ('completed', 'error')]
                for key in finished[:max(1, len(_records) - _MAX_RECORDS + 1)]:
                    del _records[key]
                if len(_records) >= _MAX_RECORDS:
                    return jsonify(success=False, error='Too many active analysis requests.'), 503
            _records[request_id] = record
        g.runtime_record = record
        g.runtime_finished = False
        g.runtime_token = _current.set(record)

    def finish(response=None):
        record = getattr(g, 'runtime_record', None)
        if record is None or getattr(g, 'runtime_finished', False):
            return
        g.runtime_finished = True
        data = response.get_json(silent=True) if response is not None and response.is_json else {}
        data = data if isinstance(data, dict) else {}
        success = response is not None and response.status_code < 400 and data.get('success') is True
        set_runtime_phase('completed' if success else 'error')
        record['total_ms'] = round((perf_counter() - record['_started']) * 1000, 2)
        record['success'] = success
        record['http_status'] = response.status_code if response is not None else 500
        public = {key: value for key, value in record.items() if not key.startswith('_')}
        public['phase_durations_ms'] = {key: round(value, 2) for key, value in public['phase_durations_ms'].items()}
        # Only values returned by the server are recorded as applied conditions.
        public['result'] = {key: data[key] for key in ('session_id', 'original_filename', 'original_width', 'original_height', 'width', 'height', 'prompt', 'confidence_threshold', 'mask_threshold', 'text_count', 'visual_count', 'box_count') if key in data}
        if isinstance(data.get('processing'), dict):
            public['server_processing'] = data['processing']
        if not success:
            public['error'] = str(data.get('error', 'Request failed'))[:512]
        try:
            writes.put_nowait(public)
        except queue.Full:
            logging.getLogger(__name__).warning('Performance log queue is full; inference result is unchanged.')

    @app.after_request
    def finish_runtime_record(response):
        finish(response)
        return response

    @app.teardown_request
    def release_runtime_context(error):
        try:
            finish()
        finally:
            token = g.pop('runtime_token', None)
            if token is not None:
                _current.reset(token)

    @app.get('/api/runtime/requests/<request_id>')
    def runtime_request_status(request_id):
        with _lock:
            record = _records.get(request_id)
            if record is None:
                payload = {'success': True, 'found': False, 'phase': 'not_received'}
            else:
                elapsed = (perf_counter() - record['_started']) * 1000
                payload = {'success': True, 'found': True, 'request_id': request_id, 'stage': record['stage'], 'phase': record['phase'], 'elapsed_ms': round(record.get('total_ms', elapsed), 2)}
        response = jsonify(payload)
        response.headers['Cache-Control'] = 'no-store'
        return response
