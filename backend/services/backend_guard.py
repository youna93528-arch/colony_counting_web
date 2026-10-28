"""Single backend ownership, checked before importing the model stack."""
import atexit
import os
from pathlib import Path
import socket


def acquire_backend_guard(project_root, host='127.0.0.1', port=7860):
    directory = Path(project_root) / 'runtime'
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / ('backend-%s.lock' % port), 'a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit('CosMi-SAM backend is already starting or running. No second model was loaded.')
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == 'nt':
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((host, port))
    except OSError:
        handle.close()
        raise SystemExit('Port %s is already in use. Stop the old backend first. No model was loaded.' % port)
    finally:
        probe.close()
    # The OS releases the lock after normal exit, Ctrl+C, or process termination.
    atexit.register(handle.close)
    return handle
