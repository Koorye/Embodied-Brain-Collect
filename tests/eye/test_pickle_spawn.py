"""Launcher pickling regression — no hardware needed.

The launcher runs each recorder in its own process; on Windows (spawn
start method) that pickles the recorder.  NeonEyeAsyncRecorder carries
six plain ``threading.Event``s that once crashed the spawn with
``TypeError: cannot pickle '_thread.lock'``.  These tests pin the fix:
every plain Event is stripped by ``__getstate__`` and rebuilt fresh by
``__setstate__``, while the launcher's multiprocessing ``stop_event``
passes through untouched.

Runs under pytest (``pytest tests/eye/test_pickle_spawn.py``) or
directly (``python tests/eye/test_pickle_spawn.py``).
"""

import pickle
import multiprocessing as mp
import threading
from pathlib import Path

from loguru import logger as _loguru

try:
    import pytest
except ImportError:    # direct run (`python tests/eye/test_pickle_spawn.py`)
    pytest = None

if pytest is not None:
    pytest.importorskip("pupil_labs")
else:
    import importlib
    importlib.import_module("pupil_labs")

from embodied_brain_collect.recorders.eye import (   # noqa: E402
    NeonEyeAsyncRecorder, EyeRecorderConfig)

_CTX = mp.get_context("spawn")

_EVENTS = ("_ready_evt", "_start_evt", "_loop_done",
           "_first_gaze_evt", "_first_imu_evt", "_first_scene_evt")


def _child(rec):
    """Real spawn child: the unpickled recorder must be fully usable."""
    missing = [k for k in _EVENTS if not hasattr(rec, k)]
    ok = (not missing
          and all(type(getattr(rec, k)) is threading.Event for k in _EVENTS)
          and type(rec.stop_event) is not threading.Event  # mp.Event survives
          and rec.logger is not None)
    return 0 if ok else 1


def _make_recorder(tmp_path):
    return NeonEyeAsyncRecorder(
        EyeRecorderConfig(session_dir=str(tmp_path / "eye")))


def _close_sinks(*recs):
    """Close loguru file sinks so Windows temp-dir cleanup can proceed."""
    for rec in recs:
        if rec._log_sink_id is not None:
            _loguru.remove(rec._log_sink_id)
            rec._log_sink_id = None


def test_pickle_roundtrip_rebuilds_events(tmp_path):
    rec = _make_recorder(tmp_path)
    rec2 = None
    try:
        rec2 = pickle.loads(pickle.dumps(rec))
        for k in _EVENTS:
            assert type(getattr(rec2, k)) is threading.Event
        assert type(rec2.stop_event) is threading.Event
        assert rec2.logger is not None
    finally:
        _close_sinks(rec)
        if rec2 is not None:
            _close_sinks(rec2)


def test_spawn_child_gets_usable_recorder(tmp_path):
    rec = _make_recorder(tmp_path)
    try:
        rec.stop_event = _CTX.Event()   # exactly what launch() does
        rec._hb_queue = _CTX.Queue()
        p = _CTX.Process(target=_child, args=(rec,))
        p.start()
        p.join(timeout=60)
        assert p.exitcode == 0
    finally:
        _close_sinks(rec)


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_pickle_roundtrip_rebuilds_events(tmp)
        test_spawn_child_gets_usable_recorder(tmp)
    print("[test:pickle_spawn] both tests passed")


if __name__ == "__main__":
    main()
