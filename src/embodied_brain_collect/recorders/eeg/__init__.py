"""EEG recorders -- Neuroscan Curry NetStream + dummy for testing.

The EEG<->PC alignment lives in ``BaseEegRecorder._close()``: Curry events
carry the marker code with the amp sample index, and ``markers/markers.npz``
holds the same codes with PC-clock receipt times; pairing them by code
sequence and fitting ``t_pc = a * t_eeg + b`` maps every EEG sample onto the
PC clock.  The fit result and the per-sample ``eeg_timestamps_pc`` are saved
inside ``eeg/eeg.npz``.

Usage::

    from embodied_brain_collect.recorders.eeg import (
        EegRecorderConfig, DummyEegRecorder, CurryEegRecorder)

    cfg = EegRecorderConfig(session_dir="./out", duration=60)
    rec = CurryEegRecorder(cfg)
    rec.run()
"""

from .eeg_recorder_config import EegRecorderConfig
from .base_eeg_recorder import BaseEegRecorder
from .dummy_eeg_recorder import DummyEegRecorder
from .curry_eeg_recorder import CurryEegRecorder

__all__ = [
    "EegRecorderConfig",
    "BaseEegRecorder",
    "DummyEegRecorder",
    "CurryEegRecorder",
]
