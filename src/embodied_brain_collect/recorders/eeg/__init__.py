"""EEG recorders -- Curry NetStream, Blackrock Cerebus, Intan RHX + dummy.

The EEG<->PC alignment lives in ``BaseEegRecorder._close()``: every real
recorder turns its amplifier's TTL/marker pathway into events carrying the
marker code plus the amp sample index (Curry: event packets; Blackrock: NSP
digital-in packets; Intan: the 16-bit digital-in word in each frame), and
``marker*/*.npz`` holds the same codes with PC-clock times.  Pairing them by
code sequence and fitting ``t_pc = a * t_eeg + b`` maps every EEG sample
onto the PC clock.  The fit result and the per-sample ``eeg_timestamps_pc``
are saved inside the slot's npz.

Usage::

    from embodied_brain_collect.recorders.eeg import (
        EegRecorderConfig, BlackrockEegRecorderConfig, IntanEegRecorderConfig,
        DummyEegRecorder, CurryEegRecorder, BlackrockEegRecorder,
        IntanEegRecorder)

    cfg = IntanEegRecorderConfig(session_dir="./out", duration=60)
    rec = IntanEegRecorder(cfg)
    rec.run()
"""

from .eeg_recorder_config import (EegRecorderConfig,
                                  BlackrockEegRecorderConfig,
                                  IntanEegRecorderConfig)
from .base_eeg_recorder import BaseEegRecorder
from .dummy_eeg_recorder import DummyEegRecorder
from .curry_eeg_recorder import CurryEegRecorder
from .blackrock_eeg_recorder import BlackrockEegRecorder
from .intan_eeg_recorder import IntanEegRecorder

__all__ = [
    "EegRecorderConfig",
    "BlackrockEegRecorderConfig",
    "IntanEegRecorderConfig",
    "BaseEegRecorder",
    "DummyEegRecorder",
    "CurryEegRecorder",
    "BlackrockEegRecorder",
    "IntanEegRecorder",
]
