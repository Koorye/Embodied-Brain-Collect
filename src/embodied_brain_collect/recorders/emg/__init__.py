"""EMG recorders -- WAVELETECH 8-channel armband + dummy for testing.

Usage::

    from recorders.emg import EmgRecorderConfig, DummyEmgRecorder, WeiliEmgRecorder

    cfg = EmgRecorderConfig(port="COM12", duration=10)
    rec = WeiliEmgRecorder(cfg)
    rec.run()
"""

from .emg_recorder_config import EmgRecorderConfig
from .base_emg_recorder import BaseEmgRecorder
from .dummy_emg_recorder import DummyEmgRecorder
from .weili_emg_recorder import WeiliEmgRecorder

__all__ = [
    "EmgRecorderConfig",
    "BaseEmgRecorder",
    "DummyEmgRecorder",
    "WeiliEmgRecorder",
]
