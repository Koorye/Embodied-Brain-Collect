"""Position tracker recorders.

The OpenVR recorder imports ``openvr`` at module top; keep that export lazy
so the base/dummy classes stay importable on machines without the SDK.
"""
from .position_recorder_config import PositionRecorderConfig
from .base_position_recorder import BasePositionRecorder
from .dummy_position_recorder import DummyPositionRecorder

_LAZY_EXPORTS = {"OpenvrPositionRecorder": ".openvr_position_recorder"}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PositionRecorderConfig",
    "BasePositionRecorder",
    "DummyPositionRecorder",
    "OpenvrPositionRecorder",
]
