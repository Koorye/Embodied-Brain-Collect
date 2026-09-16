"""Eye tracker recorders.

Neon recorders import ``pupil_labs`` at module top; keep those exports lazy
so the base/dummy classes stay importable on machines without the SDK.
"""
from .eye_recorder_config import EyeRecorderConfig
from .base_eye_recorder import BaseEyeRecorder
from .dummy_eye_recorder import DummyEyeRecorder

_LAZY_EXPORTS = {
    "NeonEyeRecorder": ".neon_eye_recorder",
    "NeonEyeAsyncRecorder": ".neon_eye_async_recorder",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EyeRecorderConfig",
    "BaseEyeRecorder",
    "DummyEyeRecorder",
    "NeonEyeRecorder",
    "NeonEyeAsyncRecorder",
]
