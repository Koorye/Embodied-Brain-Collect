"""Recorders — one sub-package per modality.

Each modality follows the convention::

    recorders/<name>/
        <name>_recorder_config.py   # Config dataclass
        base_<name>_recorder.py     # Abstract base
        dummy_<name>_recorder.py    # Dummy for testing
        <vendor>_<name>_recorder.py # Concrete hardware implementation

Exports below are LAZY: vendor modules like neon_eye / openvr_position import
their SDK at module top, and eagerly pulling every SDK in just to use one
recorder breaks partial deployments (and the test suite on dev machines).
Names resolve on first access; a missing SDK then raises at the slot that
actually needs it, not at package import.
"""
from .base import BaseRecorder, BaseRecorderConfig

_LAZY_EXPORTS = {
    "DummyCameraRecorder": ".camera",
    "OpencvCameraRecorder": ".camera",
    "DepthaiCameraRecorder": ".camera",
    "RealsenseCameraRecorder": ".camera",
    "DummyEegRecorder": ".eeg",
    "CurryEegRecorder": ".eeg",
    "DummyEmgRecorder": ".emg",
    "WeiliEmgRecorder": ".emg",
    "DummyEyeRecorder": ".eye",
    "NeonEyeRecorder": ".eye",
    "NeonEyeAsyncRecorder": ".eye",
    "DummyHandPoseRecorder": ".hand_pose",
    "ManusHandPoseRecorder": ".hand_pose",
    "DummyPositionRecorder": ".position",
    "OpenvrPositionRecorder": ".position",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
