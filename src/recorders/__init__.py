"""Recorders — one sub-package per modality.

Each modality follows the convention::

    recorders/<name>/
        <name>_recorder_config.py   # Config dataclass
        base_<name>_recorder.py     # Abstract base
        dummy_<name>_recorder.py    # Dummy for testing
        <vendor>_<name>_recorder.py # Concrete hardware implementation
"""
from .base import BaseRecorder, BaseRecorderConfig
from .camera import DummyCameraRecorder, OpencvCameraRecorder, DepthaiCameraRecorder
from .emg import DummyEmgRecorder, WeiliEmgRecorder
from .eye import DummyEyeRecorder, NeonEyeRecorder
from .hand_pose import DummyHandPoseRecorder, ManusHandPoseRecorder
from .position import DummyPositionRecorder, OpenvrPositionRecorder