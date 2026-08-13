"""Camera recorders.

Usage::

    from recorders.camera import (
        CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
        BaseCameraRecorder, DummyCameraRecorder,
        OpencvCameraRecorder, DepthaiCameraRecorder,
    )
"""
from .camera_recorder_config import (
    CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
)
from .base_camera_recorder import BaseCameraRecorder
from .dummy_camera_recorder import DummyCameraRecorder
from .opencv_camera_recorder import OpencvCameraRecorder
from .depthai_camera_recorder import DepthaiCameraRecorder

__all__ = [
    "CameraRecorderConfig", "OpencvCameraConfig", "DepthaiCameraConfig",
    "BaseCameraRecorder", "DummyCameraRecorder",
    "OpencvCameraRecorder", "DepthaiCameraRecorder",
]
