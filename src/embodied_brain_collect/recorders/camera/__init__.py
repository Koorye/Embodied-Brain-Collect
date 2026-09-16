"""Camera recorders.

Usage::

    from recorders.camera import (
        CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
        RealsenseCameraConfig,
        BaseCameraRecorder, DummyCameraRecorder,
        OpencvCameraRecorder, DepthaiCameraRecorder, RealsenseCameraRecorder,
    )
"""
from .camera_recorder_config import (
    CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
    RealsenseCameraConfig,
)
from .base_camera_recorder import BaseCameraRecorder
from .dummy_camera_recorder import DummyCameraRecorder
from .opencv_camera_recorder import OpencvCameraRecorder
from .depthai_camera_recorder import DepthaiCameraRecorder
from .realsense_camera_recorder import RealsenseCameraRecorder

__all__ = [
    "CameraRecorderConfig", "OpencvCameraConfig", "DepthaiCameraConfig",
    "RealsenseCameraConfig",
    "BaseCameraRecorder", "DummyCameraRecorder",
    "OpencvCameraRecorder", "DepthaiCameraRecorder", "RealsenseCameraRecorder",
]
