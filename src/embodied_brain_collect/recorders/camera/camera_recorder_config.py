"""Camera recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class CameraRecorderConfig(BaseRecorderConfig):
    """Base camera config.  Subclass for specific hardware."""
    crf: int = 23        # libx265 quality (lower = better / bigger)
    preset: str = "medium"  # libx265 speed preset


@dataclass
class OpencvCameraConfig(CameraRecorderConfig):
    """OpenCV USB camera."""
    idx: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    warmup: float = 1.0


@dataclass
class DepthaiCameraConfig(CameraRecorderConfig):
    """OAK-D / Luxonis depthai camera."""
    cam_w: int = 640       # output size (on-device downscale from sensor)
    cam_h: int = 460
    cam_fps_hint: float = 30.0


@dataclass
class RealsenseCameraConfig(CameraRecorderConfig):
    """Intel RealSense depth camera (D400 series)."""
    serial: str = ""      # device serial; "" = auto-pick first available
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = False  # also record depth frames aligned to color
