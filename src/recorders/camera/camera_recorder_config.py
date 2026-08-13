"""Camera recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class CameraRecorderConfig(BaseRecorderConfig):
    """Base camera config.  Subclass for specific hardware."""
    pass


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
    cam_w: int = 640
    cam_h: int = 360
    cam_fps_hint: float = 30.0
