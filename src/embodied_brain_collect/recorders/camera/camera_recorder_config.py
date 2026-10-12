"""Camera recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class CameraRecorderConfig(BaseRecorderConfig):
    """Base camera config.  Subclass for specific hardware."""
    crf: int = 23        # quality (x265 crf / nvenc cq; lower = better / bigger)
    preset: str = "medium"  # x265 speed preset (maps to nvenc p1-p7)
    encoder: str = "auto"   # "auto"=hevc_nvenc if available, else libx265;
                            # "gpu"/"nvenc"=force GPU (warn+fallback if absent);
                            # "cpu"=libx265. 16-bit depth stays libx265 either
                            # way (nvenc has no 12-bit gray mode).


@dataclass
class OpencvCameraConfig(CameraRecorderConfig):
    """OpenCV USB camera."""
    idx: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    # 辅助员控制台的实时预览:把降采样 JPEG 经 UDP 推到本机端口
    # (session/assist_console 消费)。0 = 关闭。
    preview_port: int = 0
    preview_hz: float = 30.0   # 预览帧率上限(采集帧率之上无意义)
    preview_width: int = 640   # 预览缩放宽度(等比缩高)


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
