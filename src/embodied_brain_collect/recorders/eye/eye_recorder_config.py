"""Eye tracker recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig

@dataclass
class EyeRecorderConfig(BaseRecorderConfig):
    # The device is discovered over mDNS (pupil_labs Network) — no IP/port
    # configuration needed.
    no_scene_video: bool = False
    crf: int = 23            # libx265 quality for eye.mp4
    preset: str = "medium"   # libx265 speed preset
    link_timeout: float = 0.1  # 链路静默超时:某路流连续这么久没有新样本 →
                               # 连接已死(断电/断网不发 FIN 的半开 TCP 上,
                               # 流任务只是挂住,永远不会有显式错误)。gaze/imu
                               # 持续 100Hz+ 出流,5s 静默 = 必死。0 = 关闭
