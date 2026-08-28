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
