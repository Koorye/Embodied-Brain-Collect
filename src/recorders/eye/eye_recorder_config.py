"""Eye tracker recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig

@dataclass
class EyeRecorderConfig(BaseRecorderConfig):
    neon_ip: str = "172.16.20.10"
    port: int = 8080
    no_scene_video: bool = False
