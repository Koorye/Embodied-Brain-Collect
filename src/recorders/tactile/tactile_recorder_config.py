"""Tactile glove config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig

@dataclass
class TactileRecorderConfig(BaseRecorderConfig):
    port: str = "COM3"; baud: int = 921600; hand: str = "rh"
    glove_startup_timeout: float = 5.0
    bundle_path: str = ""; license_path: str = ""
