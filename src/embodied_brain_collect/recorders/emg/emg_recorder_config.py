"""EMG recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class EmgRecorderConfig(BaseRecorderConfig):
    port: str = ""
    baud: int = 921600
