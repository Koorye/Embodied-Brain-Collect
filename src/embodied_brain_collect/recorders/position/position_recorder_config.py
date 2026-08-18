"""Position tracker config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class PositionRecorderConfig(BaseRecorderConfig):
    device_classes: str = "tracker"  # comma-separated device classes
