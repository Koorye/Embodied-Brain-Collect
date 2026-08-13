"""Position tracker config."""
from dataclasses import dataclass, field
from ..base import BaseRecorderConfig


@dataclass
class PositionRecorderConfig(BaseRecorderConfig):
    device_classes: str = "tracker"
    device_serials: list[str] = field(default_factory=list)
