"""Marker recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class MarkerRecorderConfig(BaseRecorderConfig):
    host: str = "127.0.0.1"
    port: int = 9999
