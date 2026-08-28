"""Abstract position tracker base."""
from ..base import BaseRecorder


class BasePositionRecorder(BaseRecorder):
    name = "position"
    output_dir = "position"
