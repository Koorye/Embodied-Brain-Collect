"""Abstract eye tracker base."""
from ..base import BaseRecorder


class BaseEyeRecorder(BaseRecorder):
    name = "eye"
    output_dir = "eye"
