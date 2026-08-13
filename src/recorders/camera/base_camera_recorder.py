"""Abstract camera base."""
from ..base import BaseRecorder


class BaseCameraRecorder(BaseRecorder):
    name = "camera"
    output_dir = "camera"
    role: str = ""
