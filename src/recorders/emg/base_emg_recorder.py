"""Abstract EMG base."""
from ..base import BaseRecorder


class BaseEmgRecorder(BaseRecorder):
    name = "emg"
    output_dir = "emg"
