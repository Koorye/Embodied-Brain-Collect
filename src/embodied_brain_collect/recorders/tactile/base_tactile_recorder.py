"""Abstract tactile base."""
from ..base import BaseRecorder


class BaseTactileRecorder(BaseRecorder):
    name = "tactile"
    output_dir = "tactile_glove"
