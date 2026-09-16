"""Abstract EGO headband base."""
from ..base import BaseRecorder


class BaseEgoHeadbandRecorder(BaseRecorder):
    name = "ego_headband"
    output_dir = "ego_headband"
