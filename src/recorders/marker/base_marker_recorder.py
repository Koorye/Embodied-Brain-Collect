"""Abstract base marker recorder."""
from ..base import BaseRecorder


class BaseMarkerRecorder(BaseRecorder):
    name = "marker"
    output_dir = "markers"

    def mark(self, code: int, tag: str, ts: float) -> float:
        self._acc("marker_tag", tag)
        self._acc("marker_code", code)
        self._acc("marker_t_local_recv", ts)
