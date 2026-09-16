"""Abstract base marker recorder."""
from ..base import BaseRecorder


class BaseMarkerRecorder(BaseRecorder):
    name = "marker"
    output_dir = "markers"

    def mark(self, code: int, tag: str, ts: float) -> float:
        """Record one marker; ``ts`` is the authoritative PC time.

        UDP recorders override with the sender-side stamp from the packet;
        local (dummy) recorders pass their own clock, where sending and
        receiving are the same instant anyway.
        """
        self._acc("marker_tag", tag)
        self._acc("marker_code", code)
        self._acc("marker_t_sent_pc", ts)
        self._acc("marker_t_local_recv", ts)
