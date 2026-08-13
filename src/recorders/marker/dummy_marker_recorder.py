"""Dummy marker recorder — generates periodic fake markers for testing."""

import time
from .base_marker_recorder import BaseMarkerRecorder
from .marker_recorder_config import MarkerRecorderConfig


class DummyMarkerRecorder(BaseMarkerRecorder):
    """Generates a fake marker every second, useful for testing recorders."""

    config: MarkerRecorderConfig

    def __init__(self, config: MarkerRecorderConfig):
        super().__init__(config)
        self._count = 0

    def _open(self) -> bool:
        self._log("[marker:dummy] generating fake markers @ 1 Hz")
        return True

    def _close(self) -> None:
        self._log(f"[marker:dummy] stopped ({self._count} markers)")

    def _poll(self, ts: float) -> None:
        time.sleep(1.0)
        self._count += 1
        self.mark(0x01, f"DUMMY_{self._count}", ts)
