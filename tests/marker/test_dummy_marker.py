"""Test dummy marker."""

from tests.base import BaseTest, SESSION_DIR
from src.recorders.marker import DummyMarkerRecorder, MarkerRecorderConfig


class TestDummyMarker(BaseTest):
    name = "Dummy Marker"

    def _build_layout(self, fig):
        self.ax = fig.add_subplot(111)

    def _update(self, rec, elapsed):
        codes = rec._buf.get("marker_code", [])
        tags = rec._buf.get("marker_tag", [])
        if codes:
            self.ax.clear()
            self.ax.vlines(range(len(codes)), 0, codes, linewidth=2)
            for i in range(len(codes)):
                if i % 3 == 0:
                    self.ax.annotate(tags[i], (i, codes[i]),
                                    fontsize=6, rotation=45, ha='right')
        self.ax.set_title(f"markers ({len(codes)}) [Q to stop]")


if __name__ == "__main__":
    rec = DummyMarkerRecorder(MarkerRecorderConfig(session_dir=f"{SESSION_DIR}/marker"))
    TestDummyMarker(rec).run()
