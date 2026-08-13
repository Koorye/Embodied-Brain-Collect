"""Test UDP marker recorder — real stim PC sends ``EVT|...`` packets."""

from tests.marker.test_dummy_marker import TestDummyMarker
from tests.base import SESSION_DIR
from src.recorders.marker import UdpMarkerRecorder, MarkerRecorderConfig


class TestUdpMarker(TestDummyMarker):
    name = "UDP Marker"


def main():
    cfg = MarkerRecorderConfig(session_dir=f"{SESSION_DIR}/marker")
    rec = UdpMarkerRecorder(cfg)
    print(f"Waiting for EVT markers on udp://{cfg.host}:{cfg.port} "
          "(send from stim PC) ...")
    TestUdpMarker(rec).run()


if __name__ == "__main__":
    main()
