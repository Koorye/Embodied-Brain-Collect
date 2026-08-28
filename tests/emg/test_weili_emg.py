"""Test Weili EMG — real hardware via serial port."""

from serial.tools import list_ports
from tests.emg.test_dummy_emg import TestDummyEmg
from tests.base import SESSION_DIR
from embodied_brain_collect.recorders.emg import WeiliEmgRecorder, EmgRecorderConfig


def _auto_detect_port() -> str | None:
    for p in list_ports.comports():
        h = (p.hwid or "").upper()
        d = p.description or ""
        tags = ("VID:PID=10C4", "CP210", "Silicon Labs",
                "VID:PID=1A86:55D3", "CH343")
        if any(t in h or t in d for t in tags):
            print(p)


class TestWeiliEmg(TestDummyEmg):
    name = "Weili EMG"


def main():
    _auto_detect_port()
    cfg = EmgRecorderConfig(session_dir=f"{SESSION_DIR}/emg", port="COM9")
    rec = WeiliEmgRecorder(cfg)
    # rec.run()
    TestWeiliEmg(rec).run()


if __name__ == "__main__":
    main()
