"""Test Weili EMG — real hardware via serial port."""

from tests.emg.test_dummy_emg import TestDummyEmg
from tests.base import SESSION_DIR
from src.recorders.emg import WeiliEmgRecorder, EmgRecorderConfig


class TestWeiliEmg(TestDummyEmg):
    name = "Weili EMG"


def main():
    from src.recorders.emg.weili_emg_recorder import _auto_detect_port
    print("Auto-detected Weili EMG port:", _auto_detect_port())
    cfg = EmgRecorderConfig(session_dir=f"{SESSION_DIR}/emg")
    rec = WeiliEmgRecorder(cfg)
    # rec.run()
    TestWeiliEmg(rec).run()


if __name__ == "__main__":
    main()
