"""Test neon eye."""

from tests.base import SESSION_DIR
from .test_dummy_eye import TestDummyEye
from embodied_brain_collect.recorders.eye import NeonEyeRecorder, EyeRecorderConfig


class TestNeonEye(TestDummyEye):
    name = "Neon Eye"


def main():
    cfg = EyeRecorderConfig(session_dir=f"{SESSION_DIR}/eye")
    rec = NeonEyeRecorder(cfg)
    TestNeonEye(rec).run()


if __name__ == "__main__":
    main()
