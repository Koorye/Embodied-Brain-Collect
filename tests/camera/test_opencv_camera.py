"""Test OpenCV USB camera — real hardware."""

from tests.camera.test_dummy_camera import TestDummyCamera
from tests.base import SESSION_DIR
from embodied_brain_collect.recorders.camera import OpencvCameraRecorder, OpencvCameraConfig


class TestOpencvCamera(TestDummyCamera):
    name = "OpenCV Camera"


def main():
    cfg = OpencvCameraConfig(session_dir=f"{SESSION_DIR}/camera", idx=0)
    rec = OpencvCameraRecorder(cfg)
    TestOpencvCamera(rec).run()


if __name__ == "__main__":
    main()
