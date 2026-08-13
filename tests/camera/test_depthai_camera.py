"""Test OAK-D depthai camera — real hardware."""

from tests.camera.test_dummy_camera import TestDummyCamera
from tests.base import SESSION_DIR
from src.recorders.camera import DepthaiCameraRecorder, DepthaiCameraConfig


class TestDepthaiCamera(TestDummyCamera):
    name = "DepthAI Camera"


def main():
    cfg = DepthaiCameraConfig(session_dir=f"{SESSION_DIR}/camera")
    rec = DepthaiCameraRecorder(cfg)
    TestDepthaiCamera(rec).run()


if __name__ == "__main__":
    main()
