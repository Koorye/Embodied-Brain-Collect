"""Test dummy camera."""

import numpy as np
import matplotlib.pyplot as plt
from tests.base import BaseTest, SESSION_DIR
from src.recorders.camera import DummyCameraRecorder, CameraRecorderConfig


class TestDummyCamera(BaseTest):
    name = "Dummy Camera"

    def _build_layout(self, fig):
        self.ax = fig.add_subplot(111)
        self.ax.axis('off')

    def _update(self, rec, elapsed):
        frames = rec._arr_buf.get("frames", [])
        if frames:
            self.ax.clear()
            self.ax.imshow(frames[-1])
            self.ax.set_title(f"camera ({len(frames)} frames) [Q to stop]")
            self.ax.axis('off')


if __name__ == "__main__":
    rec = DummyCameraRecorder(CameraRecorderConfig(session_dir=f"{SESSION_DIR}/camera"))
    TestDummyCamera(rec).run()
