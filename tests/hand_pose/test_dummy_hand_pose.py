"""Test dummy hand pose."""

import numpy as np
from tests.base import BaseTest, SESSION_DIR
from src.recorders.hand_pose import DummyHandPoseRecorder, HandPoseRecorderConfig


class TestDummyHandPose(BaseTest):
    name = "Dummy Hand Pose"

    def _build_layout(self, fig):
        self.ax = fig.add_subplot(111)

    def _update(self, rec, elapsed):
        data = rec._arr_buf.get("ergo_data", [])
        if len(data) > 1:
            n = len(data)
            ts = [i / max(n - 1, 1) * elapsed for i in range(n)] if n > 1 else [0.0]
            sl = self._rolling(ts, elapsed, window=5.0)
            d = np.stack(data[sl])
            self.ax.clear()
            self.ax.plot(d[:, :10], linewidth=0.3)
            self.ax.legend([f"ch{i}" for i in range(10)], fontsize=5, ncol=5)
            self.ax.set_title(f"hand pose ({sl.stop - sl.start})")


if __name__ == "__main__":
    rec = DummyHandPoseRecorder(HandPoseRecorderConfig(session_dir=f"{SESSION_DIR}/hand_pose"))
    TestDummyHandPose(rec).run()
