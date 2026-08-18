"""Test dummy tactile."""

import numpy as np
from matplotlib.gridspec import GridSpec
from tests.base import BaseTest, SESSION_DIR
from embodied_brain_collect.recorders.tactile import DummyTactileRecorder, TactileRecorderConfig

LABELS = ["quat(4)", "gyro(3)", "accel(3)", "bend(5)", "finger(60)", "palm(60)"]
SLICES = [(0,4), (4,7), (7,10), (10,15), (15,75), (75,135)]


class TestDummyTactile(BaseTest):
    name = "Dummy Tactile"

    def _build_layout(self, fig):
        gs = GridSpec(2, 3, figure=fig)
        self.axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
        for ax in self.axes: ax.tick_params(labelsize=6)

    def _update(self, rec, elapsed):
        data = rec._arr_buf.get("glove_data", [])
        if len(data) > 1:
            d = np.stack(data[-300:])
            for i, (s, e) in enumerate(SLICES):
                self.axes[i].clear()
                self.axes[i].plot(d[:, s:e], linewidth=0.3)
                self.axes[i].set_title(LABELS[i], fontsize=8)


if __name__ == "__main__":
    rec = DummyTactileRecorder(TactileRecorderConfig(session_dir=f"{SESSION_DIR}/tactile"))
    TestDummyTactile(rec).run()
