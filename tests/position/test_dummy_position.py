"""Test dummy position."""

import numpy as np
from matplotlib.gridspec import GridSpec
from tests.base import BaseTest, SESSION_DIR
from src.recorders.position import DummyPositionRecorder, PositionRecorderConfig


class TestDummyPosition(BaseTest):
    name = "Dummy Position"

    def _build_layout(self, fig):
        gs = GridSpec(1, 3, figure=fig)
        self.ax_xy = fig.add_subplot(gs[0, 0])
        self.ax_xyz = fig.add_subplot(gs[0, 1])
        self.ax_quat = fig.add_subplot(gs[0, 2])

    def _update(self, rec, elapsed):
        pos = rec._arr_buf.get("positions_m", [])
        quat = rec._arr_buf.get("quaternions_wxyz", [])
        sl = self._rolling(list(range(len(pos))), len(pos)) if pos else slice(0, 0)

        if len(pos) > 1:
            p = np.stack(pos[sl])
            self.ax_xy.clear(); self.ax_xy.plot(p[:, 0], p[:, 1], linewidth=0.5)
            self.ax_xy.set_title(f"XY ({sl.stop - sl.start})")
            self.ax_xyz.clear(); self.ax_xyz.plot(p, linewidth=0.5)
            self.ax_xyz.legend(["x","y","z"], fontsize=6); self.ax_xyz.set_title("XYZ")
        if quat:
            q = np.stack(quat[sl])
            self.ax_quat.clear(); self.ax_quat.plot(q, linewidth=0.5)
            self.ax_quat.legend(["w","x","y","z"], fontsize=6); self.ax_quat.set_title("quat")


if __name__ == "__main__":
    rec = DummyPositionRecorder(PositionRecorderConfig(session_dir=f"{SESSION_DIR}/position"))
    TestDummyPosition(rec).run()
