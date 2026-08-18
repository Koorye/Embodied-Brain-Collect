"""Test SteamVR / OpenVR 6-DOF position tracker — real hardware."""

import numpy as np

from tests.position.test_dummy_position import TestDummyPosition
from tests.base import SESSION_DIR
from embodied_brain_collect.recorders.position import OpenvrPositionRecorder, PositionRecorderConfig


class TestOpenvrPosition(TestDummyPosition):
    name = "OpenVR Position"

    def _update(self, rec, elapsed):
        # OpenVR records D devices per frame: stacks are (T, D, 3) / (T, D, 4).
        pos = rec._arr_buf.get("positions_m", [])
        quat = rec._arr_buf.get("quaternions_wxyz", [])
        classes = rec._buf.get("device_classes", [])
        ts = rec._buf.get("perf_counter_s", [])
        sl = self._rolling(ts, ts[-1], window=5.0) if len(ts) > 1 else slice(0, 0)

        if len(pos) > 1:
            p = np.stack(pos[sl])  # (T, D, 3)
            labels = classes or [f"dev{d}" for d in range(p.shape[1])]
            valid = rec._arr_buf.get("valid", [])
            n_valid = int(np.sum(valid[-1])) if valid else "-"

            self.ax_xy.clear()
            for d in range(p.shape[1]):
                self.ax_xy.plot(p[:, d, 0], p[:, d, 1], linewidth=0.5, label=labels[d])
            self.ax_xy.legend(fontsize=6)
            self.ax_xy.set_title(f"XY ({sl.stop - sl.start}) valid={n_valid}/{len(labels)}")
            self.ax_xyz.clear()
            self.ax_xyz.plot(p[:, 0, :], linewidth=0.5)
            self.ax_xyz.legend(["x", "y", "z"], fontsize=6)
            self.ax_xyz.set_title("XYZ (dev0)")
        if quat:
            q = np.stack(quat[sl])  # (T, D, 4)
            self.ax_quat.clear()
            for d in range(q.shape[1]):
                self.ax_quat.plot(q[:, d, :], linewidth=0.5)
            self.ax_quat.set_title("quat")


def main():
    cfg = PositionRecorderConfig(session_dir=f"{SESSION_DIR}/position")
    rec = OpenvrPositionRecorder(cfg)
    TestOpenvrPosition(rec).run()


if __name__ == "__main__":
    main()
