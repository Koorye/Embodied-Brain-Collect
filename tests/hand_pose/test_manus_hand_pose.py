"""Test Manus hand pose — real hardware via SDK."""

import numpy as np
from matplotlib.gridspec import GridSpec
from tests.base import BaseTest, SESSION_DIR
from src.recorders.hand_pose import ManusHandPoseRecorder, HandPoseRecorderConfig


class TestManusHandPose(BaseTest):
    name = "Manus Hand Pose"

    def _build_layout(self, fig):
        gs = GridSpec(6, 4, figure=fig)
        self.ax_ergo = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(20)]
        self.ax_skel = fig.add_subplot(gs[5, :])
        for ax in self.ax_ergo:
            ax.tick_params(labelsize=5)

    @staticmethod
    def _downsample(arr, max_pts=500):
        n = len(arr)
        if n <= max_pts:
            return arr
        return arr[::n // max_pts]

    def _update(self, rec, elapsed):
        ergo = rec._arr_buf.get("ergo_data", [])
        skel = rec._arr_buf.get("skeleton_positions", [])

        if ergo:
            n = len(ergo)
            ts = [i / max(n - 1, 1) * elapsed for i in range(n)] if n > 1 else [0.0]
            sl = self._rolling(ts, elapsed, window=5.0)
            e = self._downsample(np.stack(ergo[sl]))
            # Show first 20 channels (left hand)
            for ch in range(min(20, e.shape[1])):
                self.ax_ergo[ch].clear()
                self.ax_ergo[ch].plot(e[:, ch], linewidth=0.3)
                self.ax_ergo[ch].set_ylabel(f"ch{ch}", fontsize=5)

        if skel:
            n = len(skel)
            ts = [i / max(n - 1, 1) * elapsed for i in range(n)] if n > 1 else [0.0]
            sl = self._rolling(ts, elapsed, window=5.0)
            s = self._downsample(np.stack(skel[sl]))
            self.ax_skel.clear()
            self.ax_skel.plot(s[:, :, 0], linewidth=0.3)  # X positions
            self.ax_skel.set_title("skeleton node positions (X)", fontsize=7)


def main():
    cfg = HandPoseRecorderConfig(session_dir=f"{SESSION_DIR}/hand_pose")
    rec = ManusHandPoseRecorder(cfg)
    TestManusHandPose(rec).run()


if __name__ == "__main__":
    main()
