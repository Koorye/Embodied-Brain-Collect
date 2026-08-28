"""Test dummy EMG."""

import numpy as np
from matplotlib.gridspec import GridSpec
from tests.base import BaseTest, SESSION_DIR
from embodied_brain_collect.recorders.emg import DummyEmgRecorder, EmgRecorderConfig


class TestDummyEmg(BaseTest):
    name = "Dummy EMG"

    def _build_layout(self, fig):
        gs = GridSpec(4, 4, figure=fig)
        self.ax_emg = [fig.add_subplot(gs[i//2, i%2]) for i in range(8)]
        self.ax_gyro = fig.add_subplot(gs[2:, 2:])
        self.ax_accel = fig.add_subplot(gs[0:2, 2:])
        for ax in self.ax_emg: ax.tick_params(labelsize=6)

    @staticmethod
    def _downsample(arr, max_pts=500):
        """Decimate *arr* to at most *max_pts* rows for lighter rendering."""
        n = len(arr)
        if n <= max_pts:
            return arr
        step = n // max_pts
        return arr[::step]

    def _update(self, rec, elapsed):
        emg = rec._arr_buf.get("emg_data", [])
        gyro = rec._arr_buf.get("imu_gyro", [])
        accel = rec._arr_buf.get("imu_accel", [])

        if emg:
            sl = self._window_slice(len(emg), elapsed, window=5.0)
            e = self._downsample(np.stack(emg[sl]))
            for ch in range(8):
                self.ax_emg[ch].clear()
                self.ax_emg[ch].plot(e[:, ch], linewidth=0.3)
                self.ax_emg[ch].set_ylabel(f"ch{ch}", fontsize=6)
        if gyro:
            sl = self._window_slice(len(gyro), elapsed, window=5.0)
            self.ax_gyro.clear(); self.ax_gyro.plot(np.stack(gyro[sl]), linewidth=0.5)
            self.ax_gyro.legend(["gx","gy","gz"], fontsize=6); self.ax_gyro.set_title("IMU gyro")
        if accel:
            sl = self._window_slice(len(accel), elapsed, window=5.0)
            self.ax_accel.clear(); self.ax_accel.plot(np.stack(accel[sl]), linewidth=0.5)
            self.ax_accel.legend(["ax","ay","az"], fontsize=6); self.ax_accel.set_title("IMU accel")


if __name__ == "__main__":
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=f"{SESSION_DIR}/emg"))
    TestDummyEmg(rec).run()
