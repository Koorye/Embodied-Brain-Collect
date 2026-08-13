"""Test dummy eye."""

import numpy as np
from matplotlib.gridspec import GridSpec
from tests.base import BaseTest, SESSION_DIR
from src.recorders.eye import DummyEyeRecorder, EyeRecorderConfig


class TestDummyEye(BaseTest):
    name = "Dummy Eye"

    def _build_layout(self, fig):
        gs = GridSpec(2, 3, figure=fig)
        self.ax_gaze = fig.add_subplot(gs[0, 0])
        self.ax_gyro = fig.add_subplot(gs[0, 1])
        self.ax_accel = fig.add_subplot(gs[0, 2])
        self.ax_scene = fig.add_subplot(gs[1, :])

    def _update(self, rec, elapsed):
        gaze = rec._arr_buf.get("gaze_xy", [])
        gyro = rec._arr_buf.get("imu_gyro", [])
        accel = rec._arr_buf.get("imu_accel", [])
        scene = rec._arr_buf.get("scene_frames", [])

        if len(gaze) > 1:
            sl = self._rolling(list(range(len(gaze))), len(gaze))
            g = np.stack(gaze[sl])
            self.ax_gaze.clear(); self.ax_gaze.plot(g[:, 0], g[:, 1], linewidth=0.2)
            self.ax_gaze.set_title(f"gaze XY ({sl.stop - sl.start})")
        if gyro:
            sl = self._rolling(list(range(len(gyro))), len(gyro))
            self.ax_gyro.clear(); self.ax_gyro.plot(np.stack(gyro[sl]), linewidth=0.5)
            self.ax_gyro.legend(["gx","gy","gz"], fontsize=6); self.ax_gyro.set_title("IMU gyro")
        if accel:
            sl = self._rolling(list(range(len(accel))), len(accel))
            self.ax_accel.clear(); self.ax_accel.plot(np.stack(accel[sl]), linewidth=0.5)
            self.ax_accel.legend(["ax","ay","az"], fontsize=6); self.ax_accel.set_title("IMU accel")
        if scene:
            self.ax_scene.clear(); self.ax_scene.imshow(scene[-1])
            self.ax_scene.set_title(f"scene ({len(scene)})"); self.ax_scene.axis('off')


if __name__ == "__main__":
    rec = DummyEyeRecorder(EyeRecorderConfig(session_dir=f"{SESSION_DIR}/eye"))
    TestDummyEye(rec).run()
