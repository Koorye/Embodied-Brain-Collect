"""Dummy eye tracker — synthetic gaze, IMU, and scene-camera data."""

import numpy as np
from .base_eye_recorder import BaseEyeRecorder
from .eye_recorder_config import EyeRecorderConfig

GAZE_HZ = 200
IMU_HZ = 110
SCENE_HZ = 30
SCENE_W, SCENE_H = 640, 480


def _color_bar(t):
    img = np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)
    n = 8
    bw = SCENE_W // n
    shift = int(t * 60) % SCENE_W
    for i in range(n):
        x0 = (i * bw + shift) % (SCENE_W + bw) - bw
        hue = (i / 8. + t * 0.1) % 1.0
        h6 = int(hue * 6)
        f = hue * 6 - h6
        c = 255
        x = int(c * (1 - abs(f - 1)))
        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x),
                    (0, x, c), (x, 0, c), (c, 0, x), (c, 0, 0)][min(h6, 7)]
        img[:, max(0, x0):min(SCENE_W, x0 + bw)] = [r, g, b]
    return img


class DummyEyeRecorder(BaseEyeRecorder):
    config: EyeRecorderConfig

    def __init__(self, config: EyeRecorderConfig):
        super().__init__(config)

    def _open(self) -> bool:
        self._t_gaze = 0.0
        self._t_imu = 0.0
        self._t_scene = 0.0
        self._t0: float | None = None   # first absolute ts -> relative baseline
        self._log("[eye:dummy] synthetic gaze@200Hz IMU@110Hz scene@30fps")
        return True

    def _close(self) -> None:
        self._log(f"[eye:dummy] stopped")

    def _poll(self, ts):
        # ts is absolute unix time; generation pace is relative to the
        # first sample (t = ts - t0), otherwise the first poll would try
        # to generate ~ts seconds of data at once.
        if self._t0 is None:
            self._t0 = ts
            return
        t = ts - self._t0
        while t >= self._t_gaze:
            self._acc("gaze_timestamps", ts)
            x = 0.5 + 0.1 * np.sin(self._t_gaze * 3.0)
            y = 0.5 + 0.1 * np.cos(self._t_gaze * 2.5)
            self._acc_arr("gaze_xy", np.array([x, y], dtype=np.float32))
            self._t_gaze += 1.0 / GAZE_HZ
        while t >= self._t_imu:
            self._acc("imu_timestamps", ts)
            self._acc_arr("imu_gyro",
                np.array([np.sin(self._t_imu*5.0)*0.02, np.cos(self._t_imu*4.0)*0.01, 0.0], dtype=np.float32))
            self._acc_arr("imu_accel",
                np.array([0.01, -0.01, 9.8+np.sin(self._t_imu)*0.1], dtype=np.float32))
            self._t_imu += 1.0 / IMU_HZ
        while t >= self._t_scene:
            self._acc("scene_timestamps", ts)
            self._acc_arr("scene_frames", _color_bar(self._t_scene))
            self._t_scene += 1.0 / SCENE_HZ

    def _heartbeat_stats(self, elapsed: float) -> str:
        return super()._heartbeat_stats(elapsed)
