"""Dummy EGO headband — 4 synthetic cameras + 2 synthetic IMUs.

Emits the same output schema as the real network recorder so downstream
tools and tests can run without hardware.  Cameras are small moving
color-bar patterns at ``cam_fps``; IMUs are smooth sinusoids at
``imu_rate_hz``.
"""

import time
import numpy as np
from .base_ego_headband_recorder import BaseEgoHeadbandRecorder
from .ego_headband_recorder_config import EgoHeadbandRecorderConfig


def _bar(t: float, w: int, h: int, hue0: float) -> np.ndarray:
    """Small moving color-bar test pattern, shape (H, W, 3) uint8."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    n = 8
    bw = max(w // n, 1)
    shift = int(t * 30) % w
    for i in range(n):
        x0 = (i * bw + shift) % (w + bw) - bw
        hue = (hue0 + i / n + t * 0.05) % 1.0
        r = int(127 + 127 * np.sin(2 * np.pi * hue))
        g = int(127 + 127 * np.sin(2 * np.pi * (hue + 1 / 3)))
        b = int(127 + 127 * np.sin(2 * np.pi * (hue + 2 / 3)))
        img[:, max(0, x0):min(w, x0 + bw)] = (r, g, b)
    return img


class DummyEgoHeadbandRecorder(BaseEgoHeadbandRecorder):
    config: EgoHeadbandRecorderConfig

    def __init__(self, config: EgoHeadbandRecorderConfig):
        super().__init__(config)
        self._cam_t = 0.0
        self._imu_t = 0.0
        self._poll_i = 0

    def _open(self) -> bool:
        cfg = self.config
        self._cam_t = 0.0
        self._imu_t = 0.0
        self._poll_i = 0
        self._log(f"[ego_headband:dummy] synthetic {cfg.n_cameras} cams "
                  f"{cfg.cam_width}x{cfg.cam_height}@{cfg.cam_fps:.0f}fps + "
                  f"{cfg.n_imus} IMU @{cfg.imu_rate_hz:.0f}Hz")
        return True

    def _close(self) -> None:
        n0 = len(self._ts_buf.get("cam0", []))
        m0 = len(self._buf.get("imu0_ts", []))
        self._log(f"[ego_headband:dummy] stopped (cam0={n0} imu0={m0})")

    def _poll(self, ts):
        cfg = self.config
        imu_dt = 1.0 / max(cfg.imu_rate_hz, 1.0)
        # cameras run slower than IMUs — emit a frame every `cam_every` polls
        cam_every = max(1, int(round(cfg.imu_rate_hz / max(cfg.cam_fps, 1.0))))
        self._poll_i += 1

        # ---- IMUs (every poll, ~imu_rate_hz) ----
        p = 2 * np.pi * 2.0 * self._imu_t
        for j in range(cfg.n_imus):
            off = j * 0.7
            self._acc(f"imu{j}_ts", ts)
            self._acc_arr(f"imu{j}_gyro",
                np.array([np.sin(p + off), np.cos(p + off), 0.1 * np.sin(p)],
                         dtype=np.float32))
            self._acc_arr(f"imu{j}_accel",
                np.array([0.0, 0.0, 9.8 + 0.3 * np.sin(p * 0.5 + off)],
                         dtype=np.float32))
        self._imu_t += imu_dt

        # ---- cameras (every `cam_every` polls, ~cam_fps) ----
        if self._poll_i % cam_every == 0:
            hue_step = 1.0 / max(cfg.n_cameras, 1)
            for i in range(cfg.n_cameras):
                self._acc_ts(f"cam{i}", ts)
                self._acc_arr(f"cam{i}_frames",
                    _bar(self._cam_t, cfg.cam_width, cfg.cam_height,
                         hue0=i * hue_step))
            self._cam_t += 1.0 / max(cfg.cam_fps, 1.0)

        time.sleep(imu_dt)
