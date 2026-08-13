"""Dummy camera — color-bar test pattern + mp4."""

import time
import numpy as np
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import CameraRecorderConfig

W, H = 640, 480


def _bar(t):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    n = 8
    bw = W // n
    shift = int(t * 60) % W
    for i in range(n):
        x0 = (i * bw + shift) % (W + bw) - bw
        hue = (i / 8. + t * 0.1) % 1.0
        h6 = int(hue * 6)
        f = hue * 6 - h6
        c = 255
        x = int(c * (1 - abs(f - 1)))
        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x),
                    (0, x, c), (x, 0, c), (c, 0, x), (c, 0, 0)][min(h6, 7)]
        img[:, max(0, x0):min(W, x0 + bw)] = [b, g, r]
    return img


class DummyCameraRecorder(BaseCameraRecorder):
    config: CameraRecorderConfig

    def _open(self) -> bool:
        self._log(f"[camera:dummy] {W}x{H} @ 30fps (role={self.role or '-'})")
        return True

    def _close(self) -> None:
        self._log(f"[camera:dummy] stopped")

    def _poll(self, ts):
        time.sleep(0.033)
        self._acc_ts("cam", ts)
        self._acc_arr("frames", _bar(ts))
