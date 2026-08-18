"""Dummy tactile glove — 135-channel synthetic data @ 200 Hz."""

import time
import numpy as np
from .base_tactile_recorder import BaseTactileRecorder
from .tactile_recorder_config import TactileRecorderConfig


class DummyTactileRecorder(BaseTactileRecorder):
    config: TactileRecorderConfig

    def _open(self) -> bool:
        self._log("[tactile:dummy] simulated 135-ch glove @ 200Hz")
        return True

    def _close(self) -> None:
        self._log(f"[tactile:dummy] stopped")

    def _poll(self, ts):
        time.sleep(0.005)
        data = np.zeros(135, dtype=np.float32)
        data[:4] = [1, 0, 0, 0]
        data[4:7] = np.sin(ts * np.array([3, 4, 5])).astype(np.float32) * 0.01
        data[7:10] = [0, 0, 9.8]
        data[10:15] = (
            np.sin(ts * 2 + np.arange(5) * 0.3).astype(np.float32) * 0.5 + 0.5)
        data[15:75] = np.random.rand(60).astype(np.float32) * 0.05
        data[75:135] = np.random.rand(60).astype(np.float32) * 0.05
        self._acc("glove_timestamps", ts)
        self._acc_arr("glove_data", data)
