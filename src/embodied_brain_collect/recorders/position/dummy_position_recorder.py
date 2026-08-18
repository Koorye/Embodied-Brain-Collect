"""Dummy position tracker — synthetic 6-DOF data."""

import time
import numpy as np
from .base_position_recorder import BasePositionRecorder
from .position_recorder_config import PositionRecorderConfig


class DummyPositionRecorder(BasePositionRecorder):
    config: PositionRecorderConfig

    def _open(self) -> bool:
        self._log("[position:dummy] simulated tracker")
        return True

    def _close(self) -> None:
        self._log(f"[position:dummy] stopped")

    def _poll(self, ts):
        time.sleep(0.016)
        r = 0.5
        self._acc_arr("positions_m",
            np.array([r * np.sin(ts * 0.5), r * np.cos(ts * 0.5), 0.0], dtype=np.float32))
        self._acc_arr("quaternions_wxyz",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
