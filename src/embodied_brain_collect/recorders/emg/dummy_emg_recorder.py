"""Dummy EMG — 8ch sine + noise + IMU."""

import time
import numpy as np
from .base_emg_recorder import BaseEmgRecorder
from .emg_recorder_config import EmgRecorderConfig


class DummyEmgRecorder(BaseEmgRecorder):
    config: EmgRecorderConfig

    def __init__(self, config: EmgRecorderConfig):
        super().__init__(config)
        # One counter for both frame types, as on the wire: the armband bumps
        # the sequence number once per transmitted frame, whichever type.
        self._frame_i = 0

    def _open(self) -> bool:
        self._t_emg = 0.0
        self._t_imu = 0.0
        self._log("[emg:dummy] synthetic 8-ch EMG + IMU")
        return True

    def _close(self) -> None:
        n = len(self._buf.get("emg_timestamps", []))
        self._log(f"[emg:dummy] stopped (emg={n})")

    def _poll(self, ts):
        phase = np.arange(8) * 0.5
        raw = (
            np.sin(2 * np.pi * 60 * self._t_emg + phase) * 500
            + np.random.randn(8).astype(np.float32) * 20
        )
        p = 2 * np.pi * 5 * self._t_imu

        self._acc("emg_timestamps", ts)
        self._acc("emg_sn", self._frame_i & 0xFF)
        self._acc_arr("emg_data", raw.astype(np.int32))
        self._frame_i += 1

        self._acc("imu_timestamps", ts)
        self._acc("imu_sn", self._frame_i & 0xFF)
        self._acc_arr("imu_gyro",
            np.array([np.sin(p), np.cos(p), 0.0], dtype=np.float32))
        self._acc_arr("imu_accel",
            np.array([0.0, 0.0, 9.8 + np.sin(p * 0.3) * 0.2], dtype=np.float32))
        self._frame_i += 1

        self._t_emg += 0.01
        self._t_imu += 0.01
        time.sleep(0.01)
