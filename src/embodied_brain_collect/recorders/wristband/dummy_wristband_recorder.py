"""Dummy wristband — synthetic pressure/PPG/IMU at the real device's rates.

Feeds the same NPZ field names as ``WristbandRecorder`` so the checker and
QC page need no special case for dummy sessions.  Timestamps are the host
clock, not a synced device clock.
"""

import numpy as np

from ..base import BaseRecorder
from .protocol import ACCEL_SCALE_M_S2, GYRO_SCALE_RAD_S
from .wristband_recorder_config import WristbandRecorderConfig

_FRAME_HZ = 50.0        # 0x15 data frames
_PRESSURE_PER_FRAME = 3  # 150 Hz
_PPG_PER_FRAME = 2       # 100 Hz


class DummyWristbandRecorder(BaseRecorder):
    name = "wristband"
    output_dir = "wristband"
    config: WristbandRecorderConfig

    def __init__(self, config: WristbandRecorderConfig):
        super().__init__(config)
        self._t_start: float | None = None
        self._frame_i = 0
        self._data_frames_saved = 0

    def _open(self) -> bool:
        self._log(f"[wristband:dummy] synthetic {_FRAME_HZ:g} Hz frames "
                  f"(pressure 150 / ppg 100 / imu 50 Hz)")
        return True

    def _poll(self, ts: float) -> None:
        if self._t_start is None:
            self._t_start = ts
        # 帧号按真实时间积分(round),与 dummy eeg 同理:逐块 int 截断会
        # 累积出可见的时钟漂移
        target = int(round((ts - self._t_start) * _FRAME_HZ))
        while self._frame_i < target:
            self._emit_frame(self._t_start + self._frame_i / _FRAME_HZ)
            self._frame_i += 1

    def _emit_frame(self, tf: float) -> None:
        p = 2 * np.pi * 1.2 * tf  # ~72 bpm 的脉搏相位
        self._acc("frame_timestamp_unix_ms", int(tf * 1000))
        self._acc("frame_timestamps", tf)
        self._acc("host_receive_timestamps", tf)

        for k in range(_PRESSURE_PER_FRAME):
            self._acc("pressure_timestamps", tf + k / 150.0)
            self._acc("pressure_raw",
                      int(1000 + np.sin(p + k * 0.1) * 200
                          + np.random.randn() * 5))
        for k in range(_PPG_PER_FRAME):
            self._acc("ppg_timestamps", tf + k / 100.0)
            self._acc_arr("ppg_raw", np.asarray(
                [int(20000 + np.sin(p + c * 0.7) * 2000
                     + np.random.randn() * 100) for c in range(3)],
                dtype=np.int32))

        accel = np.array([np.sin(p * 0.5) * 0.1,
                          np.cos(p * 0.5) * 0.1,
                          9.81 + np.sin(p) * 0.05], dtype=np.float32)
        gyro = np.array([np.sin(p) * 0.05,
                         np.cos(p) * 0.05,
                         np.sin(p * 0.25) * 0.02], dtype=np.float32)
        self._acc("imu_timestamps", tf)
        self._acc_arr("accel_raw",
                      np.round(accel / ACCEL_SCALE_M_S2).astype(np.int16))
        self._acc_arr("accel_m_s2", accel)
        self._acc_arr("gyro_raw",
                      np.round(gyro / GYRO_SCALE_RAD_S).astype(np.int16))
        self._acc_arr("gyro_rad_s", gyro)
        # 温度/血氧固件 1 Hz 更新、其余帧重复最近值 —— dummy 同样逐帧存
        self._acc("temperature_timestamps", tf)
        self._acc("temperature_raw", int((36.5 + np.sin(p * 0.02) * 0.2) * 100))
        self._acc("temperature_c", 36.5 + np.sin(p * 0.02) * 0.2)
        self._acc("spo2_timestamps", tf)
        self._acc("spo2_raw", 98)
        self._acc("spo2_percent", 97.0 + np.random.rand())

        self._data_frames_saved += 1

    def _close(self) -> None:
        self._log(f"[wristband:dummy] stopped "
                  f"(frames={self._data_frames_saved})")
