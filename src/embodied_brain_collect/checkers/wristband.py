"""Wristband checker — BLE physiological band: pressure/PPG/IMU.

All timing checks run on the device timestamps (the clock is synced at
open via the 0x20 command).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput
from .checks import DeadChannel, ValueBounds, ts_checks


class WristbandChecker(BaseChecker):
    """BLE wristband: four device-clocked streams."""

    name = "wristband"
    matches = ("wristband",)
    default_series = "pressure"

    checks = [
        ts_checks("frame", expected_rate=50.0),
        ts_checks("pressure", expected_rate=150.0),
        ts_checks("ppg", expected_rate=100.0),
        ts_checks("imu", expected_rate=50.0),
        DeadChannel("ppg_raw"),
        DeadChannel("accel_m_s2"),
        DeadChannel("gyro_rad_s"),
        # 0 = 设备的"本次无读数"(无效样本标记),不是量程越界
        ValueBounds("spo2_percent", lo=0.0, hi=100.0),
        ValueBounds("temperature_c", lo=0.0, hi=45.0),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("frame", key="frame_timestamps")
        ctx.add_series("pressure", key="pressure_timestamps")
        ctx.add_series("ppg", key="ppg_timestamps")
        ctx.add_series("imu", key="imu_timestamps")
