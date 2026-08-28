"""EMG armband checker — Weili 8-channel + IMU on one serial link."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput
from .checks import DeadChannel, MadOutlier, ts_checks


@dataclass(frozen=True)
class SnGap(BaseCheck):
    """Frames the device transmitted that never reached us.

    EMG and IMU frames share ONE 8-bit counter that advances once per
    transmitted frame, so two consecutive EMG frames legitimately step by 2
    whenever an IMU frame sat between them.  Treating every step != 1 as a
    gap therefore reports one false gap per IMU frame — on session4 that was
    3396 "gaps" against 37 real ones.

    Unwrapping either stream gives the number of frames the device sent
    across the recording; subtracting what both streams actually delivered
    gives the frames genuinely lost.  No arrival order needed.

    The span runs first-EMG-frame to last-EMG-frame, so IMU frames arriving
    outside it are counted as received without being counted as sent — an
    error of at most a frame or two, which the warning fraction absorbs.
    """

    modulo: int = 256
    warn_frac: float = 0.001      # lost frames as a share of those sent

    def applies(self, ctx: CheckContext) -> bool:
        esn, isn = ctx.arr("emg_sn"), ctx.arr("imu_sn")
        return esn is not None and isn is not None and len(esn) > 1

    def run(self, ctx: CheckContext) -> CheckOutput:
        esn = np.asarray(ctx.arr("emg_sn"), dtype=np.int64).ravel()
        isn = np.asarray(ctx.arr("imu_sn"), dtype=np.int64).ravel()
        out = CheckOutput()

        sent = int((np.diff(esn) % self.modulo).sum()) + 1
        received = esn.size + isn.size
        dropped = max(sent - received, 0)
        frac = dropped / sent if sent else 0.0
        out.stats.update({"frames_sent": sent, "frames_received": received,
                          "frames_dropped": dropped, "dropped_fraction": frac})

        if frac > self.warn_frac:
            out.findings.append(self.finding(
                "WARN",
                f"设备发出 {sent} 帧,收到 {received} 帧 — 丢失 {dropped} 帧"
                f"({frac:.2%})",
                field="emg_sn", threshold=self.warn_frac, observed=frac,
                detail={"sent": sent, "received": received}))
        return out


class EmgChecker(BaseChecker):
    """Weili armband: two frame types interleaved on one wire.

    Timestamps are read-arrival times, so long runs of identical values are
    expected rather than a fault — the sequence number, not the clock, is
    what says whether anything was lost.
    """

    name = "emg"
    matches = ("emg",)
    default_series = "emg"

    checks = [
        ts_checks("emg"),
        ts_checks("imu"),
        SnGap(),
        MadOutlier("emg_data"),
        DeadChannel("emg_data"),
        MadOutlier("imu_gyro"),
        MadOutlier("imu_accel"),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("emg", key="emg_timestamps")
        ctx.add_series("imu", key="imu_timestamps")
