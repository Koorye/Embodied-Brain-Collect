"""Eye tracker checker — Pupil Labs Neon, three streams plus scene video."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput
from .checks import BlackFrame, FrameCountMatch, Freeze, ValueBounds, ts_checks


@dataclass(frozen=True)
class ClockOffset(BaseCheck):
    """Size of the phone-to-PC clock correction, for the record.

    Always informational: the recorder applies this offset when stamping, so
    however large it is the timestamps are already in the PC domain.  What
    would be a real fault is the correction not being applied at all, and
    that shows up as a stream whose samples all land outside the run window
    — which ``TimestampSanity`` reports.
    """

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.arr("pc_to_phone_offset_ms") is not None

    def run(self, ctx: CheckContext) -> CheckOutput:
        ms = float(np.asarray(ctx.arr("pc_to_phone_offset_ms")).ravel()[0])
        out = CheckOutput(stats={"pc_to_phone_offset_ms": ms})
        out.findings.append(self.finding(
            "INFO", f"手机与 PC 时钟相差 {ms:+.1f}ms(已在录制时校正)",
            field="pc_to_phone_offset_ms", observed=abs(ms)))
        return out


class EyeChecker(BaseChecker):
    """Neon: gaze, IMU and scene run on independent clocks and rates, so
    each gets its own series rather than one combined timeline."""

    name = "eye"
    matches = ("eye",)

    checks = [
        ts_checks("gaze"),
        ts_checks("scene", expected_rate=30.0),
        ts_checks("imu"),
        ClockOffset(),
        ValueBounds("gaze_xy", lo=0.0, hi=2000.0),
        FrameCountMatch(series="scene"),
        BlackFrame(series="scene"),
        Freeze(series="scene"),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("gaze", key="gaze_timestamps")
        ctx.add_series("scene", key="scene_timestamps", expected_rate=30.0)
        ctx.add_series("imu", key="imu_timestamps")
