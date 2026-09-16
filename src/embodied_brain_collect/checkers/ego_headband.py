"""EGO headband checker — network device streaming N cameras + M IMUs.

Camera and IMU counts are configurable, so the streams are named: each
camera is ``{name}.mp4`` with ``{name}_timestamps`` in the NPZ (``name``
from ``camera_names``, e.g. ``left``/``right``), each IMU is ``imu{j}_ts``
/ ``imu{j}_gyro`` / ``imu{j}_accel``.  ``prepare`` discovers
however many the recording actually holds and registers one timestamp series
each, then builds the check list to match — a fixed ClassVar set would
false-alarm "缺少时间戳" on the streams this recording does not have.

Cameras get the same video battery as the standalone camera checker (frame
count vs timestamps, black, freeze), each keyed to its own ``{name}.mp4`` and
series; a video check skips cleanly when its stream produced no mp4.  The IMUs
carry value distributions worth a MAD-outlier pass.
"""

from __future__ import annotations

import re

from .base import BaseChecker, CheckContext
from .checks import (BlackFrame, FrameCountMatch, Freeze, MadOutlier,
                     ts_checks)

_CAM_TS = re.compile(r"^(\w+)_timestamps$")
_IMU_TS = re.compile(r"^imu(\d+)_ts$")


class EgoHeadbandChecker(BaseChecker):
    """EGO headband: N cameras + M IMUs, each stream on its own clock."""

    name = "ego_headband"
    matches = ("ego_headband",)
    default_series = "left"

    def prepare(self, ctx: CheckContext) -> None:
        keys = list(ctx.npz.files) if ctx.npz is not None else []
        cams = sorted(m.group(1) for k in keys if (m := _CAM_TS.match(k)))
        imus = sorted(int(m.group(1)) for k in keys if (m := _IMU_TS.match(k)))

        checks: list = []
        for cam in cams:
            ctx.add_series(cam, key=f"{cam}_timestamps")
            checks += ts_checks(cam)
            # One mp4 per camera in this single directory, so each video check
            # is pinned to its own file + series (skips if that mp4 is absent).
            checks += [
                FrameCountMatch(video=f"{cam}.mp4", series=cam),
                BlackFrame(video=f"{cam}.mp4", series=cam),
                Freeze(video=f"{cam}.mp4", series=cam),
            ]
        for j in imus:
            ctx.add_series(f"imu{j}", key=f"imu{j}_ts")
            checks += ts_checks(f"imu{j}")
            checks += [MadOutlier(f"imu{j}_gyro"), MadOutlier(f"imu{j}_accel")]

        # BaseChecker.run reads self.checks right after prepare, so mirroring
        # the discovered streams here is what makes the count dynamic.
        self.checks = checks
