"""EGO headband checker — network device streaming N cameras + M IMUs.

Camera and IMU counts are configurable, so the NPZ fields are numbered
(``cam{i}_timestamps`` / ``cam{i}_frames`` / ``imu{j}_ts`` / ``imu{j}_gyro``
/ ``imu{j}_accel``).  ``prepare`` discovers however many streams the file
actually holds and registers one timestamp series each, then builds the check
list to match — a fixed ClassVar set would false-alarm "缺少时间戳" on the
streams this particular recording does not have.

Frames live in the NPZ as raw uint8 arrays, not an mp4, so there is no video
decode here: camera health is judged from its frame clock alone, and only the
IMUs carry value distributions worth an outlier pass.
"""

from __future__ import annotations

import re

from .base import BaseChecker, CheckContext
from .checks import MadOutlier, ts_checks

_CAM_TS = re.compile(r"^cam(\d+)_timestamps$")
_IMU_TS = re.compile(r"^imu(\d+)_ts$")


class EgoHeadbandChecker(BaseChecker):
    """EGO headband: N cameras + M IMUs, each stream on its own clock."""

    name = "ego_headband"
    matches = ("ego_headband",)
    default_series = "cam0"

    def prepare(self, ctx: CheckContext) -> None:
        keys = list(ctx.npz.files) if ctx.npz is not None else []
        cams = sorted(int(m.group(1)) for k in keys if (m := _CAM_TS.match(k)))
        imus = sorted(int(m.group(1)) for k in keys if (m := _IMU_TS.match(k)))

        checks: list = []
        for i in cams:
            ctx.add_series(f"cam{i}", key=f"cam{i}_timestamps")
            checks += ts_checks(f"cam{i}")
        for j in imus:
            ctx.add_series(f"imu{j}", key=f"imu{j}_ts")
            checks += ts_checks(f"imu{j}")
            checks += [MadOutlier(f"imu{j}_gyro"), MadOutlier(f"imu{j}_accel")]

        # BaseChecker.run reads self.checks right after prepare, so mirroring
        # the discovered streams here is what makes the count dynamic.
        self.checks = checks
