"""Camera checker — one mp4 per stream plus a frame-timestamp NPZ."""

from __future__ import annotations

from .base import BaseChecker, CheckContext
from .checks import BlackFrame, FrameCountMatch, Freeze, ts_checks


class CameraChecker(BaseChecker):
    """One camera stream.

    Frame timestamps live in ``<stream>.npz`` under ``frames_timestamps``,
    written 1:1 with the frames that actually reached the container — so a
    mismatch against the decoded frame count means the writer dropped some.
    """

    name = "camera"
    matches = ("cam",)
    default_series = "frames"

    checks = [
        ts_checks("frames", expected_rate=30.0),
        FrameCountMatch(),
        BlackFrame(),
        Freeze(),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("frames", key="frames_timestamps", expected_rate=30.0)
