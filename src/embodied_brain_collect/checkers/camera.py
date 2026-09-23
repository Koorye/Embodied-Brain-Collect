"""Camera checker — one mp4 per stream plus a frame-timestamp NPZ."""

from __future__ import annotations

from .base import BaseChecker, CheckContext
from .checks import (BlackFrame, ContainerIntegrity, FrameCountMatch,
                     Freeze, ts_checks)


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
        # 视频检查必须钉死 frames.mp4:深度槽位目录里还有 depth_frames.mp4,
        # 空 video 名会按字典序取到它 —— 深度图天然趋黑,BlackFrame 必然误报
        # "100% 黑屏";深度流不做黑屏/冻结检查。
        ContainerIntegrity(video="frames.mp4"),
        FrameCountMatch(video="frames.mp4"),
        BlackFrame(video="frames.mp4"),
        Freeze(video="frames.mp4"),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("frames", key="frames_timestamps", expected_rate=30.0)
