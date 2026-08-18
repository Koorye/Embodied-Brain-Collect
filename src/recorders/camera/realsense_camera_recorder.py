"""Intel RealSense depth camera via pyrealsense2."""

import sys
import time
import numpy as np
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import RealsenseCameraConfig

sys.stdout.reconfigure(line_buffering=True)


class RealsenseCameraRecorder(BaseCameraRecorder):
    """Real Intel RealSense (D400 series) camera.

    Records color into ``frames`` (RGB — same convention as the other camera
    recorders) and, when ``depth=True``, depth into ``depth_frames``
    (uint16, millimeters).  Depth is aligned to the color stream; if the align
    processing block fails at runtime it is disabled and the raw (unaligned)
    depth stream is recorded for the rest of the session instead.  If a depth
    frame is dropped, ``depth_frames`` may be shorter than ``frames``.
    """

    config: RealsenseCameraConfig

    def __init__(self, config: RealsenseCameraConfig):
        super().__init__(config)
        self._pipeline = None
        self._align = None
        self._last_frame_at = 0.0

    def _open(self) -> bool:
        import pyrealsense2 as rs

        cfg = self.config
        pipeline = rs.pipeline()
        rscfg = rs.config()
        if cfg.serial:
            rscfg.enable_device(cfg.serial)
        rscfg.enable_stream(rs.stream.color, cfg.width, cfg.height,
                            rs.format.rgb8, cfg.fps)
        if cfg.depth:
            rscfg.enable_stream(rs.stream.depth, cfg.width, cfg.height,
                                rs.format.z16, cfg.fps)

        profile = pipeline.start(rscfg)

        dev = profile.get_device()
        self._log(
            f"[camera:realsense] {dev.get_info(rs.camera_info.name)} "
            f"sn={dev.get_info(rs.camera_info.serial_number)} "
            f"{cfg.width}x{cfg.height}@{cfg.fps}fps depth={cfg.depth} "
            f"(role={self.role or '-'})"
        )
        self._align = rs.align(rs.stream.color) if cfg.depth else None
        self._pipeline = pipeline

        # First-data gate: one complete frameset proves the streams are live.
        self._log("[camera:realsense] waiting for first frameset ...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            try:
                fs = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            if not fs or not fs.get_color_frame():
                continue
            if not cfg.depth:
                self._log("[camera:realsense] first frameset received — ready")
                return True
            if self._align.process(fs).get_depth_frame():
                self._log("[camera:realsense] first frameset (+depth) "
                          "received — ready")
                return True
        self._open_error = "no frameset within 10 s"
        self._log(f"[camera:realsense] open failed — {self._open_error}")
        pipeline.stop()
        self._pipeline = None
        self._align = None
        return False

    def _poll(self, ts):
        assert self._pipeline is not None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1)
        except RuntimeError:
            return

        color = frames.get_color_frame()
        now = time.time()
        self._last_frame_at = now
        # Absolute host wall-clock at frame grab (unix seconds).
        self.arr_video("frames", now, np.array(color.get_data()))
        if not self.config.depth:
            return

        depth = self._align.process(frames).get_depth_frame()
        self.arr_video("depth_frames", now, np.array(depth.get_data()))

    def _check_stall(self) -> None:
        """Log (rate-limited) when no frameset arrives for several seconds."""
        now = time.time()
        if self._last_frame_at and now - self._last_frame_at > 5.0:
            self._log(f"[camera:realsense] no frames for "
                      f"{now - self._last_frame_at:.1f}s — USB/driver issue?")
            self._last_frame_at = now  # re-arm; logs at most once per ~5 s

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None
