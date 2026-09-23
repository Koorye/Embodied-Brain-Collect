"""Intel RealSense depth camera via pyrealsense2."""

import time
import numpy as np
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import RealsenseCameraConfig


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
        # 不读首帧:能否出流由 launcher 的确认阶段判断
        return True

    def _poll(self, ts):
        assert self._pipeline is not None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1)
        except RuntimeError:
            return

        color = frames.get_color_frame()
        now = time.time()
        # Absolute host wall-clock at frame grab (unix seconds).
        self.arr_video("frames", now, np.array(color.get_data()))
        if not self.config.depth:
            return

        depth = self._align.process(frames).get_depth_frame()
        self.arr_video("depth_frames", now, np.array(depth.get_data()))

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None
