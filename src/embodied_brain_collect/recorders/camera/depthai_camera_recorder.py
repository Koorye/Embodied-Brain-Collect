"""OAK-D camera via Luxonis depthai SDK."""

import sys
import time
import numpy as np
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import DepthaiCameraConfig

sys.stdout.reconfigure(line_buffering=True)


def _pick_resolution(cam_w: int, cam_h: int):
    """Nearest supported OAK-D color sensor resolution for the config."""
    import depthai
    R = depthai.ColorCameraProperties.SensorResolution
    if cam_w >= 1920 or cam_h >= 1080:
        return R.THE_1080_P, (1920, 1080)
    return R.THE_720_P, (1280, 720)


class DepthaiCameraRecorder(BaseCameraRecorder):
    """Real OAK-D via Luxonis depthai."""

    config: DepthaiCameraConfig

    def __init__(self, config: DepthaiCameraConfig):
        super().__init__(config)
        self._queue = None
        self._device = None
        self._pipeline = None

    def _open(self) -> bool:
        import depthai

        cfg = self.config
        res, (w, h) = _pick_resolution(cfg.cam_w, cfg.cam_h)
        self._log(
            f"[camera:depthai] {w}x{h} @ {cfg.cam_fps_hint:.0f}fps "
            f"(role={self.role or '-'})"
        )

        pipeline = depthai.Pipeline()
        cam = pipeline.create(depthai.node.ColorCamera)
        cam.setResolution(res)
        cam.setFps(int(cfg.cam_fps_hint))
        self._queue = cam.video.createOutputQueue(maxSize=30, blocking=False)

        self._pipeline = pipeline
        pipeline.start()
        self._device = pipeline.getDefaultDevice()
        self._log(f"[camera:depthai] usb_speed={self._device.getUsbSpeed()} — "
                  f"waiting for first frame ...")

        # First-data gate: one packet proves the camera is streaming.
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._queue.tryGet() is not None:
                self._log("[camera:depthai] first frame received — ready")
                return True
            time.sleep(0.05)
        self._open_error = "no frame within 10 s"
        self._log(f"[camera:depthai] open failed — {self._open_error}")
        self._device.close()
        self._device = None
        self._pipeline = None
        self._queue = None
        return False

    def _poll(self, ts):
        assert self._queue is not None
        pkt = self._queue.tryGet()
        if pkt is None:
            return
        # Absolute host wall-clock at frame grab (unix seconds).
        now = time.time()
        # getCvFrame() is BGR and shares the SDK buffer: copy out (np.array)
        # and swap channels to record RGB.
        self.arr_video("frames", now, np.array(pkt.getCvFrame())[:, :, ::-1])

    def _close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
            self._pipeline = None
            self._queue = None

