"""OAK-D camera via Luxonis depthai SDK."""

import sys
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import DepthaiCameraConfig

sys.stdout.reconfigure(line_buffering=True)


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
        self._log(
            f"[camera:depthai] {cfg.cam_w}x{cfg.cam_h} "
            f"@ {cfg.cam_fps_hint:.0f}fps (role={self.role or '-'})"
        )

        pipeline = depthai.Pipeline()
        cam = pipeline.create(depthai.node.ColorCamera)
        cam.setResolution(
            depthai.ColorCameraProperties.SensorResolution.THE_720_P)
        self._queue = cam.video.createOutputQueue(maxSize=30, blocking=False)

        self._pipeline = pipeline
        self._device = depthai.Device(pipeline)
        self._log(f"[camera:depthai] usb_speed={self._device.getUsbSpeed()}")
        return True

    def _poll(self, ts):
        assert self._queue is not None
        pkt = self._queue.tryGet()
        if pkt is None:
            return
        self._acc_ts("cam", ts)
        self._acc_arr("frames", pkt.getCvFrame())

    def _close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
            self._pipeline = None
            self._queue = None

