"""OAK-D camera via Luxonis depthai SDK.

The recorder was capping at ~21.8 fps instead of the configured 30, at
EVERY resolution (640x360 included): ``tryGet()`` is non-blocking, but the
generic ``BaseRecorder._loop`` polls once per ``1/hz`` (hz=30 in the slot
config) and then sleeps the rest of the period.  With no frame waiting,
every poll eats the full ~33 ms sleep, which the coarse Windows timer
rounds up to ~31-47 ms — the loop drains at ~21-24 Hz, slower than the
sensor pushes, so the SDK queue (maxSize=30) overflows and frames drop.
Blocking sources (OpenCV ``read()``) self-pace to frame arrival and never
hit this; non-blocking ``tryGet`` does.  Fix: override ``_loop`` with a
tight drain — grab immediately while frames are waiting, sleep 5 ms only
when idle (same shape as the reference recorder that holds 30 fps on this
rig).

USB link speed is reported at open purely as a diagnostic: USB2 may
additionally throttle large resolutions (1280x720 NV12 @ 30fps ≈ 332 Mbps
vs USB2's effective XLink throughput), but that ceiling was never observed
on this rig — the drain bug alone explained every 21.8 fps reading.  If
fps still falls short of the target on a USB2 link, the fix is a USB3
port + USB3 cable.
"""

import time
import numpy as np
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import DepthaiCameraConfig

#: Idle poll sleep of the tight drain loop (matches the reference recorder).
_IDLE_SLEEP_S = 0.005


def _is_usb2(usb_speed) -> bool:
    import depthai
    return usb_speed in (depthai.UsbSpeed.HIGH,
                         depthai.UsbSpeed.FULL,
                         depthai.UsbSpeed.LOW)


def _build_pipeline(w: int, h: int):
    """Camera node + NV12 output at ``(w, h)`` (on-device downscale)."""
    import depthai
    pipeline = depthai.Pipeline()
    cam = pipeline.create(depthai.node.Camera).build(
        depthai.CameraBoardSocket.CAM_A)
    out = cam.requestOutput((w, h), depthai.ImgFrame.Type.NV12)
    queue = out.createOutputQueue(maxSize=30, blocking=False)
    return pipeline, queue


class DepthaiCameraRecorder(BaseCameraRecorder):
    """Real OAK-D via Luxonis depthai."""

    config: DepthaiCameraConfig

    def __init__(self, config: DepthaiCameraConfig):
        super().__init__(config)
        self._queue = None
        self._device = None
        self._pipeline = None

    def _open(self) -> bool:
        cfg = self.config
        # 分辨率当前固定 640x360(cam_w/cam_h 暂不生效,接入更多档位后再放开)
        w, h = 640, 360

        pipeline, queue = _build_pipeline(w, h)
        pipeline.start()
        device = pipeline.getDefaultDevice()
        usb = device.getUsbSpeed()

        self._queue = queue
        self._pipeline = pipeline
        self._device = device
        self._log(f"[camera:depthai] {w}x{h} @ {cfg.cam_fps_hint:.0f}fps "
                  f"usb_speed={usb} (role={self.role or '-'})")
        if _is_usb2(usb):
            self._log(
                f"[camera:depthai] USB2 link — if fps stays below "
                f"{cfg.cam_fps_hint:.0f}, plug into a USB3 port "
                f"with a USB3 cable.", level="WARNING")

        # 不读首帧:能否出流由 launcher 的确认阶段判断
        return True

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Tight drain loop — see the module docstring (drain cadence).

        The generic loop's ``1/hz`` throttle is meant for pollers that
        block or sleep on their own; ``tryGet`` is non-blocking, so under
        it the throttle turns into a frame-dropping rate limiter.
        """
        self._loop_t0 = self._ts()
        last = self._loop_t0
        while True:
            if self._maybe_commit():
                last = self._loop_t0
            ts = self._ts()
            elapsed = ts - self._loop_t0
            if self._stop_by_duration(elapsed) or self.stop_event.is_set():
                break
            self._poll(ts)
            if ts - last > 0.5:
                last = ts
                self._heartbeat(elapsed)

    def _poll(self, ts):
        pkt = self._queue.tryGet() if self._queue is not None else None
        if pkt is None:
            time.sleep(_IDLE_SLEEP_S)   # idle: brief sleep before next drain
            return
        # Absolute host wall-clock at frame grab (unix seconds).
        # getCvFrame() is BGR and shares the SDK buffer: copy out (np.array)
        # and swap channels to record RGB.
        now = time.time()
        self.arr_video("frames", now, np.array(pkt.getCvFrame())[:, :, ::-1])

    def _close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
            self._pipeline = None
            self._queue = None
