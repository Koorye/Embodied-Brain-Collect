"""OpenCV USB camera."""

import sys
import time
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import OpencvCameraConfig

sys.stdout.reconfigure(line_buffering=True)


class OpencvCameraRecorder(BaseCameraRecorder):
    config: OpencvCameraConfig

    def __init__(self, config: OpencvCameraConfig):
        super().__init__(config)
        self._cap = None
        self._cv2 = None

    def _open(self) -> bool:
        import cv2
        self._cv2 = cv2

        cfg = self.config
        cap = cv2.VideoCapture(cfg.idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self._open_error = f"camera idx={cfg.idx} not opened"
            self._log(f"[camera:opencv] open failed — {self._open_error}")
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        dg = (aw, ah) != (cfg.width, cfg.height) or abs(afps - cfg.fps) > 1
        self._log(f"[camera:opencv] idx={cfg.idx} {aw}x{ah}@{afps:.1f}fps "
                  f"(role={self.role or '-'})" + (" (DOWNGRADED!)" if dg else ""))

        if cfg.warmup > 0:
            deadline = time.time() + cfg.warmup
            while time.time() < deadline:
                cap.read()

        # First-data gate: one successful read proves the stream is live.
        self._log("[camera:opencv] waiting for first frame ...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            ok, _ = cap.read()
            if ok:
                self._cap = cap
                self._log("[camera:opencv] first frame received — ready")
                return True
        cap.release()
        self._open_error = "no frame within 10 s"
        self._log(f"[camera:opencv] open failed — {self._open_error}")
        return False

    def _poll(self, ts):
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        # Absolute host wall-clock at frame grab (unix seconds).
        now = time.time()
        # VideoCapture delivers BGR; record RGB like the other cameras.
        self.arr_video("frames", now,
                       self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
