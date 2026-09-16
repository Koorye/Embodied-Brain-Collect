"""OpenCV USB camera."""

import sys
import time
from .base_camera_recorder import BaseCameraRecorder
from .camera_recorder_config import OpencvCameraConfig


def preferred_backend(cv2) -> int:
    """按平台选 VideoCapture 后端。

    CAP_DSHOW 只存在于 Windows;在 Linux 上指定它会直接打不开(设备明明
    在线却报读不到帧)。Linux 用 V4L2,其余平台交给 OpenCV 自动选。
    """
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


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
        cap = cv2.VideoCapture(cfg.idx, preferred_backend(cv2))
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

        self._cap = cap
        # 不读首帧:能否出流由 launcher 的确认阶段判断,自动曝光的
        # 预热帧由 commit 丢弃
        return True

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
