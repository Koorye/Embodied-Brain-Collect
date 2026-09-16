"""Dummy EGO headband — 4 synthetic cameras (mp4) + 2 synthetic IMUs (npz).

Emits the SAME output schema as the real network recorder so downstream tools,
QC and tests run without hardware: cameras are small moving color-bar patterns
at ``cam_fps`` written to ``{name}.mp4`` via the shared HEVC pipeline; IMUs are
smooth sinusoids at ``imu_rate_hz`` accumulated into the npz.
"""

import time
import numpy as np
from .base_ego_headband_recorder import BaseEgoHeadbandRecorder
from .ego_headband_recorder_config import EgoHeadbandRecorderConfig


def _bar(t: float, w: int, h: int, hue0: float) -> np.ndarray:
    """Small moving color-bar test pattern, shape (H, W, 3) uint8 RGB."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    n = 8
    bw = max(w // n, 1)
    shift = int(t * 30) % w
    for i in range(n):
        x0 = (i * bw + shift) % (w + bw) - bw
        hue = (hue0 + i / n + t * 0.05) % 1.0
        r = int(127 + 127 * np.sin(2 * np.pi * hue))
        g = int(127 + 127 * np.sin(2 * np.pi * (hue + 1 / 3)))
        b = int(127 + 127 * np.sin(2 * np.pi * (hue + 2 / 3)))
        img[:, max(0, x0):min(w, x0 + bw)] = (r, g, b)
    return img


class DummyEgoHeadbandRecorder(BaseEgoHeadbandRecorder):
    config: EgoHeadbandRecorderConfig

    def __init__(self, config: EgoHeadbandRecorderConfig):
        super().__init__(config)
        self._cam_t = 0.0
        self._imu_t = 0.0
        self._poll_i = 0
        self._audio_index = 0

    def _open(self) -> bool:
        cfg = self.config
        self._cam_t = 0.0
        self._imu_t = 0.0
        self._poll_i = 0
        self._log(f"[ego_headband:dummy] synthetic {cfg.n_cameras} cams "
                  f"{cfg.cam_width}x{cfg.cam_height}@{cfg.cam_fps:.0f}fps -> mp4 "
                  f"+ {cfg.n_imus} IMU @{cfg.imu_rate_hz:.0f}Hz")
        return True

    def _close(self) -> None:
        self._close_microphone()
        m0 = len(self._buf.get("imu0_ts", []))
        # cam frame counts are logged by the base _save (per {name}.mp4), after
        # the writer threads flush — they are still queued at _close time.
        self._log(f"[ego_headband:dummy] stopped (imu0={m0})")

    def _poll(self, ts):
        cfg = self.config
        imu_dt = 1.0 / max(cfg.imu_rate_hz, 1.0)
        # cameras run slower than IMUs — emit a frame every `cam_every` polls
        cam_every = max(1, int(round(cfg.imu_rate_hz / max(cfg.cam_fps, 1.0))))
        self._poll_i += 1

        # ---- IMUs (every poll, ~imu_rate_hz) -> npz ----
        p = 2 * np.pi * 2.0 * self._imu_t
        for j in range(cfg.n_imus):
            off = j * 0.7
            self._acc(f"imu{j}_ts", ts)
            self._acc_arr(f"imu{j}_gyro",
                np.array([np.sin(p + off), np.cos(p + off), 0.1 * np.sin(p)],
                         dtype=np.float32))
            self._acc_arr(f"imu{j}_accel",
                np.array([0.0, 0.0, 9.8 + 0.3 * np.sin(p * 0.5 + off)],
                         dtype=np.float32))
        self._imu_t += imu_dt

        # ---- cameras (every `cam_every` polls, ~cam_fps) -> {name}.mp4 ----
        if self._poll_i % cam_every == 0:
            hue_step = 1.0 / max(cfg.n_cameras, 1)
            for i in range(cfg.n_cameras):
                self.arr_video(self._cam_stem(i), ts,
                               _bar(self._cam_t, cfg.cam_width, cfg.cam_height,
                                    hue0=i * hue_step))
            self._cam_t += 1.0 / max(cfg.cam_fps, 1.0)

        if cfg.audio_enabled:
            # Use the same packet schema and output path as the device.
            target_samples = int(self._imu_t * 16000)
            while self._audio_index + 320 <= target_samples:
                positions = np.arange(self._audio_index, self._audio_index + 320)
                pcm = (2000 * np.sin(2 * np.pi * 440 * positions / 16000)).astype("<i2")
                self._write_microphone({
                    "type": "audio/PCM", "stream_seq": self._audio_index // 320 + 1,
                    "read_complete_ns": time.time_ns(),
                    "timestamp_basis": "read_complete_not_hardware_capture",
                    "audio": {"sample_rate": 16000, "channels": 1, "format": "S16_LE",
                              "samples": 320, "sample_index": self._audio_index},
                }, pcm.tobytes())
                self._audio_index += 320
        time.sleep(imu_dt)
