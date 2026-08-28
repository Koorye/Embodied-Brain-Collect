"""Abstract eye tracker base."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..base import BaseRecorder

# _arr_buf keys streamed to disk as PNG sequences while recording (crash-safe).
_PNG_SUBDIRS = {"scene_frames": "scene"}


class BaseEyeRecorder(BaseRecorder):
    name = "eye"
    output_dir = "eye"

    def __init__(self, config):
        super().__init__(config)
        self._png_counters: dict[str, int] = {}   # frames streamed per key
        self._png_dirs: dict[str, Path] = {}
        self._write_failed: set[str] = set()      # keys with OSError so far

    # ---- streaming writes (called from _acc_arr / _acc during recording) ----

    def _acc_arr(self, key: str, arr: np.ndarray) -> None:
        super()._acc_arr(key, arr)
        subdir = _PNG_SUBDIRS.get(key)
        if subdir is None or not self.config.session_dir:
            return
        if key not in self._png_dirs:
            try:
                png_dir = Path(self.config.session_dir) / self.output_dir / subdir
                png_dir.mkdir(parents=True, exist_ok=True)
                self._png_dirs[key] = png_dir
            except OSError as exc:
                self._warn_write(key, exc)
                return
        try:
            img = arr
            # cv2.imwrite expects BGR (or single-channel); scene frames are RGB.
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            n = self._png_counters.get(key, 0)
            cv2.imwrite(str(self._png_dirs[key] / f"frame_{n:08d}.png"), img)
            self._png_counters[key] = n + 1
        except (OSError, cv2.error) as exc:
            # Keep accumulating in memory; _save() retries the unwritten tail.
            self._warn_write(key, exc)

    def _warn_write(self, key: str, exc: Exception) -> None:
        if key not in self._write_failed:
            self._log(f"[{self.name}] {key} disk write failed — {exc}; "
                      "keeping data in memory (retried at save)")
            self._write_failed.add(key)

    # ---- finalize at close --------------------------------------------------

    def _save(self) -> None:
        """Write the tail of any unwritten scene PNGs, close the timestamp
        streams, then save the (small, scene-free) eye.npz."""
        if not self.config.session_dir:
            return

        # unwritten tail (normally empty; covers disk errors and last frames)
        for key, subdir in _PNG_SUBDIRS.items():
            arrays = self._arr_buf.get(key, [])
            start = self._png_counters.get(key, 0)
            if len(arrays) <= start:
                continue
            try:
                if key not in self._png_dirs:
                    png_dir = Path(self.config.session_dir) / self.output_dir / subdir
                    png_dir.mkdir(parents=True, exist_ok=True)
                    self._png_dirs[key] = png_dir
                png_dir = self._png_dirs[key]
                for i in range(start, len(arrays)):
                    img = arrays[i]
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(png_dir / f"frame_{i:08d}.png"), img)
                    self._png_counters[key] = i + 1
                self._log(f"[{self.name}] wrote {len(arrays) - start} remaining "
                          f"{key} PNGs -> {png_dir}")
            except (OSError, cv2.error) as exc:
                self._log(f"[{self.name}] {key} PNG write failed — {exc}")

        super()._save()  # eye.npz via _build_output (scene_frames excluded)

    def _build_output(self) -> dict[str, np.ndarray]:
        out = super()._build_output()
        out.pop("scene_frames", None)  # already on disk as PNGs
        return out
