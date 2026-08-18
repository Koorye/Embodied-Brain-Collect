"""Frame-exact video encoding via an ffmpeg/libx265 subprocess.

Frames are piped as rawvideo over stdin.  The container is CFR with
``-g fps -keyint_min fps -x265-params bframes=0`` — no inter-frame
interpolation and one keyframe per second — so the decoded frame at
index *i* is exactly the *i*-th frame written, and every frame keeps its
1:1 correspondence with the recorded timestamp array.

Pixel formats: 8-bit RGB color in (``rgb24``) -> ``yuv420p`` out;
16-bit depth in (``gray16le``) -> 12-bit HEVC (``gray12le``) out.
"""

import shutil
import subprocess

import numpy as np


class FFmpegWriter:
    def __init__(
        self,
        path,
        width: int,
        height: int,
        fps: float,
        input_pix_fmt: str = "rgb24",
        output_pix_fmt: str = "yuv420p",
        crf: int = 23,
        preset: str = "medium",
        loglevel: str = "error",
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        args = [
            "ffmpeg", "-y", "-loglevel", loglevel,
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-pix_fmt", input_pix_fmt, "-i", "-",
            "-r", str(fps), "-fps_mode", "cfr",
            "-video_track_timescale", "90000",
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", output_pix_fmt,
            "-g", str(int(fps)), "-keyint_min", str(int(fps)),
            "-x265-params", "bframes=0",
            str(path),
        ]
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        """Pipe one raw frame to ffmpeg (blocks on backpressure — that is
        the pacing: the caller's queue drops frames at the arrival side)."""
        if self._closed:
            raise RuntimeError("ffmpeg writer already closed")
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg died: {self._stderr_tail()}") from exc

    def close(self) -> None:
        """Close stdin, wait for ffmpeg to finalize the container."""
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin:
            self._proc.stdin.close()
        rc = self._proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg exited rc={rc}: {self._stderr_tail()}")

    def _stderr_tail(self) -> str:
        try:
            if self._proc.stderr:
                return self._proc.stderr.read().decode("utf-8",
                                                        errors="replace").strip()
        except Exception:
            pass
        return ""
