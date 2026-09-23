"""Frame-exact video encoding via an ffmpeg/libx265 subprocess.

Frames are piped as rawvideo over stdin.  The container is CFR with
``-g fps -keyint_min fps -x265-params bframes=0`` — no inter-frame
interpolation and one keyframe per second — so the decoded frame at
index *i* is exactly the *i*-th frame written, and every frame keeps its
1:1 correspondence with the recorded timestamp array.

Pixel formats: 8-bit RGB color in (``rgb24``) -> ``yuv420p`` out;
16-bit depth in (``gray16le``) -> 12-bit HEVC (``gray12le``) out.
"""

import subprocess
import sys
import threading
from collections import deque

import numpy as np

from ..utils.media import media_tool


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
        args = [
            media_tool("ffmpeg"), "-y", "-loglevel", loglevel,
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
        # 独立进程组/会话:控制台 Ctrl+C(Windows 控制台事件 / POSIX 前台
        # 进程组 SIGINT)绝不能直达 ffmpeg —— 那会让它死在写 moov 之前,
        # 容器永久没有索引。收尾只由 stdin EOF 驱动(close())。
        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **popen_kwargs)
        self._closed = False
        # stderr 必须持续排空:ffmpeg 往 stderr 写超过管道缓冲(几 KB)就会
        # 阻塞整个编码管线。后台线程只留最后若干行,供失败时报错。
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="ffmpeg-stderr")
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                self._stderr_lines.append(
                    raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

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
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass    # ffmpeg 已死:close 只负责收尸,失败在下面的 rc 检查里报
        rc = self._proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg exited rc={rc}: {self._stderr_tail()}")

    def _stderr_tail(self) -> str:
        try:
            if self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)
        except Exception:
            pass
        return "\n".join(self._stderr_lines).strip()
