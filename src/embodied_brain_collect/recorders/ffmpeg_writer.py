"""Frame-exact video encoding via an ffmpeg subprocess (libx265 / hevc_nvenc).

Frames are piped as rawvideo over stdin.  The container is CFR with one
keyframe per second and NO B-frames — ``-g fps -keyint_min fps`` plus
``-x265-params bframes=0`` (CPU) or ``-bf 0`` (GPU) — so the decoded frame at
index *i* is exactly the *i*-th frame written, and every frame keeps its
1:1 correspondence with the recorded timestamp array.  This invariant holds
identically on both encoders; pack_daily's smart cut and reqc depend on it.

Encoding runs on the GPU (``hevc_nvenc``) when ``encoder`` allows it and the
probe says nvenc works here — CPU libx265 cannot keep up with multiple
concurrent color streams and frames drop at the write queue.  Depth streams
(``gray12le`` out) always stay on libx265: nvenc tops out at 10-bit and
cannot express the 12-bit monochrome depth contract.

Pixel formats: 8-bit RGB color in (``rgb24``/``bgr24``) -> ``yuv420p`` out;
16-bit depth in (``gray16le``) -> 12-bit HEVC (``gray12le``) out.
"""

import subprocess
import sys
import threading
from collections import deque

import numpy as np
from loguru import logger

from ..utils.media import media_tool

# One-time, process-wide verdict on whether hevc_nvenc works on this machine
# (``None`` = not probed yet).  A lock serializes the probe: the camera writer
# threads all reach it on their first frame at once, and without the lock the
# losers would race the publisher (same lesson as the EGO headband's probe).
_NVENC_OK: bool | None = None
_NVENC_PROBE_LOCK = threading.Lock()
# Keep the two "requested GPU, got CPU" warnings to once per process instead
# of once per stream (color: no nvenc here; depth: nvenc has no 12-bit).
_NVENC_FALLBACK_WARNED = False
_DEPTH_CPU_WARNED = False

# x265 preset names -> nvenc p-levels (p4 == jpeg writer's proven default).
_NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p3", "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7",
}


def _probe_nvenc() -> bool:
    """Can ffmpeg encode with ``hevc_nvenc`` here?

    Runs a few tiny lavfi frames through the real encoder and caches the
    verdict.  320x240 must stay >= nvenc's minimum encode size (145x49) or
    the probe would report False even on a healthy GPU.  Any failure (no
    NVIDIA GPU, old driver, nvenc missing from the build, concurrent-session
    limit) -> ``False`` so callers stay on the CPU.
    """
    global _NVENC_OK
    if _NVENC_OK is not None:
        return _NVENC_OK
    with _NVENC_PROBE_LOCK:
        if _NVENC_OK is not None:      # another thread finished while we waited
            return _NVENC_OK
        ok = False
        try:
            chk = subprocess.run(
                [media_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=0.1",
                 "-c:v", "hevc_nvenc", "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30)
            ok = (chk.returncode == 0)
        except Exception:
            ok = False
        _NVENC_OK = ok                  # publish the FINAL verdict only
        return _NVENC_OK


class FFmpegWriter:
    """Raw frames in -> frame-exact HEVC ``.mp4``, GPU when available.

    ``encoder``: ``"auto"`` (default) uses hevc_nvenc when the probe says it
    works and falls back to CPU silently otherwise; ``"gpu"``/``"nvenc"``
    warns loudly before falling back (the operator asked for the GPU and CPU
    encoding drops frames); ``"cpu"``/``"libx265"`` forces CPU.  Independent
    of that choice, non-``yuv420p`` output (the depth streams' 12-bit
    ``gray12le``) always encodes with libx265 — nvenc has no 12-bit mode and
    the depth precision contract must not change.
    """

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
        encoder: str = "auto",
        loglevel: str = "error",
    ):
        global _NVENC_FALLBACK_WARNED, _DEPTH_CPU_WARNED
        want = (encoder or "auto").lower()
        force_cpu = want in ("cpu", "libx265", "x265")
        # nvenc encodes 8-bit yuv420p (and 10-bit p010) only; gray12le depth
        # is structurally CPU-bound, not an environment fallback.
        nvenc_fmt = output_pix_fmt in ("yuv420p", "nv12")
        self.gpu = (not force_cpu) and nvenc_fmt and _probe_nvenc()
        self.encoder_used = "hevc_nvenc" if self.gpu else "libx265"
        if want not in ("auto",) and not force_cpu:
            if nvenc_fmt and not self.gpu and not _NVENC_FALLBACK_WARNED:
                _NVENC_FALLBACK_WARNED = True
                logger.warning(
                    f"[ffmpeg_writer] encoder={encoder!r} requested but "
                    "hevc_nvenc is unavailable here — falling back to CPU "
                    "libx265 (color streams may drop frames)")
            elif not nvenc_fmt and not _DEPTH_CPU_WARNED:
                _DEPTH_CPU_WARNED = True
                logger.warning(
                    f"[ffmpeg_writer] encoder={encoder!r} requested but "
                    f"{output_pix_fmt} (16-bit depth) cannot use hevc_nvenc "
                    "(no 12-bit mode) — encoding depth with libx265 on the CPU")

        # Common output timing: CFR, 1 keyframe/s, no reorder -> frame i == ts i.
        g = str(int(fps))
        tail = ["-r", str(fps), "-fps_mode", "cfr",
                "-video_track_timescale", "90000", "-g", g, "-keyint_min", g]
        if self.gpu:
            # Raw frames arrive in system memory; ffmpeg converts to yuv420p
            # and uploads to CUDA for nvenc.  -b:v 0 lifts the default ~2M
            # bitrate cap so -cq alone controls quality (same combo as the
            # EGO jpeg writer).  -bf 0 keeps decode order == display order.
            enc = ["-c:v", "hevc_nvenc",
                   "-preset", _NVENC_PRESETS.get(preset.lower(), "p4"),
                   "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
                   "-pix_fmt", output_pix_fmt, "-bf", "0"]
        else:
            enc = ["-c:v", "libx265", "-crf", str(crf), "-preset", preset,
                   "-pix_fmt", output_pix_fmt,
                   "-x265-params", "bframes=0"]
        args = [
            media_tool("ffmpeg"), "-y", "-loglevel", loglevel,
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-pix_fmt", input_pix_fmt, "-i", "-",
            *enc, *tail,
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
