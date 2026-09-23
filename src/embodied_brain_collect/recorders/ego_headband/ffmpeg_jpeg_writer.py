"""EGO headband GPU video writer — Motion-JPEG in, frame-exact HEVC ``.mp4`` out.

The headband's four fisheye cameras stream Motion-JPEG.  Decoding each JPEG on
the CPU (``cv2.imdecode``) and re-encoding with libx265 tops out near 17 fps for
a SINGLE 1920x1088 stream — four of them cannot keep up and frames drop.  So the
compressed frame is piped straight into ffmpeg, which decodes it on the GPU
(``mjpeg_cuvid``) and re-encodes on the GPU (``hevc_nvenc``): both the decode and
the encode stay off the CPU (measured ~113 fps per stream on an RTX 3050).

This lives beside the headband recorder rather than in the shared
``recorders.ffmpeg_writer`` because it is specific to JPEG-streaming devices —
the shared ``FFmpegWriter`` keeps serving raw-frame cameras unchanged.
``BaseCameraRecorder`` reaches this writer through its ``_make_writer`` hook,
which ``BaseEgoHeadbandRecorder`` overrides; nothing in the generic camera base
needs to know about JPEG or the GPU.

GPU is used when ``_probe_gpu_transcode()`` says the path works on this machine
(probed once per process, cached); otherwise it falls back to CPU ``mjpeg`` +
``libx265`` so recording still works without NVIDIA/nvenc.
"""

import subprocess
import sys
import threading
from collections import deque

from loguru import logger

from ...utils.media import media_tool

# One-time, process-wide verdict on whether the GPU transcode path works here.
# ``None`` = not probed yet.  Guarded by a lock because the four camera writer
# threads all reach this on their first frame at once: without the lock they
# race, and the old "publish False before probing" shortcut made the losers read
# a not-yet-real False and needlessly fall back to CPU (only camera 0 got the
# GPU).  Now exactly one thread probes; the others block, then read the same
# cached verdict.  ``_GPU_FALLBACK_WARNED`` keeps a genuine fallback warning to
# once per process instead of once per camera.
_GPU_TRANSCODE: bool | None = None
_GPU_PROBE_LOCK = threading.Lock()
_GPU_FALLBACK_WARNED = False


def _probe_gpu_transcode() -> bool:
    """Can ffmpeg do ``mjpeg_cuvid`` (GPU decode) -> ``hevc_nvenc`` (GPU encode)?

    Runs the exact GPU pipeline on a few tiny ffmpeg-generated JPEGs and caches
    the verdict.  Any failure (no NVIDIA GPU, old driver, nvenc/cuvid missing,
    concurrent-session limit) -> ``False`` so callers fall back to the CPU.  Uses
    only ffmpeg (no cv2/numpy), so it is safe to call from any recorder process.
    """
    global _GPU_TRANSCODE
    if _GPU_TRANSCODE is not None:          # fast path: already decided
        return _GPU_TRANSCODE
    with _GPU_PROBE_LOCK:                    # serialize the one-time probe
        if _GPU_TRANSCODE is not None:       # another thread finished while we waited
            return _GPU_TRANSCODE
        result = False
        try:
            ff = media_tool("ffmpeg")
            # 1) mint a handful of small JPEGs with ffmpeg itself (no decoder dep).
            gen = subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "testsrc2=size=320x240:rate=30:duration=0.1",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
            # 2) push them through the real GPU decode+encode path.
            if gen.stdout:
                chk = subprocess.run(
                    [ff, "-hide_banner", "-loglevel", "error",
                     "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                     "-f", "image2pipe", "-framerate", "30", "-vcodec", "mjpeg_cuvid",
                     "-i", "-", "-c:v", "hevc_nvenc", "-preset", "p4",
                     "-vf", "scale_cuda=format=yuv420p", "-f", "null", "-"],
                    input=gen.stdout, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=40)
                result = (chk.returncode == 0)
        except Exception:
            result = False
        _GPU_TRANSCODE = result              # publish the FINAL verdict only
        return _GPU_TRANSCODE


class FFmpegJpegWriter:
    """Motion-JPEG in -> frame-exact HEVC ``.mp4``, GPU when available.

    ``encoder``: ``"gpu"``/``"nvenc"`` (the EGO headband default) uses the GPU
    and, if ``_probe_gpu_transcode()`` says it is unavailable here, warns loudly
    before falling back to CPU; ``"auto"`` does the same but adapts silently;
    ``"cpu"``/``"libx265"`` forces CPU.  Each JPEG written becomes exactly one
    output frame — CFR, no B-frames, one keyframe per second — so frame *i*
    keeps its 1:1 correspondence with the recorded timestamp array, matching the
    shared ``FFmpegWriter``.
    """

    def __init__(
        self,
        path,
        fps: float,
        encoder: str = "auto",
        crf: int = 23,
        preset: str = "medium",
        loglevel: str = "error",
    ):
        global _GPU_FALLBACK_WARNED
        ff = media_tool("ffmpeg")
        want = (encoder or "auto").lower()
        force_cpu = want in ("cpu", "libx265", "x265")
        # GPU whenever it is wanted (explicit "gpu"/"nvenc", or "auto") AND the
        # probe says it works here.  A forced "gpu" that can't run falls back to
        # CPU with a loud warning rather than writing a broken/empty mp4 — but
        # that fallback is ~17 fps/stream, far too slow for four 1088p30 fisheye
        # streams, so the operator must know the GPU they asked for is not used.
        self.gpu = (not force_cpu) and _probe_gpu_transcode()
        if not self.gpu and not force_cpu and want != "auto":
            if not _GPU_FALLBACK_WARNED:
                _GPU_FALLBACK_WARNED = True
                logger.warning("[ego_headband] encoder={!r} requested but GPU "
                               "transcode (mjpeg_cuvid/hevc_nvenc) is unavailable "
                               "here — falling back to CPU libx265 (too slow for "
                               "4x1088p30; frames will drop)", encoder)
        g = str(int(fps))
        # Common output timing: CFR, 1 keyframe/s, no reorder -> frame i == ts i.
        tail = ["-r", str(fps), "-fps_mode", "cfr",
                "-video_track_timescale", "90000", "-g", g, "-keyint_min", g]
        if self.gpu:
            # mjpeg_cuvid decodes on the GPU and stays in CUDA memory; scale_cuda
            # converts the JPEG's full-range yuvj420p to yuv420p for nvenc (a
            # plain -pix_fmt yuv420p is rejected on cuda frames — rc -22).
            pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                   "-f", "image2pipe", "-framerate", str(fps),
                   "-vcodec", "mjpeg_cuvid"]
            enc = ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-cq", str(crf), "-b:v", "0",
                   "-vf", "scale_cuda=format=yuv420p", "-bf", "0"]
            self.encoder_used = "hevc_nvenc"
        else:
            pre = ["-f", "image2pipe", "-framerate", str(fps), "-vcodec", "mjpeg"]
            enc = ["-c:v", "libx265", "-crf", str(crf), "-preset", preset,
                   "-pix_fmt", "yuv420p", "-x265-params", "bframes=0"]
            self.encoder_used = "libx265"
        args = [ff, "-y", "-loglevel", loglevel, *pre, "-i", "-",
                *enc, *tail, str(path)]

        # Same process-group isolation + stderr drain as the shared FFmpegWriter:
        # Ctrl+C must not reach ffmpeg before it writes the moov index, and a
        # full stderr pipe would otherwise stall the whole encode.
        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **popen_kwargs)
        self._closed = False
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="ffmpeg-jpeg-stderr")
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                self._stderr_lines.append(
                    raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    def write(self, jpeg: bytes) -> None:
        """Pipe one JPEG frame to ffmpeg (blocks on backpressure — the caller's
        bounded queue drops the oldest frame at the arrival side)."""
        if self._closed:
            raise RuntimeError("ffmpeg jpeg writer already closed")
        try:
            self._proc.stdin.write(jpeg)
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
            pass
        rc = self._proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited rc={rc}: {self._stderr_tail()}")

    def _stderr_tail(self) -> str:
        try:
            if self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)
        except Exception:
            pass
        return "\n".join(self._stderr_lines).strip()
