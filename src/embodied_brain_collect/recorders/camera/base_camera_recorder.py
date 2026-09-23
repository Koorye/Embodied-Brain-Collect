"""Abstract camera base.

Frame writing is decoupled from the poll loop: concrete recorders just call
``arr_video(key, ts, arr)`` per frame.  A per-stream bounded queue + a
dedicated writer thread turn frames into ``{key}.mp4`` (libx265 HEVC via the
shared FFmpegWriter — 8-bit color as yuv420p, 16-bit depth as gray12le)
*without ever blocking the poll thread on disk I/O*.

Timestamps are recorded only for frames that are actually written, so
``{key}_timestamps`` in the NPZ stays 1:1 with the container's frame indices
— when the writer falls behind, the oldest queued frames are dropped instead
of letting the recording drift behind real time.

Output (under ``<session>/<output_dir>/``)::

    {key}.mp4                    one per video stream
    {output_dir}.npz             {key}_timestamps (N,) float64, PC clock

The NPZ is written once at close, so a hard kill before ``_save`` loses the
timestamps even though the mp4 survives.
"""

import queue
import threading
from pathlib import Path

import numpy as np

from ..base import BaseRecorder
from ..ffmpeg_writer import FFmpegWriter

_FPS_FALLBACK = 30.0


class BaseCameraRecorder(BaseRecorder):
    name = "camera"
    output_dir = "camera"
    role: str = ""

    def __init__(self, config):
        super().__init__(config)
        self._write_failed: set[str] = set()      # keys with OSError so far
        # Async write pipeline: per-stream bounded queue + writer thread,
        # so a slow stream (16-bit depth) never stalls another (color).
        self._write_queues: dict[str, queue.Queue] = {}
        self._write_threads: dict[str, threading.Thread] = {}
        self._video_writers: dict[str, FFmpegWriter] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped: dict[str, int] = {}
        self._written: dict[str, int] = {}
        self._last_shapes: dict[str, tuple] = {}
        self._frames_enqueued = 0
        self._frames_precommit = 0

    # ---- async frame write pipeline (libx265 HEVC) --------------------------

    def _start_write_worker(self, key: str) -> None:
        if key in self._write_threads:
            return
        q = self._write_queues.setdefault(key, queue.Queue(maxsize=8))
        t = threading.Thread(
            target=self._write_worker, args=(key, q),
            name=f"{self.name}-{key}-writer", daemon=True)
        self._write_threads[key] = t
        t.start()

    def _write_fps(self) -> float:
        """Container framerate for the mp4 (also its keyframe interval).
        Subclasses whose config names it differently (the EGO headband's
        ``cam_fps``) override this."""
        return float(getattr(self.config, "fps", None)
                     or getattr(self.config, "cam_fps_hint", None)
                     or _FPS_FALLBACK)

    def _write_worker(self, key: str, q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is None:  # shutdown sentinel
                break
            ts, data = item
            try:
                w = self._video_writers.get(key)
                if w is None:
                    out_dir = Path(self.config.session_dir) / self.output_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{key}.mp4"
                    w = self._make_writer(path, self._write_fps(), data)
                    self._video_writers[key] = w
                    self._video_paths[key] = path
                w.write(data)
                # Timestamps ONLY for frames actually written -> 1:1 with
                # the container's frame indices even when frames drop.
                self._acc_ts(key, ts)
                self._written[key] = self._written.get(key, 0) + 1
                if isinstance(data, np.ndarray):
                    self._last_shapes[key] = data.shape
            except (OSError, RuntimeError) as exc:
                self._warn_write(key, exc)

    def _make_writer(self, path: Path, fps: float, data: np.ndarray):
        """Build the mp4 writer for a stream's first frame.

        The base handles raw frames (HxWx3 RGB -> yuv420p, or 16-bit depth ->
        gray12le) via the shared FFmpegWriter/libx265.  A subclass that receives
        compressed frames (the EGO headband's Motion-JPEG) overrides this to add
        its own writer — see ``BaseEgoHeadbandRecorder._make_writer`` — so the
        generic camera base never needs to know about JPEG or GPU encoding.
        """
        # 8-bit RGB color -> yuv420p; 16-bit depth -> 12-bit HEVC (gray12le).
        # bframes=0 + keyint=fps (see FFmpegWriter): frame i in the container
        # == timestamp i — no inter-frame interpolation.
        if data.ndim == 3 and data.shape[2] == 3:
            inp, outp = "rgb24", "yuv420p"
        else:
            inp, outp = "gray16le", "gray12le"
        h, width = data.shape[:2]
        w = FFmpegWriter(
            path, width, h, fps,
            input_pix_fmt=inp, output_pix_fmt=outp,
            crf=int(self.config.crf), preset=str(self.config.preset))
        self._log(f"[{self.name}] {path.stem} mp4 -> {path} ({width}x{h})")
        return w

    def _recording_signal(self) -> int:
        """帧走异步写盘管线,不进 _count_samples 的三类缓冲(写入失败时
        更不可见)—— 确认阶段以投喂进管线的帧数为准;预热期直接丢弃的
        commit 前帧也计入(证明数据在流动)。"""
        return self._frames_enqueued + self._frames_precommit

    def _enqueue(self, key: str, ts: float, data) -> None:
        """Non-blocking put of one frame (raw ndarray OR JPEG bytes) into the
        stream's write queue.  The poll loop never waits on disk I/O; when the
        writer falls behind, the OLDEST queued frame is dropped so the recording
        stays synchronized with real time.

        两段式启动:commit 前的预热帧**不进 mp4**——容器一旦写入就无法
        回退,而 commit 会清掉预热时间戳,先写后清会让 mp4 永远比
        ``{key}_timestamps`` 多出预热帧数(历史教训)。丢弃后容器的第一帧
        就是 commit 后第一帧,1:1 严格成立;确认阶段照常计数。
        """
        if self._launch_mode and not self._committed:
            self._frames_precommit += 1
            return
        self._start_write_worker(key)
        self._frames_enqueued += 1
        q = self._write_queues[key]
        try:
            q.put_nowait((ts, data))
        except queue.Full:
            try:
                q.get_nowait()  # drop the oldest
            except queue.Empty:
                pass
            q.put_nowait((ts, data))
            self._dropped[key] = self._dropped.get(key, 0) + 1

    def arr_video(self, key: str, ts: float, arr: np.ndarray) -> None:
        """Feed one decoded frame (HxWx3 RGB or HxW depth) into the stream's
        async write pipeline.  See ``_enqueue`` for backpressure/commit rules."""
        self._enqueue(key, ts, arr)

    def arr_jpeg(self, key: str, ts: float, jpeg: bytes) -> None:
        """Feed one Motion-JPEG frame into the stream's async GPU-transcode
        pipeline.  The frame stays compressed — ffmpeg decodes+encodes it (on
        the GPU when available, see FFmpegJpegWriter), so a JPEG-streaming
        device (the EGO headband's four fisheye cameras) never pays a Python
        ``cv2.imdecode`` per frame.  Same backpressure / 1:1-timestamp contract
        as ``arr_video`` (see ``_enqueue``)."""
        self._enqueue(key, ts, jpeg)

    def _stop_write_worker(self) -> None:
        for key, q in list(self._write_queues.items()):
            # Make room for the sentinel (dropping the oldest queued frames
            # — the bounded tail is lost at shutdown by design).
            while True:
                try:
                    q.put_nowait(None)  # sentinel
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        for t in self._write_threads.values():
            t.join(timeout=10.0)
        self._write_threads.clear()
        self._write_queues.clear()
        for key, w in self._video_writers.items():
            try:
                w.close()   # finalize the HEVC container
            except Exception as exc:
                # finalize 失败 = 容器没写 moov,文件读不出。必须当场喊出来:
                # 否则下面的汇总照印 "(N frames, X MB)",坏文件要等到打包
                # 预扫才暴露,且现场再无线索。
                self._log(f"[{self.name}] {key} mp4 finalize FAILED — "
                          f"file may be unplayable: {exc}")
        self._video_writers.clear()
        for key, n in sorted(self._dropped.items()):
            if n:
                self._log(f"[{self.name}] {key} writer behind — dropped "
                          f"{n} frames (timestamps stay 1:1 with written "
                          f"frames)")

    def _warn_write(self, key: str, exc: Exception) -> None:
        if key not in self._write_failed:
            self._log(f"[{self.name}] {key} disk write failed — {exc}; "
                      "affected video frames are lost (timestamps stay 1:1 "
                      "with written frames)")
            self._write_failed.add(key)

    def _heartbeat_stats(self, elapsed: float) -> str:
        """Written-frame counts + latest frame shapes (mp4 streams)."""
        parts = [super()._heartbeat_stats(elapsed)]
        for key, n in sorted(self._written.items()):
            shape = tuple(self._last_shapes.get(key, ()))
            parts.append(f"{key}={n}{shape}")
        return "  ".join(parts)

    # ---- finalize at close --------------------------------------------------

    def _save(self) -> None:
        """Stop the write worker (flush tail, finalize mp4s), then write the
        timestamps NPZ and the final report."""
        if not self.config.session_dir:
            return

        # The writer threads own the mp4s — stop them BEFORE reading the
        # timestamp buffers they feed.
        self._stop_write_worker()

        ts_payload = {}
        for cam, ts_list in self._ts_buf.items():
            ts_payload[f"{cam}_timestamps"] = (
                np.asarray(ts_list, dtype=np.float64)
                if ts_list else np.zeros(0, dtype=np.float64))
        if ts_payload:
            out_dir = Path(self.config.session_dir) / self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ts_path = out_dir / f"{self.output_dir}.npz"
            np.savez(ts_path, **ts_payload)
            self._log(f"[{self.name}] saved timestamps -> {ts_path}")

        for key, path in sorted(self._video_paths.items()):
            size = path.stat().st_size / 1e6 if path.is_file() else 0.0
            n = self._written.get(key, 0)
            self._log(f"[{self.name}] {key} -> {path} "
                      f"({n} frames, {size:.1f} MB)")

        if not (self._written or ts_payload):
            self._log(f"[{self.name}] nothing to save.")
