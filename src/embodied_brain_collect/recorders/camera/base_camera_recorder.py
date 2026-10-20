"""Abstract camera base.

Frame writing is decoupled from the poll loop: concrete recorders just call
``arr_video(key, ts, arr)`` per frame.  A per-stream bounded queue + a
dedicated writer thread turn frames into ``{key}.mp4`` (HEVC via the shared FFmpegWriter — hevc_nvenc on the GPU when
available, libx265 otherwise; 8-bit color as yuv420p, 16-bit depth as
gray12le always on libx265)
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
import socket
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
        # 辅助员控制台预览:每个流一个 UDP 目标端口(_preview_ports),失败
        # 从不影响录制。socket 不可 pickle —— spawn 时由 __getstate__ 剔除,
        # 子进程里按需惰性重建。
        self._preview_sock = None
        self._preview_last: dict[str, float] = {}
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

    def _write_queue_size(self) -> int:
        """Depth of the per-stream write queue (frames held between the poll
        loop and ffmpeg).  8 keeps RAM tight for raw-frame cameras (a depth
        frame is tens of MB); anything past it drops the oldest frame by
        design — the recording must never fall behind real time."""
        return 8

    def _start_write_worker(self, key: str, warm_data=None) -> None:
        """启动 ``key`` 的写盘线程。``warm_data`` 非 None 时(该流第一帧),
        线程先按这帧的 shape/像素格式把 FFmpegWriter(nvenc 探测 + ffmpeg
        进程拉起,本机实测 ~0.8s)构造好,**不写这一帧** —— 预热期暖管,
        commit 后第一帧即写,不再有冷启动空窗。"""
        if key in self._write_threads:
            return
        q = self._write_queues.setdefault(
            key, queue.Queue(maxsize=self._write_queue_size()))
        t = threading.Thread(
            target=self._write_worker, args=(key, q, warm_data),
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

    def _write_worker(self, key: str, q: queue.Queue,
                      warm_data=None) -> None:
        # 暖管:预热期第一帧只用来确定 shape/像素格式,先构造 writer、
        # 不写帧 —— 构造失败按写失败处理(_write_failed → _enqueue 抛错)。
        if warm_data is not None:
            try:
                self._build_writer(key, warm_data)
            except (OSError, RuntimeError) as exc:
                self._warn_write(key, exc)
        while True:
            item = q.get()
            if item is None:  # shutdown sentinel
                break
            ts, data = item
            try:
                w = self._video_writers.get(key)
                if w is None:
                    w = self._build_writer(key, data)
                w.write(data)
                # Timestamps ONLY for frames actually written -> 1:1 with
                # the container's frame indices even when frames drop.
                self._acc_ts(key, ts)
                self._written[key] = self._written.get(key, 0) + 1
                if isinstance(data, np.ndarray):
                    self._last_shapes[key] = data.shape
            except (OSError, RuntimeError) as exc:
                self._warn_write(key, exc)

    def _build_writer(self, key: str, data) -> FFmpegWriter:
        """按 ``data`` 的 shape/格式构造并登记 ``key`` 的 mp4 writer。"""
        out_dir = Path(self.config.session_dir) / self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{key}.mp4"
        w = self._make_writer(path, self._write_fps(), data)
        self._video_writers[key] = w
        self._video_paths[key] = path
        return w

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
            crf=int(self.config.crf), preset=str(self.config.preset),
            encoder=str(getattr(self.config, "encoder", "auto")))
        self._log(f"[{self.name}] {path.stem} mp4 -> {path} "
                  f"({width}x{h}, {w.encoder_used})")
        return w

    def _recording_signal(self) -> int:
        """帧走异步写盘管线,不进 _count_samples 的三类缓冲(写入失败时
        更不可见)—— 确认阶段以投喂进管线的帧数为准;预热期直接丢弃的
        commit 前帧也计入(证明数据在流动)。"""
        return self._frames_enqueued + self._frames_precommit

    def _ready_to_commit(self) -> bool:
        """所有已启动的写盘线程都完成了 writer 构造(ffmpeg 已拉起、
        nvenc 探测已缓存)。launcher 的确认阶段等它翻真才广播 commit,
        保证 commit→RUN_START 之间没有冷启动空窗。"""
        return all(key in self._video_writers for key in self._write_threads)

    def discard_recording(self) -> None:
        """abort 路径:预热期可能已经暖管拉起了 ffmpeg —— 收掉写盘线程,
        删掉 0 帧 mp4。否则中止的 session 目录里留着空 mp4,has_any_data
        会把"空录制"误判成"有数据",汇总口径失真。"""
        super().discard_recording()      # _save_enabled = False(此后 _enqueue 直接丢帧)
        self._stop_write_worker()
        for path in list(self._video_paths.values()):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._video_paths.clear()

    def _enqueue(self, key: str, ts: float, data) -> None:
        """Non-blocking put of one frame (raw ndarray OR JPEG bytes) into the
        stream's write queue.  The poll loop never waits on disk I/O; when the
        writer falls behind, the OLDEST queued frame is dropped so the recording
        stays synchronized with real time.

        两段式启动:commit 前的预热帧**不进 mp4**——容器一旦写入就无法
        回退,而 commit 会清掉预热时间戳,先写后清会让 mp4 永远比
        ``{key}_timestamps`` 多出预热帧数(历史教训)。丢弃后容器的第一帧
        就是 commit 后第一帧,1:1 严格成立;确认阶段照常计数。

        预热第一帧只用来**暖管**:写盘线程先按它的 shape 把 FFmpegWriter
        构造好(nvenc 探测 + ffmpeg 进程,~0.8s,在预热期消化),确认
        阶段经 :meth:`_ready_to_commit` 等 writer 就绪 —— launcher 才在
        管线全热时广播 commit。commit 后第一帧即写,RUN_START(commit 后
        ~0.6s)不再落进冷启动空窗(历史教训:懒构造期间 8 帧队列溢出,
        每路丢 ~30 帧,QC 报首端 TimestampGap)。
        """
        if not self._save_enabled:
            # 已 discard(launch 中止):帧直接扔掉,不再重启写盘线程
            return
        if self._write_failed:
            # 写盘线程已报过显式错误(ffmpeg died / 磁盘写失败)→ 原地
            # 抛错终止:继续录只会产出没有视频内容、只剩时间戳的数据
            raise RuntimeError(
                "视频写盘管线已故障"
                f"({', '.join(sorted(self._write_failed))})— 见写失败日志,"
                "停止录制")
        self._start_write_worker(key, warm_data=data)
        if self._launch_mode and not self._committed:
            self._frames_precommit += 1
            return
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

    # ---- assist-console preview (UDP JPEG, best effort) --------------------

    def _preview_ports(self) -> dict[str, int]:
        """``{流名: UDP 端口}`` —— 要推预览的流。默认取单口配置
        ``preview_port``(映射到本 recorder 的首个流名,见
        :attr:`preview_stream`);多流设备(头环四路鱼眼)覆盖本方法按
        流名分别给端口。0/缺省 = 该流不推。"""
        port = int(getattr(self.config, "preview_port", 0) or 0)
        if port <= 0:
            return {}
        return {self.preview_stream: port}

    #: 单流相机的流名(与 ``arr_video`` 的 key 一致)
    preview_stream = "frames"

    def _preview_due(self, key: str, now: float) -> bool:
        """按 ``preview_hz`` 限流 —— 预览帧率在采集帧率之上没有意义。"""
        hz = float(getattr(self.config, "preview_hz", 30.0) or 30.0)
        if now - self._preview_last.get(key, 0.0) < 1.0 / max(hz, 0.1):
            return False
        self._preview_last[key] = now
        return True

    #: UDP 单包上限(IPv4 datagram 载荷 65507)。超限的 sendto 会直接报错,
    #: 预览就永远停在"等待预览" —— 宁可丢这一帧也不能静默失效。
    _PREVIEW_MAX_DATAGRAM = 65000

    def _preview_send(self, key: str, jpeg: bytes) -> None:
        """把一帧 JPEG 推给该流配置的端口;任何失败都静默(预览非录制)。"""
        port = self._preview_ports().get(key, 0)
        if port <= 0 or not jpeg:
            return
        if len(jpeg) > self._PREVIEW_MAX_DATAGRAM:
            return
        try:
            if self._preview_sock is None:
                self._preview_sock = socket.socket(socket.AF_INET,
                                                   socket.SOCK_DGRAM)
            self._preview_sock.sendto(jpeg, ("127.0.0.1", port))
        except OSError:
            pass          # 单包超限/瞬断:预览丢帧无碍录制

    def preview_raw(self, key: str, now: float, frame_bgr) -> None:
        """原始帧入口:降采样 + JPEG 编码后推送(限流后才编码)。"""
        self._preview_encode(key, now, frame_bgr, rgb=False)

    def preview_rgb(self, key: str, now: float, frame_rgb) -> None:
        """RGB 帧入口(头环 H.264 解码输出是 rgb24):同 :meth:`preview_raw`,
        只是编码前补一次 RGB→BGR —— 控制台按 BGR 解 JPEG,不转红蓝互换。"""
        self._preview_encode(key, now, frame_rgb, rgb=True)

    def _preview_encode(self, key: str, now: float, frame, *,
                        rgb: bool, throttled: bool = False) -> None:
        """降采样 + JPEG 编码后推送。``throttled`` = 调用方已过限流闸门
        (:meth:`_preview_due` 会记时,再问一次必然为假、白丢这一帧)。"""
        if self._preview_ports().get(key, 0) <= 0:
            return
        if not throttled and not self._preview_due(key, now):
            return
        try:
            import cv2
            h, w = frame.shape[:2]
            pw = min(int(getattr(self.config, "preview_width", 640)), w)
            ph = max(1, round(h * pw / w))
            small = cv2.resize(frame, (pw, ph)) if pw != w else frame
            if rgb:
                small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            # 640 宽 + q70 通常是 30-60KB,但高细节画面(鱼眼全景、噪点)
            # 能到 120KB —— 一个 UDP 包装不下就整帧消失。降质量重编码直到
            # 塞得进去:预览画质可以让,预览出不出画面不能让。
            for quality in (70, 50, 30):
                ok, enc = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    return
                if len(enc) <= self._PREVIEW_MAX_DATAGRAM:
                    self._preview_send(key, enc.tobytes())
                    return
        except Exception:
            pass

    def preview_jpeg(self, key: str, now: float, jpeg: bytes) -> None:
        """已压缩帧入口(头环鱼眼 MJPEG):能塞进一个 UDP 包就直接转发,
        不解码不重编码;装不下才解码降采样重编码 —— 1920x1088 的鱼眼帧
        有 200-400KB,远超单包上限,不降采样等于预览永远不出画面。
        限流同 :meth:`preview_raw`。"""
        if self._preview_ports().get(key, 0) <= 0:
            return
        if not self._preview_due(key, now):
            return
        if len(jpeg) <= self._PREVIEW_MAX_DATAGRAM:
            self._preview_send(key, jpeg)
            return
        try:
            import cv2
            arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        except Exception:
            return
        if arr is not None:
            self._preview_encode(key, now, arr, rgb=False, throttled=True)

    def _close_preview(self) -> None:
        if self._preview_sock is not None:
            try:
                self._preview_sock.close()
            except OSError:
                pass
            self._preview_sock = None

    def __getstate__(self) -> dict:
        # socket 不可 pickle(spawn 下父进程会把 recorder 整体传给子进程)
        state = super().__getstate__()
        state["_preview_sock"] = None
        return state

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
