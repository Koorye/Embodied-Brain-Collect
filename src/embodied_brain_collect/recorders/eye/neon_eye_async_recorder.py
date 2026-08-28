"""Pupil Labs Neon — async full-rate API with independent stream clocks.

Uses asyncio + 3 concurrent tasks so every gaze/IMU/scene sample is captured
without the simple-API queue-of-size-1 frame loss.
"""

import asyncio
import logging
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import signal
import threading
from .base_eye_recorder import BaseEyeRecorder
from .eye_recorder_config import EyeRecorderConfig
from ..ffmpeg_writer import FFmpegWriter

sys.stdout.reconfigure(line_buffering=True)


from pupil_labs.realtime_api.discovery import Network
from pupil_labs.realtime_api import Device
from pupil_labs.realtime_api.streaming import (
    receive_gaze_data, receive_imu_data, receive_video_frames)
from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator


def install_asyncio_signal_shutdown() -> threading.Event:
    """Return a ``threading.Event`` that is set when SIGBREAK / SIGTERM
    arrives, so an asyncio-based recorder can shut down gracefully.

    Background
    ----------
    On Windows ``signal.default_int_handler`` only schedules a
    ``KeyboardInterrupt`` to be raised at the next Python bytecode
    boundary.  Inside the ``ProactorEventLoop`` the main thread is
    typically blocked in an IOCP wait while real network I/O is flowing
    (pupil_labs gaze/IMU/scene streams in our case), and that wait
    doesn't return to bytecode for long stretches.  Result: the
    ``KeyboardInterrupt`` never fires and the recorder is hard-killed by
    the launcher's flush timeout, losing all of its data.

    Workaround
    ----------
    Replace the SIGBREAK / SIGTERM handlers with ones that *also* set a
    ``threading.Event``.  An asyncio task (which by definition runs
    between bytecode-level await points) polls that event every ~100 ms
    and triggers ``stop.set()`` from inside the loop, so the recorder
    can reach its ``finally`` clause and flush data to disk.

    The returned Event is safe to import / inspect from any thread.
    Setting ``KeyboardInterrupt`` is still attempted as a fallback in
    case the recorder happens to be at a bytecode boundary when the
    signal arrives.
    """
    evt = threading.Event()

    def _handler(signum, _frame):
        evt.set()
        # Best-effort: also raise KeyboardInterrupt at the next bytecode
        # boundary.  On Windows asyncio this is almost certain to be a
        # no-op (see docstring) but it's free and helps the POSIX path.
        raise KeyboardInterrupt

    for name in ("SIGBREAK", "SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
    return evt


class NeonEyeAsyncRecorder(BaseEyeRecorder):
    """Pupil Labs Neon — async, full-rate.  Overrides ``_record`` for asyncio."""

    config: EyeRecorderConfig

    def __init__(self, config: EyeRecorderConfig):
        super().__init__(config)
        self._writer: FFmpegWriter | None = None
        self._out_mp4: Path | None = None
        self._last_scene_frame: np.ndarray | None = None  # BGR, for GUI tests
        self._device_info = None   # cached by _open(), reused by _record()
        self._clock_offset_s: float = 0.0    # pc_to_phone offset, applied
        # to every device timestamp so eye stays in the PC clock domain
        # shared by all other recorders.

        # Warm-standby plumbing: one persistent event-loop thread spans
        # _open() (connect + first-data gate, streams kept RUNNING) and
        # _record() (go -> start recording).  The scene camera cold start
        # (~5 s) therefore happens exactly once, during the open gate.
        # 流任务只负责采集并把样本投喂进队列;go 时队列清空,录制循环
        # _drain() 唯一的工作就是把队列 acc 下来——go 后第一个样本立刻
        # 入库,录制阶段没有任何启动开销。
        self._recording: bool = False
        # 仅常驻循环路径开启:直接驱动流任务(如 tests/eye GUI 测试)时,
        # 任务不受 standby 丢弃逻辑影响,行为与旧版一致。
        self._standby_mode: bool = False
        self._ready_evt = threading.Event()     # open result available
        self._open_ok: bool = False             # written by the loop thread
        self._start_evt = threading.Event()     # go: 开始录制
        self._loop_done = threading.Event()     # persistent loop exited
        self._record_rc: int = 0
        self._first_gaze_evt = threading.Event()
        self._first_imu_evt = threading.Event()
        self._first_scene_evt = threading.Event()
        # 采集队列(主循环线程内使用,无需加锁):standby 期间也在投喂,
        # go 时整体清空,保证录制从 go 那一刻才开始。
        self._gaze_q: deque = deque(maxlen=4096)    # (ts_pc, xy float32)
        self._imu_q: deque = deque(maxlen=4096)     # (ts_pc, gyro, accel)
        self._scene_ts_q: deque = deque(maxlen=4096)  # no_scene_video 时间戳

    # ==================================================================
    # _record() — asyncio entry point
    # ==================================================================

    def _record(self) -> None:
        """Go: 通知主循环开始录制(采集队列 → _drain → acc),然后等待
        录制结束(stop_event / duration / signal)。主循环在 _open() 时已
        启动并持续采集,这里没有任何连接/启动开销。"""
        self._start_evt.set()
        self._loop_done.wait()
        if self._record_rc != 0:
            self.logger.error(
                f"[eye:neon] recording ended with rc={self._record_rc}")

    # ==================================================================
    # Persistent loop thread (open gate + recording share one event loop)
    # ==================================================================

    def _loop_main(self) -> None:
        """Entry point of the persistent event-loop thread."""
        try:
            asyncio.run(self._standby_and_record())
        except Exception as exc:
            self._log(f"[eye:neon] loop died: {type(exc).__name__}: {exc}", level="ERROR")
            traceback.print_exc()
            if not self._ready_evt.is_set():
                self._open_error = f"{type(exc).__name__}: {exc}"
                self._open_ok = False
                self._ready_evt.set()
        finally:
            self._loop_done.set()

    async def _standby_and_record(self) -> None:
        """Connect once, start the streams in discard mode, run the
        first-data gate, then wait for go and accumulate until stop.

        Signal handlers are installed in _open() on the MAIN thread
        (signal.signal fails silently in this worker thread) and arrive
        here via self._shutdown.
        """
        rc = 0

        try:
            # Reuse the device found during discovery when possible.
            device_info = self._device_info
            if device_info is None:
                device_info = await Network().wait_for_new_device(
                    timeout_seconds=10)
                self._device_info = device_info
            async with Device(address=device_info.addresses[0], port=device_info.port) as dev:
                status = await dev.get_status()
                offset_ms = await self._sync_clock(status)
                self._acc("pc_to_phone_offset_ms", offset_ms)
                # Device timestamps live on the phone clock; convert them to
                # the PC clock domain (t_pc = t_phone + offset_ms/1000) so
                # eye aligns with every other recorder on the common timeline.
                self._clock_offset_s = offset_ms / 1000.0

                sensors = await self._await_sensors(dev)
                if sensors is None:
                    self._open_error = "sensor(s) not connected"
                    self._ready_evt.set()
                    return
                gs, ims, ws = sensors  # (gaze, imu, world)

                stop = asyncio.Event()
                self._standby_mode = True    # 采集任务持续投喂队列(go 时
                                             # 清空);scene 视频 go 前不解码
                named = [
                    ("gaze",     self._gaze_task(gs.url, stop)),
                    ("imu",      self._imu_task(ims.url, stop)),
                    ("scene",    self._scene_task(ws.url, stop)),
                    ("progress", self._progress_task(stop)),
                    ("signal",   self._signal_task(stop, self._shutdown)),
                ]
                tasks = [asyncio.create_task(self._guarded(n, c))
                         for n, c in named]

                # ---- first-data gate (streams stay running afterwards) ----
                for evt, err in (
                    (self._first_gaze_evt, "no gaze sample within 20 s"),
                    (self._first_imu_evt, "no IMU sample within 20 s"),
                    (self._first_scene_evt, "no scene frame within 20 s"),
                ):
                    if not await self._wait_first(evt, 20.0, err):
                        self._open_ok = False
                        self._ready_evt.set()
                        stop.set()
                        for t in tasks:
                            t.cancel()
                        await asyncio.gather(*tasks,
                                             return_exceptions=True)
                        return
                self._open_ok = True
                self._ready_evt.set()
                self._log("[eye:neon] standby warm — streams running, "
                          "waiting for go (capture-only, dropped at go)")

                # ---- wait for go, or abort ----
                while not self._start_evt.is_set():
                    if stop.is_set() or self.stop_event.is_set():
                        return    # 中止:standby 期间不写任何数据
                    await asyncio.sleep(0.05)
                self._recording = True
                # go 之前队列里全是 standby 数据——清空,录制从 go 开始。
                self._gaze_q.clear()
                self._imu_q.clear()
                self._scene_ts_q.clear()
                self._log("[eye:neon] go — recording")

                # ---- 录制循环:唯一的数据工作就是 _drain() 把采集队列
                # acc 下来,外加监督流任务是否异常退出 ----
                while True:
                    if stop.is_set() or self.stop_event.is_set():
                        break
                    self._drain()
                    for t in tasks:
                        if t.done() and t.exception() is not None:
                            self._log(f"[eye:neon] stream task error — {type(t.exception()).__name__}", level="ERROR")
                            rc = 1
                            stop.set()
                            break
                    await asyncio.sleep(0.01)
                self._drain()    # 停表前的最后一小批也收下

        except KeyboardInterrupt:
            self._log("[eye:neon] Ctrl+C (outer)", level="WARNING")
        except Exception as exc:
            self._log(f"[eye:neon] {type(exc).__name__}: {exc}", level="ERROR")
            traceback.print_exc()
            rc = 1
            if not self._ready_evt.is_set():
                self._open_error = f"{type(exc).__name__}: {exc}"
                self._ready_evt.set()
        finally:
            if self._start_evt.is_set():
                self._teardown()    # 正式开录过:关流 + 保存
            else:
                self._close()       # 中止路径:不写空数据文件
            self._record_rc = rc

    def _drain(self) -> None:
        """录制循环唯一的数据工作:把采集任务投喂进队列的样本 acc 存下来。

        采集任务(主循环在 open 时启动)在 standby 期间也持续投喂;go 时
        队列已清空,所以这里 acc 的全是 go 之后的数据——go 后第一个样本
        立刻入库,没有任何启动延迟。单线程 asyncio 内无 await,遍历安全。
        """
        for ts, xy in self._gaze_q:
            self._acc("gaze_timestamps", ts)
            self._acc_arr("gaze_xy", xy)
        self._gaze_q.clear()
        for ts, gyro, accel in self._imu_q:
            self._acc("imu_timestamps", ts)
            self._acc_arr("imu_gyro", gyro)
            self._acc_arr("imu_accel", accel)
        self._imu_q.clear()
        for ts in self._scene_ts_q:
            self._acc("scene_timestamps", ts)
        self._scene_ts_q.clear()

    async def _wait_first(self, evt: threading.Event, timeout: float,
                          err: str) -> bool:
        """Poll a first-sample event; False on timeout or abort."""
        deadline = time.time() + timeout
        while not evt.is_set():
            if time.time() > deadline:
                self._open_error = err
                self.logger.error(f"[eye:neon] open failed — {err}")
                return False
            if self.stop_event.is_set():
                return False
            await asyncio.sleep(0.05)
        return True

    async def _await_sensors(self, dev, timeout: float = 10.0):
        """Poll status until gaze/imu/world sensors all report connected.

        direct_*_sensor() returns a default Sensor(connected=False) instead
        of None when missing, so an ``is None`` check never fires.  A world
        sensor that connects a moment after the HTTP API responds would
        otherwise start the scene stream with url=None and die silently.
        """
        names = ("gaze", "imu", "world")
        t0 = time.time()
        while True:
            status = await dev.get_status()
            got = (status.direct_gaze_sensor(), status.direct_imu_sensor(),
                   status.direct_world_sensor())
            missing = [n for n, s in zip(names, got)
                       if s is None or not s.connected or s.url is None]
            if not missing:
                return got
            print(f"\r[eye:neon] waiting for sensor(s): {', '.join(missing)} "
                  f"...", end="", flush=True)
            if time.time() - t0 >= timeout:
                self._log(
                    f"[eye:neon] sensor(s) not connected after {timeout:g}s: "
                    f"{', '.join(missing)} — open the Neon Camera app on the phone",
                    level="WARNING")
                return None
            await asyncio.sleep(1.0)

    async def _guarded(self, name: str, coro):
        """Print immediately when a stream task dies; re-raise so the
        gather inspection also sees it."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"[eye:neon] {name} stream failed: {type(exc).__name__}: {exc}", level="ERROR")
            traceback.print_exc()
            raise

    # ==================================================================
    # Stream tasks
    # ==================================================================

    async def _sync_clock(self, status) -> float:
        est = await TimeOffsetEstimator(
            status.phone.ip, status.phone.time_echo_port,
        ).estimate(number_of_measurements=10)
        ms = float(est.time_offset_ms.mean) if est else 0.0
        self._log(f"[eye:neon] {status.phone.device_name} "
                      f"battery={status.phone.battery_level}%  "
                      f"pc_to_phone_offset={ms:.2f}ms", echo=False)
        print(
            f"[eye:neon] {status.phone.device_name} "
            f"battery={status.phone.battery_level}%  "
            f"pc_to_phone_offset={ms:.2f}ms"
        )
        return ms

    async def _gaze_task(self, url, stop):
        # run_loop=True: aiortsp auto-reconnects on transient drops (the
        # same option the simple API's stream manager uses).
        async for g in receive_gaze_data(url, run_loop=True,
                                         log_level=logging.WARNING):
            if stop.is_set():
                break
            self._first_gaze_evt.set()
            ts = g.timestamp_unix_seconds + self._clock_offset_s
            xy = np.array([g.x, g.y], dtype=np.float32)
            if self._standby_mode:
                self._gaze_q.append((ts, xy))    # 由录制循环 _drain 存
            else:
                # 直接驱动路径(如 tests/eye GUI 测试):像旧版一样直接 acc
                self._acc("gaze_timestamps", ts)
                self._acc_arr("gaze_xy", xy)

    async def _imu_task(self, url, stop):
        async for d in receive_imu_data(url, run_loop=True,
                                        log_level=logging.WARNING):
            if stop.is_set():
                break
            self._first_imu_evt.set()
            ts = d.timestamp_unix_seconds + self._clock_offset_s
            gyro = np.array([d.gyro_data.x, d.gyro_data.y, d.gyro_data.z],
                            dtype=np.float32)
            accel = np.array([d.accel_data.x, d.accel_data.y, d.accel_data.z],
                             dtype=np.float32)
            if self._standby_mode:
                self._imu_q.append((ts, gyro, accel))    # 由 _drain 存
            else:
                # 直接驱动路径(如 tests/eye GUI 测试):像旧版一样直接 acc
                self._acc("imu_timestamps", ts)
                self._acc_arr("imu_gyro", gyro)
                self._acc_arr("imu_accel", accel)

    async def _scene_task(self, url, stop):
        loop = asyncio.get_running_loop()
        self._log("[eye:neon] scene stream opening — first frame takes ~5 s "
                      "while the camera starts ...", echo=False)
        if self.config.no_scene_video:
            async for f in receive_video_frames(url, run_loop=True,
                                                log_level=logging.WARNING):
                if stop.is_set():
                    break
                self._first_scene_evt.set()
                ts = f.timestamp_unix_seconds + self._clock_offset_s
                if self._standby_mode:
                    self._scene_ts_q.append(ts)    # 由录制循环 _drain 存
                else:
                    # 直接驱动路径:像旧版一样直接 acc
                    self._acc("scene_timestamps", ts)
            return

        # Decode and mp4-write are decoupled: the decode loop below only
        # feeds the freshest frame to the preview and a bounded queue; the
        # writer task drains at its own pace and drops the OLDEST frames
        # when it falls behind.  A serial decode->write pipeline instead
        # replays its backlog FIFO — the preview (and the recorded mp4)
        # lag real time more and more and never catch up.
        queue = asyncio.Queue(maxsize=4)
        writer_task = asyncio.create_task(self._scene_writer(queue, stop))
        dropped = 0
        try:
            async for f in receive_video_frames(url, run_loop=True,
                                                log_level=logging.WARNING):
                if stop.is_set():
                    break
                self._first_scene_evt.set()
                if self._standby_mode and not self._recording:
                    continue    # standby: discard (no decode, no write)
                # Decode in a worker thread (loop stays free for UDP).
                img = await loop.run_in_executor(None, f.bgr_buffer)
                self._last_scene_frame = img
                ts_pc = f.timestamp_unix_seconds + self._clock_offset_s
                try:
                    queue.put_nowait((ts_pc, img))
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()  # drop the oldest
                    except asyncio.QueueEmpty:
                        pass
                    queue.put_nowait((ts_pc, img))
                    dropped += 1
        finally:
            if dropped:
                self._log(f"[eye:neon] scene writer behind — dropped {dropped} frames "
                      f"from eye.mp4 (timestamps stay 1:1 with written frames)",
                      level="WARNING")
            print(f"\n[eye:neon] scene writer behind — dropped "
                      f"{dropped} frames from eye.mp4 (timestamps stay 1:1 "
                      f"with written frames)")
            await writer_task

    async def _scene_writer(self, queue, stop):
        """Drain the decoded-frame queue into eye.mp4 at its own pace.

        Timestamps are recorded HERE, for written frames only, so
        eye.mp4 stays 1:1 with scene_timestamps even when frames drop.
        """
        loop = asyncio.get_running_loop()
        while not stop.is_set() or not queue.empty():
            try:
                ts, img = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self._writer is None:
                h, w = img.shape[:2]
                out_dir = Path(self.config.session_dir) / self.output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                self._out_mp4 = out_dir / "eye.mp4"
                # Frame-exact libx265 (bframes=0, CFR, keyint=fps) — same
                # FFmpegWriter the camera recorders use.  Frames are BGR
                # (f.bgr_buffer()).
                self._writer = FFmpegWriter(
                    self._out_mp4, w, h, 30.0,
                    input_pix_fmt="bgr24", output_pix_fmt="yuv420p",
                    crf=int(self.config.crf), preset=str(self.config.preset))
                print(f"\n[eye:neon] scene mp4 -> {self._out_mp4} ({w}x{h})")
            self._acc("scene_timestamps", ts)
            await loop.run_in_executor(None, self._writer.write, img)
            queue.task_done()

    async def _progress_task(self, stop):
        t0 = None    # duration clock starts at the go signal
        while not stop.is_set():
            await asyncio.sleep(0.5)
            if not self._recording:
                print("\r[eye:neon] standby (warm) ...", end="", flush=True)
                continue
            if t0 is None:
                t0 = time.time()
            elapsed = time.time() - t0
            ng = len(self._buf.get("gaze_timestamps", []))
            ni = len(self._buf.get("imu_timestamps", []))
            ns = len(self._buf.get("scene_timestamps", []))
            print(
                f"\r[eye:neon] t={elapsed:5.1f}s  "
                f"gaze={ng:>6} imu={ni:>6} scene={ns:>5}",
                end="", flush=True,
            )
            if 0 < self.config.duration <= elapsed:
                stop.set()

    async def _signal_task(self, stop, shutdown):
        """Watch the signal-handler event AND the launcher's stop_event.

        The launcher runs run() in a worker thread (signals unavailable),
        so it requests shutdown via ``self.stop_event`` instead.
        """
        while not stop.is_set():
            if shutdown.is_set() or self.stop_event.is_set():
                print("\n[eye:neon] shutdown requested; stopping cleanly")
                stop.set()
                return
            await asyncio.sleep(0.1)

    # ==================================================================
    # Hooks
    # ==================================================================

    def _open(self) -> bool:
        """First-data gate for the launcher, warm-standby style: start the
        persistent event-loop thread, which connects the device, waits for
        all sensors and the first gaze/IMU/scene samples (the scene camera
        takes ~5 s to start), then KEEPS the streams running in discard
        mode until the go signal arrives in _record().  The scene camera
        cold start therefore happens exactly once."""
        self._setup()
        # Install signal handlers in the MAIN thread (the loop runs in a
        # worker thread where signal.signal is not allowed).
        self._shutdown = install_asyncio_signal_shutdown()
        threading.Thread(target=self._loop_main, name="eye-neon-loop",
                         daemon=True).start()
        # Discovery 10 s + sensors 10 s + samples 3 x 20 s, worst case.
        if not self._ready_evt.wait(timeout=90.0):
            self._open_error = "warm-up timed out after 90 s"
            self.logger.error(f"[eye:neon] open failed — {self._open_error}")
            self.stop_event.set()
            return False
        return self._open_ok

    def _close(self) -> None:
        # Request the persistent loop to exit (launch-abort path).  The
        # device itself is closed by its async context manager when the
        # loop unwinds.
        self.stop_event.set()

    def _teardown(self) -> None:
        # Release the mp4 BEFORE _save(): a hard kill at the npz step at
        # least leaves a playable mp4 (same ordering as tests/eye/test2.py).
        if self._writer is not None:
            try:
                self._writer.close()
                self._writer = None
                size = (self._out_mp4.stat().st_size / 1e6
                        if self._out_mp4 and self._out_mp4.is_file() else 0.0)
                print(f"\n[eye:neon] mp4 finalized -> {self._out_mp4} "
                      f"({size:.1f} MB)")
            except Exception as exc:
                print(f"[eye:neon] writer.close failed: {exc}")
        super()._teardown()
        # Per-stream counts + rates (mirrors tests/eye/test2.py).
        for key in ("gaze_timestamps", "imu_timestamps", "scene_timestamps"):
            ts = self._buf.get(key, [])
            fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            print(f"[eye:neon] {key}: {len(ts):>6}  fps={fps:6.1f}")

    def probe_data_flow(self, timeout: float = 5.0) -> tuple[bool, str]:
        """preflight 数据流探测:standby 队列由常驻事件循环持续投喂,只要
        gaze/imu 任一队列在增长,数据就是活的(与 _poll 无关,故覆写)。"""
        n0 = len(self._gaze_q) + len(self._imu_q)
        t0 = time.time()
        while time.time() - t0 < timeout:
            n = len(self._gaze_q) + len(self._imu_q)
            if n > n0:
                return True, (f"standby 队列持续进样 "
                              f"(gaze={len(self._gaze_q)}, imu={len(self._imu_q)})")
            time.sleep(0.1)
        return False, f"{timeout:g}s 内 gaze/imu 队列无新增(standby 未进样)"

    def _heartbeat_stats(self, elapsed: float) -> str:
        return super()._heartbeat_stats(elapsed)

