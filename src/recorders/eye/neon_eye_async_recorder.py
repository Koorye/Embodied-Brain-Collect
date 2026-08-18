"""Pupil Labs Neon — async full-rate API with independent stream clocks.

Uses asyncio + 3 concurrent tasks so every gaze/IMU/scene sample is captured
without the simple-API queue-of-size-1 frame loss.
"""

import asyncio
import logging
import sys
import time
import traceback
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

    # ==================================================================
    # _record() — asyncio entry point
    # ==================================================================

    def _record(self) -> None:
        rc = asyncio.run(self._async_run())
        if rc != 0:
            self.logger.error(f"[eye:neon] recording ended with rc={rc}")

    # ==================================================================
    # Async main
    # ==================================================================

    async def _async_run(self) -> int:
        self._setup()
        # Installed AFTER _setup(): BaseRecorder._setup() re-installs the
        # sync SIGBREAK/SIGTERM handlers, which would clobber ours.
        shutdown = install_asyncio_signal_shutdown()
        rc = 0

        try:
            # Reuse the device found during the _open() gate when possible.
            device_info = self._device_info
            if device_info is None:
                device_info = await Network().wait_for_new_device(
                    timeout_seconds=10)
                self._device_info = device_info
            async with Device(address=device_info.addresses[0], port=device_info.port) as dev:
                status = await dev.get_status()
                self._acc("pc_to_phone_offset_ms",
                    await self._sync_clock(status))

                sensors = await self._await_sensors(dev)
                if sensors is None:
                    return 1
                gs, ims, ws = sensors  # (gaze, imu, world)

                stop = asyncio.Event()
                named = [
                    ("gaze",     self._gaze_task(gs.url, stop)),
                    ("imu",      self._imu_task(ims.url, stop)),
                    ("scene",    self._scene_task(ws.url, stop)),
                    ("progress", self._progress_task(stop)),
                    ("signal",   self._signal_task(stop, shutdown)),
                ]
                tasks = [asyncio.create_task(self._guarded(n, c))
                         for n, c in named]
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for (name, _c), res in zip(named, results):
                        if isinstance(res, Exception):
                            print(f"\n[eye:neon] {name} task ended with an "
                                  f"error — {type(res).__name__}: {res}")
                            rc = 1
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n[eye:neon] Ctrl+C")
                finally:
                    stop.set()
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except KeyboardInterrupt:
            print("\n[eye:neon] Ctrl+C (outer)")
        except Exception as exc:
            print(f"\n[eye:neon] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            rc = 1
        finally:
            self._teardown()

        return rc

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
                print(f"\n[eye:neon] sensor(s) not connected after {timeout:g}s: "
                      f"{', '.join(missing)} — open the Neon Camera app on "
                      f"the phone")
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
            print(f"\n[eye:neon] {name} stream failed: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise

    # ==================================================================
    # Stream tasks
    # ==================================================================

    @staticmethod
    async def _sync_clock(status) -> float:
        est = await TimeOffsetEstimator(
            status.phone.ip, status.phone.time_echo_port,
        ).estimate(number_of_measurements=10)
        ms = float(est.time_offset_ms.mean) if est else 0.0
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
            self._acc("gaze_timestamps", g.timestamp_unix_seconds)
            self._acc_arr("gaze_xy",
                np.array([g.x, g.y], dtype=np.float32))

    async def _imu_task(self, url, stop):
        async for d in receive_imu_data(url, run_loop=True,
                                        log_level=logging.WARNING):
            if stop.is_set():
                break
            self._acc("imu_timestamps", d.timestamp_unix_seconds)
            self._acc_arr("imu_gyro",
                np.array([d.gyro_data.x, d.gyro_data.y, d.gyro_data.z], dtype=np.float32))
            self._acc_arr("imu_accel",
                np.array([d.accel_data.x, d.accel_data.y, d.accel_data.z], dtype=np.float32))

    async def _scene_task(self, url, stop):
        loop = asyncio.get_running_loop()
        print(f"[eye:neon] scene stream opening — first frame takes "
              f"~5 s while the camera starts ...")
        if self.config.no_scene_video:
            async for f in receive_video_frames(url, run_loop=True,
                                                log_level=logging.WARNING):
                if stop.is_set():
                    break
                self._acc("scene_timestamps", f.timestamp_unix_seconds)
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
                # Decode in a worker thread (loop stays free for UDP).
                img = await loop.run_in_executor(None, f.bgr_buffer)
                self._last_scene_frame = img
                try:
                    queue.put_nowait((f.timestamp_unix_seconds, img))
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()  # drop the oldest
                    except asyncio.QueueEmpty:
                        pass
                    queue.put_nowait((f.timestamp_unix_seconds, img))
                    dropped += 1
        finally:
            if dropped:
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
        t0 = time.time()
        while not stop.is_set():
            await asyncio.sleep(0.5)
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
        """First-data gate for the launcher: discover + connect, wait for
        all sensors, then pull one gaze sample and one scene frame (the
        scene camera takes ~5 s to start).  Nothing is buffered here —
        run() records fresh."""

        async def _check() -> bool:
            try:
                device_info = await asyncio.wait_for(
                    Network().wait_for_new_device(timeout_seconds=10),
                    timeout=12.0)
            except Exception as exc:
                self._open_error = (f"device discovery failed: "
                                    f"{type(exc).__name__}: {exc}")
                self.logger.error(f"[eye:neon] open failed — "
                                  f"{self._open_error}")
                return False
            self._device_info = device_info  # _record() reuses this
            try:
                async with Device(address=device_info.addresses[0],
                                  port=device_info.port) as dev:
                    sensors = await self._await_sensors(dev)
                    if sensors is None:
                        self._open_error = "sensor(s) not connected"
                        return False
                    gs, _ims, ws = sensors

                    # First gaze sample (bounded).
                    gaze_gen = receive_gaze_data(gs.url, run_loop=True,
                                                 log_level=logging.WARNING)
                    try:
                        await asyncio.wait_for(gaze_gen.__anext__(),
                                               timeout=8.0)
                    except (asyncio.TimeoutError, StopAsyncIteration):
                        self._open_error = "no gaze sample within 8 s"
                        self.logger.error(f"[eye:neon] open failed — "
                                          f"{self._open_error}")
                        return False
                    finally:
                        await gaze_gen.aclose()

                    # First scene frame (scene camera cold start ~5 s).
                    video_gen = receive_video_frames(ws.url, run_loop=True,
                                                     log_level=logging.WARNING)
                    try:
                        await asyncio.wait_for(video_gen.__anext__(),
                                               timeout=15.0)
                    except (asyncio.TimeoutError, StopAsyncIteration):
                        self._open_error = "no scene frame within 15 s"
                        self.logger.error(f"[eye:neon] open failed — "
                                          f"{self._open_error}")
                        return False
                    finally:
                        await video_gen.aclose()

                self.logger.info("[eye:neon] first gaze + scene frame "
                                 "received — ready")
                return True
            except Exception as exc:
                self._open_error = f"{type(exc).__name__}: {exc}"
                self.logger.error(f"[eye:neon] open failed — "
                                  f"{self._open_error}")
                return False

        try:
            return asyncio.run(_check())
        except Exception as exc:
            self._open_error = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[eye:neon] open failed — {self._open_error}")
            return False

    def _close(self) -> None:
        pass  # device closed by async context manager

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

    def _heartbeat_stats(self, elapsed: float) -> str:
        return super()._heartbeat_stats(elapsed)

