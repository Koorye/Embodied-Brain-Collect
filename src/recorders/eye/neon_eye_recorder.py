"""Pupil Labs Neon — async full-rate API with independent stream clocks.

Uses asyncio + 3 concurrent tasks so every gaze/IMU/scene sample is captured
without the simple-API queue-of-size-1 frame loss.
"""

import asyncio
import sys
import time
import numpy as np
from .base_eye_recorder import BaseEyeRecorder
from .eye_recorder_config import EyeRecorderConfig

sys.stdout.reconfigure(line_buffering=True)


class NeonEyeRecorder(BaseEyeRecorder):
    """Pupil Labs Neon — async, full-rate.  Overrides ``run()`` for asyncio."""

    config: EyeRecorderConfig

    def __init__(self, config: EyeRecorderConfig):
        super().__init__(config)
        self._offset_ms = 0.0

    # ==================================================================
    # run() — asyncio entry point
    # ==================================================================

    def run(self) -> int:
        print(f"[eye:neon] starting {self.config.neon_ip}:{self.config.port} ...")
        try:
            return asyncio.run(self._async_run())
        except KeyboardInterrupt:
            print("\n[eye:neon] interrupted")
            return 130
        except Exception as exc:
            print(f"\n[eye:neon] FATAL: {type(exc).__name__}: {exc}")
            return 1

    # ==================================================================
    # Async main
    # ==================================================================

    async def _async_run(self) -> int:
        from pupil_labs.realtime_api import Device
        from pupil_labs.realtime_api.streaming import (
            receive_gaze_data, receive_imu_data, receive_video_frames)
        from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator

        install_asyncio_signal_shutdown()
        self._setup()
        rc = 0

        try:
            async with Device(
                address=self.config.neon_ip, port=self.config.port,
            ) as dev:
                status = await dev.get_status()
                self._acc("pc_to_phone_offset_ms",
                    await self._sync_clock(status))

                gs = status.direct_gaze_sensor()
                ws = status.direct_world_sensor()
                ims = status.direct_imu_sensor()
                for s, name in [(gs, "gaze"), (ws, "world"), (ims, "imu")]:
                    if s is None:
                        print(f"[eye:neon] missing {name} sensor")
                        return 1

                stop = asyncio.Event()
                shutdown = install_asyncio_signal_shutdown()

                tasks = [
                    asyncio.create_task(self._gaze_task(gs.url, stop)),
                    asyncio.create_task(self._imu_task(ims.url, stop)),
                    asyncio.create_task(self._scene_task(ws.url, stop)),
                    asyncio.create_task(self._progress_task(stop)),
                    asyncio.create_task(self._signal_task(stop, shutdown)),
                ]
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
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
            rc = 1
        finally:
            self._teardown()

        return rc

    # ==================================================================
    # Stream tasks
    # ==================================================================

    @staticmethod
    async def _sync_clock(status) -> float:
        from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator

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
        from pupil_labs.realtime_api.streaming import receive_gaze_data

        async for g in receive_gaze_data(url):
            if stop.is_set():
                break
            self._acc("gaze_timestamps", g.timestamp_unix_seconds)
            self._acc_arr("gaze_xy",
                np.array([g.x, g.y], dtype=np.float32))

    async def _imu_task(self, url, stop):
        from pupil_labs.realtime_api.streaming import receive_imu_data

        async for d in receive_imu_data(url):
            if stop.is_set():
                break
            self._acc("imu_timestamps", d.timestamp_unix_seconds)
            self._acc_arr("imu_gyro",
                np.array([d.gyro_data.x, d.gyro_data.y, d.gyro_data.z], dtype=np.float32))
            self._acc_arr("imu_accel",
                np.array([d.accel_data.x, d.accel_data.y, d.accel_data.z], dtype=np.float32))

    async def _scene_task(self, url, stop):
        from pupil_labs.realtime_api.streaming import receive_video_frames

        loop = asyncio.get_running_loop()
        async for f in receive_video_frames(url):
            if stop.is_set():
                break
            self._acc("scene_timestamps", f.timestamp_unix_seconds)
            img = await loop.run_in_executor(None, f.bgr_buffer)
            self._acc_arr("scene_frames", img)

    async def _progress_task(self, stop):
        t0 = time.time()
        while not stop.is_set():
            await asyncio.sleep(0.5)
            elapsed = time.time() - t0
            ng = len(self._buf.get("gaze_timestamps", []))
            ni = len(self._buf.get("imu_timestamps", []))
            ns = len(self._buf.get("scene_timestamps", []))
            m = self._marker_sub.count() if self._marker_sub else 0
            print(
                f"\r[eye:neon] t={elapsed:5.1f}s  "
                f"gaze={ng:>6} imu={ni:>6} scene={ns:>5} mark={m}",
                end="", flush=True,
            )
            if 0 < self.config.duration <= elapsed:
                stop.set()

    @staticmethod
    async def _signal_task(stop, shutdown):
        while not stop.is_set():
            if shutdown.is_set():
                stop.set()
                return
            await asyncio.sleep(0.1)

    # ==================================================================
    # Hooks
    # ==================================================================

    def _open(self) -> bool:
        return True  # device connection handled in _async_run

    def _close(self) -> None:
        pass  # device closed by async context manager

    def _heartbeat_stats(self, elapsed: float) -> str:
        ng = len(self._buf.get("gaze_timestamps", []))
        ni = len(self._buf.get("imu_timestamps", []))
        ns = len(self._buf.get("scene_timestamps", []))
        return (
            f"gaze={ng:>5} ({ng/elapsed:.0f}/s)  "
            f"imu={ni:>5} ({ni/elapsed:.0f}/s)  "
            f"scene={ns:>4}"
        )

