"""Test Pupil Labs Neon eye tracker — real hardware.

Neon drives its own asyncio stream tasks (there is no ``_poll``), so this test
overrides ``run()``: the recorder's own gaze/IMU/scene tasks run in the event
loop while a matplotlib window runs in the same thread (``plt.pause`` pumps the
GUI, ``await asyncio.sleep(0)`` yields to the stream tasks).
"""

import asyncio
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

from tests.eye.test_dummy_eye import TestDummyEye
from tests.base import SESSION_DIR
from src.recorders.eye import NeonEyeAsyncRecorder, EyeRecorderConfig


class TestNeonEye(TestDummyEye):
    name = "Neon Eye"

    # ---- custom run: asyncio instead of the BaseTest poll thread -----------

    def run(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        from pupil_labs.realtime_api.device import Device
        from pupil_labs.realtime_api.discovery import Network

        rec = self.rec
        rec._setup()
        stop = asyncio.Event()
        tasks: list[asyncio.Task] = []
        
        try:
            device_info = await Network().wait_for_new_device(timeout_seconds=10)
            async with Device(address=device_info.addresses[0], port=device_info.port) as dev:
                status = await dev.get_status()
                rec._acc("pc_to_phone_offset_ms", await rec._sync_clock(status))

                gs = status.direct_gaze_sensor()
                ws = status.direct_world_sensor()
                ims = status.direct_imu_sensor()
                # direct_*_sensor() returns a default Sensor(connected=False)
                # instead of None when missing — check connected/url, not None.
                missing = [name for name, s in
                           (("gaze", gs), ("world", ws), ("imu", ims))
                           if s is None or not s.connected or s.url is None]
                if missing:
                    print(f"[test:neon] missing/disconnected sensor(s): "
                          f"{missing} — check the Neon Camera app on the phone")
                    return

                def _watch(name):
                    def _cb(t):
                        if not t.cancelled() and t.exception() is not None:
                            exc = t.exception()
                            print(f"\n[test:neon] {name} task failed: "
                                  f"{type(exc).__name__}: {exc}")
                            traceback.print_exception(
                                type(exc), exc, exc.__traceback__)
                    return _cb

                tasks = []
                for name, coro in (("scene", rec._scene_task(ws.url, stop)),
                                   ("gaze",  rec._gaze_task(gs.url, stop)),
                                   ("imu",   rec._imu_task(ims.url, stop))):
                    t = asyncio.create_task(coro)
                    t.add_done_callback(_watch(name))
                    tasks.append(t)

                fig = plt.figure(figsize=(14, 8))
                fig.canvas.manager.set_window_title(f"{self.name} — Q to stop")
                self._build_layout(fig)

                running = True

                def on_key(e):
                    nonlocal running
                    if e.key == "q":
                        running = False

                fig.canvas.mpl_connect("key_press_event", on_key)

                # One-time pause: maps the window + first draw (flush_events
                # alone never maps an unmapped Tk window).
                plt.pause(0.05)

                t0 = time.time()
                last_redraw = 0.0
                while running:
                    now = time.time() - t0
                    # Redraw at ~10 Hz max: full matplotlib redraws block the
                    # thread and starve the asyncio stream tasks — the scene
                    # stream yields ZERO frames under a blocked event loop
                    # (verified 2026-08-17: 0 frames under GUI load vs 344
                    # frames in 8 s once the loop was unblocked).
                    if now - last_redraw >= 0.1:
                        self._update(rec, now)
                        last_redraw = now
                        # flush_events() never redraws; draw_idle() schedules
                        # the redraw (non-blocking) and flush_events runs it.
                        fig.canvas.draw_idle()
                    # Non-blocking GUI pump (still delivers keypress events);
                    # plt.pause() blocks and must not be used inside asyncio.
                    fig.canvas.flush_events()
                    await asyncio.sleep(0.01)  # feed the stream tasks
                plt.close()
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            rec._teardown()

    # ---- update: window on real sensor timestamps (200 Hz streams) ---------

    @staticmethod
    def _windowed(ts_list, arr_list, window=5.0):
        """Stack the last ``window`` seconds of (timestamp, array) pairs."""
        if len(ts_list) < 2:
            return None
        t0 = ts_list[-1] - window
        keep = [i for i, t in enumerate(ts_list) if t >= t0]
        return np.stack([arr_list[i] for i in keep])

    def _update(self, rec, elapsed):
        gaze = self._windowed(
            rec._buf.get("gaze_timestamps", []), rec._arr_buf.get("gaze_xy", []))
        gyro = self._windowed(
            rec._buf.get("imu_timestamps", []), rec._arr_buf.get("imu_gyro", []))
        accel = self._windowed(
            rec._buf.get("imu_timestamps", []), rec._arr_buf.get("imu_accel", []))
        # Scene frames stream to eye.mp4 now; the recorder keeps only the
        # latest frame (BGR — flip for matplotlib imshow).
        scene = rec._last_scene_frame

        if gaze is not None:
            self.ax_gaze.clear()
            self.ax_gaze.plot(gaze[:, 0], gaze[:, 1], linewidth=0.2)
            self.ax_gaze.set_title(f"gaze XY ({len(gaze)})")
        if gyro is not None:
            self.ax_gyro.clear()
            self.ax_gyro.plot(gyro, linewidth=0.5)
            self.ax_gyro.legend(["gx", "gy", "gz"], fontsize=6)
            self.ax_gyro.set_title("IMU gyro")
        if accel is not None:
            self.ax_accel.clear()
            self.ax_accel.plot(accel, linewidth=0.5)
            self.ax_accel.legend(["ax", "ay", "az"], fontsize=6)
            self.ax_accel.set_title("IMU accel")
        if scene is not None:
            self.ax_scene.clear()
            # Downsample 4x + nearest: antialiased resampling of 1600x1200
            # on every redraw is slow enough to lag the GUI (and feed back
            # into the event loop).  400x300 is visually identical at the
            # panel's 374x280 size.
            self.ax_scene.imshow(
                scene[::4, ::4, ::-1], interpolation="nearest")
            n_scene = len(rec._buf.get("scene_timestamps", []))
            self.ax_scene.set_title(f"scene ({n_scene})")
            self.ax_scene.axis("off")


def main():
    cfg = EyeRecorderConfig(session_dir=f"{SESSION_DIR}/eye")
    rec = NeonEyeAsyncRecorder(cfg)
    TestNeonEye(rec).run()


if __name__ == "__main__":
    main()
