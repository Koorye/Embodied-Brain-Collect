"""Test Pupil Labs Neon eye tracker — real hardware.

Neon drives its own asyncio stream tasks (there is no ``_poll``), so this test
overrides ``run()``: the recorder's own gaze/IMU/scene tasks run in the event
loop while a matplotlib window runs in the same thread (``plt.pause`` pumps the
GUI, ``await asyncio.sleep(0)`` yields to the stream tasks).
"""

import asyncio
import time

import numpy as np
import matplotlib.pyplot as plt

from tests.eye.test_dummy_eye import TestDummyEye
from tests.base import SESSION_DIR
from src.recorders.eye import NeonEyeRecorder, EyeRecorderConfig


class TestNeonEye(TestDummyEye):
    name = "Neon Eye"

    # ---- custom run: asyncio instead of the BaseTest poll thread -----------

    def run(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        from pupil_labs.realtime_api import Device

        rec = self.rec
        rec._setup()
        stop = asyncio.Event()
        tasks: list[asyncio.Task] = []
        try:
            async with Device(
                address=rec.config.neon_ip, port=rec.config.port,
            ) as dev:
                status = await dev.get_status()
                rec._acc("pc_to_phone_offset_ms", await rec._sync_clock(status))

                gs = status.direct_gaze_sensor()
                ws = status.direct_world_sensor()
                ims = status.direct_imu_sensor()
                if gs is None or ws is None or ims is None:
                    print(f"[test:neon] missing sensor — "
                          f"gaze={gs} world={ws} imu={ims}")
                    return

                tasks = [
                    asyncio.create_task(rec._gaze_task(gs.url, stop)),
                    asyncio.create_task(rec._imu_task(ims.url, stop)),
                    asyncio.create_task(rec._scene_task(ws.url, stop)),
                ]

                fig = plt.figure(figsize=(14, 8))
                fig.canvas.manager.set_window_title(f"{self.name} — Q to stop")
                self._build_layout(fig)

                running = True

                def on_key(e):
                    nonlocal running
                    if e.key == "q":
                        running = False

                fig.canvas.mpl_connect("key_press_event", on_key)

                t0 = time.time()
                while running:
                    self._update(rec, time.time() - t0)
                    plt.pause(0.01)
                    await asyncio.sleep(0)  # yield to the stream tasks
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
        scene = rec._arr_buf.get("scene_frames", [])

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
        if scene:
            self.ax_scene.clear()
            self.ax_scene.imshow(scene[-1])
            self.ax_scene.set_title(f"scene ({len(scene)})")
            self.ax_scene.axis("off")


def main():
    cfg = EyeRecorderConfig(session_dir=f"{SESSION_DIR}/eye")
    rec = NeonEyeRecorder(cfg)
    TestNeonEye(rec).run()


if __name__ == "__main__":
    main()
