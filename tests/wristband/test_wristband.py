"""Interactive real-hardware test for the BLE physiological wristband.

Examples:

    python -m tests.wristband.test_wristband
    python -m tests.wristband.test_wristband --name MY_DEVICE
    python -m tests.wristband.test_wristband --address XX:XX:XX:XX:XX:XX

Press Q or close the window to stop.  The same NPZ output used by a formal
session is saved, making this both a live signal check and an end-to-end test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

from embodied_brain_collect.recorders.wristband import (
    WristbandRecorder,
    WristbandRecorderConfig,
)

SESSION_DIR = str(Path(__file__).resolve().parents[1] / "sessions")


class TestWristband:
    """Live plots plus connection/data-quality status for one wristband."""

    def __init__(self, recorder: WristbandRecorder, window_seconds: float = 5.0):
        self.recorder = recorder
        self.window_seconds = window_seconds
        self._running = True
        self.return_code: int | None = None

    def run(self) -> int:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.canvas.manager.set_window_title("Physiological wristband — Q to stop")
        ax_pressure, ax_ppg, ax_accel, ax_gyro = axes.flat

        def stop(_event=None):
            self._running = False

        def on_key(event):
            if event.key and event.key.lower() == "q":
                stop()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", stop)

        worker = threading.Thread(target=self._run_recorder, daemon=True)
        worker.start()

        try:
            while self._running and worker.is_alive() and plt.fignum_exists(fig.number):
                self._update_plot(ax_pressure, ax_ppg, ax_accel, ax_gyro)
                plt.pause(0.10)
        finally:
            self.recorder.stop_event.set()
            worker.join(timeout=10.0)
            if plt.fignum_exists(fig.number):
                plt.close(fig)

        return 1 if self.return_code is None else self.return_code

    def _run_recorder(self) -> None:
        self.return_code = self.recorder.run()

    def _update_plot(self, ax_pressure, ax_ppg, ax_accel, ax_gyro) -> None:
        rec = self.recorder

        pressure_t = np.asarray(
            rec._buf.get("pressure_timestamps", [])[-1000:], dtype=float
        )
        pressure = np.asarray(rec._buf.get("pressure_raw", [])[-1000:])
        ppg_t = np.asarray(rec._buf.get("ppg_timestamps", [])[-700:], dtype=float)
        ppg_values = rec._arr_buf.get("ppg_raw", [])[-700:]
        imu_t = np.asarray(rec._buf.get("imu_timestamps", [])[-400:], dtype=float)
        accel_values = rec._arr_buf.get("accel_m_s2", [])[-400:]
        gyro_values = rec._arr_buf.get("gyro_rad_s", [])[-400:]

        self._plot_scalar(ax_pressure, pressure_t, pressure, "Pressure raw", ["P"])
        self._plot_vectors(ax_ppg, ppg_t, ppg_values, "PPG raw", ["G", "R", "IR"])
        self._plot_vectors(
            ax_accel, imu_t, accel_values, "Acceleration (m/s²)", ["ax", "ay", "az"]
        )
        self._plot_vectors(
            ax_gyro, imu_t, gyro_values, "Angular velocity (rad/s)", ["gx", "gy", "gz"]
        )

        temp = rec._buf.get("temperature_c", [])
        spo2 = rec._buf.get("spo2_percent", [])
        status = (
            f"device={rec._selected_device_name or 'searching'}  "
            f"frames={rec._data_frames_saved}  "
            f"temp={self._latest(temp, '%.2f°C')}  "
            f"SpO₂={self._latest(spo2, '%.0f%%')}"
        )
        ax_pressure.figure.suptitle(status, fontsize=11)

    def _relative_window(self, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if timestamps.size == 0:
            return timestamps, np.zeros(0, dtype=bool)
        latest = timestamps[-1]
        keep = timestamps >= latest - self.window_seconds
        return timestamps[keep] - latest, keep

    def _plot_scalar(self, ax, timestamps, values, title, labels) -> None:
        ax.clear()
        rel_t, keep = self._relative_window(timestamps)
        if rel_t.size and len(values) == len(timestamps):
            ax.plot(rel_t, values[keep], linewidth=0.7, label=labels[0])
        ax.set_title(title)
        ax.set_xlabel("Seconds before latest sample")
        ax.grid(alpha=0.2)

    def _plot_vectors(self, ax, timestamps, values, title, labels) -> None:
        ax.clear()
        rel_t, keep = self._relative_window(timestamps)
        if rel_t.size and len(values) == len(timestamps):
            array = np.stack(values)
            for channel, label in enumerate(labels):
                ax.plot(rel_t, array[keep, channel], linewidth=0.7, label=label)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("Seconds before latest sample")
        ax.grid(alpha=0.2)

    @staticmethod
    def _latest(values, fmt: str) -> str:
        if not values:
            return "--"
        value = float(values[-1])
        return "invalid" if np.isnan(value) else fmt % value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", help="BLE name or name substring")
    parser.add_argument("--address", default="", help="BLE MAC/CoreBluetooth ID")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to record; 0 means stop with Q",
    )
    parser.add_argument(
        "--session-dir",
        default="",
        help="Session output directory (default: timestamped tests/sessions path)",
    )
    args = parser.parse_args(argv)

    session_dir = args.session_dir
    if not session_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(SESSION_DIR) / f"wristband_{stamp}")

    recorder = WristbandRecorder(
        WristbandRecorderConfig(
            session_dir=session_dir,
            duration=args.duration,
            device_name=args.name,
            address=args.address,
        )
    )
    print(f"Wristband test output: {Path(session_dir).resolve()}")
    return TestWristband(recorder).run()


if __name__ == "__main__":
    raise SystemExit(main())
