"""Base test class for recorder tests — handles poll thread, Q-stop, session dir."""

import sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SESSION_DIR = str(Path(__file__).resolve().parent / "sessions")


class BaseTest:
    """Subclass and override ``_build_layout(fig)`` and ``_update(rec, now)``."""

    name: str = "test"

    def __init__(self, recorder):
        self.rec = recorder
        self._stop = threading.Event()

    def run(self):
        if not self.rec._open():
            reason = self.rec._open_error or "unknown reason"
            print(f"[{self.name}] device open FAILED — {reason}")
            return

        fig = plt.figure(figsize=(14, 8))
        fig.canvas.manager.set_window_title(f"{self.name} — Q to stop")
        self._build_layout(fig)

        running = True
        def on_key(e):
            nonlocal running
            if e.key == 'q':
                running = False
        fig.canvas.mpl_connect('key_press_event', on_key)

        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

        t0 = time.time()
        last_ver = -1
        while running:
            # Redraw only when new data arrived (serial devices deliver in
            # bursts, so a fixed 10 ms redraw mostly re-renders the same
            # frame and stutters).
            ver = self._data_version()
            if ver != last_ver:
                self._update(self.rec, time.time() - t0)
                last_ver = ver
            try:
                plt.pause(0.03)
            except Exception:
                # Window closed with the mouse (X) instead of Q — stop
                # cleanly instead of dumping a backend traceback.
                break

        self._stop.set()
        t.join(timeout=0.5)
        plt.close()

    def _data_version(self) -> int:
        """Total recorded sample count — cheap change detector."""
        rec = self.rec
        return (sum(len(v) for v in rec._buf.values())
                + sum(len(v) for v in rec._arr_buf.values())
                + sum(len(v) for v in rec._ts_buf.values()))

    def _poll_loop(self):
        t0 = time.time()
        while not self._stop.is_set():
            self.rec._poll(time.time() - t0)
            time.sleep(0.001)

    def _build_layout(self, fig):
        pass

    def _update(self, rec, elapsed):
        pass

    @staticmethod
    def _window_slice(n: int, elapsed: float, window: float = 5.0):
        """O(1) slice of the last `window` seconds for *n* samples spread
        uniformly over [0, elapsed] (the sample-index time axis the EMG
        tests use).  Equivalent to ``_rolling`` on such a synthetic axis,
        without building/scaning an O(n) timestamp list every redraw."""
        if n <= 1 or elapsed <= 0:
            return slice(0, n)
        start = int(np.ceil((n - 1) * (elapsed - window) / elapsed))
        return slice(max(start, 0), n)

    @staticmethod
    def _rolling(vals, t0, window=5.0):
        """Return slice for last `window` seconds of data.

        ``vals`` are timestamps (float seconds), ``t0`` is the current
        elapsed time reference.  Returns a slice that selects the last
        ``window`` seconds of samples.
        """
        if not vals:
            return slice(0, 0)
        cutoff = t0 - window
        for i, v in enumerate(vals):
            if v >= cutoff:
                return slice(i, len(vals))
        return slice(-1, len(vals))
