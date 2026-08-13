"""Base test class for recorder tests — handles poll thread, Q-stop, session dir."""

import sys, time, threading
from pathlib import Path

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
        self.rec._open()

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
        while running:
            self._update(self.rec, time.time() - t0)
            plt.pause(0.01)

        self._stop.set()
        t.join(timeout=0.5)
        plt.close()

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
