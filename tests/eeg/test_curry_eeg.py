"""Test Curry EEG — real Curry 9 NetStream over TCP (default 127.0.0.1:4455).

Start Curry's NetStream service first, then run::

    python -m tests.eeg.test_curry_eeg            # raw view
    python -m tests.eeg.test_curry_eeg --filter   # host-style display filters

The plots stay empty until the first EEG block arrives.  The window is
fixed to the last 5 blocks, and the first 20 channels are plotted.

The raw view matches the host with its display filters switched off.
``--filter`` enables a zero-phase IIR cascade that mimics the host's
default filters (Low Filter 10 Hz, High Filter 70 Hz, 50 Hz notch +
harmonics); press F at any time to toggle.  The recorder itself always
keeps raw data.

On Q the recorder closes: the close-time alignment runs against
markers/markers.npz (present when a marker recorder ran in the same session
dir) and eeg.npz is saved.
"""

import sys

import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import signal

from tests.base import SESSION_DIR
from tests.eeg.test_dummy_eeg import TestDummyEeg
from embodied_brain_collect.recorders.eeg import CurryEegRecorder, EegRecorderConfig

WINDOW_BLOCKS = 5   # fixed window: plot the last N blocks
N_CHANNELS = 20     # plot the first N channels

_FILTER_LO = 10.0   # Curry "Low Filter" (highpass), Hz
_FILTER_HI = 70.0   # Curry "High Filter" (lowpass), Hz
_NOTCH_BASE = 50.0  # notch fundamental + harmonics up to Nyquist
_NOTCH_Q = 30.0     # IIR notch quality factor


class TestCurryEeg(TestDummyEeg):
    name = "Curry EEG"

    def __init__(self, recorder, filtered=False):
        super().__init__(recorder)
        self._filtered = filtered   # mimic the Curry display filters; F toggles
        self._sos = None
        self._sos_fs = 0.0

    def _build_layout(self, fig):
        gs = GridSpec(5, 5, figure=fig)
        self.ax_ch = [fig.add_subplot(gs[i // 4, i % 4])
                      for i in range(N_CHANNELS)]
        self.ax_ev = fig.add_subplot(gs[:, 4])
        for i, ax in enumerate(self.ax_ch):
            ax.tick_params(labelsize=6)
            if i < N_CHANNELS - 4:   # all rows but the last
                ax.set_xticklabels([])
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, e):
        if e.key == "f":
            self._filtered = not self._filtered

    def _filter_sos(self, fs):
        """SOS cascade: 4th-order Butterworth bandpass 10-70 Hz + notch
        at 50 Hz and each harmonic below Nyquist."""
        if self._sos is not None and fs == self._sos_fs:
            return self._sos
        sos = signal.butter(4, [_FILTER_LO, _FILTER_HI], "bandpass",
                            fs=fs, output="sos")
        for f0 in np.arange(_NOTCH_BASE, fs / 2, _NOTCH_BASE):
            b, a = signal.iirnotch(f0, Q=_NOTCH_Q, fs=fs)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
        self._sos, self._sos_fs = sos, fs
        return sos

    def _filter(self, data, fs):
        return signal.sosfiltfilt(self._filter_sos(fs), data, axis=0)

    @staticmethod
    def _decimate(x, y, max_pts=400):
        """Min-max decimation: keep both extrema of each bin.

        Strided sampling of a 50 Hz signal at 1 kHz aliases into noise-like
        jags; min-max keeps the wave envelope intact like the Curry display.
        """
        if len(x) <= max_pts:
            return x, y
        n = len(x)
        edges = np.linspace(0, n, max_pts + 1).astype(np.int64)
        xs, ys = [], []
        for b in range(max_pts):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            seg = y[lo:hi]
            for i in sorted({lo + int(np.argmin(seg)), lo + int(np.argmax(seg))}):
                xs.append(x[i])
                ys.append(y[i])
        return np.array(xs), np.array(ys)

    def _update(self, rec, elapsed):
        if rec._blocks:
            data = np.concatenate(rec._blocks[-WINDOW_BLOCKS:])
            fs = max(rec._sample_rate, 1.0)
            data = data[:, :N_CHANNELS]
            if self._filtered and fs > 2 * _FILTER_HI:
                data = self._filter(data, fs)
            t_axis = np.arange(len(data)) / fs
            labels = rec._channel_labels
            for ch in range(min(N_CHANNELS, data.shape[1])):
                tt, dd = self._decimate(t_axis, data[:, ch])
                ax = self.ax_ch[ch]
                ax.clear()
                ax.plot(tt, dd, linewidth=0.3)
                ax.set_ylabel(labels[ch] if ch < len(labels) else f"ch{ch}",
                              fontsize=6)
            mode = "10-70+50n" if self._filtered else "raw"
            self.ax_ch[0].set_title(
                f"eeg {rec._sample_rate:g} Hz  "
                f"window=last {min(WINDOW_BLOCKS, len(rec._blocks))}/"
                f"{len(rec._blocks)} blocks  filter={mode} [F]",
                fontsize=8)
        ev_codes = rec._buf.get("eeg_event_code", [])
        if ev_codes:
            self.ax_ev.clear()
            self.ax_ev.vlines(range(len(ev_codes)), 0, ev_codes, linewidth=2)
            for i, code in enumerate(ev_codes):
                if i % 2 == 0:
                    self.ax_ev.annotate(str(code), (i, code),
                                        fontsize=6, rotation=45, ha="right")
        self.ax_ev.set_title(f"events ({len(ev_codes)}) [Q to stop]")
        self.ax_ev.set_xlim(0, max(10, len(ev_codes) + 1))


def main():
    filtered = "--filter" in sys.argv
    cfg = EegRecorderConfig(session_dir=f"{SESSION_DIR}/eeg",
                            host="127.0.0.1", port=4455)
    rec = CurryEegRecorder(cfg)
    print(f"Connecting to Curry NetStream {cfg.host}:{cfg.port} ...")
    TestCurryEeg(rec, filtered=filtered).run()


if __name__ == "__main__":
    main()
