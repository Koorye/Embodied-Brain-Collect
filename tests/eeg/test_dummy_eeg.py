"""Test dummy EEG — synthetic 8-ch traces + events; Q to stop.

On stop the recorder closes: the close-time EEG<->PC alignment runs (skipped
when no markers/markers.npz exists in the test session dir) and eeg.npz is
saved under tests/sessions/eeg/.
"""

import numpy as np
from matplotlib.gridspec import GridSpec

from tests.base import BaseTest, SESSION_DIR
from embodied_brain_collect.recorders.eeg import DummyEegRecorder, EegRecorderConfig


class TestDummyEeg(BaseTest):
    name = "Dummy EEG"

    def _build_layout(self, fig):
        gs = GridSpec(3, 3, figure=fig)
        self.ax_ch = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
        self.ax_ev = fig.add_subplot(gs[:, 1:])
        for ax in self.ax_ch:
            ax.tick_params(labelsize=6)

    @staticmethod
    def _downsample(arr, max_pts=800):
        n = len(arr)
        if n <= max_pts:
            return arr
        return arr[:: n // max_pts]

    def _update(self, rec, elapsed):
        if rec._blocks:
            data = np.concatenate(rec._blocks)
            t_axis = np.arange(len(data)) / max(rec._sample_rate, 1.0)
            sl = self._rolling(t_axis.tolist(), elapsed, window=5.0)
            idx = self._downsample(np.arange(len(data))[sl])
            d = data[idx]
            tt = t_axis[idx]
            for ch in range(6):
                self.ax_ch[ch].clear()
                self.ax_ch[ch].plot(tt, d[:, ch], linewidth=0.3)
                self.ax_ch[ch].set_ylabel(f"ch{ch}", fontsize=6)
            self.ax_ch[0].set_title(
                f"eeg {rec._sample_rate:g} Hz  "
                f"blocks={len(rec._blocks)}", fontsize=8)
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

    def run(self):
        try:
            super().run()
        finally:
            # stop the poll thread first so hardware close doesn't race it
            self._stop.set()
            # close -> alignment -> save, so the stop path is exercised too
            self.rec._close()
            fit = self.rec._fit or {}
            if fit.get("fitted"):
                print(f"[test] align: slope={fit['slope_pc_per_eeg']:.7f} "
                      f"resid_rms={fit['resid_rms_ms']:.2f}ms "
                      f"resid_max={fit['resid_max_ms']:.2f}ms "
                      f"n={fit['n']} inliers={fit['n_inliers']}")
            else:
                print(f"[test] align skipped — {fit.get('reason', '?')}")
            self.rec._save()


if __name__ == "__main__":
    rec = DummyEegRecorder(EegRecorderConfig(session_dir=f"{SESSION_DIR}/eeg"))
    TestDummyEeg(rec).run()
