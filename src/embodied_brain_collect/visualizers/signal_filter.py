"""Zero-phase signal filtering for QC page display copies.

The page embeds a filtered copy of EEG/EMG channel traces so a reader can
toggle between the raw truth and the classic research view (bandpass +
50 Hz mains notch with harmonics).  Raw NPZ data is never modified — the
filtered copy exists only inside qc.html, and only for display.

scipy is optional: without it ``scipy_ok()`` is False and the page simply
ships raw-only.  The SOS cascade mirrors the Curry preview filter in
tests/eeg/test_curry_eeg.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

try:
    from scipy import signal as _sp_signal
except ImportError:          # scipy is not in requirements-hard environments
    _sp_signal = None


@dataclass(frozen=True)
class FilterPreset:
    """One bandpass + mains-notch design for a stream family."""

    lo: float              # bandpass low edge, Hz
    hi: float              # bandpass high edge, Hz
    notch_base: float      # mains fundamental; harmonics up to Nyquist
    notch_q: float = 30.0
    order: int = 4         # Butterworth order


PRESETS: dict[str, FilterPreset] = {
    # 经典科研预处理参数:EEG 0.5-70 Hz,EMG 20-450 Hz,均陷 50 Hz 及谐波
    "eeg": FilterPreset(0.5, 70.0, 50.0),
    "emg": FilterPreset(20.0, 450.0, 50.0),
}


def scipy_ok() -> bool:
    return _sp_signal is not None


def preset_from_dict(raw: dict | None, base: FilterPreset) -> FilterPreset:
    """Merge a checker.yaml ``filter:`` mapping onto a base preset.

    Unknown keys are ignored; only ``lo/hi/notch_base/notch_q/order`` are
    FilterPreset fields.
    """
    if not raw:
        return base
    return replace(base, **{k: v for k, v in raw.items()
                            if k in ("lo", "hi", "notch_base", "notch_q",
                                     "order")})


def preset_for(name: str, overrides: dict | None = None) -> FilterPreset | None:
    """Preset for a stream name, e.g. ``"emg_left"`` -> the ``emg`` preset.

    *overrides* is the checker.yaml ``filter:`` section: each key must match
    a PRESETS family ("eeg"/"emg") and is merged onto it via
    :func:`preset_from_dict`.  The longest matching prefix wins, so a future
    ``"eeg_left"`` family in PRESETS would beat ``"eeg"`` for that stream.
    """
    merged = dict(PRESETS)
    if overrides:
        for k, v in overrides.items():
            if k in merged and isinstance(v, dict):
                merged[k] = preset_from_dict(v, merged[k])
    for key in sorted(merged, key=len, reverse=True):
        if name.startswith(key):
            return merged[key]
    return None


@lru_cache(maxsize=16)
def _sos_cached(fs: int, lo: float, hi: float, notch_base: float,
                notch_q: float, order: int) -> np.ndarray:
    """Butterworth bandpass plus one notch per mains harmonic below Nyquist."""
    sos = _sp_signal.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")
    for f0 in np.arange(notch_base, fs / 2.0, notch_base):
        b, a = _sp_signal.iirnotch(f0, Q=notch_q, fs=fs)
        sos = np.vstack([sos, _sp_signal.tf2sos(b, a)])
    return sos


def design_sos(fs: float, preset: FilterPreset) -> np.ndarray:
    """Design the SOS cascade for *preset* at sampling rate *fs*.

    *fs* is rounded to an integer before design: EMG's derived rate is
    ~2000 Hz (mean of timestamp diffs), and rounding gives cache hits plus
    notch positions that stay put between sessions.
    """
    if not scipy_ok():
        raise RuntimeError("scipy 未安装,无法设计滤波器")
    fs = int(round(float(fs)))
    if fs <= 2 * preset.hi:
        raise ValueError(f"fs={fs} 低于带通上边 {preset.hi} Hz 的两倍")
    return _sos_cached(fs, preset.lo, preset.hi, preset.notch_base,
                       preset.notch_q, preset.order)


def apply_filter(data, fs: float, preset: FilterPreset) -> np.ndarray:
    """Zero-phase filtered copy of an (N, C) matrix.  Never raises.

    Any channel that cannot be filtered safely — NaN present (sosfiltfilt
    would smear one NaN across the whole IIR cascade), the array too short
    for the filtfilt padlen, or *fs* below the bandpass edge — is copied
    through unfiltered, so the page always has something drawable and a QC
    run can never die on one odd channel.
    """
    data = np.asarray(data)
    out = data.astype(np.float64, copy=True)
    if (_sp_signal is None or data.ndim != 2 or data.shape[0] < 32
            or fs <= 2 * preset.hi):
        return out
    try:
        sos = design_sos(fs, preset)
    except (ValueError, RuntimeError):
        return out
    if data.shape[0] <= 4 * (2 * len(sos) + 1):   # shorter than filtfilt padlen
        return out
    bad = ~np.isfinite(data).all(axis=0)
    if bad.all():
        return out
    try:
        out[:, ~bad] = _sp_signal.sosfiltfilt(sos, out[:, ~bad], axis=0)
    except ValueError:                             # any residual edge case
        return data.astype(np.float64, copy=True)
    return out
