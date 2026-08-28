"""Turn a QC report plus its raw recordings into a drawing payload.

The QC report says *what* went wrong and, through ``Finding.spans``, *when*.
This module adds the third thing a reader needs — what the data actually looked
like at that moment — by pulling traces out of the NPZs and frames out of the
mp4s, embedded in one HTML file.

Everything is expressed relative to RUN_START, so a single time axis lines all
the streams up: the point of the page is seeing that an EMG dropout and a
camera freeze happened at the same instant.

Two reductions still apply:

* **int16 quantisation** with a NaN sentinel, base64'd into the JSON and
  decoded to typed arrays on the page.  Y values keep their shape 1:1 — the
  quantisation is per-series over its own range (32767 levels), far finer
  than a pixel.
* **pre-reduction of the giant arrays** — 115700x50x3 of skeleton is a
  centroid and a mean node speed; nobody can read 50 joints at once anyway.

Every series keeps every sample: decimation (min/max envelopes at 100 pts/s)
was dropped in v1.1.0 so the page can zoom into per-sample detail.  A uniform
x axis is shipped as a stride instead of one float per point when possible.
Series carry a *slot index*, never a hex colour, so the page can resolve them
against whichever theme is active.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

# v1.1.0: no decimation.  Every series keeps every sample — the point budget
# is the sample count itself.  The old 100 pts/s min/max envelopes made the
# page blind to anything finer than ~10 ms no matter how far you zoomed in.
PTS_PER_SEC = 100        # kept only as a documented floor for opt-in coarse
MIN_PTS = 600           # rendering via explicit max_pts (nothing uses it)

FRAME_FPS = 1.0         # thumbnails per second
FRAME_W = 240           # thumbnail width, px
JPEG_Q = 60


# =============================================================================
# Encoding
# =============================================================================

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def encode_f32(a) -> str:
    return _b64(np.asarray(a, dtype=np.float32))


def encode_i16(y) -> tuple[str, float, float]:
    """Quantise to int16 over the series' own range; NaN -> -32768."""
    y = np.asarray(y, dtype=np.float64)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return _b64(np.full(y.size, -32768, dtype=np.int16)), 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    q = np.full(y.size, -32768, dtype=np.int16)
    ok = np.isfinite(y)
    q[ok] = np.clip(np.round((y[ok] - lo) / (hi - lo) * 32767), 0, 32767)
    return _b64(q), lo, hi


def target_points(t) -> int:
    """Point budget for a series.

    v1.1.0 policy: everything is kept, so the budget is the series itself.
    ``PTS_PER_SEC``/``MIN_PTS`` remain only as a documented floor for any
    future opt-in coarse rendering (pass an explicit ``max_pts`` to
    ``series()`` to use them).
    """
    return int(np.asarray(t, dtype=np.float64).size)


def minmax_downsample(t, y, max_pts: int | None = None):
    """Decimate to <= max_pts points, keeping each bucket's min and max.

    Only runs when a caller passes an explicit budget; the default path keeps
    every sample.  Spikes survive, which was the whole point of the envelope:
    a jump or a dropout is one or two samples wide and plain striding would
    walk straight past it.  An all-NaN bucket emits a single NaN so the
    stroke breaks there.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_pts is None:
        return t, y
    n = t.size
    if n <= max_pts:
        return t, y

    nb = max(1, max_pts // 2)
    edges = np.linspace(0, n, nb + 1).astype(np.int64)
    out_t: list[float] = []
    out_y: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        chunk = y[lo:hi]
        if not np.isfinite(chunk).any():
            out_t.append(float(t[lo]))
            out_y.append(float("nan"))
            continue
        i_min = lo + int(np.nanargmin(chunk))
        i_max = lo + int(np.nanargmax(chunk))
        first, second = min(i_min, i_max), max(i_min, i_max)
        out_t.append(float(t[first]))
        out_y.append(float(y[first]))
        if second != first:
            out_t.append(float(t[second]))
            out_y.append(float(y[second]))
    return np.asarray(out_t), np.asarray(out_y)


# =============================================================================
# Payload pieces
# =============================================================================

def series(t, y, *, label: str = "", slot: int = 1, unit: str = "",
           max_pts: int | None = None, uniform_ts: bool = False,
           y_f: np.ndarray | None = None) -> dict | None:
    """One decimated, quantised trace.  None when there is nothing to draw.

    ``uniform_ts`` may only be used when *t* is strictly uniform (no
    decimation ran): the x axis then ships as a stride instead of one float32
    per point, which at 2000 Hz over minutes is most of a trace's size.

    *y_f* is an optional filtered display copy (same length as *y*): it is
    quantised over its own range and carried as extra ``yf/flo/fhi`` keys,
    sharing the x axis.  The page shows it only when the reader asks — raw
    data stays the default and the npz is never modified.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size < 2 or t.size != y.size or not np.isfinite(y).any():
        return None
    td, yd = minmax_downsample(t, y, max_pts)
    q, lo, hi = encode_i16(yd)
    out = {"label": label, "slot": slot, "unit": unit,
           "y": q, "lo": lo, "hi": hi}
    if y_f is not None:
        yf = np.asarray(y_f, dtype=np.float64)
        if yf.size == y.size and np.isfinite(yf).any():
            yfd = minmax_downsample(t, yf, max_pts)[1]
            qf, flo, fhi = encode_i16(yfd)
            out["yf"], out["flo"], out["fhi"] = qf, flo, fhi
    if uniform_ts:
        dt = np.diff(td)
        # Rebuilt timestamps carry float64's epoch-grid sawtooth (±0.12 us,
        # see timestamp_rebuild), and old fitted ones alternate between
        # adjacent floats.  A stride stays honest while the deviation is far
        # below what the page's float32 x axis (~1.5 us at session scale) can
        # show — real non-uniformity (e.g. 66 ms arrival batches) is orders
        # of magnitude above this band and correctly falls back.
        if dt.size and dt[0] > 0 and float((dt - dt.mean()).max()) < 1e-6:
            out["t"] = encode_f32(td[:2])
            out["tstride"] = float(dt.mean())
            return out
    out["t"] = encode_f32(td)
    return out


def row(label: str, ser: list, *, h: int = 56, unit: str = "",
        src: str = "") -> dict | None:
    """One plotted row.

    ``src`` names the series or device this row came from, and must match the
    ``subject`` a finding carries.  That is what lets an IMU dropout band
    cover only the IMU rows: drawn across the whole card it would claim the
    EMG channels went missing too, which is both wrong and unreadable once a
    stream has dozens of events.
    """
    ser = [s for s in ser if s]
    if not ser:
        return None
    return {"label": label, "h": h, "unit": unit, "kind": "lines",
            "src": src, "ser": ser}


def channel_rows(t, data, *, names=None, prefix: str = "ch", unit: str = "",
                 src: str = "", data_f: np.ndarray | None = None) -> list[dict]:
    """One thin row per channel of an (N, C) array.

    Each row carries a single series, so colour never has to distinguish 132
    things — the row label does, and a flat line reads as a dead channel at a
    glance.  Only the row *height* shrinks with the channel count; the point
    budget does not, so a 132-channel montage keeps the same time resolution
    as an 8-channel one.

    *data_f*, when given, is the filtered display copy of *data*; a shape
    mismatch is ignored rather than an error.
    """
    t = np.asarray(t, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != t.size or t.size < 2:
        return []
    if data_f is not None:
        data_f = np.asarray(data_f, dtype=np.float64)
        if data_f.shape != data.shape:
            data_f = None
    n_ch = data.shape[1]
    h = 14 if n_ch > 32 else (20 if n_ch > 12 else 26)

    out: list[dict] = []
    for c in range(n_ch):
        name = (str(names[c]) if names is not None and c < len(names)
                else f"{prefix}{c}")
        r = row(name, [series(t, data[:, c], label=name, slot=1, unit=unit,
                              y_f=None if data_f is None else data_f[:, c])],
                h=h, unit=unit, src=src)
        if r:
            out.append(r)
    return out


# =============================================================================
# Generic rows every stream gets
# =============================================================================

def timing_rows(t_rel: np.ndarray, src: str = "", max_pts=None) -> list[dict]:
    """Sampling interval, and duplicate-stamp density when there is any.

    Both are omitted when the stream is clean, so a healthy recording costs
    no vertical space and the eye goes straight to the streams that aren't.
    ``max_pts`` passes through to the interval trace (e.g. fine EMG mode).
    """
    out: list[dict] = []
    if t_rel.size < 3:
        return out
    label = f"采样间隔 {src}".strip()
    dt_ms = np.diff(t_rel) * 1000.0
    r = row(label, [series(t_rel[1:], dt_ms, label="dt", slot=1, unit="ms",
                           max_pts=max_pts)],
            h=52, unit="ms", src=src)
    if r:
        out.append(r)

    dup = np.diff(t_rel) == 0
    if dup.any():
        edges = np.arange(np.floor(t_rel[0]), np.ceil(t_rel[-1]) + 1.0, 1.0)
        if edges.size >= 2:
            counts, _ = np.histogram(t_rel[1:][dup], edges)
            r = row(f"重复时间戳 {src}".strip(),
                    [series(edges[:-1], counts.astype(float),
                            label="dup/s", slot=2, unit="/s")],
                    h=44, unit="/s", src=src)
            if r:
                out.append(r)
    return out


def xyz_rows(label: str, t, arr, *, unit: str = "", h: int = 64,
             src: str = "") -> list[dict]:
    """An (N, 3) array as one row with three direct-labelled series.

    Slots 1-3 are the three that validate on the all-pairs gate, and the x/y/z
    labels are the relief the light-mode contrast warning requires.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return []
    ser = [series(t, arr[:, i], label=ax, slot=i + 1, unit=unit)
           for i, ax in enumerate("xyz")]
    r = row(label, ser, h=h, unit=unit, src=src)
    return [r] if r else []
