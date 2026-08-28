"""Rebuild per-frame EMG timestamps from the armband's sequence number.

The recorder stamps every frame carried by one ``Serial.read()`` with that
read's arrival time.  At 2 kHz on the wire against ~15 Hz of reads that puts
~140 frames on a single timestamp: 99.3% of the series are duplicates, and
every frame carries up to one batch-period of latency (mean: half a batch).

The armband's crystal is far steadier than the poll loop, and the 8-bit
sequence number shared by EMG and IMU counts transmitted frames exactly.  So
unwrap it into a global frame index ``k``, fit ``t = a*k + b`` against arrival
time, and read per-frame timestamps off the fit.

Anchors are the LAST frame of each read batch: within a batch the last frame is
the one that had just arrived while the earlier ones sat in the driver buffer,
so fitting against every frame would tilt the line by the batch's own delay
ramp.

Measured on a 25.8 s run: residual std 0.48 ms, fitted rate 1999.997 Hz against
a 2000 Hz nominal (3 ppm).  The residual's lag-1 autocorrelation is -0.11 and
splitting the run into five independently fitted segments does not reduce it
(0.477 vs 0.478 ms) — the arrival noise is white, so one global line is the
right model.  No piecewise fit, and no online tracking: a per-batch refit makes
the line wander between batches, which turns duplicate timestamps into
*backwards* ones.

What the fit cannot recover is the constant delay between the device sampling
an instant and the PC reading it; that needs a hardware sync pulse.  Jitter and
the batch offset do come out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SN_MODULO = 256

_MIN_ANCHORS = 8            # fewer than this and the slope is mostly noise
_OUTLIER_SIGMA = 3.0
_MIN_RATE_HZ = 50.0         # sanity band for the combined EMG+IMU frame rate
_MAX_RATE_HZ = 100_000.0
_MAX_ORIGIN_SHIFT_S = 2.0   # rebuilt t0 may not wander further from arrival t0


@dataclass
class RebuildResult:
    """Outcome of one fit.  ``ok`` False means: keep the arrival timestamps."""

    ok: bool
    note: str
    emg_timestamps: np.ndarray | None = None
    imu_timestamps: np.ndarray | None = None
    period: float = 0.0        # seconds per transmitted frame (EMG and IMU)
    emg_rate_hz: float = 0.0   # EMG frame rate implied by the fit
    n_anchors: int = 0
    n_outliers: int = 0
    residual_ms: float = 0.0   # std of the anchor residual about the line
    shift_ms: float = 0.0      # mean(arrival - rebuilt): the batch delay removed

    def summary(self) -> str:
        if not self.ok:
            return f"timestamp rebuild skipped — {self.note}"
        return (f"timestamp rebuild ok — {self.emg_rate_hz:.3f} Hz emg, "
                f"anchors={self.n_anchors} (-{self.n_outliers} outliers), "
                f"residual={self.residual_ms:.3f} ms, "
                f"removed {self.shift_ms:+.2f} ms of batch delay")


def _unwrap(sn: np.ndarray) -> np.ndarray:
    """Sequence number -> frames elapsed since this stream's first frame.

    Steps are taken modulo 256, which is what makes a counter shared with the
    other frame type usable: an EMG frame following an IMU frame steps by 2,
    and that step *is* the number of frames the device sent in between.  A gap
    of 256 frames or more looks like no gap at all and would silently compress
    the timeline — the caller checks the frame budget for that.
    """
    if sn.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.cumsum(np.r_[0, np.diff(sn.astype(np.int64)) % SN_MODULO])


def _imu_origin(emg_sn: np.ndarray, imu_sn: np.ndarray) -> int:
    """Frames from the first EMG frame to the first IMU frame (may be negative).

    Each stream unwraps to its own first frame, so one offset ties them to a
    common origin.  The two first frames are at most a few tens of frames apart
    (an IMU frame lands every ~18th), so the shorter way round the 8-bit circle
    is the right one.
    """
    d = int(imu_sn[0] - emg_sn[0]) % SN_MODULO
    return d - SN_MODULO if d > SN_MODULO // 2 else d


def _anchors(ts: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick the last frame of each read batch as (arrival, frame index).

    Frames sharing a timestamp came out of one read; the largest ``k`` among
    them is the one that had just arrived.
    """
    order = np.lexsort((k, ts))
    ts_s, k_s = ts[order], k[order]
    last = np.r_[np.diff(ts_s) > 0, True]
    return ts_s[last], k_s[last]


def _fit(k: np.ndarray, t_rel: np.ndarray) -> tuple[float, float]:
    """Least squares ``t_rel = a*k + b``.

    Callers pass time relative to a reference epoch: Unix seconds need 16
    significant digits to hold microseconds, which is exactly float64's limit.
    """
    a, b = np.polyfit(k.astype(np.float64), t_rel, 1)
    return float(a), float(b)


def _sec_from_ns(x_ns: np.ndarray) -> np.ndarray:
    """Whole nanoseconds -> seconds, per-element error < 1e-14.

    ``x * 1e-9`` on an int64 array rounds every element to its own ulp, which
    at 2.6e10 ns is ~3.7 us — enough to make some neighbours jump two grid
    lines.  Integer divmod keeps the quotient exact and confines the float
    work to a remainder below 1e9, where the rounding error is < 1e-15.
    """
    x = np.asarray(x_ns, dtype=np.int64)
    q, r = np.divmod(x, 10**9)
    return q.astype(np.float64) + r.astype(np.float64) * 1e-9


def rebuild(emg_ts: np.ndarray, emg_sn: np.ndarray,
            imu_ts: np.ndarray, imu_sn: np.ndarray) -> RebuildResult:
    """Fit the device clock and return per-frame timestamps for both streams.

    Never raises: anything unexpected comes back as ``ok=False`` with a reason,
    because losing a recording to a failed cosmetic fix would be absurd.
    """
    try:
        emg_ts = np.asarray(emg_ts, dtype=np.float64).ravel()
        emg_sn = np.asarray(emg_sn, dtype=np.int64).ravel()
        imu_ts = np.asarray(imu_ts, dtype=np.float64).ravel()
        imu_sn = np.asarray(imu_sn, dtype=np.int64).ravel()

        if emg_ts.size < 2 or emg_ts.size != emg_sn.size:
            return RebuildResult(False, "no usable emg_timestamps/emg_sn pair")
        if imu_ts.size != imu_sn.size:
            return RebuildResult(False, "imu_timestamps/imu_sn length mismatch")
        if not np.all(np.isfinite(emg_ts)):
            return RebuildResult(False, "non-finite emg_timestamps")

        k_emg = _unwrap(emg_sn)
        if imu_sn.size:
            k_imu = _unwrap(imu_sn) + _imu_origin(emg_sn, imu_sn)
        else:
            k_imu = np.zeros(0, dtype=np.int64)

        # A frame budget that cannot fit what we received means the unwrap lost
        # a 256-frame wrap somewhere, and every timestamp after it would be
        # pulled early.  Better to ship arrival times than a warped timeline.
        sent = int(k_emg[-1]) + 1
        received = emg_ts.size + imu_ts.size
        if sent < received:
            return RebuildResult(
                False, f"sequence unwrap inconsistent ({sent} sent < "
                       f"{received} received) — likely a >=256-frame gap")

        all_ts = np.concatenate([emg_ts, imu_ts])
        all_k = np.concatenate([k_emg, k_imu])
        a_ts, a_k = _anchors(all_ts, all_k)
        if a_ts.size < _MIN_ANCHORS:
            return RebuildResult(
                False, f"only {a_ts.size} read batches (need {_MIN_ANCHORS})")

        # Fit, then refit without the batches whose arrival was an outlier —
        # a scheduling stall delays a read without moving the device clock.
        t_ref = float(a_ts[0])
        a, b = _fit(a_k, a_ts - t_ref)
        res = (a_ts - t_ref) - (a * a_k + b)
        sigma = float(res.std())
        n_out = 0
        if sigma > 0:
            keep = np.abs(res) <= _OUTLIER_SIGMA * sigma
            n_out = int((~keep).sum())
            if n_out and keep.sum() >= _MIN_ANCHORS:
                a, b = _fit(a_k[keep], (a_ts - t_ref)[keep])
                res = (a_ts - t_ref)[keep] - (a * a_k[keep] + b)
            else:
                n_out = 0

        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
            return RebuildResult(False, f"degenerate fit (period={a})")
        rate = 1.0 / a
        if not (_MIN_RATE_HZ <= rate <= _MAX_RATE_HZ):
            return RebuildResult(
                False, f"implausible frame rate {rate:.1f} Hz from fit")

        # The device transmits at a fixed crystal rate, so the true spacing
        # is a constant — express the period in whole nanoseconds and it stays
        # one (473308 ns here).  The quantized slope differs from the fitted
        # one by << 1 ppm, far under the fit's own ~40 ppm uncertainty.
        # The remaining ±0.12 us sawtooth is float64's floor: absolute times
        # sit on the Unix epoch's 0.238 us ulp grid and a 473 us step is not a
        # whole number of grid lines, so neighbours alternate between the two
        # nearest lines.  Nothing after this stage has that resolution — the
        # fit itself is only good to ~0.4 ms — and the frontend's stride
        # compression tolerates it explicitly.
        ns = round(a * 1e9)
        if ns <= 0:
            return RebuildResult(False, f"degenerate quantized period {ns} ns")
        base_ns = round(b * 1e9)
        new_emg = t_ref + _sec_from_ns(base_ns + k_emg * ns)
        new_imu = (t_ref + _sec_from_ns(base_ns + k_imu * ns)
                   if k_imu.size else imu_ts.copy())

        shift = float(np.mean(emg_ts - new_emg))
        if abs(new_emg[0] - emg_ts[0]) > _MAX_ORIGIN_SHIFT_S:
            return RebuildResult(
                False, f"rebuilt origin drifts {new_emg[0] - emg_ts[0]:+.3f}s "
                       f"from arrival — refusing to move the session timeline")

        # The whole point was a strictly increasing series; a positive slope
        # guarantees it, but assert rather than trust the arithmetic.
        if new_emg.size > 1 and not np.all(np.diff(new_emg) > 0):
            return RebuildResult(False, "rebuilt emg series is not increasing")

        emg_rate = rate * (emg_ts.size / received) if received else rate
        return RebuildResult(
            ok=True, note="fitted",
            emg_timestamps=new_emg, imu_timestamps=new_imu,
            period=a, emg_rate_hz=emg_rate,
            n_anchors=int(a_ts.size), n_outliers=n_out,
            residual_ms=float(res.std()) * 1e3, shift_ms=shift * 1e3)
    except Exception as exc:  # noqa: BLE001 - must never break a recording
        return RebuildResult(False, f"{type(exc).__name__}: {exc}")
