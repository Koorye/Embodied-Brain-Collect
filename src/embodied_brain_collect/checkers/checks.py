"""Reusable checks, shared by every modality.

Each class is one independent verdict with its thresholds as constructor
defaults, so a modality's ``checks`` list doubles as its configuration::

    checks = [*ts_checks("frames", expected_rate=30.0),
              BlackFrame(lum_thr=8.0, run_s=2.0)]

Timing checks read a registered series by label; data checks read an NPZ
field by name.  Both skip cleanly (``applies``) when what they need is
absent, so a modality can declare a check for an optional field.

Anything genuinely expensive goes through ``ctx.artifact`` — notably the
video decode, which three separate checks consume but which must happen
exactly once per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import BaseCheck, CheckContext, CheckOutput, Span

# =============================================================================
# Numeric helpers
# =============================================================================


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode a boolean mask -> [(start_index, length), ...]."""
    out: list[tuple[int, int]] = []
    if mask.size == 0:
        return out
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    for lo, hi in zip(np.r_[0, idx], np.r_[idx, mask.size]):
        if mask[lo]:
            out.append((int(lo), int(hi - lo)))
    return out


def mad_outliers(a, k: float = 20.0) -> dict:
    """Robust outlier stats: how much of ``a`` sits beyond median +/- k*MAD.

    Scale-free, so it catches garbage bursts (the false-header corruption
    seen in EMG) without knowing the units.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "fraction": 0.0, "n_out": 0, "median": float("nan"),
                "mad": float("nan"), "min": float("nan"), "max": float("nan")}
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) or float(np.std(a)) or 1e-9
    n_out = int(np.sum(np.abs(a - med) > k * mad))
    return {"n": n, "fraction": n_out / n, "n_out": n_out, "median": med,
            "mad": mad, "min": float(np.min(a)), "max": float(np.max(a))}


def gap_segments(t: np.ndarray, gap_mask: np.ndarray, med: float) -> list[dict]:
    """Merge consecutive over-long intervals into one segment each."""
    out: list[dict] = []
    for lo, length in runs(gap_mask):
        hi = lo + length - 1
        gap_s = float(t[hi + 1] - t[lo])
        out.append({"t": float(t[lo]), "offset": float(t[lo] - t[0]),
                    "gap_s": gap_s,
                    "missed": int(round(gap_s / med)) if med > 0 else 0})
    return out


def window_counts(t: np.ndarray, window_s: float = 1.0,
                  stride_s: float | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Sample counts in fixed-width windows -> (window_starts, counts).

    ``stride_s=None`` means non-overlapping windows (the classic per-second
    histogram); a smaller stride gives overlapping ones.

    The grid starts at the first sample, not at a rounded second, and stops
    before the last partial window.  Every returned window is therefore
    fully covered by the recording — otherwise the leading and trailing
    fragments show up as rate drops that are really just the edges of the
    data, which is a false alarm on every single stream.
    """
    stride = window_s if stride_s is None else stride_s
    t = np.asarray(t, dtype=np.float64)
    t = np.sort(t[np.isfinite(t)])
    if t.size == 0 or stride <= 0 or window_s <= 0:
        return np.empty(0), np.empty(0, dtype=int)
    base = float(t[0])
    n_win = int(np.floor((float(t[-1]) - base - window_s) / stride)) + 1
    if n_win <= 0:
        return np.empty(0), np.empty(0, dtype=int)
    starts = base + np.arange(n_win) * stride
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, starts + window_s, side="left")
    return starts, (hi - lo).astype(int)


# =============================================================================
# Timestamp checks
# =============================================================================

@dataclass(frozen=True)
class TimestampSanity(BaseCheck):
    """Is this series usable at all: present, finite, ordered, distinct."""

    def run(self, ctx: CheckContext) -> CheckOutput:
        label = self.series or ctx.default_series
        s = self.target(ctx)
        out = CheckOutput()
        if s is None:
            # The modality declared this series but the file has no such
            # field.  A stream with no timeline cannot be aligned to
            # anything else in the session — that is an error, not a
            # warning: recorder policy is "no trustworthy timestamps, no
            # timestamps at all", and QC must surface that loudly.
            if label:
                out.findings.append(self.finding(
                    "ERROR", f"缺少 {label} 时间戳 — 无法与其他流对齐",
                    subject=label))
            return out
        if s.n == 0:
            # "Nothing in the window" and "nothing at all" point at very
            # different faults — a clock that disagrees with the rest of the
            # session versus a sensor that never produced data.
            if s.raw is not None and s.raw.size:
                out.findings.append(self.finding(
                    "ERROR",
                    f"{s.raw.size} 个样本全部落在运行窗口之外 — 时钟与本次会话不一致",
                    subject=s.label, observed=float(s.raw.size),
                    detail={"t0": float(s.raw[0]), "t1": float(s.raw[-1])}))
            else:
                out.findings.append(self.finding(
                    "ERROR", "没有任何样本", subject=s.label))
            return out
        if s.n_nan:
            out.findings.append(self.finding(
                "ERROR", f"{s.n_nan}/{s.n} 个时间戳为 NaN",
                subject=s.label, observed=float(s.n_nan)))
        if s.n < 2:
            out.findings.append(self.finding(
                "WARN", f"仅 {s.n} 个样本 — 无法统计时序", subject=s.label))
            return out

        n_back = int(np.sum(s.dt < 0))
        n_dup = int(np.sum(s.dt == 0))
        if n_back:
            out.findings.append(self.finding(
                "WARN", f"{n_back} 个时间戳回退(最大回退 {-float(s.dt.min()):.3f}s)",
                subject=s.label, observed=float(n_back),
                detail={"max_backstep_s": -float(s.dt.min())}))
        if n_dup:
            out.findings.append(self.finding(
                "WARN", f"{n_dup} 个重复时间戳({n_dup / s.dt.size:.1%})",
                subject=s.label, observed=float(n_dup),
                detail={"fraction": n_dup / s.dt.size}))
        return out


@dataclass(frozen=True)
class TimestampJump(BaseCheck):
    """Intervals far above the median — a stall the recorder did not notice."""

    mult: float = 3.0        # interval > mult x median = a jump

    def run(self, ctx: CheckContext) -> CheckOutput:
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n < 2 or s.interval_median <= 0:
            return out
        thr = self.mult * s.interval_median
        idx = np.flatnonzero(s.dt > thr)
        if idx.size == 0:
            return out
        worst = np.sort(s.dt[idx])[-3:][::-1]
        out.findings.append(self.finding(
            "WARN",
            f"{idx.size} 处跳变 > {self.mult:g}× 中位间隔,最大 "
            + ", ".join(f"{x:.3f}s" for x in worst),
            subject=s.label, threshold=thr, observed=float(s.dt[idx].max()),
            spans=[Span(float(s.t[i]), float(s.dt[i]),
                        f"跳变 {s.dt[i]:.3f}s") for i in idx[:50]]))
        return out


@dataclass(frozen=True)
class TimestampGap(BaseCheck):
    """Stretches with no samples at all — data that never arrived.

    A hole longer than ``min_s`` is an ERROR, not a warning: unlike jitter or a
    rate wobble, missing data cannot be recovered or worked around, and a
    tenth of a second is already tens of samples on every stream here.  The
    ``mult`` floor only keeps the threshold above one sampling interval, so
    an unusually slow stream cannot report its own cadence as a gap.
    """

    min_s: float = 0.1
    mult: float = 1.5
    note: str = ""              # appended to the message by subclasses

    def scan_targets(self, ctx: CheckContext
                     ) -> list[tuple[str, np.ndarray, float]]:
        """What to scan for holes: ``(subject, timestamps, median interval)``.

        One entry per thing that can independently go missing.  Subclasses
        override this to feed the same detection a different notion of
        "present" — see ``TrackingGap``, where the timeline is unbroken but
        the poses in it are not usable.
        """
        s = self.target(ctx)
        if s is None or s.n < 2 or s.interval_median <= 0:
            return []
        return [(s.label, s.t, s.interval_median)]

    def run(self, ctx: CheckContext) -> CheckOutput:
        out = CheckOutput()
        for subject, t, med in self.scan_targets(ctx):
            if t.size < 2 or med <= 0:
                continue
            thr = max(self.min_s, self.mult * med)
            segs = gap_segments(t, np.diff(t) > thr, med)
            total = sum(g["gap_s"] for g in segs)
            out.stats[subject] = {"n_gaps": len(segs), "total_gap_s": total}
            if not segs:
                continue
            worst = max(g["gap_s"] for g in segs)
            out.stats[subject]["worst_gap_s"] = worst
            out.findings.append(self.finding(
                "ERROR",
                f"{len(segs)} 段数据缺失 > {thr:.3f}s,共 {total:.2f}s,"
                f"最长 {worst:.2f}s{self.note}",
                subject=subject, threshold=thr, observed=worst,
                detail={"n_segments": len(segs), "total_s": total,
                        "segments": [{"t_offset": g["offset"],
                                      "gap_s": g["gap_s"],
                                      "missed": g["missed"]}
                                     for g in segs[:20]]},
                spans=[Span(g["t"], g["gap_s"],
                            f"缺 {g['missed']} 个样本") for g in segs[:50]]))
        return out


@dataclass(frozen=True)
class IntervalJitter(BaseCheck):
    """Spread of the sampling interval — steady cadence or not."""

    cv_warn: float = 0.3

    def run(self, ctx: CheckContext) -> CheckOutput:
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n < 2 or not np.isfinite(s.interval_cv):
            return out
        if s.interval_cv > self.cv_warn:
            out.findings.append(self.finding(
                "WARN", f"间隔抖动 CV {s.interval_cv:.2f} — 时序不稳定",
                subject=s.label, threshold=self.cv_warn,
                observed=s.interval_cv))
        return out


# =============================================================================
# Rate checks
# =============================================================================

@dataclass(frozen=True)
class SlidingWindowRate(BaseCheck):
    """Sample rate measured in windows, to catch drops a mean would hide.

    Defaults to non-overlapping 1-second windows.  Set ``stride_s`` below
    ``window_s`` for genuinely sliding windows, which localise a brief drop
    that a bucket boundary would otherwise split in half.
    """

    window_s: float = 1.0
    stride_s: float | None = None
    dev_frac: float = 0.5     # +/- this far from the median rate = deviating

    def run(self, ctx: CheckContext) -> CheckOutput:
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n < 2:
            return out
        starts, counts = window_counts(s.t, self.window_s, self.stride_s)
        if counts.size == 0:
            return out

        nonzero = counts[counts > 0]
        out.stats.update({
            "window_s": self.window_s, "n_windows": int(counts.size),
            "per_window_min": int(counts.min()),
            "per_window_max": int(counts.max()),
            "per_window_avg": float(counts.mean()),
            "per_window_std": float(counts.std()),
            "empty_windows": int(np.sum(counts == 0)),
        })
        if nonzero.size == 0:
            out.findings.append(self.finding(
                "ERROR", "所有窗口都没有样本", subject=s.label))
            return out

        med = float(np.median(nonzero))
        lo, hi = med * (1 - self.dev_frac), med * (1 + self.dev_frac)
        dev = (counts > 0) & ((counts < lo) | (counts > hi))
        empty = counts == 0

        if dev.any():
            idx = np.flatnonzero(dev)
            worst = idx[np.argsort(-np.abs(counts[idx] - med))[:3]]
            out.findings.append(self.finding(
                "WARN",
                f"{dev.sum()}/{counts.size} 个窗口偏离中位速率 "
                f"±{self.dev_frac:.0%}(中位 {med:.0f}/窗口): "
                + ", ".join(f"t={starts[i]:.0f}s:{counts[i]}" for i in worst),
                subject=s.label, threshold=med, observed=float(counts[idx].min()),
                spans=[Span(float(starts[i]), self.window_s,
                            f"{counts[i]}/窗口") for i in idx[:50]]))
        if empty.any():
            idx = np.flatnonzero(empty)
            out.findings.append(self.finding(
                "WARN", f"{empty.sum()}/{counts.size} 个窗口没有任何样本",
                subject=s.label, observed=float(empty.sum()),
                spans=[Span(float(starts[i]), self.window_s, "无样本")
                       for i in idx[:50]]))
        return out


@dataclass(frozen=True)
class WindowSampleCount(BaseCheck):
    """Samples inside the run window against what the nominal rate implies.

    A 20.21 s window at 1000 Hz should hold ~20210 samples.  A shortfall means
    samples went missing between the device and the file — which the EEG
    block-continuity check only catches when the loss lands on a block
    boundary, and which a rate check misses entirely because a stream that
    drops a chunk and then continues still has a healthy median interval.

    Only meaningful with a marker window; without one the span and the count
    are derived from the same array and the comparison is circular.
    """

    expected: float | None = None    # None = take it from the series
    warn_frac: float = 0.01          # relative deviation worth reporting

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.window is not None

    def run(self, ctx: CheckContext) -> CheckOutput:
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n == 0:
            return out
        rate = self.expected or s.expected_rate
        if not rate:
            return out

        span = float(ctx.window["t1"] - ctx.window["t0"])
        expected = span * rate
        if expected <= 0:
            return out
        diff = s.n - expected
        frac = diff / expected
        out.stats.update({"window_s": span, "expected_samples": expected,
                          "actual_samples": s.n, "deviation": frac})

        if abs(frac) > self.warn_frac:
            out.findings.append(self.finding(
                "WARN",
                f"窗口 {span:.2f}s @{rate:g}Hz 应有 {expected:.0f} 个样本,"
                f"实际 {s.n} ({diff:+.0f}, {frac:+.2%})",
                subject=s.label, threshold=self.warn_frac, observed=frac,
                detail={"expected": expected, "actual": s.n,
                        "window_s": span, "rate": rate}))
        return out


@dataclass(frozen=True)
class ExpectedRate(BaseCheck):
    """Mean rate against what the device was configured to deliver."""

    expected: float | None = None   # None = take it from the series
    min_frac: float = 0.5

    def run(self, ctx: CheckContext) -> CheckOutput:
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n < 2:
            return out
        expected = self.expected or s.expected_rate
        if not expected or not np.isfinite(s.mean_rate):
            return out
        if s.mean_rate < self.min_frac * expected:
            out.findings.append(self.finding(
                "WARN",
                f"平均速率 {s.mean_rate:.0f}/s 低于预期 {expected:.0f}/s",
                subject=s.label, threshold=self.min_frac * expected,
                observed=s.mean_rate, detail={"expected_rate": expected}))
        return out


# =============================================================================
# Data-value checks
# =============================================================================

@dataclass(frozen=True)
class MadOutlier(BaseCheck):
    """Values far outside the robust spread — a corrupt parse, a bad contact.

    Reported per channel wherever the array has channels: "1.16% of the
    values are outliers" is not actionable, whereas "ch2 12.69%" points at
    one electrode.  A whole-array figure would also hide a single ruined
    channel behind seven healthy ones.
    """

    field: str = ""
    frac_warn: float = 0.001
    k: float = 20.0
    top_n: int = 5              # channels named in the message

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.size > 0

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field))
        out = CheckOutput()
        overall = ctx.artifact(f"mad:{self.field}:{self.k}",
                               lambda: mad_outliers(a, self.k))
        out.stats.update({"shape": list(a.shape),
                          "range": [float(np.nanmin(a)), float(np.nanmax(a))],
                          "outlier_fraction": overall["fraction"]})

        if a.ndim != 2:
            if overall["fraction"] > self.frac_warn:
                out.findings.append(self.finding(
                    "WARN",
                    f"{overall['fraction']:.2%} 的值为 MAD 异常值 — 可能是解析损坏",
                    field=self.field, threshold=self.frac_warn,
                    observed=overall["fraction"]))
            return out

        per = [mad_outliers(a[:, c], self.k)["fraction"]
               for c in range(a.shape[1])]
        bad = sorted(((c, f) for c, f in enumerate(per) if f > self.frac_warn),
                     key=lambda x: -x[1])
        out.stats["per_channel_fraction"] = per
        out.stats["bad_channels"] = [c for c, _ in bad]
        if not bad:
            return out

        named = ", ".join(f"ch{c} {f:.2%}" for c, f in bad[:self.top_n])
        more = f" 等 {len(bad)} 个通道" if len(bad) > self.top_n else ""
        out.findings.append(self.finding(
            "WARN",
            f"{len(bad)}/{a.shape[1]} 个通道存在 MAD 异常值: {named}{more} "
            "— 可能是解析损坏或接触不良",
            field=self.field, subject=f"ch{bad[0][0]}",
            threshold=self.frac_warn, observed=bad[0][1],
            detail={"channels": {f"ch{c}": round(f, 6) for c, f in bad[:20]},
                    "n_bad_channels": len(bad),
                    "overall_fraction": overall["fraction"]}))
        return out


@dataclass(frozen=True)
class DeadChannel(BaseCheck):
    """Columns that never move — an electrode that fell off, a dead sensor."""

    field: str = ""
    cap: int = 10               # list at most this many in the message

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.ndim == 2 and a.size > 0

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field), dtype=np.float64)
        out = CheckOutput()
        dead = [c for c in range(a.shape[1]) if np.nanstd(a[:, c]) == 0]
        out.stats["dead_channels"] = dead
        if dead:
            shown = dead[:self.cap]
            tail = "…" if len(dead) > self.cap else ""
            out.findings.append(self.finding(
                "WARN", f"{len(dead)} 个死通道(无波动): {shown}{tail}",
                field=self.field, observed=float(len(dead)),
                detail={"channels": dead}))
        return out


@dataclass(frozen=True)
class NanFraction(BaseCheck):
    """How much of an array is NaN — a sensor dropping out mid-recording."""

    field: str = ""
    frac_warn: float = 0.001

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.size > 0 and a.dtype.kind == "f"

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field))
        out = CheckOutput()
        frac = float(np.isnan(a).sum()) / a.size
        out.stats.update({"shape": list(a.shape), "nan_fraction": frac})
        if frac > self.frac_warn:
            out.findings.append(self.finding(
                "WARN", f"{frac:.1%} 为 NaN(设备断连?)", field=self.field,
                threshold=self.frac_warn, observed=frac))
        return out


@dataclass(frozen=True)
class ValueBounds(BaseCheck):
    """Values outside the physically plausible range."""

    field: str = ""
    lo: float = float("-inf")
    hi: float = float("inf")

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.size > 0

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field), dtype=np.float64)
        out = CheckOutput()
        finite = a[np.isfinite(a)]
        n_oob = int(np.sum((finite < self.lo) | (finite > self.hi)))
        out.stats["n_out_of_bounds"] = n_oob
        if n_oob:
            out.findings.append(self.finding(
                "WARN",
                f"{n_oob} 个值超出合理范围 [{self.lo:g}, {self.hi:g}]",
                field=self.field, observed=float(n_oob),
                detail={"range": [float(finite.min()), float(finite.max())]}))
        return out


@dataclass(frozen=True)
class ValueJump(BaseCheck):
    """Teleports: a tracked point moving further between two samples than
    it physically could.  Shared by the VR trackers and the glove skeleton —
    only the threshold and the array layout differ."""

    field: str = ""
    thr: float = 0.2            # metres between consecutive samples
    valid_field: str = ""       # optional boolean mask of usable samples
    per_node: bool = False      # (T, N, 3) -> check every node

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.ndim >= 2 and len(a) > 2

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field), dtype=np.float64)
        s = self.target(ctx)
        out = CheckOutput()

        if self.per_node and a.ndim == 3:
            disp = np.linalg.norm(np.diff(a, axis=0), axis=-1)   # (T-1, N)
            hit = disp > self.thr
            n = int(hit.sum())
            idx = np.flatnonzero(hit.any(axis=1))
        else:
            flat = a.reshape(len(a), -1)
            disp = np.linalg.norm(np.diff(flat, axis=0), axis=-1)
            hit = disp > self.thr
            n = int(hit.sum())
            idx = np.flatnonzero(hit)

        out.stats["n_jumps"] = n
        if n:
            spans = []
            if s is not None and s.n == len(a):
                spans = [Span(float(s.t[i + 1]), 0.0, "瞬移") for i in idx[:50]]
            out.findings.append(self.finding(
                "WARN", f"{n} 处瞬移(相邻样本位移 > {self.thr:g})",
                field=self.field, threshold=self.thr,
                observed=float(disp.max()), spans=spans))
        return out


# =============================================================================
# Video
# =============================================================================

@dataclass
class VideoDecode:
    """One decode pass over an mp4, shared by every video check.

    Frames are sampled at ``rate_hz`` rather than decoded in full detail:
    two samples a second is plenty to spot a freeze or a black stretch, and
    it keeps a 33 MB file to a couple of seconds.
    """

    file: str
    n_frames: int = 0
    fps: float = 30.0
    rate_hz: float = 2.0
    lums: np.ndarray = None          # mean luminance per sampled frame
    diffs: np.ndarray = None         # |frame - prev sampled frame|
    t_samp: np.ndarray = None        # timestamp per sampled frame
    n_window: int | None = None      # frames inside the run window
    opened: bool = True


def decode_video(mp4: Path, frame_ts, window: dict | None,
                 rate_hz: float = 2.0) -> VideoDecode:
    """Decode ``mp4`` once, collecting everything the video checks need."""
    import cv2

    out = VideoDecode(file=mp4.name, rate_hz=rate_hz)
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        out.opened = False
        return out
    out.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    stride = max(1, int(round(out.fps / rate_hz)))

    ts = np.asarray(frame_ts, dtype=np.float64) if frame_ts is not None else None
    i0 = i1 = None
    if window is not None and ts is not None and ts.size:
        inside = np.flatnonzero((ts >= window["t0"]) & (ts <= window["t1"]))
        if inside.size:
            i0, i1 = int(inside[0]), int(inside[-1])

    lums, diffs, t_samp, prev, n = [], [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if i0 is not None and not (i0 < n <= i1 + 1):
            continue                      # outside the run window
        if n % stride != 1 and stride > 1:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lums.append(float(gray.mean()))
        t_samp.append(float(ts[n - 1]) if ts is not None and n - 1 < ts.size
                      else (n - 1) / out.fps)
        if prev is not None:
            diffs.append(float(np.abs(gray.astype(np.float32) - prev).mean()))
        prev = gray
    cap.release()

    out.n_frames = n
    out.n_window = (i1 - i0 + 1) if i0 is not None else None
    out.lums = np.asarray(lums, dtype=np.float64)
    out.diffs = np.asarray(diffs, dtype=np.float64)
    out.t_samp = np.asarray(t_samp, dtype=np.float64)
    return out


@dataclass(frozen=True)
class _VideoCheck(BaseCheck):
    """Shared plumbing: locate the mp4 and reuse the one decode."""

    video: str = ""              # file name; "" = the only mp4 in the dir

    def _path(self, ctx: CheckContext) -> Path | None:
        if self.video:
            p = ctx.dir / self.video
            return p if p.exists() else None
        found = sorted(ctx.dir.glob("*.mp4"))
        return found[0] if found else None

    def applies(self, ctx: CheckContext) -> bool:
        return self._path(ctx) is not None

    def decode(self, ctx: CheckContext) -> VideoDecode:
        """The one decode of this file, shared via the context cache.

        Frame indices count from the start of the container, so the decode
        gets the UNwindowed timestamps plus the window and works out the
        in-window frame range itself.
        """
        p = self._path(ctx)
        s = self.target(ctx)
        raw = s.raw if s is not None else None
        return ctx.artifact(
            f"video:{p.name}", lambda: decode_video(p, raw, ctx.window))


@dataclass(frozen=True)
class FrameCountMatch(_VideoCheck):
    """Container frames against timestamp count — a writer that fell behind
    drops frames silently, and only this comparison shows it."""

    tol_frac: float = 0.02

    def run(self, ctx: CheckContext) -> CheckOutput:
        dec = self.decode(ctx)
        s = self.target(ctx)
        out = CheckOutput()
        if not dec.opened:
            out.findings.append(self.finding(
                "ERROR", f"无法打开视频 {dec.file}", field=dec.file))
            return out
        out.stats.update({"file": dec.file, "n_frames": dec.n_frames,
                          "fps": dec.fps})
        if s is None or s.n == 0:
            return out
        n_ts = s.n
        # Inside a run window the comparison is window-vs-window; the total
        # decoded count stays in the stats but is not what is compared.
        n_frames = dec.n_window if dec.n_window is not None else dec.n_frames
        out.stats.update({"n_timestamps": n_ts, "n_compared": n_frames})
        if abs(n_frames - n_ts) > max(2, self.tol_frac * n_ts):
            out.findings.append(self.finding(
                "WARN", f"视频帧数({n_frames})与时间戳数({n_ts})不一致",
                field=dec.file, observed=float(abs(n_frames - n_ts)),
                threshold=max(2.0, self.tol_frac * n_ts)))
        return out


@dataclass(frozen=True)
class BlackFrame(_VideoCheck):
    """Stretches with no light reaching the sensor — lens cap, dead feed."""

    lum_thr: float = 8.0        # mean luminance below this = black (0-255)
    run_s: float = 2.0          # a black stretch longer than this is a WARN
    fail_frac: float = 0.5      # this much of the file black = ERROR

    def run(self, ctx: CheckContext) -> CheckOutput:
        dec = self.decode(ctx)
        out = CheckOutput()
        if not dec.opened or dec.lums.size == 0:
            return out
        black = dec.lums < self.lum_thr
        frac = float(black.mean())
        mean_lum = float(dec.lums.mean())
        out.stats.update({"mean_luminance": mean_lum, "black_fraction": frac})

        long = [(lo, length / dec.rate_hz) for lo, length in runs(black)
                if length / dec.rate_hz > self.run_s]
        if long:
            out.findings.append(self.finding(
                "WARN",
                f"视频黑屏: {len(long)} 段("
                + ", ".join(f"{d:.1f}s" for _, d in long) + ")",
                field=dec.file, threshold=self.lum_thr, observed=mean_lum,
                spans=[Span(float(dec.t_samp[lo]), d, f"黑屏 {d:.1f}s")
                       for lo, d in long[:50]]))
        if frac > self.fail_frac:
            out.findings.append(self.finding(
                "ERROR", f"{frac:.0%} 的视频为黑屏", field=dec.file,
                threshold=self.fail_frac, observed=frac))
        return out


@dataclass(frozen=True)
class Freeze(_VideoCheck):
    """Stretches where the image stops changing — a stalled capture thread
    still produces frames, so only pixel differences reveal it."""

    diff_thr: float = 1.0       # mean |frame diff| below this = frozen
    run_s: float = 2.0

    def run(self, ctx: CheckContext) -> CheckOutput:
        dec = self.decode(ctx)
        out = CheckOutput()
        if not dec.opened or dec.diffs.size == 0:
            return out
        frozen = dec.diffs < self.diff_thr
        frac = float(frozen.mean())
        out.stats["frozen_fraction"] = frac
        long = [(lo, length / dec.rate_hz) for lo, length in runs(frozen)
                if length / dec.rate_hz > self.run_s]
        if long:
            out.findings.append(self.finding(
                "WARN",
                f"视频冻结: {len(long)} 段("
                + ", ".join(f"{d:.1f}s" for _, d in long) + ")",
                field=dec.file, threshold=self.diff_thr, observed=frac,
                spans=[Span(float(dec.t_samp[lo]), d, f"冻结 {d:.1f}s")
                       for lo, d in long[:50]]))
        return out


# =============================================================================
# Bundles
# =============================================================================

def ts_checks(series: str | None = None, *,
              expected_rate: float | None = None) -> list[BaseCheck]:
    """The standard timestamp battery for one series.

    Every modality wants the same five verdicts about its clock, so they are
    bundled rather than repeated seven times; ``checks`` lists may nest.
    """
    out: list[BaseCheck] = [
        TimestampSanity(series=series),
        TimestampJump(series=series),
        TimestampGap(series=series),
        IntervalJitter(series=series),
        SlidingWindowRate(series=series),
    ]
    if expected_rate:
        out.append(ExpectedRate(series=series, expected=expected_rate))
    return out
