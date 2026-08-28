"""Core types for the session quality checkers.

A checker is a pure, offline object: hand it a recorder's output directory and
it loads the saved NPZ / MP4 / log files and reports what looks wrong.
Nothing here touches hardware or the recording lifecycle.

Three layers::

    BaseChecker     one modality (emg, camera, ...).  Declares `checks`;
                    owns no checking logic of its own.
      BaseCheck     one independent check — a jump detector, a black-frame
                    detector.  Reusable across modalities; its constructor
                    arguments are its thresholds, so a modality's `checks`
                    list is also its configuration.
      CheckContext  the per-run working set the checks share: lazy NPZ
                    access, windowed timestamp series, and a memo cache so
                    an expensive artifact (a video decode) is produced once
                    no matter how many checks want it.

Checks return ``Finding`` objects instead of printing, so one run can be
rendered as a console report or serialized to JSON.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Sequence

import numpy as np

# =============================================================================
# Levels
# =============================================================================

LEVELS = ("INFO", "WARN", "ERROR")
_LEVEL_ORDER = {name: i for i, name in enumerate(LEVELS)}


def worst_level(levels: Iterable[str]) -> str:
    """The most severe level present, or INFO when there is nothing."""
    return max(levels, key=lambda l: _LEVEL_ORDER.get(l, 0), default="INFO")


# =============================================================================
# Findings
# =============================================================================

@dataclass(frozen=True)
class Span:
    """A time-anchored slice of a problem, for report timelines."""

    t: float          # epoch seconds
    dur: float = 0.0
    msg: str = ""

    def to_dict(self) -> dict:
        return {"t": float(self.t), "dur": float(self.dur), "msg": self.msg}


@dataclass(frozen=True)
class Finding:
    """One thing a check decided is worth reporting.

    ``threshold`` / ``observed`` carry the numbers behind the verdict so a
    reader can judge how marginal it was without re-deriving anything, and so
    the JSON is useful without parsing ``message``.
    """

    level: str                        # INFO | WARN | ERROR
    message: str                      # human-readable, Chinese
    check: str = ""                   # producing check's class name
    field: str = ""                   # npz field / file the finding concerns
    subject: str = ""                 # sub-stream: tracker serial, channel, ...
    threshold: float | None = None
    observed: float | None = None
    detail: dict = dataclasses.field(default_factory=dict)
    spans: list[Span] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-ready; empty optional fields are dropped to keep it legible."""
        out: dict[str, Any] = {"level": self.level, "check": self.check,
                               "message": self.message}
        for key in ("field", "subject"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        for key in ("threshold", "observed"):
            if getattr(self, key) is not None:
                out[key] = float(getattr(self, key))
        if self.detail:
            out["detail"] = _jsonable(self.detail)
        if self.spans:
            out["spans"] = [s.to_dict() for s in self.spans]
        return out


@dataclass
class CheckOutput:
    """What one check produces: findings plus numbers worth reporting."""

    findings: list[Finding] = dataclasses.field(default_factory=list)
    stats: dict = dataclasses.field(default_factory=dict)

    @property
    def level(self) -> str:
        return worst_level(f.level for f in self.findings)


def _jsonable(obj: Any) -> Any:
    """numpy scalars/arrays -> plain Python, recursively."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# =============================================================================
# Timestamp series
# =============================================================================

@dataclass
class Series:
    """One timestamp series, already restricted to the run window.

    Derived numbers are computed once and cached.  Intervals use only the
    POSITIVE diffs: burst-write recorders stamp every frame of a read with
    the same time, so the raw median interval is 0 and hides the cadence.
    """

    label: str
    t: np.ndarray
    n_outside: int = 0
    expected_rate: float | None = None
    #: the series before window restriction — video checks need it to map
    #: container frame indices, which are numbered from the start of the file
    raw: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.t.size)

    @cached_property
    def n_nan(self) -> int:
        return int(np.isnan(self.t).sum()) if self.n else 0

    @cached_property
    def t0(self) -> float | None:
        return float(self.t[0]) if self.n else None

    @cached_property
    def t1(self) -> float | None:
        return float(self.t[-1]) if self.n else None

    @cached_property
    def duration(self) -> float:
        return (self.t1 - self.t0) if self.n else 0.0

    @cached_property
    def rate(self) -> float:
        return (self.n - 1) / self.duration if self.duration > 0 else 0.0

    @cached_property
    def dt(self) -> np.ndarray:
        return np.diff(self.t) if self.n >= 2 else np.empty(0, dtype=np.float64)

    @cached_property
    def dt_pos(self) -> np.ndarray:
        d = self.dt
        return d[d > 0]

    @cached_property
    def interval_median(self) -> float:
        return float(np.median(self.dt_pos)) if self.dt_pos.size else 0.0

    @cached_property
    def interval_cv(self) -> float:
        med = self.interval_median
        return float(np.std(self.dt_pos) / med) if med > 0 else float("nan")

    @cached_property
    def mean_rate(self) -> float:
        med = self.interval_median
        return 1.0 / med if med > 0 else float("nan")

    def summary(self) -> dict:
        """The numbers a report shows for this series."""
        out = {"n": self.n, "t0": self.t0, "t1": self.t1,
               "duration": self.duration, "rate": self.rate,
               "interval_median": self.interval_median,
               "interval_cv": self.interval_cv, "mean_rate": self.mean_rate}
        if self.n_outside:
            out["n_outside_window"] = self.n_outside
        if self.expected_rate:
            out["expected_rate"] = self.expected_rate
        return out


@dataclass(frozen=True)
class _SeriesSource:
    key: str | None = None
    loader: Callable[[], np.ndarray | None] | None = None
    expected_rate: float | None = None


# =============================================================================
# Context
# =============================================================================

class CheckContext:
    """Per-run working set shared by one modality's checks.

    Holds the NPZ handle, the registered timestamp series, and a memo cache.
    The cache is what lets independent checks compose without paying twice
    for the same work — most of it is cheap and cached only so every check
    quotes the same number, but the video decode is genuinely expensive and
    must happen exactly once.
    """

    def __init__(self, *, stream: str, directory: Path,
                 window: dict | None = None,
                 default_series: str | None = None) -> None:
        self.stream = stream
        self.dir = Path(directory)
        self.window = window
        #: series a check reads when it names none
        self.default_series = default_series
        self._npz = None
        self._npz_loaded = False
        self._arrays: dict[str, np.ndarray | None] = {}
        self._sources: dict[str, _SeriesSource] = {}
        self._series: dict[str, Series | None] = {}
        self._cache: dict[str, Any] = {}

    # ---- files ---------------------------------------------------------

    @property
    def npz_path(self) -> Path | None:
        files = sorted(p for p in self.dir.glob("*.npz"))
        return files[0] if files else None

    @property
    def npz(self):
        """The modality's NPZ, opened once.  None when the dir has none."""
        if not self._npz_loaded:
            self._npz_loaded = True
            p = self.npz_path
            if p is not None:
                try:
                    self._npz = np.load(p, allow_pickle=False)
                except (OSError, ValueError):
                    self._npz = None
        return self._npz

    @property
    def has_npz(self) -> bool:
        return self.npz is not None

    def arr(self, key: str | None) -> np.ndarray | None:
        """One NPZ field, decompressed once and remembered.

        Missing fields return None rather than raising: a check that wants a
        field the recorder never wrote should skip, not crash the session.
        """
        if key is None:
            return None
        if key not in self._arrays:
            z = self.npz
            try:
                self._arrays[key] = (z[key] if z is not None
                                     and key in z.files else None)
            except (KeyError, OSError, ValueError):
                self._arrays[key] = None
        return self._arrays[key]

    def files(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.name for p in self.dir.iterdir() if p.is_file())

    # ---- timestamp series ----------------------------------------------

    def add_series(self, label: str, *, key: str | None = None,
                   loader: Callable[[], np.ndarray | None] | None = None,
                   expected_rate: float | None = None) -> None:
        """Register a timestamp series a check can ask for by label.

        ``key`` names an NPZ field; ``loader`` covers anything else (camera
        reads a sidecar txt in preference to its NPZ).
        """
        self._sources[label] = _SeriesSource(key, loader, expected_rate)

    @property
    def series_labels(self) -> list[str]:
        return list(self._sources)

    def series(self, label: str) -> Series | None:
        """The window-restricted series, built once per label."""
        if label in self._series:
            return self._series[label]
        src = self._sources.get(label)
        raw = None
        if src is not None:
            raw = src.loader() if src.loader is not None else self.arr(src.key)
        if raw is None:
            self._series[label] = None
            return None
        full = np.asarray(raw, dtype=np.float64).ravel()
        t, n_outside = full, 0
        mask = self.mask(full)
        if mask is not None:
            n_outside = int((~mask).sum())
            t = full[mask]
        s = Series(label=label, t=t, n_outside=n_outside,
                   expected_rate=src.expected_rate, raw=full)
        self._series[label] = s
        return s

    def mask(self, t: np.ndarray | None) -> np.ndarray | None:
        """Boolean mask selecting the run window, or None when unrestricted."""
        if self.window is None or t is None or len(t) == 0:
            return None
        t = np.asarray(t, dtype=np.float64)
        return (t >= self.window["t0"]) & (t <= self.window["t1"])

    # ---- memoized artifacts --------------------------------------------

    def artifact(self, key: str, factory: Callable[[], Any]) -> Any:
        """Produce ``key`` once and hand the same object to every caller."""
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    # ---- lifecycle ------------------------------------------------------

    def close(self) -> None:
        if self._npz is not None:
            try:
                self._npz.close()
            except Exception:
                pass
            self._npz = None


# =============================================================================
# Checks
# =============================================================================

@dataclass(frozen=True)
class BaseCheck:
    """One independent quality check.

    Subclasses are frozen dataclasses whose fields are the thresholds, which
    is why a modality's ``checks`` list doubles as its configuration.  A
    check is pure: it reads the context and returns findings, mutating
    nothing, so instances are safe to share between runs.
    """

    #: which registered series to read; None = the checker's default.
    #: keyword-only so a subclass's own first field stays positional —
    #: ``MadOutlier("emg_data")`` reads better than forcing ``field=``.
    series: str | None = dataclasses.field(default=None, kw_only=True)

    @property
    def name(self) -> str:
        return type(self).__name__

    def applies(self, ctx: CheckContext) -> bool:
        """Skip cleanly when the data this check needs is absent."""
        return True

    def run(self, ctx: CheckContext) -> CheckOutput:
        raise NotImplementedError

    # ---- helpers for subclasses ----------------------------------------

    def finding(self, level: str, message: str, **kw: Any) -> Finding:
        """A Finding already stamped with this check's name."""
        kw.setdefault("check", self.name)
        return Finding(level=level, message=message, **kw)

    def target(self, ctx: CheckContext) -> Series | None:
        """The series this check reads: its own, else the checker's default."""
        label = self.series or ctx.default_series
        return ctx.series(label) if label else None


def flatten_checks(checks: Sequence) -> list[BaseCheck]:
    """Allow ``checks`` to nest lists, so bundles read as one entry."""
    out: list[BaseCheck] = []
    for item in checks:
        if isinstance(item, BaseCheck):
            out.append(item)
        else:
            out.extend(flatten_checks(item))
    return out


# =============================================================================
# Reports
# =============================================================================

@dataclass
class StreamReport:
    """Result for one recorder directory."""

    stream: str
    files: list[str] = dataclasses.field(default_factory=list)
    findings: list[Finding] = dataclasses.field(default_factory=list)
    series: dict = dataclasses.field(default_factory=dict)
    stats: dict = dataclasses.field(default_factory=dict)

    @property
    def level(self) -> str:
        return worst_level(f.level for f in self.findings)

    def to_dict(self) -> dict:
        return {"level": self.level, "files": self.files,
                "series": _jsonable(self.series), "stats": _jsonable(self.stats),
                "findings": [f.to_dict() for f in self.findings]}


@dataclass
class SessionReport:
    """Result for a whole session directory."""

    session_dir: str
    window: dict | None = None
    streams: dict = dataclasses.field(default_factory=dict)
    findings: list[Finding] = dataclasses.field(default_factory=list)
    empty_dirs: list[str] = dataclasses.field(default_factory=list)

    @property
    def level(self) -> str:
        return worst_level(
            [f.level for f in self.findings]
            + [r.level for r in self.streams.values()])

    def to_dict(self) -> dict:
        return {"session_dir": self.session_dir, "level": self.level,
                "window": _jsonable(self.window) if self.window else None,
                "empty_dirs": self.empty_dirs,
                "findings": [f.to_dict() for f in self.findings],
                "streams": {k: v.to_dict() for k, v in self.streams.items()}}


# =============================================================================
# Log scan (a fixed pipeline step, not a composable check)
# =============================================================================

_LOG_ERROR = re.compile(r"ERROR|FAILED|Traceback|crash", re.IGNORECASE)
_LOG_MAX = 5


def scan_logs(d: Path) -> list[Finding]:
    """Surface ERROR-ish lines from the recorder's own log files."""
    out: list[Finding] = []
    for log in sorted(d.glob("*.log")):
        try:
            lines = [ln.strip() for ln in log.read_text(
                encoding="utf-8", errors="replace").splitlines()
                if _LOG_ERROR.search(ln)]
        except OSError:
            continue
        if lines:
            out.append(Finding(
                "WARN", f"{log.name} 有 {len(lines)} 行错误日志",
                check="LogScan", field=log.name,
                observed=float(len(lines)),
                detail={"lines": lines[:_LOG_MAX]}))
    return out


# =============================================================================
# Checkers
# =============================================================================

class BaseChecker:
    """One modality.  Owns no checking logic — it composes checks.

    ``run`` is the only template method: build the context, let ``prepare``
    register this modality's series, run every check, scan the logs, and let
    ``finalize`` add anything that isn't a check (a marker listing).
    """

    name: ClassVar[str] = "base"
    #: directory-name prefixes this checker claims
    matches: ClassVar[tuple[str, ...]] = ()
    #: ERROR when the directory has no NPZ (camera works from a txt alone)
    requires_npz: ClassVar[bool] = True
    #: series a check reads when it names none
    default_series: ClassVar[str | None] = None
    checks: ClassVar[Sequence] = ()

    def __init__(self, checker_cfg: dict | None = None) -> None:
        #: ``configs/checker.yaml`` threshold overrides.  Keys are normalized
        #: to the check class name (lower-case, underscores stripped), so
        #: ``timestamp_gap`` and ``TimestampGap`` both match ``TimestampGap``.
        self.checker_cfg = {k.lower().replace("_", ""): v
                            for k, v in (checker_cfg or {}).items()}

    def run(self, directory: Path, window: dict | None = None) -> StreamReport:
        directory = Path(directory)
        report = StreamReport(stream=directory.name)
        ctx = CheckContext(stream=directory.name, directory=directory,
                           window=window,
                           default_series=self.default_series)
        try:
            report.files = ctx.files()
            if self.requires_npz and not ctx.has_npz:
                report.findings.append(Finding(
                    "ERROR", "未找到 npz 数据文件", check="NpzPresent"))
            else:
                self.prepare(ctx)
                for check in flatten_checks(self.checks):
                    # 配置键名按 check 类名匹配(不区分大小写、忽略下划线),
                    # 所以 checker.yaml 里写 timestamp_gap 或 TimestampGap 均可
                    overrides = self.checker_cfg.get(
                        check.name.lower().replace("_", ""))
                    if isinstance(overrides, dict):
                        try:
                            check = dataclasses.replace(check, **overrides)
                        except TypeError:
                            pass    # 未知参数名:保持类默认,不炸整个会话
                    if not check.applies(ctx):
                        continue
                    out = check.run(ctx)
                    report.findings.extend(out.findings)
                    if out.stats:
                        # Same check, different series (EMG runs the timing
                        # battery over both emg and imu) must not overwrite
                        # each other's numbers.
                        key = (f"{check.name}[{check.series}]" if check.series
                               else check.name)
                        report.stats.setdefault(key, {}).update(out.stats)
                for label in ctx.series_labels:
                    s = ctx.series(label)
                    if s is not None:
                        report.series[label] = s.summary()
                self.finalize(ctx, report)
            report.findings.extend(scan_logs(directory))
        finally:
            ctx.close()
        return report

    # ---- modality hooks -------------------------------------------------

    def prepare(self, ctx: CheckContext) -> None:
        """Register timestamp series and normalize any awkward arrays."""

    def finalize(self, ctx: CheckContext, report: StreamReport) -> None:
        """Add modality extras that aren't checks (e.g. a marker listing)."""
