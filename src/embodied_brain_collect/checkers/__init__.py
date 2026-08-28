"""Session quality checkers.

Point :func:`qc_session` at a session directory and it loads whatever each
recorder saved — NPZ, mp4, log — and reports what looks wrong.  Purely
offline; nothing here touches hardware.

The analysis window comes from the marker stream (RUN_START -> RUN_END), and
every modality clips to it, so streams that opened early or ran late are
compared over the same span.  When no such marker pair exists the full data
range is used and the report says so.

Usage::

    from embodied_brain_collect.checkers import qc_session, print_report
    report = qc_session("data/session4")
    print_report(report)
    report.to_dict()      # JSON-ready

Adding a modality means writing a ``BaseChecker`` subclass and listing it in
``CHECKERS`` — the table is ordered, so a more specific prefix must come
before a broader one.
"""

from __future__ import annotations

from pathlib import Path

from .base import (BaseCheck, BaseChecker, CheckContext, CheckOutput, Finding,
                   SessionReport, Span, StreamReport, worst_level)
from .camera import CameraChecker
from .eeg import EegChecker
from .emg import EmgChecker
from .eye import EyeChecker
from .hand_pose import HandPoseChecker
from .marker import MarkerChecker, find_run_window
from .position import PositionChecker

__all__ = [
    "BaseCheck", "BaseChecker", "CheckContext", "CheckOutput", "Finding",
    "SessionReport", "Span", "StreamReport", "worst_level",
    "CHECKERS", "checker_for", "qc_session", "print_report",
]

#: Ordered prefix table.  ``hand_pose`` must precede nothing broader here,
#: but the order is load-bearing in general, so keep specific names first.
CHECKERS: tuple[type[BaseChecker], ...] = (
    EmgChecker,
    EegChecker,
    EyeChecker,
    HandPoseChecker,
    PositionChecker,
    MarkerChecker,
    CameraChecker,
)

_LEVEL_ZH = {"INFO": "提示", "WARN": "警告", "ERROR": "错误"}


def checker_for(dir_name: str) -> type[BaseChecker] | None:
    for cls in CHECKERS:
        if dir_name.startswith(cls.matches):
            return cls
    return None


def _has_data(d: Path) -> bool:
    return any(d.glob("*.npz")) or any(d.glob("*.mp4"))


# =============================================================================
# Session run
# =============================================================================

def qc_session(session_dir: Path | str,
               checker_cfg: dict | None = None) -> SessionReport:
    """Check every recorder directory in one session.

    ``checker_cfg`` carries ``configs/checker.yaml`` threshold overrides
    (``{check_class_lower: {param: value}}``); anything missing keeps the
    check's own default.
    """
    checker_cfg = checker_cfg or {}
    root = Path(session_dir)
    window = find_run_window(root)
    report = SessionReport(session_dir=str(root), window=window)

    if window is None:
        report.findings.append(Finding(
            "WARN", "未找到 RUN_START/RUN_END 标记对 — 按全部数据范围检查",
            check="RunWindow"))

    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if not _has_data(d):
            # A recorder that opened but never saved leaves its log behind.
            # A missing modality is an error, not a warning: the session is
            # incomplete and nothing downstream can reconstruct it.
            report.empty_dirs.append(d.name)
            report.findings.append(Finding(
                "ERROR", f"{d.name}/ 无数据文件 — recorder 未保存",
                check="StreamPresent", subject=d.name))
            continue
        cls = checker_for(d.name)
        if cls is None:
            report.streams[d.name] = StreamReport(
                stream=d.name,
                files=sorted(p.name for p in d.iterdir() if p.is_file()),
                findings=[Finding("INFO", "未知模态 — 已跳过",
                                  check="Dispatch")])
            continue
        report.streams[d.name] = cls(checker_cfg).run(d, window)

    return report


# =============================================================================
# Console report
# =============================================================================

def _zh(level: str) -> str:
    return _LEVEL_ZH.get(level, level)


def _is_scalar(v) -> bool:
    return v is not None and isinstance(v, (int, float, str, bool))


def _fmt(v) -> str:
    if isinstance(v, bool) or not isinstance(v, float):
        return str(v)
    # Sample counts and durations read better in full than as 2.021e+04.
    return f"{v:.1f}" if abs(v) >= 1000 else f"{v:.4g}"


def _print_markers(r: StreamReport) -> None:
    """The marker listing is the whole point of that stream's report."""
    m = r.stats.get("markers")
    if not m or not m.get("items"):
        return
    print(f"    · 标记 {m['n_in_window']}/{m['n_total']} 在窗口内")
    for it in m["items"]:
        print(f"        {it['name']:<14} t=+{it['t_offset']:6.2f}s  "
              f"code={it['code']:>3}")


def _primary(r: StreamReport) -> dict | None:
    """The series that best represents a stream in the summary table."""
    for s in r.series.values():
        if s.get("n"):
            return s
    return next(iter(r.series.values()), None)


def print_report(report: SessionReport) -> None:
    root = report.session_dir
    print(f"\n{'=' * 16} 会话质量检查: {root} {'=' * 16}")

    w = report.window
    if w:
        print(f"运行窗口: RUN_START → RUN_END  {w['t1'] - w['t0']:.2f}s  "
              f"({w['n_markers']} 个标记)")
    for f in report.findings:
        print(f"  {_zh(f.level)}  {f.message}")

    # ---- summary table ----
    print(f"\n{'流':<18}{'样本':>9}{'时长(s)':>10}{'速率/s':>9}{'等级':>8}")
    print("-" * 56)
    for name, r in report.streams.items():
        s = _primary(r) or {}
        n = s.get("n")
        dur = s.get("duration")
        rate = s.get("rate")
        print(f"{name:<18}{(n if n is not None else '-'):>9}"
              f"{(f'{dur:.1f}' if dur else '-'):>10}"
              f"{(f'{rate:.1f}' if rate else '-'):>9}"
              f"{_zh(r.level):>8}")

    # ---- per-stream detail ----
    for name, r in report.streams.items():
        if not r.findings and not r.stats:
            continue
        print(f"\n[{name}]  {_zh(r.level)}")

        # Numbers worth seeing even when nothing tripped a threshold — a
        # frozen fraction of 0.24 is only reassuring if you can read it.
        for check, stats in sorted(r.stats.items()):
            if check == "markers":       # rendered in full just below
                continue
            bits = [f"{k}={_fmt(v)}" for k, v in stats.items()
                    if _is_scalar(v)]
            if bits:
                print(f"    · {check:<18} {'  '.join(bits)}")
        _print_markers(r)

        for f in r.findings:
            head = f"  {_zh(f.level):<4} {f.check:<18} {f.message}"
            print(head)
            bits = []
            if f.field:
                bits.append(f.field)
            if f.subject and f.subject != f.field:
                bits.append(f.subject)
            if f.threshold is not None:
                bits.append(f"阈值={f.threshold:g}")
            if f.observed is not None:
                bits.append(f"实测={f.observed:g}")
            if f.spans:
                bits.append(f"{len(f.spans)} 处")
            if bits:
                print(f"       {'  '.join(bits)}")

    # ---- timeline ----
    rows = []
    for name, r in report.streams.items():
        t0s = [s["t0"] for s in r.series.values() if s.get("t0") is not None]
        t1s = [s["t1"] for s in r.series.values() if s.get("t1") is not None]
        if t0s:
            rows.append((name, min(t0s), max(t1s)))
    if rows:
        base = w["t0"] if w else min(t0 for _, t0, _ in rows)
        end = w["t1"] if w else max(t1 for _, _, t1 in rows)
        ref = "RUN_START" if w else "会话起点"
        print(f"\n时间线(相对{ref})")
        for name, t0, t1 in sorted(rows, key=lambda x: x[1]):
            print(f"  {name:<18} 起点 {t0 - base:+7.2f}s  "
                  f"时长 {t1 - t0:6.2f}s  尾部缺口 {end - t1:6.2f}s")

    # ---- 错误汇总:所有 ERROR 级别的问题(按时间先后) ----
    errors: list[tuple[str, Finding]] = []
    for name, r in report.streams.items():
        for f in r.findings:
            if f.level == "ERROR":
                errors.append((name, f))
    for f in report.findings:
        if f.level == "ERROR":
            errors.append(("session", f))
    errors.sort(key=lambda x: (x[1].spans[0].t if x[1].spans else float("inf")))

    if errors:
        print(f"\n{'=' * 16} 错误汇总: {len(errors)} 处 {'=' * 16}")
        for name, f in errors:
            if f.spans:
                t_rel = f.spans[0].t - (w["t0"] if w else f.spans[0].t)
                when = f"+{t_rel:7.2f}s"
            else:
                when = f"{'全局':>9}"
            bits = [when, f"[{name}]", f.check, f.message]
            print("  " + "  ".join(bits))
    else:
        print(f"\n{'=' * 16} 错误汇总: 无错误 {'=' * 16}")
    print()
