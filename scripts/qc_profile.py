#!/usr/bin/env python3
"""Profile the session quality checks: wall time per stream and per check.

    python scripts/qc_profile.py data/session-night            # 一个 day 目录
    python scripts/qc_profile.py data/session-day data/session-night --json prof.json

Replicates ``BaseChecker.run``'s loop exactly (same dispatch, same
checker.yaml overrides) but times each step, so the numbers say where QC's
wall time actually goes — and, together with the finding counts, which
checks never fire and are candidates for removal.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from embodied_brain_collect.checkers import checker_for  # noqa: E402
from embodied_brain_collect.checkers.base import (  # noqa: E402
    CheckContext, flatten_checks, scan_logs, worst_level)
from embodied_brain_collect.checkers.marker import find_run_window  # noqa: E402


class TimedContext(CheckContext):
    """CheckContext that records seconds spent decompressing each NPZ key."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.load_s: dict[str, float] = {}

    def arr(self, key):
        if key is not None and key not in self._arrays:
            t0 = time.perf_counter()
            try:
                return super().arr(key)
            finally:
                self.load_s[key] = (self.load_s.get(key, 0.0)
                                    + time.perf_counter() - t0)
        return super().arr(key)


def _has_data(d: Path) -> bool:
    return any(d.glob("*.npz")) or any(d.glob("*.mp4"))


def profile_stream(cls, d: Path, window: dict | None,
                   checker_cfg: dict) -> dict:
    inst = cls(checker_cfg)
    ctx = TimedContext(stream=d.name, directory=d, window=window,
                       default_series=cls.default_series)
    checks: list[dict] = []
    t_all = time.perf_counter()
    try:
        if cls.requires_npz and not ctx.has_npz:
            checks.append({"check": "NpzPresent", "series": "", "s": 0.0,
                           "applies": True, "n_findings": 1, "level": "ERROR"})
        else:
            t0 = time.perf_counter()
            inst.prepare(ctx)
            checks.append({"check": "(prepare)", "series": "", "s":
                           time.perf_counter() - t0, "applies": True,
                           "n_findings": 0, "level": "-"})
            for check in flatten_checks(inst.checks):
                overrides = inst.checker_cfg.get(
                    check.name.lower().replace("_", ""))
                if isinstance(overrides, dict):
                    try:
                        check = dataclasses.replace(check, **overrides)
                    except TypeError:
                        pass
                t0 = time.perf_counter()
                applies = check.applies(ctx)
                out = check.run(ctx) if applies else None
                dt = time.perf_counter() - t0
                checks.append({
                    "check": check.name,
                    "series": check.series or cls.default_series or "",
                    "s": dt, "applies": applies,
                    "n_findings": len(out.findings) if out else 0,
                    "level": out.level if out else "-"})
            for label in ctx.series_labels:      # keep parity with real run
                ctx.series(label)
        t0 = time.perf_counter()
        logs = scan_logs(d)
        log_s = time.perf_counter() - t0
    finally:
        ctx.close()
    return {"stream": d.name, "checker": cls.name,
            "total_s": time.perf_counter() - t_all,
            "log_scan_s": log_s, "n_log_findings": len(logs),
            "load_s": dict(ctx.load_s), "checks": checks}


def profile_session(root: Path, checker_cfg: dict) -> dict | None:
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    window = find_run_window(root)
    window_s = time.perf_counter() - t0
    if window is None:
        return {"session": str(root), "total_s": time.perf_counter() - t_all,
                "window_s": window_s, "streams": []}
    streams = [profile_stream(checker_for(d.name), d, window, checker_cfg)
               for d in sorted(x for x in root.iterdir() if x.is_dir())
               if _has_data(d) and checker_for(d.name) is not None]
    return {"session": str(root), "total_s": time.perf_counter() - t_all,
            "window_s": window_s, "window_span": window["t1"] - window["t0"],
            "streams": streams}


def aggregate(runs: list[dict]) -> dict:
    """Roll the per-session profiles up: per checker and per (checker, check)."""
    by_checker: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "total_s": 0.0, "load_s": 0.0, "log_s": 0.0})
    by_check: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_runs": 0, "n_applies": 0, "total_s": 0.0, "max_s": 0.0,
                 "n_findings": 0, "levels": defaultdict(int)})
    loads: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total_s": 0.0, "max_s": 0.0})
    for r in runs:
        for st in r["streams"]:
            c = by_checker[st["checker"]]
            c["n"] += 1
            c["total_s"] += st["total_s"]
            c["load_s"] += sum(st["load_s"].values())
            c["log_s"] += st["log_scan_s"]
            for ck in st["checks"]:
                k = (st["checker"], ck["check"])
                a = by_check[k]
                a["n_runs"] += 1
                a["n_applies"] += int(ck["applies"])
                a["total_s"] += ck["s"]
                a["max_s"] = max(a["max_s"], ck["s"])
                a["n_findings"] += ck["n_findings"]
                if ck["n_findings"]:
                    a["levels"][ck["level"]] += 1
            for key, s in st["load_s"].items():
                k = (st["checker"], f"load:{key}")
                loads[k]["total_s"] += s
                loads[k]["max_s"] = max(loads[k]["max_s"], s)
    return {"by_checker": dict(by_checker),
            "by_check": {f"{k[0]}/{k[1]}": v for k, v in by_check.items()},
            "loads": {f"{k[0]}/{k[1]}": v for k, v in loads.items()}}


def print_summary(runs: list[dict]) -> None:
    agg = aggregate(runs)
    n_sess = len(runs)
    total_qc = sum(r["total_s"] for r in runs)
    print(f"\n{len(runs)} 个会话,QC 总耗时 {total_qc:.1f}s,"
          f"平均 {total_qc / max(n_sess, 1):.1f}s/会话\n")

    print(f"{'checker':<14}{'次数':>6}{'总耗时s':>10}{'均值s':>9}"
          f"{'其中加载s':>10}{'日志s':>8}")
    print("-" * 60)
    for name, c in sorted(agg["by_checker"].items(),
                          key=lambda kv: -kv[1]["total_s"]):
        print(f"{name:<14}{c['n']:>6}{c['total_s']:>10.1f}"
              f"{c['total_s'] / max(c['n'], 1):>9.2f}"
              f"{c['load_s']:>10.1f}{c['log_s']:>8.1f}")

    print(f"\n逐 check 明细(按累计耗时排序;命中= 有 finding 的运行次数)")
    print(f"{'checker/check':<44}{'跑过':>6}{'生效':>6}{'总s':>9}"
          f"{'均值ms':>9}{'最大ms':>9}{'命中':>6}")
    print("-" * 96)
    rows = []
    for key, a in agg["by_check"].items():
        if key.startswith("load:"):
            continue
        rows.append((key, a))
    for key, a in sorted(rows, key=lambda kv: -kv[1]["total_s"]):
        print(f"{key:<44}{a['n_runs']:>6}{a['n_applies']:>6}"
              f"{a['total_s']:>9.2f}{a['total_s'] / max(a['n_runs'], 1) * 1e3:>9.1f}"
              f"{a['max_s'] * 1e3:>9.1f}{a['n_findings']:>6}")

    print(f"\nNPZ 字段解压耗时")
    print(f"{'checker/field':<44}{'总s':>9}{'最大ms':>9}")
    print("-" * 64)
    for key, a in sorted(agg["loads"].items(), key=lambda kv: -kv[1]["total_s"]):
        print(f"{key:<44}{a['total_s']:>9.2f}{a['max_s'] * 1e3:>9.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path, nargs="+",
                    help="一个或多个 session(或含多个 session 的 day)目录")
    ap.add_argument("--json", type=Path, default=None,
                    help="把逐会话剖析结果写成 JSON")
    ap.add_argument("--quiet", action="store_true", help="只出汇总不逐会话打印")
    args = ap.parse_args(argv)

    from embodied_brain_collect.session.config import load_checker
    try:
        checker_cfg = load_checker()
    except FileNotFoundError:
        checker_cfg = {}

    roots: list[Path] = []
    for p in args.session_dir:
        if p.is_dir() and any(x.is_dir() and x.name[:2] == "20"
                              for x in p.iterdir()):
            roots.extend(sorted(x for x in p.iterdir()
                                if x.is_dir() and x.name[:2] == "20"))
        else:
            roots.append(p)

    runs = []
    for root in roots:
        t0 = time.perf_counter()
        r = profile_session(root, checker_cfg)
        runs.append(r)
        if not args.quiet:
            st = ", ".join(f"{s['stream']}={s['total_s']:.2f}s"
                           for s in r["streams"])
            print(f"{root.name}  {time.perf_counter() - t0:6.2f}s   {st}")

    print_summary(runs)
    if args.json:
        args.json.write_text(json.dumps(
            {"runs": runs, "aggregate": aggregate(runs)},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
