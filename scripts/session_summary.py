#!/usr/bin/env python3
"""现有数据汇总 —— 对已采集的 session 做统计分析总结。

    python scripts/session_summary.py                       # data/session-night 全部
    python scripts/session_summary.py --date 2026-08-24     # 只统计这一天
    python scripts/session_summary.py --date 2026-08        # 整个 8 月(--date 是前缀匹配)
    python scripts/session_summary.py -o summary.json       # 同时写 JSON

统计口径与 run_session.py 的会话汇总一致:读每个 session 目录的
qc_report.json(没有则标「未跑 QC」)与 meta.yaml,给出:

  * 数据量:session 数、无误数据比例、有效录制时长合计
  * 每日产量:按日期分组的 session 数与时长
  * 任务覆盖:任务库哪些 task 已录、哪些缺失、哪些录了多次
  * 质量问题:各 QC 错误/警告的条数与涉及 session 占比
  * 明细表:每个 session 的日期、任务、QC 等级、时长

扫描范围是 {session_dir}/yyyy-MM-dd-HH-mm-ss/ 下的全部录制目录;
--date 用目录名前缀过滤(2026-08-24 只取当天,2026-08 取整月)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embodied_brain_collect.config.load import load_tasks  # noqa: E402
from run_session import (_collect, _load_qc, _meta_task,   # noqa: E402
                         print_summary, _session_span)


def _scan(session_dir: Path, date: str | None) -> list[Path]:
    """{session_dir}/yyyy-MM-dd-HH-mm-ss/ 下的录制目录,date 前缀过滤。"""
    if not session_dir.is_dir():
        print(f"{session_dir}: 目录不存在", file=sys.stderr)
        return []
    runs = sorted(p for p in session_dir.iterdir()
                  if p.is_dir() and len(p.name) == 19
                  and p.name[4] == p.name[7] == "-" and p.name[10] == "-"
                  and p.name[13] == p.name[16] == "-"
                  and (date is None or p.name.startswith(date)))
    if date is not None and not runs:
        print(f"{session_dir}: 没有匹配 '{date}' 的录制目录(前缀匹配)",
              file=sys.stderr)
    return runs


def _daily(summary: dict) -> dict[str, dict]:
    """按日期分组的 session 数与有效时长。"""
    daily: dict[str, dict] = {}
    for s in summary["sessions"]:
        day = s["dir"][:10]
        d = daily.setdefault(day, {"sessions": 0, "span_s": 0.0, "clean": 0})
        d["sessions"] += 1
        d["span_s"] += s.get("span_s", 0.0)
        d["clean"] += int(s.get("clean", False))
    return daily


def _task_coverage(runs: list[Path]) -> dict:
    """任务库全集 vs 已录任务的覆盖情况。"""
    try:
        library = load_tasks()
    except FileNotFoundError:
        library = []
    library_ids = {int(t["task_id"]): t.get("task_name", "") for t in library}

    recorded: Counter = Counter()
    for run_dir in runs:
        tid = _meta_task(run_dir)
        if tid is not None:
            recorded[tid] += 1

    missing = sorted(set(library_ids) - set(recorded))
    repeated = {tid: n for tid, n in recorded.items() if n > 1}
    return {
        "library": library_ids, "recorded": dict(recorded),
        "missing": missing, "repeated": repeated,
    }


def print_extra(summary: dict, runs: list[Path], cov: dict) -> None:
    """每日产量与任务覆盖 —— print_summary 之外的补充小节。"""
    daily = _daily(summary)
    if len(daily) > 1:
        print("  每日产量:")
        for day, d in sorted(daily.items()):
            print(f"    {day}  {d['sessions']:>3} 个 session"
                  f"  {d['clean']}/{d['sessions']} 无误"
                  f"  {d['span_s'] / 60:6.1f} 分钟")

    lib, rec = cov["library"], cov["recorded"]
    print(f"\n  任务覆盖: 任务库 {len(lib)} 个 · 已录 {len(rec)} 个")
    if cov["missing"]:
        names = "、".join(f"#{i} {lib[i]}" for i in cov["missing"][:8])
        more = f" 等 {len(cov['missing'])} 个" if len(cov["missing"]) > 8 else ""
        print(f"    未录制: {names}{more}")
    if cov["repeated"]:
        for tid, n in sorted(cov["repeated"].items()):
            print(f"    录了 {n} 次的: #{tid} {lib.get(tid, '')}")
    if not cov["missing"] and not cov["repeated"]:
        print("    全覆盖且无重复。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, default=Path("data/session-night"),
                    help="会话根目录(默认 data/session-night)")
    ap.add_argument("--date", default=None,
                    help="日期前缀过滤:2026-08-24 当天 / 2026-08 整月 / 2026 全年")
    ap.add_argument("-o", "--json", type=Path, default=None,
                    help="把汇总写成 JSON")
    args = ap.parse_args(argv)

    runs = _scan(args.session_dir.resolve(), args.date)
    if not runs:
        return 1
    print(f"{args.session_dir} — {len(runs)} 个录制目录"
          + (f"(匹配 '{args.date}')" if args.date else ""))

    summary = _collect(runs)
    summary["scanned"] = [str(r) for r in runs]
    summary["daily"] = _daily(summary)
    summary["tasks"] = _task_coverage(runs)

    print_summary(summary, title=args.date or "全部数据")
    print_extra(summary, runs, summary["tasks"])

    if args.json:
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"汇总已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
