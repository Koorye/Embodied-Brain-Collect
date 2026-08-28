#!/usr/bin/env python3
"""Run the session quality checks over one or more session directories.

    python scripts/qc.py data/session4
    python scripts/qc.py data/session* --json report.json

Exit code is the worst level found: 0 = all clear, 1 = warnings, 2 = failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.checkers import (  # noqa: E402
    print_report, qc_session)

_EXIT = {"INFO": 0, "WARN": 1, "ERROR": 2}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path, nargs="+",
                    help="一个或多个 session 目录")
    ap.add_argument("--json", type=Path, default=None,
                    help="把完整报告写成 JSON(多个 session 时为数组)")
    ap.add_argument("--quiet", action="store_true",
                    help="不打印控制台报告,只用退出码")
    args = ap.parse_args(argv)

    # configs/checker.yaml 覆盖检查阈值;文件缺失则全部用代码默认值
    from embodied_brain_collect.config.load import load_checker
    try:
        checker_cfg = load_checker()
    except FileNotFoundError:
        checker_cfg = {}

    reports = []
    worst = 0
    for root in args.session_dir:
        if not root.is_dir():
            print(f"{root}: 不是目录 — 跳过", file=sys.stderr)
            worst = max(worst, 2)
            continue
        report = qc_session(root, checker_cfg=checker_cfg)
        reports.append(report)
        if not args.quiet:
            print_report(report)
        worst = max(worst, _EXIT.get(report.level, 0))

    if args.json:
        payload = ([r.to_dict() for r in reports] if len(reports) != 1
                   else reports[0].to_dict())
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"\nJSON 报告 -> {args.json}")

    return worst


if __name__ == "__main__":
    sys.exit(main())
