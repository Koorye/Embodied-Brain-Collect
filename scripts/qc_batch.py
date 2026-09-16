#!/usr/bin/env python3
"""批量重跑 QC:对某日期的全部会话刷新 qc_report.json(必选)与 qc.html(可选)。

    python scripts/qc_batch.py data/session-day --date 2026-09-16
    python scripts/qc_batch.py data/session-day data/session-night --date 2026-09-16
    python scripts/qc_batch.py data/session-day --date 2026-09-16 --no-html

repair_videos.py 修完视频、或检查口径变更之后,用它一键刷新已保存会话的
QC 报告。每个会话:重跑检查器 → 覆盖写 ``qc_report.json``;qc.html 按
``checker.yaml`` 的 ``html:`` 开关渲染(渲染要解码视频抽缩略图,慢),
``--html`` / ``--no-html`` 可临时强制,优先于配置。

输出一行/会话:目录名、整体等级、ERROR/WARN 计数。

Exit code 与 qc.py 一致:2 = 存在 ERROR 会话,1 = 只有 WARN,0 = 全部干净。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.checkers import qc_session  # noqa: E402

_EXIT = {"INFO": 0, "WARN": 1, "ERROR": 2}


def _findings_of(report_dict: dict) -> list[tuple[str, str]]:
    """[(level, check)] —— 流级 + 会话级 findings 摊平。"""
    out: list[tuple[str, str]] = []
    for st in (report_dict.get("streams") or {}).values():
        for f in st.get("findings", []):
            out.append((f.get("level", "INFO"), f.get("check", "")))
    for f in report_dict.get("findings", []):
        out.append((f.get("level", "INFO"), f.get("check", "")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="班次根目录(如 data/session-day),其下 <日期>-* "
                         "的目录逐个重跑 QC")
    ap.add_argument("--date", default=None,
                    help="只处理 <date>-* 的会话(默认全部)")
    ap.add_argument("--html", dest="html", action="store_true", default=None,
                    help="强制渲染 qc.html(优先于 checker.yaml)")
    ap.add_argument("--no-html", dest="html", action="store_false",
                    help="强制不渲染 qc.html(优先于 checker.yaml)")
    a = ap.parse_args(argv)

    from embodied_brain_collect.session.config import load_checker
    try:
        checker_cfg = load_checker()
    except FileNotFoundError:
        checker_cfg = {}
    html_on = a.html if a.html is not None \
        else bool((checker_cfg or {}).get("html", True))

    sessions: list[Path] = []
    for root in a.roots:
        pattern = f"{a.date}-*" if a.date else "*"
        sessions += [p for p in sorted(root.glob(pattern)) if p.is_dir()]
    if not sessions:
        print("没有找到 session 目录", file=sys.stderr)
        return 1

    build_page = None
    if html_on:
        try:
            from embodied_brain_collect.visualizers.qc_page import (
                Options, build_page)
        except Exception as exc:    # noqa: BLE001
            print(f"[warn] qc.html 渲染组件不可用({exc})— 只更新 JSON",
                  file=sys.stderr)
            html_on = False

    worst = 0
    n_err = n_warn = 0
    for i, sd in enumerate(sessions, 1):
        report = qc_session(sd, checker_cfg=checker_cfg)
        d = report.to_dict()
        (sd / "qc_report.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

        findings = _findings_of(d)
        e = sum(1 for lv, _ in findings if lv == "ERROR")
        w = sum(1 for lv, _ in findings if lv == "WARN")
        level = report.level
        worst = max(worst, _EXIT.get(level, 0))
        n_err += 1 if e else 0
        n_warn += 1 if w and not e else 0

        note = ""
        if html_on:
            try:
                (sd / "qc.html").write_text(
                    build_page(d, sd, Options()), encoding="utf-8")
                note = "  +qc.html"
            except Exception as exc:    # noqa: BLE001
                note = f"  qc.html 渲染失败: {exc}"
        print(f"[{i}/{len(sessions)}] {sd.name}: {level}"
              f"  ({e} ERROR / {w} WARN){note}")

    tag = {0: "全部干净", 1: "仅 WARN", 2: "存在 ERROR"}[worst]
    print(f"\n完成: {len(sessions)} 个会话已刷新 qc_report.json"
          f"({n_err} 个含 ERROR,{n_warn} 个仅含 WARN)— {tag}"
          + ("" if html_on else ";qc.html 按 checker.yaml 关闭/不可用,未渲染"))
    return worst


if __name__ == "__main__":
    sys.exit(main())
