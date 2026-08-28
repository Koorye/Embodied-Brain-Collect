#!/usr/bin/env python3
"""Render a QC report as a self-contained HTML page.

    python scripts/qc_report.py report.json -o qc.html   # reuse a report
    python scripts/qc_report.py data/session4            # run the checks first

The page embeds its own data, so the output is one file you can open from
disk or hand to someone else.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.checkers import qc_session          # noqa: E402
from embodied_brain_collect.visualizers.qc_page import (        # noqa: E402
    Options, build_page)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path,
                    help="session 目录,或 qc.py --json 产出的报告")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出 html(默认 <session>/qc.html)")
    ap.add_argument("--no-frames", action="store_true", help="不嵌入相机画面")
    ap.add_argument("--fps", type=float, default=1.0, help="缩略图每秒张数")
    ap.add_argument("--thumb-width", type=int, default=240)
    ap.add_argument("--jpeg-quality", type=int, default=60)
    ap.add_argument("--no-filter", action="store_true",
                    help="不嵌入滤波副本(减小体积;原始 npz 数据永不修改)")
    ap.add_argument("--open", action="store_true", help="生成后用浏览器打开")
    args = ap.parse_args(argv)

    if args.source.is_dir():
        report = qc_session(args.source).to_dict()
        session = args.source
    elif args.source.is_file():
        report = json.loads(args.source.read_text(encoding="utf-8"))
        if isinstance(report, list):
            print("该报告含多个 session,请分别渲染", file=sys.stderr)
            return 2
        session = Path(report.get("session_dir", "."))
        if not session.is_dir():
            print(f"报告里的 session_dir 不存在: {session}", file=sys.stderr)
            return 2
    else:
        print(f"{args.source}: 既不是目录也不是文件", file=sys.stderr)
        return 2

    # 滤波参数覆盖来自 checker.yaml 的 "filter:" 节(可选)。
    filter_presets = None
    if not args.no_filter:
        try:
            from embodied_brain_collect.config.load import load_checker
            filter_presets = (load_checker() or {}).get("filter")
        except FileNotFoundError:
            filter_presets = None

    out = args.out or session / "qc.html"
    html = build_page(report, session, Options(
        frames=not args.no_frames, fps=args.fps,
        thumb_w=args.thumb_width, jpeg_q=args.jpeg_quality,
        filter=not args.no_filter, filter_presets=filter_presets))
    out.write_text(html, encoding="utf-8")

    mb = out.stat().st_size / 1e6
    print(f"{out}  ({mb:.1f} MB)")
    if mb > 25:
        print("  体积偏大 — 可用 --no-filter,或 --fps 0.5 --thumb-width 180 或 "
              "--no-frames")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
