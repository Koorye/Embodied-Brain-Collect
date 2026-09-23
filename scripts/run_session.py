#!/usr/bin/env python3
"""兼容入口 —— 转发到拆分后的模式脚本(旧命令不失效)。

入口已按模式拆分::

    scripts/run_session_env.py     # 图纸模式(原 --mode env)
    scripts/run_session_tasks.py   # 任务列表模式(原 --mode tasks)
    scripts/run_session_video.py   # video_rate(RLHF 视频打分)

本文件只做转发:按 --mode(configs/session.yaml 的 mode,缺省 env)把参数
原样交给对应脚本。公共编排逻辑在 src/embodied_brain_collect/session/
run_base.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode", default=None)
    args, rest = ap.parse_known_args(argv)

    if args.mode is None:
        from embodied_brain_collect.session.config import load_session_run
        args.mode = str(load_session_run().get("mode") or "env")

    print("[run_session] 提示: 入口已按模式拆分 — run_session_env.py(图纸) / "
          "run_session_tasks.py(任务) / run_session_video.py(视频);"
          "本命令仅为兼容转发", file=sys.stderr)

    if args.mode == "env":
        import run_session_env
        return run_session_env.main(rest)
    if args.mode == "tasks":
        import run_session_tasks
        return run_session_tasks.main(rest)
    print(f"[run_session] 未知 mode: {args.mode!r} "
          "(env=图纸模式 / tasks=任务列表模式;视频用 run_session_video.py)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
