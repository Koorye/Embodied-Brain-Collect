#!/usr/bin/env python3
"""任务列表模式入口 —— tasks.yaml 随机采样 → 逐条录制+QC → 汇总。

    python scripts/run_session_tasks.py              # 完整会话(交互式)
    python scripts/run_session_tasks.py --seed 42    # 指定队列种子(可复现)
    python scripts/run_session_tasks.py --auto-keep  # 不询问,录完即留
    python scripts/run_session_tasks.py --dummy      # 假设备试跑整条链路

流程:

  1. 队列来自 configs/tasks.yaml 随机采样(seed 可固定);打印完整队列,
     按 Enter 才开始采集
  2. 每条录制:提示任务编号 → 按 Enter 开录 → launcher 启动(所有 recorder
     多进程 + 刺激程序,指令屏显示任务名)→ 自动 QC
  3. 每条录完输入 字母+Enter 确认:n = 成功,下一条 / r = 失败,重跑 /
     q = 成功并退出;结局写进该条 meta.yaml 的 status 字段
  4. 全部完成后打印会话汇总

刺激程序 --stim(paradigm1 / rgb / simple / sync_test;缺省读
configs/session.yaml 的 stim)。与图纸/视频入口共享的编排逻辑见
session/run_base.py。

Windows 注意:recorder 子进程以 spawn 启动,本脚本必须作为主入口运行。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.session.config import (  # noqa: E402
    load_session_run, load_tasks)
from embodied_brain_collect.session.run_base import (  # noqa: E402
    ModeHooks, add_common_cli, resolve_collect, run_queue, seed_or_now)
from embodied_brain_collect.stim.factory import (  # noqa: E402
    STIM_KINDS, build_stim_cmd)


def _hooks(args) -> ModeHooks:
    """任务模式行为:Enter 开录 / task_id 进 meta / 无台账。"""

    def preroll(job: dict, run_dir: Path, args) -> bool:
        print(f"\n{'─' * 68}")
        print(f"  任务 #{job['task_id']}  {job.get('task_name', '')}")
        print(f"  录制目录: {run_dir}")
        print(f"{'─' * 68}")
        if not args.dummy:
            input("按 Enter 正式开始...")
        return True

    return ModeHooks(
        label="tasks",
        preroll=preroll,
        meta_kwargs=lambda job: {"task_id": job["task_id"]},
        stim_cmd=lambda job, stim, run_dir: build_stim_cmd(
            stim, task_id=job["task_id"]),
        job_id=lambda job: job["task_id"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_cli(ap)
    ap.add_argument("--auto-keep", action="store_true",
                    help="每个任务录完自动保留,不询问")
    ap.add_argument("--stim", choices=sorted(
        k for k in STIM_KINDS if k != "video_rate"), default=None,
        help="刺激程序(缺省读 configs/session.yaml 的 stim,"
             "再缺省 paradigm1);参数见 configs/stim.yaml")
    ap.add_argument("--max-runs", type=int, default=None, dest="max_runs",
                    help="单次会话最大采集条数(覆盖 configs/session.yaml "
                         "的 max_runs;0 = 不限)")
    args = ap.parse_args(argv)

    session_root = args.session_dir.resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    stim = args.stim or str(load_session_run().get("stim") or "paradigm1")
    if stim not in STIM_KINDS or stim == "video_rate":
        print(f"[run_session_tasks] 未知/不可用 stim: {stim!r} "
              f"(可用: {sorted(k for k in STIM_KINDS if k != 'video_rate')})",
              file=sys.stderr)
        return 2

    collect = resolve_collect(args)
    hooks = _hooks(args)

    # ---- 队列:tasks.yaml 随机采样 ----
    seed = seed_or_now(args)
    rng = random.Random(seed)
    try:
        tasks = load_tasks()
    except FileNotFoundError as exc:
        print(f"缺少配置文件: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print("configs/tasks.yaml 没有任务", file=sys.stderr)
        return 2
    rng.shuffle(tasks)
    queue = [{"mode": "tasks", "task_id": int(t["task_id"]),
              "task_name": t.get("task_name", "")} for t in tasks]
    print(f"\n任务库共 {len(queue)} 个任务;本次随机队列 seed={seed}")
    print(f"{'─' * 68}\n执行顺序:")
    for i, job in enumerate(queue, 1):
        print(f"  {i:>3}. #{job['task_id']:<3} {job['task_name']}")
    max_runs = (args.max_runs if args.max_runs is not None
                else int(load_session_run().get("max_runs") or 0))
    if max_runs > 0 and len(queue) > max_runs:
        queue = queue[:max_runs]
        print(f"最大采集量 max_runs={max_runs} — 队列只取前 {max_runs} 条")
    print(f"{'─' * 68}")
    print(f"[run_session_tasks] 任务列表模式  stim={stim}  "
          f"计划={len(queue)} 条"
          + (f"  采集信息: {collect}" if collect else ""))
    try:
        input("按 Enter 开始采集(Ctrl+C 取消) ...")
    except KeyboardInterrupt:
        print("\n[run_session] 已取消")
        return 0

    return run_queue(session_root, queue, stim=stim, args=args,
                     collect=collect, hooks=hooks, seed=seed)


if __name__ == "__main__":
    sys.exit(main())
