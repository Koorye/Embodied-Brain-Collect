#!/usr/bin/env python3
"""图纸模式入口 —— 一次采集会话的完整编排:图纸队列 → 摆放+录制+QC → 汇总。

    python scripts/run_session_env.py                # 完整会话(交互式)
    python scripts/run_session_env.py --seed 42      # 指定队列种子(可复现)
    python scripts/run_session_env.py --auto-keep    # 不询问,录完即留
    python scripts/run_session_env.py --dummy        # 假设备试跑整条链路

流程:

  1. 抽取队列来自 configs/environments 的图纸池(随机排序,seed 可固定),
     已采集成功的图纸自动跳过;打印完整队列,按 Enter 才开始采集
  2. 按队列逐张图纸录制:全屏显示图纸 → 采集员照图摆放实物 → 按 s + Enter
     关闭图纸 → launcher 启动(所有 recorder 多进程 + 刺激程序)→ 自动 QC;
     Esc 可在图纸阶段取消本次(不开录)
  3. 每张录完需输入 字母+Enter 确认,防止误触:
     n = 当前采集成功,下一条 / r = 当前采集失败,重跑 / q = 成功并退出
     * 结局写进该条 meta.yaml 的 status 字段(r → failed,n/q → success)
     * 只有"成功"记入图纸台账 configs/environments/<场景目录>/used.yaml
  4. 全部完成后打印会话汇总(保留/重采/退出的录制都计入)

刺激程序 --stim(paradigm1 / rgb / simple / sync_test;缺省读
configs/session.yaml 的 stim),指令屏按图纸 config.yaml 的场景/任务显示。
与任务/视频入口共享的编排逻辑见 session/run_base.py。

Windows 注意:recorder 子进程以 spawn 启动,本脚本必须作为主入口运行。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.session import environment as env  # noqa: E402
from embodied_brain_collect.session.config import (  # noqa: E402
    load_session_run)
from embodied_brain_collect.session.run_base import (  # noqa: E402
    ModeHooks, add_common_cli, resolve_collect, run_queue, seed_or_now)
from embodied_brain_collect.stim.factory import (  # noqa: E402
    STIM_KINDS, build_stim_cmd)


def _stim_cmd(job: dict, stim: str, run_dir: Path) -> list:
    return build_stim_cmd(stim, environment=job["rel"])


def _hooks(args) -> ModeHooks:
    """图纸模式行为:全屏摆放确认 / environment+layout 进 meta / 台账记账。"""

    def preroll(job: dict, run_dir: Path, args) -> bool:
        rel = job["rel"]
        print(f"\n{'─' * 68}")
        print(f"  图纸 {rel}  场景任务: {env.scene_task(rel)}")
        print(f"  录制目录: {run_dir}")
        print(f"{'─' * 68}")
        print("  图纸已全屏打开 — 照图摆放实物,按 s + Enter 关闭并正式开始"
              "(未确认直接关窗 = 取消)")
        if env.show(rel) == "abort":
            print("[run_session] 图纸阶段取消 — 未开录,目录留档")
            return False
        print("[run_session] 图纸已关闭,正式启动采集")
        return True

    def on_result(job: dict, run_dir: Path, choice: str, status: str,
                  rc: int) -> None:
        if status != "success":
            print("  QC 有 ERROR — 图纸不记台账(会重新抽到重采)")
            return
        env.mark_used(job["rel"], session=str(run_dir))
        print(f"  图纸 {job['rel']} 已记入台账"
              + (",进入下一张" if choice == "kept" else ""))

    return ModeHooks(
        label="env",
        preroll=preroll,
        meta_kwargs=lambda job: {
            "environment": job["rel"],
            "layout": env.scene_layout(job["rel"]) or None},
        stim_cmd=_stim_cmd,
        job_id=lambda job: job["rel"],
        on_result=on_result,
        qc_error_note="图纸不记入台账 — 之后会重新抽到重采",
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

    # stim:CLI 显式传参优先,否则读 configs/session.yaml 的 stim
    stim = args.stim or str(load_session_run().get("stim") or "paradigm1")
    if stim not in STIM_KINDS or stim == "video_rate":
        print(f"[run_session_env] 未知/不可用 stim: {stim!r} "
              f"(可用: {sorted(k for k in STIM_KINDS if k != 'video_rate')})",
              file=sys.stderr)
        return 2

    collect = resolve_collect(args)
    hooks = _hooks(args)

    # ---- 队列:未采集图纸(随机排序,已成功的不会再抽到) ----
    seed = seed_or_now(args)
    rng = random.Random(seed)
    unused = env.pool_unused()
    if not unused:
        print("configs/environments 里没有未采集过的图纸 — 各场景目录"
              "台账已全部记账,或目录为空", file=sys.stderr)
        return 2
    rng.shuffle(unused)
    queue = [{"mode": "env", "rel": rel} for rel in unused]
    print(f"\n图纸池共 {len(unused)} 张未采集;本次随机队列 seed={seed}")
    print(f"{'─' * 68}\n执行顺序:")
    for i, job in enumerate(queue, 1):
        print(f"  {i:>3}. {job['rel']}  [{env.scene_task(job['rel'])}]")
    max_runs = (args.max_runs if args.max_runs is not None
                else int(load_session_run().get("max_runs") or 0))
    if max_runs > 0 and len(queue) > max_runs:
        queue = queue[:max_runs]
        print(f"最大采集量 max_runs={max_runs} — 队列只取前 {max_runs} 条")
    print(f"{'─' * 68}")
    print(f"[run_session_env] 图纸模式  stim={stim}  计划={len(queue)} 条"
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
