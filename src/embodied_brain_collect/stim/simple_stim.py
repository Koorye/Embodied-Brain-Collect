"""Simple stim —— 最小刺激流程,只为给采集打一对边界 marker。

    1. 任务名 + 按 [空格] 开始采集
    2. 采集中 + 按 [空格] 结束
    3. 直接退出

整个 run 只发一对 RUN_START/RUN_END 边界码(数值 = configs/markers.yaml
码表的 run_start/run_end),与 paradigm1 的码约定一致;没有注视点/指令/
想象等其他画面。Esc 中止也会补 RUN_END,保证 EEG 对齐的码对完整。

    python -m embodied_brain_collect.stim.simple_stim --task-id 0
    python -m embodied_brain_collect.stim.simple_stim --task-id 0 --fast 10 --windowed   # 冒烟
"""
from __future__ import annotations

import argparse
import sys

from embodied_brain_collect.session.config import task_by_id
from embodied_brain_collect.stim.base_stim import BaseStim, stim_defaults
from embodied_brain_collect.stim import marker_codes as M


class SimpleStim(BaseStim):
    """任务名 → 空格开跑 → 空格结束 → 退出。"""

    title = "Simple Stim"

    @staticmethod
    def add_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--task-id", type=int, required=True,
                        help="Task to run (id in configs/tasks.yaml; "
                             "launcher 自动传入)")

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        task = task_by_id(args.task_id)
        if task is None:
            raise SystemExit(
                f"[stim] configs/tasks.yaml 没有 task_id={args.task_id}")
        self.task_id = int(task["task_id"])
        self.task_name = str(task["task_name"])

    def run_flow(self) -> None:
        header = f"任务 #{self.task_id}"
        if not self._wait_space(lambda: self._draw_message(
                header, f"{self.task_name}   按 [空格] 开始采集")):
            return                          # Esc: 未开始,无 marker,直接退
        self.marker.mark(M.RUN_START, "RUN_START")

        self._wait_space(lambda: self._draw_message(
            "采集中 ...", f"{self.task_name}   按 [空格] 结束"))
        # 空格结束或 Esc 中止都补 END,保住 EEG 对齐的码对
        self.marker.mark(M.RUN_END, "RUN_END")


def main(argv: list[str] | None = None) -> int:
    over = stim_defaults("simple")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    BaseStim.add_common_args(ap, over)
    SimpleStim.add_args(ap, over)
    args = ap.parse_args(argv)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("[stim] pygame not installed. Run: pip install pygame",
              file=sys.stderr)
        return 2

    stim = SimpleStim(args)
    print(f"[stim] task_id={stim.task_id} "
          f"task_name={stim.task_name!r} (空格开跑/空格结束/Esc 退出)")
    return stim.run()


if __name__ == "__main__":
    raise SystemExit(main())
