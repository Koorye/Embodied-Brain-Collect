"""Paradigm 1 (Pick & Place + motor imagery) -- pure Python stim using pygame.

Trial flow (per trial, experimenter-driven with SPACE):

    1. Press SPACE to start trial
    2. Red-dot fixation 2 s                         -> FIX_ON / FIX_OFF
    3. Instruction text (task_id + task_name) 10 s  -> INSTR_ON / INSTR_OFF
    4. Close eyes, motor imagery (self-paced)       -> IMG_START
    5. Open eyes, press SPACE to begin execution    -> IMG_END
    6. Perform pick & place                         -> EXEC_START
    7. Press SPACE when done                        -> EXEC_END
    8. Press SPACE for next trial

The task id itself is not a marker: a session records exactly one task, and
its identity lives in the session's meta.yaml (written by the launcher).
Marker codes carry only event timing.

Markers go through ParallelBox COM (EEG) and UDP (sync_hub), matching
record/sync/marker_codes.py.  Window / fonts / markers / key-wait plumbing
lives in ``base_stim.BaseStim`` — this module only defines the trial flow.

Example:
    python -m embodied_brain_collect.stim.paradigm1_pickplace --task-id 0

Smoke test:
    python -m embodied_brain_collect.stim.paradigm1_pickplace --task-id 0 \\
        --no-serial --fast 10 --windowed
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from embodied_brain_collect.config.load import task_by_id
from embodied_brain_collect.stim.base_stim import BaseStim, stim_defaults
from embodied_brain_collect.stim import marker_codes as M


@dataclass
class TrialSpec:
    task_id: int
    task_name: str


class Paradigm1Stim(BaseStim):
    """Pick & place + motor imagery 流程。"""

    title = "Paradigm 1 -- Pick & Place + MI"

    @staticmethod
    def add_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--task-id", type=int, required=True,
                        help="Task to run (id in configs/tasks.yaml; "
                             "launcher 自动传入)")
        ap.add_argument("--fix-pre-s", type=float,
                        default=float(over.get("fix_pre_s", 2.0)))
        ap.add_argument("--instr-s", type=float,
                        default=float(over.get("instr_s", 10.0)))
        ap.add_argument("--fix-radius", type=int,
                        default=int(over.get("fix_radius", 14)))
        ap.add_argument("--once", action="store_true",
                        help="Run exactly one trial then exit")

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        task = task_by_id(args.task_id)
        if task is None:
            raise SystemExit(
                f"[stim] configs/tasks.yaml 没有 task_id={args.task_id}")
        self.trial = TrialSpec(task_id=args.task_id,
                               task_name=task["task_name"])
        self.run_no = 0          # 轮转计数:同一 task 重跑时的显示用

    def _draw_instruction(self) -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        header = self.font_title.render(f"任务 #{self.trial.task_id}",
                                        True, self.hint_color)
        body = self.font_instr.render(self.trial.task_name, True, self.text_color)
        self.screen.blit(header,
                         header.get_rect(center=(cx, cy - self.args.font_size)))
        self.screen.blit(body,
                         body.get_rect(center=(cx, cy + self.args.font_size // 4)))

    def _draw_status(self, trial_n: int, hint: str) -> None:
        header = f"task {self.trial.task_id}  run {self.run_no}  #{trial_n}"
        self._draw_message(header, hint)

    def run_flow(self) -> None:
        run_started = False
        trial_n = 0
        trial = self.trial

        while not self.aborted:
            trial_n += 1
            hint = "按 [空格] 开始" if trial_n == 1 \
                else "按 [空格] 开始下一次  |  Esc 结束"
            if not self._wait_space(lambda: self._draw_status(trial_n, hint)):
                break

            if not run_started:
                self.marker.mark(M.RUN_START, "RUN_START")
                run_started = True

            self.marker.set_trial(trial_n)
            print(f"[trial #{trial_n}] task_id={trial.task_id} "
                  f"name='{trial.task_name}'")

            self._draw_fixation()
            self._flip_and_mark(M.FIX_ON, "FIX_ON")
            self._wait_seconds(self.args.fix_pre_s)
            if self.aborted:
                break

            self._draw_instruction()
            self._flip_and_mark(M.INSTR_ON, "INSTR_ON")
            self.marker.mark(M.FIX_OFF, "FIX_OFF")
            self._wait_seconds(self.args.instr_s)
            if self.aborted:
                break
            self.marker.mark(M.INSTR_OFF, "INSTR_OFF")

            self._draw_message("请闭眼，进行运动想象",
                               "想象完成后睁眼，按 [空格] 继续")
            self._flip_and_mark(M.IMG_START, "IMG_START")
            if not self._wait_space(lambda: self._draw_message(
                    "请闭眼，进行运动想象", "想象完成后睁眼，按 [空格] 继续")):
                break
            self.marker.mark(M.IMG_END, "IMG_END")

            if not self._wait_space(lambda: self._draw_message(
                    "请执行 pick & place", "按 [空格] 开始实际动作")):
                break
            self._draw_message("执行中 ...", "完成后按 [空格] 结束")
            self._flip_and_mark(M.EXEC_START, "EXEC_START")
            if not self._wait_space(lambda: self._draw_message(
                    "执行中 ...", "完成后按 [空格] 结束")):
                break
            self.marker.mark(M.EXEC_END, "EXEC_END")

            if self.args.once:
                break

        if run_started:
            self.marker.mark(M.RUN_END, "RUN_END")


def main(argv: list[str] | None = None) -> int:
    # configs/stim.yaml 的值作为 argparse 默认值 —— CLI 显式传参仍可覆盖
    over = stim_defaults("paradigm1")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    BaseStim.add_common_args(ap, over)
    Paradigm1Stim.add_args(ap, over)
    args = ap.parse_args(argv)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("[stim] pygame not installed. Run: pip install pygame",
              file=sys.stderr)
        return 2

    stim = Paradigm1Stim(args)
    print(f"[stim] task_id={stim.trial.task_id} "
          f"task_name={stim.trial.task_name!r} "
          f"(Esc 结束，空格重复本任务)")
    return stim.run()


if __name__ == "__main__":
    raise SystemExit(main())
