"""同步测试刺激程序 —— 画面计时与 marker 时刻的对照实验。

流程(全自动,无按键交互):

    想象阶段  画面显示「开始想象」+ 计时(带毫秒),持续 --imag-s 秒
              marker: IMG_START ... IMG_END
    执行阶段  计时归零重新开始,循环 --cycles 轮:
              0.0s 抬左手  2.5s 放左手  5.0s 抬右手  7.5s 放右手
              (每轮 --cycle-s*4 秒;每个动作切换时刻发对应 marker)
              marker: EXEC_START, LIFT/PUT_*, EXEC_END
    结束      RUN_END 后退出

marker 走与真实 stim 相同的 MarkerSender(串口 TTL + UDP 带发送端
t_sent_pc),因此录到的 marker 时间戳可用于校准各传感器时钟。

窗口 / 字体 / marker / 计时原语来自 ``base_stim.BaseStim``。

Usage::

    python -m embodied_brain_collect.stim.sync_test \\
        [--windowed] [--imag-s 10] [--cycle-s 2.5] [--cycles 3]
"""

from __future__ import annotations

import argparse
import sys
import time

from embodied_brain_collect.stim.base_stim import BaseStim, stim_defaults
from embodied_brain_collect.stim.marker_codes import (EXEC_END, EXEC_START,
                                                      IMG_END, IMG_START,
                                                      RUN_END, RUN_START,
                                                      make_hand_cue)

# 动作序列(文字);marker 码 = make_hand_cue(轮次, 序号) —— 每轮用不同的码,
# 保证整个 session 码唯一(EEG 对齐按码配对要求唯一)
_CYCLE = ("抬左手", "放左手", "抬右手", "放右手")
_LIFT_COLOR = (255, 190, 0)      # 抬=琥珀(醒目),放=白


def _fmt(seconds: float) -> str:
    """mm:ss.mmm"""
    m = int(seconds) // 60
    return f"{m:02d}:{seconds - m * 60:06.3f}"


class SyncTestStim(BaseStim):
    """全自动同步测试:想象 + 执行阶段,每个动作切换时刻发 marker。"""

    title = "同步测试"

    @staticmethod
    def add_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--imag-s", type=float,
                        default=float(over.get("imag_s", 10.0)),
                        help="想象阶段时长")
        ap.add_argument("--read-s", type=float,
                        default=float(over.get("read_s", 5.0)),
                        help="想象序列提示画面时长(受试者睁眼阅读)")
        ap.add_argument("--cycle-s", type=float,
                        default=float(over.get("cycle_s", 2.5)),
                        help="每个动作时长")
        ap.add_argument("--cycles", type=int,
                        default=int(over.get("cycles", 3)),
                        help="执行阶段循环轮数")

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.cx = args.width // 2
        self.cy = args.height // 2

    def _draw(self, title: str, big: str, elapsed: float, color=None) -> None:
        screen, pg = self.screen, self.pygame
        screen.fill(self.bg)
        t = self.font_small.render(title, True, self.hint_color)
        screen.blit(t, t.get_rect(center=(self.cx, self.cy - 260)))
        b = self.font_instr.render(big, True, color or self.text_color)
        screen.blit(b, b.get_rect(center=(self.cx, self.cy)))
        clock = self.font_instr.render(_fmt(elapsed), True, self.text_color)
        screen.blit(clock, clock.get_rect(center=(self.cx, self.cy + 260)))
        pg.display.flip()

    def _draw_read(self, text: str) -> None:
        """提示画面:逐段渲染抬/放序列,抬=琥珀、放=白。"""
        screen = self.screen
        self._clear()
        head = self.font_small.render("请记住想象序列", True, self.text_color)
        screen.blit(head, head.get_rect(center=(self.cx, self.cy - 120)))
        parts = []
        for k, t in enumerate(_CYCLE):
            color = _LIFT_COLOR if k % 2 == 0 else self.text_color
            parts.append(self.font_small.render(t, True, color))
            if k < len(_CYCLE) - 1:
                parts.append(self.font_small.render(" → ", True, self.hint_color))
        total_w = sum(p.get_width() for p in parts)
        x = self.cx - total_w // 2
        for p in parts:
            screen.blit(p, p.get_rect(midleft=(x, self.cy + 60)))
            x += p.get_width()
        self.pygame.display.flip()

    def run_flow(self) -> None:
        args, marker = self.args, self.marker

        # ---- 准备画面:闭眼前告知想象序列(睁眼阅读) ----
        t0 = time.time()
        while time.time() - t0 < args.read_s and not self.aborted:
            self._draw_read("请记住想象序列")
            if self._poll_abort():
                break
            time.sleep(0.02)

        # ---- 想象阶段:闭眼想象(画面只计时) ----
        marker.mark(RUN_START, "RUN_START")
        marker.mark(IMG_START, "IMG_START")
        t0 = time.time()
        while not self.aborted:
            elapsed = time.time() - t0
            if elapsed >= args.imag_s:
                break
            self._draw("同步测试 · 开始想象", "闭眼想象", elapsed)
            if self._poll_abort():
                break
            time.sleep(0.02)
        marker.mark(IMG_END, "IMG_END")

        # ---- 执行阶段 ----
        marker.mark(EXEC_START, "EXEC_START")
        t0 = time.time()
        action_i = -1
        while not self.aborted:
            elapsed = time.time() - t0
            cycle = int(elapsed // (args.cycle_s * 4))
            if cycle >= args.cycles:
                break
            i = int((elapsed % (args.cycle_s * 4)) // args.cycle_s)
            if i >= len(_CYCLE):
                continue
            if i != action_i:
                action_i = i
                marker.mark(make_hand_cue(cycle, i), f"SYNC_{make_hand_cue(cycle, i):02X}")
            color = _LIFT_COLOR if i % 2 == 0 else self.text_color
            self._draw("同步测试 · 开始执行",
                       f"{_CYCLE[i]}  (第 {cycle + 1}/{args.cycles} 轮)",
                       elapsed, color)
            if self._poll_abort():
                break
            time.sleep(0.02)
        marker.mark(EXEC_END, "EXEC_END")

        # ---- 结束 ----
        marker.mark(RUN_END, "RUN_END")
        if not self.aborted:
            print("[sync_test] 完成: 想象 %.1fs + 执行 %.1fs"
                  % (args.imag_s, args.cycle_s * 4 * args.cycles))


def main(argv: list[str] | None = None) -> int:
    over = stim_defaults("sync_test")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    BaseStim.add_common_args(ap, over)
    SyncTestStim.add_args(ap, over)
    args = ap.parse_args(argv)
    return SyncTestStim(args).run()


if __name__ == "__main__":
    sys.exit(main())
