"""RGB 色块刺激 —— 纯颜色三阶段流程,无任何文字提示。

    阶段1  黑底 + 5/4/3/2/1 倒数(每秒一跳,``wait_s`` 秒)等待期,数完自动切换
    阶段2  蓝色  想象期   按 [空格] 切换
    阶段3  绿色  操作期   按 [空格] 结束

单次运行 = 一个完整 run,阶段顺序与 marker 语法与 paradigm1 一致
(颜色替代文字指令):

    RUN_START → INSTR_ON(红) → INSTR_OFF
              → IMG_START(蓝) → IMG_END
              → EXEC_START(绿) → EXEC_END → RUN_END

Esc 中止也会补全未闭合的阶段码与 RUN_END,保住 EEG 对齐的码对。
颜色默认纯 RGB 原色(便于光电二极管/颜色 cue),可在 configs/stim.yaml
的 ``rgb:`` 段或 CLI 覆盖。

    python -m embodied_brain_collect.stim.rgb_cue
    python -m embodied_brain_collect.stim.rgb_cue --wait-s 2 --windowed   # 冒烟
"""

from __future__ import annotations

import argparse
import sys

from embodied_brain_collect.stim.base_stim import (
    BaseStim, _required, stim_defaults)
from embodied_brain_collect.stim import marker_codes as M


def _rgb(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """``#RRGGBB`` → (r, g, b);解析失败回退默认色。"""
    v = str(value).strip().lstrip("#")
    try:
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except (ValueError, IndexError):
        return fallback


class RgbCueStim(BaseStim):
    """红/蓝/绿三阶段颜色 cue,纯色块无文字。"""

    title = "RGB Cue"

    @staticmethod
    def add_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--wait-s", type=float,
                        default=_required(over, "wait_s"),
                        help="红色等待期秒数(到时自动切换)")
        ap.add_argument("--color-imag", default=_required(over, "color_imag", str),
                        help="阶段2 颜色(默认蓝)")
        ap.add_argument("--color-exec", default=_required(over, "color_exec", str),
                        help="阶段3 颜色(默认绿)")

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.c_imag = _rgb(args.color_imag, (0, 0, 255))
        self.c_exec = _rgb(args.color_exec, (0, 255, 0))
        # 倒数数字用大字号(基础字号的 4 倍,黑底居中)
        from .base_stim import _FONT_CANDIDATES, _find_font
        path = _find_font(_FONT_CANDIDATES)
        size = max(120, int(self.args.font_size * 4))
        self.font_digit = self.pygame.font.Font(
            path or self.pygame.font.get_default_font(), size)

    def _show_digit(self, digit: int) -> None:
        """黑底 + 居中大数字,无其他元素。"""
        self.screen.fill((0, 0, 0))
        surf = self.font_digit.render(str(digit), True, (235, 235, 235))
        self.screen.blit(surf, surf.get_rect(
            center=(self.screen.get_width() // 2,
                    self.screen.get_height() // 2)))
        self.pygame.display.flip()

    def _show(self, color: tuple[int, int, int],
              code: int | None = None, tag: str = "") -> None:
        """整屏填色 + 翻屏;给码则翻屏后立刻发(画面与标记同帧生效)。"""
        self.screen.fill(color)
        self.pygame.display.flip()
        if code is not None:
            self.marker.mark(code, tag)

    def run_flow(self) -> None:
        self.marker.mark(M.RUN_START, "RUN_START")

        # 阶段1 黑底倒数 —— 等待期,数完自动切换
        self._show((0, 0, 0), M.INSTR_ON, "INSTR_ON")
        for digit in range(max(1, int(round(self.args.wait_s))), 0, -1):
            self._show_digit(digit)
            self._wait_seconds(1.0)
            if self.aborted:
                break
        self.marker.mark(M.INSTR_OFF, "INSTR_OFF")

        if not self.aborted:
            # 阶段2 蓝色 —— 想象期,按 [空格] 切换(Esc 中止也补 END)
            self._show(self.c_imag, M.IMG_START, "IMG_START")
            self._wait_space(lambda: self._show(self.c_imag))
            self.marker.mark(M.IMG_END, "IMG_END")

        if not self.aborted:
            # 阶段3 绿色 —— 操作期,按 [空格] 结束
            self._show(self.c_exec, M.EXEC_START, "EXEC_START")
            self._wait_space(lambda: self._show(self.c_exec))
            self.marker.mark(M.EXEC_END, "EXEC_END")

        self.marker.mark(M.RUN_END, "RUN_END")


def main(argv: list[str] | None = None) -> int:
    over = stim_defaults("rgb")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    BaseStim.add_common_args(ap, over)
    RgbCueStim.add_args(ap, over)
    args = ap.parse_args(argv)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("[stim] pygame not installed. Run: pip install pygame",
              file=sys.stderr)
        return 2

    stim = RgbCueStim(args)
    print(f"[stim] RGB cue: 黑底倒数 {stim.args.wait_s:g}s → 空格(蓝)→ "
          "空格(绿)→ 空格结束;Esc 中止")
    return stim.run()


if __name__ == "__main__":
    raise SystemExit(main())
