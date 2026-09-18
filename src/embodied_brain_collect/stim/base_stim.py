"""BaseStim — 所有刺激程序的共同骨架。

每个刺激程序都是同一个形状:pygame 全屏窗口 + MarkerSender(串口 TTL 与
UDP 双路) + 一段 trial 流程。以前 paradigm1 与 sync_test 各自复制了窗口/
字体/marker 构造、Esc 中止、SPACE 等待、时间压缩、清理收尾——现在这些
都在这里,子类只写自己真正的流程(``run_flow``)。

子类契约::

    class MyStim(BaseStim):
        title = "我的实验"

        @classmethod
        def add_args(cls, ap):        # 可选:专属参数(公共参数已由
            ...                        # add_common_args 注册)
        def run_flow(self) -> None:   # 必需:设置 self.aborted 可提前退出
            ...

CLI 主函数模式(参数默认值来自 configs/stim.yaml)::

    def main(argv=None):
        over = stim_defaults("my_stim")          # stim.yaml 公共段 + my_stim 段
        ap = argparse.ArgumentParser(...)
        BaseStim.add_common_args(ap, over)
        MyStim.add_args(ap, over)
        args = ap.parse_args(argv)
        return MyStim(args).run()
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from embodied_brain_collect.stim.marker_sender import MarkerSender

# 全屏黑底、红点注视、浅灰正文、深灰提示 —— 所有范式共用同一套视觉语言
BG = (0, 0, 0)
FIX_COLOR = (220, 0, 0)
TEXT_COLOR = (230, 230, 230)
HINT_COLOR = (120, 120, 120)

_FONT_CANDIDATES = ["msyh.ttc", "simhei.ttf", "simsun.ttc", "NotoSansCJK*",
                    "wqy-microhei.ttc", "DroidSansFallbackFull.ttf",
                    "uming.ttc", "ukai.ttc"]


def _required(over: dict, key: str, cast=float):
    """stim.yaml 是参数唯一来源:缺键直接报缺哪个,不再留代码默认。

    代码字面默认会和 yaml 各存一份,两边迟早漂移(曾经 instr_s 代码
    10s / yaml 5s)。缺键是配置问题,启动即报,好过带着错默认跑一半。
    语义性默认(fast=1.0 不压缩、fullscreen 默认开)除外。
    """
    if key not in over:
        raise SystemExit(
            f"[stim] configs/stim.yaml 缺少 {key} — 补上该键(公共段或"
            "所用 kind 的段)")
    return cast(over[key])


def stim_defaults(section: str) -> dict:
    """configs/stim.yaml 的公共键 + ``section`` 专属键,作为 argparse 默认值。"""
    try:
        from embodied_brain_collect.session.config import load_stim
        stim = load_stim()
    except FileNotFoundError:
        stim = {}
    over = {k: v for k, v in stim.items() if not isinstance(v, dict)}
    over.update({k: v for k, v in stim.items()
                 if isinstance(v, dict)}.get(section, {}))
    return over


def _find_font(preferred: list[str]) -> str | None:
    dirs = ([Path(r"C:\Windows\Fonts")] if platform.system() == "Windows"
            else [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")])
    for d in dirs:
        if not d.is_dir():
            continue
        for name in preferred:
            for f in d.rglob(name):
                return str(f)
    return None


class BaseStim:
    """pygame 窗口 + 字体 + marker 发射 + 通用交互/计时原语。

    子类在 ``__init__`` 里通过 ``self.pygame`` 画屏;所有 Esc 中止与
    SPACE 等待逻辑都已内建,流程里只需检查 ``self.aborted``。
    """

    title = "刺激程序"

    # ---- CLI 公共参数:默认值一律来自 stim.yaml(_required,缺键即报);
    # 只有语义性默认留在代码(fast=1.0 不压缩、fullscreen 默认开)------

    @staticmethod
    def add_common_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--parallelbox", default=over.get("parallelbox", ""),
                        help="ParallelBox 串口(EEG TTL);多台放大器逗号分隔,"
                             '如 "COM5,COM14"')
        ap.add_argument("--baud", type=int,
                        default=_required(over, "baud", int))
        ap.add_argument("--marker-host", default=_required(over, "udp_host", str))
        ap.add_argument("--marker-port", type=int,
                        default=_required(over, "udp_port", int))
        ap.add_argument("--hold-s", type=float,
                        default=_required(over, "hold_s"),
                        help="TTL 高电平持续时间(s)")
        ap.add_argument("--no-serial", action="store_true",
                        default=not bool(over.get("serial", True)),
                        help="不发 ParallelBox 串口")
        ap.add_argument("--width", type=int,
                        default=_required(over, "width", int))
        ap.add_argument("--height", type=int,
                        default=_required(over, "height", int))
        ap.add_argument("--fullscreen", action="store_true",
                        default=bool(over.get("fullscreen", True)))
        ap.add_argument("--windowed", action="store_true",
                        help="强制窗口模式(覆盖 --fullscreen)")
        ap.add_argument("--display", type=int,
                        default=_required(over, "display", int),
                        help="启动屏幕索引(0=主屏;双屏时刺激屏通常为 1)")
        ap.add_argument("--font-size", type=int,
                        default=_required(over, "font_size", int))
        ap.add_argument("--fast", type=float,
                        default=float(over.get("fast", 1.0)),
                        help="时间压缩倍率,试跑用(10 = 快 10 倍)")
        ap.add_argument("--environment", default="",
                        help="图纸相对路径(configs/environments 下):画面"
                             "显示其场景/任务;图纸模式由 launcher 传入")

    # ---- 生命周期 ----------------------------------------------------------

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # 图纸模式(--environment):所有 stim 的画面都显示该图纸的场景/任务
        self.environment = getattr(args, "environment", "")
        self.scene_title = ""
        self.scene_task = ""
        if self.environment:
            from ..session import environment as env
            self.scene_title = env.scene_title(self.environment)
            self.scene_task = env.scene_task(self.environment)

        import pygame
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mouse.set_visible(False)
        display = max(0, int(getattr(args, "display", 0) or 0))
        n_displays = pygame.display.get_num_displays()
        if display >= n_displays:
            print(f"[stim] 配置的屏幕 {display} 不存在(共 {n_displays} 块)"
                  "— 用主屏")
            display = 0
        flags = (pygame.FULLSCREEN
                 if (args.fullscreen and not args.windowed) else 0)
        self.screen = pygame.display.set_mode(
            (args.width, args.height), flags, display=display)
        pygame.display.set_caption(self.title)

        font_path = _find_font(_FONT_CANDIDATES)
        if font_path is None and platform.system() != "Windows":
            font_path = pygame.font.get_default_font()
        self.font_instr = pygame.font.Font(font_path, args.font_size)
        self.font_small = pygame.font.Font(font_path, max(16, args.font_size // 3))
        self.font_title = pygame.font.Font(font_path, max(20, args.font_size // 2))

        self.marker = MarkerSender(
            port=args.parallelbox, baud=args.baud,
            udp_host=args.marker_host, udp_port=args.marker_port,
            hold_s=args.hold_s, enable_serial=not args.no_serial,
        )

        self.pygame = pygame
        self.bg = BG
        self.fix_color = FIX_COLOR
        self.text_color = TEXT_COLOR
        self.hint_color = HINT_COLOR
        self.aborted = False

    # ---- 绘制 --------------------------------------------------------------

    def _clear(self) -> None:
        self.screen.fill(self.bg)

    def _draw_message(self, main: str, hint: str = "") -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        main_surf = self.font_instr.render(main, True, self.text_color)
        self.screen.blit(main_surf,
                         main_surf.get_rect(center=(cx, cy - (20 if hint else 0))))
        if hint:
            hint_surf = self.font_small.render(hint, True, self.hint_color)
            self.screen.blit(hint_surf,
                             hint_surf.get_rect(center=(cx, cy + self.args.font_size)))

    def _draw_fixation(self) -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        self.pygame.draw.circle(self.screen, self.fix_color, (cx, cy),
                                self.args.fix_radius)

    def _flip_and_mark(self, code: int, tag: str) -> float:
        """翻屏并在同一步发 marker —— 画面与标记在同一帧生效。"""
        self.pygame.display.flip()
        return self.marker.mark(code, tag)

    # ---- 时间与输入 --------------------------------------------------------

    def _scale(self, seconds: float) -> float:
        return seconds / max(1.0, self.args.fast)

    def _poll_abort(self) -> bool:
        """泵事件并检测 Esc。每帧都要泵:Windows 以事件队列判定未响应。"""
        for ev in self.pygame.event.get():
            if ev.type == self.pygame.QUIT:
                self.aborted = True
                return True
            if ev.type == self.pygame.KEYDOWN and ev.key == self.pygame.K_ESCAPE:
                self.aborted = True
                return True
        return False

    def _wait_seconds(self, seconds: float) -> None:
        end_t = time.perf_counter() + self._scale(seconds)
        while time.perf_counter() < end_t and not self.aborted:
            if self._poll_abort():
                return
            time.sleep(0.005)

    def _wait_space(self, draw_fn=None) -> bool:
        """等待 SPACE;期间可用 draw_fn 重绘画面。返回 False 表示已中止。"""
        pg = self.pygame
        while not self.aborted:
            if draw_fn is not None:
                draw_fn()
                pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        self.aborted = True
                        return False
                    if ev.key == pg.K_SPACE:
                        return True
            time.sleep(0.01)
        return False

    # ---- 主循环 ------------------------------------------------------------

    def run_flow(self) -> None:
        """子类实现 trial 流程;设置 self.aborted 提前结束。"""
        raise NotImplementedError

    def run(self) -> int:
        try:
            self.run_flow()
        finally:
            self.marker.close()
            self.pygame.mouse.set_visible(True)
            self.pygame.quit()
        return 0 if not self.aborted else 2
