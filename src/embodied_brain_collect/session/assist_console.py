"""辅助员控制台 —— 独立程序:显示刺激屏实时截屏 + 任意多路相机画面。

    python scripts/assist_console.py            # 配置见 configs/assist_console.yaml
    python scripts/assist_console.py --windowed # 调试:窗口化

完全独立于主采集逻辑:自己截取刺激屏画面(mss),自己收各路相机的 UDP JPEG
预览流,随时开关都不影响录制。辅助员看本屏判断采集阶段与采集员动作,直接
按键盘空格切换阶段(空格作用于持有焦点的刺激窗口)。Esc = 关闭控制台。

布局随屏幕自适应(``layout: auto``):**竖屏(高 > 宽)上下排 —— 上 = 刺激屏
镜像,下 = 相机区**;横屏左右排(左 stim / 右相机)。``layout`` 可写死
``vertical`` / ``horizontal``,``split`` 调两块的占比。相机区内一路占满,
多路按该区域的宽高比自动选行列(让每路画面尽量大)。

相机来源由 ``cameras`` 段逐条配置(标题 + 端口),端口要与 recorders.yaml
里对应 recorder 的 ``preview_port`` / 头环的 ``preview_ports`` 一致 ——
第三视角、腕部相机、头环四路鱼眼里的任意一路或多路都能同屏显示。
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import threading

import cv2
import numpy as np

#: 没有 cameras 段时的兜底(兼容旧配置的单口 camera_port)
_DEFAULT_CAMERAS = [{"title": "third camera(采集员)", "port": 9996}]


def load_config() -> dict:
    """configs/assist_console.yaml;文件缺失 = 全部默认值。"""
    try:
        import yaml
        from ..session.config import configs_dir
        path = configs_dir() / "assist_console.yaml"
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve_cameras(cfg: dict, ports_override: list[int] | None) -> list[dict]:
    """配置 → ``[{title, port}, ...]``(保持书写顺序 = 面板顺序)。

    优先 ``--camera-port``(可重复,顺序即面板顺序);否则读 ``cameras``
    段;两者都没有时回退旧的单口 ``camera_port``。端口非法/重复的条目
    直接丢弃 —— 控制台是旁路工具,配错不该让它起不来。
    """
    if ports_override:
        return [{"title": f"camera :{p}", "port": int(p)}
                for p in ports_override]
    raw = cfg.get("cameras")
    if raw is None:
        port = int(cfg.get("camera_port", 0) or 0)
        return ([{"title": _DEFAULT_CAMERAS[0]["title"], "port": port}]
                if port > 0 else list(_DEFAULT_CAMERAS))
    out: list[dict] = []
    seen: set[int] = set()
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            title, port = item.get("title") or item.get("name"), item.get("port")
        else:                      # 只写端口的简写形式
            title, port = None, item
        try:
            p = int(port)
        except (TypeError, ValueError):
            continue
        if p <= 0 or p in seen:
            continue
        seen.add(p)
        out.append({"title": str(title) if title else f"camera :{p}",
                    "port": p})
    return out


def grid_shape(n: int, area_w: float = 16.0, area_h: float = 9.0,
               frame_aspect: float = 16 / 9) -> tuple[int, int]:
    """n 路画面在 ``area_w × area_h`` 区域里的 (列, 行)。

    穷举所有 (列, 行) 组合,取**单格实际画面面积最大**的那个 —— 竖长
    区域自然选出 1×n / 2×n 这类竖排,宽区域选出 n×1,方区域选近正方。
    ``frame_aspect`` 是画面本身的宽高比(预览按等比缩放塞进格子,格子
    富余的那一维不贡献面积)。
    """
    if n <= 1:
        return 1, 1
    best, best_area = (1, n), -1.0
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        cell_w, cell_h = area_w / cols, area_h / rows
        # 等比缩放后画面的实际尺寸:受限于格子较紧的那一维
        scale = min(cell_w / frame_aspect, cell_h)
        area = (scale * frame_aspect) * scale
        if area > best_area + 1e-9:
            best, best_area = (cols, rows), area
    return best


def capture_loop(stim_display: int, fps: float, stop: threading.Event,
                 holder: dict) -> None:
    """后台线程:持续截取刺激屏画面(mss),最新帧写进 holder。"""
    try:
        import mss
        with mss.mss() as sct:
            idx = stim_display + 1          # mss: 0=整个虚拟屏,1..N=各显示器
            while not stop.is_set():
                try:
                    mons = sct.monitors
                    mon = mons[idx] if idx < len(mons) else mons[-1]
                    shot = sct.grab(mon)
                    holder["stim"] = (
                        np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
                            shot.height, shot.width, 3),
                        (shot.width, shot.height))
                except Exception:
                    pass
                stop.wait(1.0 / max(fps, 1.0))
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--display", type=int, default=None,
                    help="控制台全屏显示的屏幕(缺省读配置文件)")
    ap.add_argument("--stim-display", type=int, default=None,
                    help="被镜像的刺激屏(缺省读配置文件)")
    ap.add_argument("--no-mirror", action="store_true", help="不显示刺激屏")
    ap.add_argument("--no-camera", action="store_true", help="不显示相机")
    ap.add_argument("--camera-port", type=int, action="append", default=None,
                    help="相机预览端口,可重复给多路(缺省读配置文件 cameras 段)")
    ap.add_argument("--fps", type=float, default=None, help="控制台刷新率")
    ap.add_argument("--layout", choices=("auto", "vertical", "horizontal"),
                    default=None,
                    help="面板排布:vertical=上下(竖屏), horizontal=左右, "
                         "auto=按屏幕宽高比自动(缺省读配置文件)")
    ap.add_argument("--windowed", action="store_true", help="调试:窗口化")
    args = ap.parse_args(argv)

    cfg = load_config()
    display = args.display if args.display is not None else int(cfg.get("display", 0))
    stim_display = (args.stim_display if args.stim_display is not None
                    else int(cfg.get("stim_display", 0)))
    show_mirror = bool(cfg.get("mirror", True)) and not args.no_mirror
    show_camera = bool(cfg.get("camera", True)) and not args.no_camera
    cameras = resolve_cameras(cfg, args.camera_port) if show_camera else []
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 30))
    layout = str(args.layout or cfg.get("layout", "auto")).lower()
    # stim 区占整屏的比例(竖屏=高度占比,横屏=宽度占比),其余给相机区
    split = float(cfg.get("split", 0.5))
    split = min(max(split, 0.2), 0.8)
    print(f"[assist] display={display} stim_display={stim_display} "
          f"mirror={show_mirror} cameras="
          + (", ".join(f"{c['title']}@{c['port']}" for c in cameras)
             if cameras else "(无)")
          + f" fps={fps}")

    import pygame
    from ..stim.base_stim import _FONT_CANDIDATES, _find_font

    pygame.init()
    n = pygame.display.get_num_displays()
    disp = display
    if disp >= n:
        disp = 0
    from .config import display_settings
    ds = display_settings(disp)
    flags = pygame.FULLSCREEN if ds["mode"] == "fullscreen" else 0
    size = ((ds["width"], ds["height"])
            if (ds["mode"] == "windowed"
                and ds["width"] > 0 and ds["height"] > 0)
            else (0, 0))
    screen = pygame.display.set_mode(size, flags, display=disp)
    sw, sh = screen.get_size()
    # 竖屏(高 > 宽)默认上下排,横屏左右排;layout 写死时以配置为准
    vertical = (sh > sw) if layout == "auto" else (layout == "vertical")
    print(f"[assist] 屏幕 {sw}x{sh} → 布局 "
          f"{'上下(竖屏)' if vertical else '左右(横屏)'} split={split:g}")
    pygame.display.set_caption("辅助员控制台")
    font_path = _find_font(_FONT_CANDIDATES)
    font_mid = pygame.font.Font(
        font_path or pygame.font.get_default_font(), max(26, sh // 26))
    font_small = pygame.font.Font(
        font_path or pygame.font.get_default_font(), max(20, sh // 36))

    # --- 各路相机预览流:一路一个 UDP socket(端口即身份) ---
    cam_socks: list[socket.socket] = []
    for cam in cameras:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(("127.0.0.1", cam["port"]))
        except OSError as exc:
            # 端口被占(常见:两个控制台同时开)不该拖垮整个面板
            print(f"[assist] 端口 {cam['port']} 绑定失败 — {exc};该路不显示")
            s.close()
            cam["dead"] = True
            cam_socks.append(None)
            continue
        s.setblocking(False)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
        cam_socks.append(s)

    # --- 刺激屏截屏线程(mss) ---
    stop_evt = threading.Event()
    holder: dict = {"stim": None}
    if show_mirror:
        threading.Thread(target=capture_loop,
                         args=(stim_display, 12.0, stop_evt, holder),
                         daemon=True).start()

    clock = pygame.time.Clock()
    frames: dict[int, np.ndarray] = {}      # 面板序号 -> 最新 RGB 帧

    def blit_fit(arr: np.ndarray, rect: pygame.Rect) -> None:
        wh = (arr.shape[1], arr.shape[0])
        surf = pygame.image.frombuffer(arr.tobytes(), wh, "RGB")
        scale = min(rect.w / wh[0], rect.h / wh[1])
        tw, th = max(1, int(wh[0] * scale)), max(1, int(wh[1] * scale))
        scaled = pygame.transform.smoothscale(surf, (tw, th))
        screen.blit(scaled, scaled.get_rect(center=rect.center))

    def label(rect: pygame.Rect, text: str) -> None:
        screen.blit(font_small.render(text, True, (170, 170, 170)),
                    (rect.x + 10, rect.y + 8))

    running = True
    while running:
        # --- 收各路相机包:每路只保留最新帧 ---
        for i, s in enumerate(cam_socks):
            if s is None:
                continue
            try:
                while True:
                    data, _ = s.recvfrom(256 * 1024)
                    arr = cv2.imdecode(
                        np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        frames[i] = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            except BlockingIOError:
                pass

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # --- 渲染:竖屏 = stim 在上/相机在下,横屏 = stim 左/相机右 ---
        screen.fill((16, 16, 18))
        gap = 12
        bar_h = int(sh * 0.07)
        body_w, body_h = sw - gap * 2, sh - gap * 2 - bar_h
        if vertical:
            stim_h = int(body_h * split)
            left = pygame.Rect(gap, gap, body_w, stim_h)
            right = pygame.Rect(gap, gap + stim_h + gap,
                                body_w, body_h - stim_h - gap)
        else:
            stim_w = int(body_w * split)
            left = pygame.Rect(gap, gap, stim_w, body_h)
            right = pygame.Rect(gap + stim_w + gap, gap,
                                body_w - stim_w - gap, body_h)

        pygame.draw.rect(screen, (28, 28, 34), left)
        stim = holder.get("stim")
        if show_mirror and stim is not None:
            blit_fit(stim[0], left)
        else:
            txt = font_mid.render(
                "刺激屏镜像关闭" if not show_mirror else "截屏启动中 …",
                True, (200, 200, 200))
            screen.blit(txt, txt.get_rect(center=left.center))
        label(left, f"stim(刺激屏 {stim_display})")

        if not cameras:
            pygame.draw.rect(screen, (28, 28, 34), right)
            txt = font_mid.render("相机预览关闭", True, (200, 200, 200))
            screen.blit(txt, txt.get_rect(center=right.center))
        else:
            cols, rows = grid_shape(len(cameras), right.w, right.h)
            cell_w = (right.w - gap * (cols - 1)) // cols
            cell_h = (right.h - gap * (rows - 1)) // rows
            for i, cam in enumerate(cameras):
                r, c = divmod(i, cols)
                cell = pygame.Rect(right.x + c * (cell_w + gap),
                                   right.y + r * (cell_h + gap),
                                   cell_w, cell_h)
                pygame.draw.rect(screen, (28, 28, 34), cell)
                frame = frames.get(i)
                if frame is not None:
                    blit_fit(frame, cell)
                else:
                    msg = ("端口被占用" if cam.get("dead")
                           else "等待预览 …")
                    txt = font_small.render(msg, True, (200, 200, 200))
                    screen.blit(txt, txt.get_rect(center=cell.center))
                label(cell, f"{cam['title']}")

        bar = pygame.Rect(0, sh - int(sh * 0.07), sw, int(sh * 0.07))
        pygame.draw.rect(screen, (30, 30, 36), bar)
        tip = font_mid.render("空格 = 切换阶段(焦点保持在刺激窗口)",
                              True, (255, 230, 120))
        screen.blit(tip, tip.get_rect(midleft=(24, bar.centery)))
        tip2 = font_small.render("Esc = 关闭控制台(不影响采集)",
                                 True, (150, 150, 150))
        screen.blit(tip2, tip2.get_rect(midright=(sw - 24, bar.centery)))
        pygame.display.flip()
        clock.tick(fps)

    stop_evt.set()
    for s in cam_socks:
        if s is not None:
            s.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
