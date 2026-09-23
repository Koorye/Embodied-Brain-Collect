"""辅助员控制台 —— 独立程序:显示刺激屏实时截屏 + third camera 画面。

    python scripts/assist_console.py            # 配置见 configs/assist_console.yaml
    python scripts/assist_console.py --windowed # 调试:窗口化

完全独立于主采集逻辑:自己截取刺激屏画面(mss),自己收相机预览流,
随时开关都不影响录制。布局:左 = 刺激屏实时镜像,右 = third camera;
辅助员看本屏判断采集阶段与采集员动作,直接按键盘空格切换阶段
(空格作用于持有焦点的刺激窗口)。Esc = 关闭控制台。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading

import cv2
import numpy as np


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
    ap.add_argument("--camera-port", type=int, default=None,
                    help="相机预览流端口(缺省读配置文件)")
    ap.add_argument("--fps", type=float, default=None, help="控制台刷新率")
    ap.add_argument("--windowed", action="store_true", help="调试:窗口化")
    args = ap.parse_args(argv)

    cfg = load_config()
    display = args.display if args.display is not None else int(cfg.get("display", 0))
    stim_display = (args.stim_display if args.stim_display is not None
                    else int(cfg.get("stim_display", 0)))
    show_mirror = bool(cfg.get("mirror", True)) and not args.no_mirror
    show_camera = bool(cfg.get("camera", True)) and not args.no_camera
    camera_port = (args.camera_port if args.camera_port is not None
                   else int(cfg.get("camera_port", 9996)))
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 30))
    print(f"[assist] display={display} stim_display={stim_display} "
          f"mirror={show_mirror} camera={show_camera}@{camera_port} fps={fps}")

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
    pygame.display.set_caption("辅助员控制台")
    font_path = _find_font(_FONT_CANDIDATES)
    font_mid = pygame.font.Font(
        font_path or pygame.font.get_default_font(), max(26, sh // 26))
    font_small = pygame.font.Font(
        font_path or pygame.font.get_default_font(), max(20, sh // 36))

    # --- 相机预览流(third camera 录制器推送) ---
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cam_sock.bind(("127.0.0.1", camera_port))
    cam_sock.setblocking(False)
    cam_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)

    # --- 刺激屏截屏线程(mss) ---
    stop_evt = threading.Event()
    holder: dict = {"stim": None}
    if show_mirror:
        threading.Thread(target=capture_loop,
                         args=(stim_display, 12.0, stop_evt, holder),
                         daemon=True).start()

    clock = pygame.time.Clock()

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
        # --- 收相机包:只保留最新帧 ---
        try:
            while True:
                data, _ = cam_sock.recvfrom(256 * 1024)
                arr = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    holder["cam"] = (frame_rgb,
                                     (arr.shape[1], arr.shape[0]))
        except BlockingIOError:
            pass

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # --- 渲染:左 = 刺激屏镜像,右 = third camera ---
        screen.fill((16, 16, 18))
        gap = 12
        panel_w = (sw - gap * 3) // 2
        panel_h = sh - gap * 2 - int(sh * 0.07)
        left = pygame.Rect(gap, gap, panel_w, panel_h)
        right = pygame.Rect(gap * 2 + panel_w, gap, panel_w, panel_h)

        pygame.draw.rect(screen, (28, 28, 34), left)
        stim = holder.get("stim")
        if show_mirror and stim is not None:
            blit_fit(stim[0], left)
        else:
            txt = font_mid.render(
                "刺激屏镜像关闭" if not show_mirror else "截屏启动中 …",
                True, (200, 200, 200))
            screen.blit(txt, txt.get_rect(center=left.center))
        lbl = font_small.render(f"stim(刺激屏 {stim_display})", True,
                                (170, 170, 170))
        screen.blit(lbl, (left.x + 10, left.y + 8))

        pygame.draw.rect(screen, (28, 28, 34), right)
        cam = holder.get("cam")
        if show_camera and cam is not None:
            blit_fit(cam[0], right)
        else:
            txt = font_mid.render("等待相机预览 …", True, (200, 200, 200))
            screen.blit(txt, txt.get_rect(center=right.center))
        lbl = font_small.render("third camera(采集员)", True, (170, 170, 170))
        screen.blit(lbl, (right.x + 10, right.y + 8))

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

    cam_sock.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
