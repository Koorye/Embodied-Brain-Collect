#!/usr/bin/env python3
"""EMG 左右手对应检查 —— 实时显示双手肌电波形,确认 emg_left/emg_right 没接反。

    python scripts/check_emg.py                     # 按记录器配置开窗检查
    python scripts/check_emg.py --list              # 只列出左右槽位配置,不开窗
    python scripts/check_emg.py --left-port COM31 --right-port COM30
    python scripts/check_emg.py --seconds 3         # 每次晃动的采样窗口(秒)

流程(窗口内按键,全程单循环,窗口不冻结):

  1. 打开左右两条 EMG(recorders.yaml 的 emg_left / emg_right 槽位)。
  2. 按 L → 提示反复晃动【左手】数秒;按 R → 晃动【右手】。
     窗口自动统计晃动期间左右两侧的肌电活动量并给出对照结论
     (对应 ✓ / 疑似接反 ✗ / 活动不明显)。
  3. 肉眼核对:左手动作时上面板(左手)波形应显著起伏,右手同理。
  4. 按 Y 确认无误退出(退出码 0);按 Q 放弃(退出码 1)。

波形不经 recorder 落盘 —— 直接驱动 weili_emg 的 open/poll(preflight 同款),
临时目录自始至终不产生 npz。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPT_DIR), str(_SCRIPT_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embodied_brain_collect.session.config import load_recorders  # noqa: E402
from embodied_brain_collect.session.recorder_presets import get_weili_emg  # noqa: E402
from embodied_brain_collect.stim.base_stim import _FONT_CANDIDATES, _find_font  # noqa: E402

PANEL_W, PANEL_H = 1280, 300          # 单侧面板尺寸
WINDOW_SAMPLES = 2000                 # 波形显示的样本数(约 1-2 s)
SHOW_ROWS = 8                         # 每只手 8 个 EMG 通道
COLOR_LEFT = (80, 220, 120)           # 左手面板波形:绿
COLOR_RIGHT = (255, 170, 60)          # 右手面板波形:橙
COLOR_TEXT = (230, 230, 230)
COLOR_HINT = (150, 150, 150)
COLOR_OK = (90, 220, 120)
COLOR_BAD = (255, 90, 90)


def configured_slots():
    """recorders.yaml 的左右 EMG 槽位配置;缺失/类型不对时给出可操作报错。"""
    cfg = load_recorders()
    out = {}
    for slot in ("emg_left", "emg_right"):
        c = cfg.get(slot)
        if not c:
            raise SystemExit(f"configs/recorders.yaml 缺少 {slot} 槽位 — 补上再检查")
        if c.get("kind") != "weili_emg":
            raise SystemExit(f"{slot} 的 kind 是 {c.get('kind')!r},check_emg 只支持 weili_emg")
        out[slot] = c
    return out


def open_side(slot: str, cfg: dict, port: str | None, tmp: str):
    """打开一侧 EMG(不经 recorder 落盘);失败返回 None。"""
    rec = get_weili_emg(session_dir=str(Path(tmp) / slot),
                        port=port if port is not None else str(cfg.get("port") or ""),
                        baud=int(cfg.get("baud") or 921600))
    try:
        ok = rec._open()
    except Exception as exc:  # noqa: BLE001 — 串口占用/不存在等,给可读报错
        rec._open_error = f"{type(exc).__name__}: {exc}"
        ok = False
    if not ok:
        print(f"[{slot}] 打开失败 — {rec._open_error or '未知原因'}")
        return None
    print(f"[{slot}] 已打开 {rec.config.port or '自动探测'} "
          f"@ {rec.config.baud} ({cfg.get('name', slot)})")
    return rec


def emg_view(rec):
    """最近的波形窗口:np.ndarray (n, 8) 或 None(还没数据)。"""
    buf = rec._arr_buf.get("emg_data")
    if not buf:
        return None
    return np.asarray(buf[-WINDOW_SAMPLES:], dtype=np.float64)


def activity_since(rec, n0: int) -> float:
    """n0 之后的新样本的平均通道标准差 —— 晃动动作的活动量度量。"""
    buf = rec._arr_buf.get("emg_data")
    new = buf[n0:] if buf else []
    if len(new) < 10:
        return 0.0
    arr = np.asarray(new, dtype=np.float64)
    return float(np.std(arr, axis=0).mean())


def judge(expected: str, a_exp: float, a_other: float) -> tuple[str, str]:
    """(结论文本, 颜色)。比值带迟滞,动作太轻给'不明显'。"""
    if a_exp <= 0 and a_other <= 0:
        return "没有检测到活动 — 加大动作重测", COLOR_HINT
    if a_exp > a_other * 1.5:
        return "对应 ✓", COLOR_OK
    if a_other > a_exp * 1.5:
        return "疑似接反 ✗", COLOR_BAD
    return "两侧活动接近 — 加大动作重测", COLOR_HINT


def draw_panel(surf, rect, rec, title: str, color, peaks: dict, font, font_s):
    """一侧面板:标题 + 8 通道滚动波形 + 活动值读数。"""
    import pygame
    pygame.draw.rect(surf, (28, 28, 32), rect)
    pygame.draw.rect(surf, (70, 70, 78), rect, 1)
    surf.blit(font.render(title, True, COLOR_TEXT), (rect.x + 12, rect.y + 8))

    data = emg_view(rec)
    rate = 0.0
    if data is not None and data.shape[0] >= 2:
        surf.blit(font_s.render(f"{data.shape[0]} 样本", True, COLOR_HINT),
                  (rect.right - 130, rect.y + 12))
        rows = SHOW_ROWS
        row_h = (rect.height - 40) // rows
        peak = peaks.get(title, 0.0)
        peak = max(float(np.abs(data).max()), peak * 0.995, 500.0)
        peaks[title] = peak
        for ch in range(rows):
            y0 = rect.y + 30 + ch * row_h + row_h // 2
            seg = data[:, ch]
            step = max(1, len(seg) // rect.width)
            pts = [(rect.x + i * rect.width // max(1, len(seg) - 1),
                    int(y0 - np.clip(seg[i] / peak, -0.9, 0.9) * row_h * 0.45))
                   for i in range(0, len(seg), step)]
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 1)
        act = float(np.std(data, axis=0).mean())
        n_all = len(rec._arr_buf.get("emg_data") or [])
        rate = _rate(rec)
        surf.blit(font_s.render(f"活动 {act:8.0f}   {rate:5.0f} 帧/s",
                                True, COLOR_HINT),
                  (rect.x + 12, rect.bottom - 26))
    else:
        surf.blit(font_s.render("等待数据流 …", True, COLOR_HINT),
                  (rect.x + 12, rect.y + rect.height // 2))
    return rate


_RATE = {}


def _rate(rec) -> float:
    """近 2 秒的 EMG 帧率,顺带证明数据流健康。"""
    ts = rec._frame_ts.get("emg_data") or []
    now = time.perf_counter()
    recent = [t for t in ts[-4000:] if now - t <= 2.0]
    _RATE[id(rec)] = len(recent) / 2.0 if len(recent) >= 2 else _RATE.get(id(rec), 0.0)
    return _RATE[id(rec)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left-port", default=None,
                    help="覆盖 emg_left 的串口(默认读 recorders.yaml)")
    ap.add_argument("--right-port", default=None,
                    help="覆盖 emg_right 的串口")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="每次晃动的采样窗口秒数(默认 3)")
    ap.add_argument("--list", action="store_true",
                    help="只列出左右槽位配置,不开窗")
    args = ap.parse_args(argv)

    slots = configured_slots()
    if args.list:
        for slot, c in slots.items():
            print(f"{slot}: port={c.get('port') or '自动探测'} "
                  f"baud={c.get('baud')} name={c.get('name', '')}")
        return 0

    try:
        import pygame
    except ImportError:
        print("pygame 未安装 — pip install pygame(刺激程序同款依赖)", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="check_emg_") as tmp:
        sides = {}
        for slot, key, port in (("emg_left", "left", args.left_port),
                                ("emg_right", "right", args.right_port)):
            rec = open_side(slot, slots[slot], port, tmp)
            if rec is None:
                print("有一侧没打开 — 检查接线/串口后重试(也可 --left-port/--right-port 指定)",
                      file=sys.stderr)
                return 1
            sides[key] = {"rec": rec, "slot": slot,
                          "label": f"{slot}({rec.config.port or 'auto'})"}

        pygame.init()
        font_path = _find_font(list(_FONT_CANDIDATES))
        font = pygame.font.Font(font_path, 26)
        font_s = pygame.font.Font(font_path, 18)
        screen = pygame.display.set_mode((PANEL_W, PANEL_H * 2 + 130))
        pygame.display.set_caption("EMG 左右手对应检查")

        state = "idle"                    # idle / test_L / done_L / test_R / done_R
        t0 = n0L = n0R = 0
        resL = resR = None                # (文本, 颜色, a_exp, a_other)
        peaks: dict = {}
        exit_code = 1
        clock = pygame.time.Clock()
        running = True

        def banner(text, color=COLOR_TEXT):
            screen.fill((16, 16, 18))
            screen.blit(font.render(text, True, color), (16, PANEL_H * 2 + 40))

        while running:
            now = time.time()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                        running = False
                    elif ev.key == pygame.K_l and state in ("idle", "done_L",
                                                            "done_R", "done_LR"):
                        state, t0 = "test_L", now
                        n0L = len(sides["left"]["rec"]._arr_buf.get("emg_data") or [])
                        n0R = len(sides["right"]["rec"]._arr_buf.get("emg_data") or [])
                        print(f"[test] 请反复晃动【左手】{args.seconds:g} 秒 …")
                    elif ev.key == pygame.K_r and state in ("idle", "done_L",
                                                            "done_R", "done_LR"):
                        state, t0 = "test_R", now
                        n0L = len(sides["left"]["rec"]._arr_buf.get("emg_data") or [])
                        n0R = len(sides["right"]["rec"]._arr_buf.get("emg_data") or [])
                        print(f"[test] 请反复晃动【右手】{args.seconds:g} 秒 …")
                    elif ev.key == pygame.K_y and state == "done_LR":
                        print("[result] 采集员确认左右对应正确 — 通过")
                        exit_code = 0
                        running = False

            for key in ("left", "right"):
                sides[key]["rec"]._poll(time.time())

            if state == "test_L" and now - t0 >= args.seconds:
                aL = activity_since(sides["left"]["rec"], n0L)
                aR = activity_since(sides["right"]["rec"], n0R)
                resL = (*judge("left", aL, aR), aL, aR)
                state = "done_LR" if resR is not None else "done_L"
                print(f"[test] 左手窗口:左侧活动 {aL:.0f} vs 右侧 {aR:.0f} → {resL[0]}")
            if state == "test_R" and now - t0 >= args.seconds:
                aL = activity_since(sides["left"]["rec"], n0L)
                aR = activity_since(sides["right"]["rec"], n0R)
                resR = (*judge("right", aR, aL), aL, aR)
                state = "done_LR" if resL is not None else "done_R"
                print(f"[test] 右手窗口:右侧活动 {aR:.0f} vs 左侧 {aL:.0f} → {resR[0]}")

            # ---- 绘制 ----
            banner_color = COLOR_TEXT
            if state == "idle":
                banner_text = "L=晃动左手测试   R=晃动右手测试   Q=放弃退出"
            elif state in ("test_L", "test_R"):
                banner_text = (f"请反复晃动【{'左手' if state == 'test_L' else '右手'}】… "
                               f"剩余 {max(0.0, args.seconds - (now - t0)):.0f} s")
                banner_color = (255, 210, 80)
            elif state == "done_L":
                banner_text = f"左手结果:{resL[0]}   R=测右手   L=重测"
            elif state == "done_R":
                banner_text = f"右手结果:{resR[0]}   L=重测左手   R=重测"
            else:  # done_LR
                ok = bool(resL and resL[1] is COLOR_OK
                          and resR and resR[1] is COLOR_OK)
                banner_text = ("两侧均对应 ✓ — 肉眼核对波形后按 Y 确认退出"
                               if ok else
                               "存在疑点 ✗ — 核对波形;接反则交换串口配置后重测;"
                               "Y=坚持确认 / Q=放弃")
                banner_color = COLOR_OK if ok else COLOR_BAD
            banner(banner_text, banner_color)

            tL = f"左手 {sides['left']['label']}   上面板"
            tR = f"右手 {sides['right']['label']}   下面板"
            if state == "test_L":
                tL = "◀ 晃动中 — 应见此面板起伏 ▶ " + tL
            if state == "test_R":
                tR = "◀ 晃动中 — 应见此面板起伏 ▶ " + tR
            draw_panel(screen, pygame.Rect(0, 0, PANEL_W, PANEL_H),
                       sides["left"]["rec"], tL, COLOR_LEFT, peaks, font, font_s)
            draw_panel(screen, pygame.Rect(0, PANEL_H + 20, PANEL_W, PANEL_H),
                       sides["right"]["rec"], tR, COLOR_RIGHT, peaks, font, font_s)

            if resL is not None:
                screen.blit(font.render(f"左手测试: {resL[0]}  ({resL[2]:.0f} vs {resL[3]:.0f})",
                                         True, resL[1]), (16, PANEL_H * 2 + 80))
            if resR is not None:
                screen.blit(font.render(f"右手测试: {resR[0]}  ({resR[3]:.0f} vs {resR[2]:.0f})",
                                         True, resR[1]), (480, PANEL_H * 2 + 80))

            pygame.display.flip()
            clock.tick(60)

        for s in sides.values():
            s["rec"]._close()
        pygame.quit()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
