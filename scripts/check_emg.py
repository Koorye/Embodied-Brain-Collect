#!/usr/bin/env python3
"""EMG 左右手对应检查 —— 实时显示双手肌电 + IMU 波形,确认 emg_left/emg_right 没接反。

    python scripts/check_emg.py                     # 按记录器配置开窗检查
    python scripts/check_emg.py --list              # 只列出左右槽位配置,不开窗
    python scripts/check_emg.py --left-port COM31 --right-port COM30
    python scripts/check_emg.py --seconds 3         # 每次晃动的采样窗口(秒)

流程(窗口内按键,全程单循环,窗口不冻结):

  1. 打开左右两条 EMG(recorders.yaml 的 emg_left / emg_right 槽位)。
  2. 按 L → 提示反复晃动【左手】数秒;按 R → 晃动【右手】。
     窗口自动统计晃动期间左右两侧的肌电活动量并给出对照结论
     (对应 ✓ / 疑似接反 ✗ / 活动不明显)。
  3. 肉眼核对:晃动时 IMU 波形(上3=加速度计、下3=陀螺仪)起伏最直观,
     8 通道 EMG 应同步出现活动;左手动作 → 上面板,右手 → 下面板。
  4. 按 Y 确认无误退出(退出码 0);按 Q 放弃(退出码 1)。

所有波形都是设备原始值 —— 不做滤波、整流、去重力等任何后处理,
画面上只做同面板跨通道共享的显示幅度自动缩放(否则画不出来)。

面板读数里的「延迟」是最新一帧从串口到达到当前的年龄:正常应 <100 ms,
>300 ms 变红 —— 那是串口缓冲积压(读取跟不上设备流量),不是设备慢。

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
from embodied_brain_collect.recorders.factory import build as build_recorder  # noqa: E402
from embodied_brain_collect.stim.base_stim import _FONT_CANDIDATES, _find_font  # noqa: E402

PANEL_W, PANEL_H = 1600, 300          # 单侧面板尺寸:左 60% EMG / 右 40% IMU
EMG_SPLIT = 0.6                       # 面板内 EMG 区宽度占比,其余给 IMU
WINDOW_SAMPLES = 2000                 # EMG 波形显示的样本数(约 1-2 s)
IMU_WINDOW_S = 2.0                    # IMU 波形显示的时长(秒,按到达时刻取)
SHOW_ROWS = 8                         # 每只手 8 个 EMG 通道
COLOR_LEFT = (80, 220, 120)           # 左手面板 EMG 波形:绿
COLOR_RIGHT = (255, 170, 60)          # 右手面板 EMG 波形:橙
COLOR_ACCEL = (110, 180, 255)         # IMU 加速度计波形:蓝
COLOR_GYRO = (255, 110, 220)          # IMU 陀螺仪波形:品红
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
    # 槽位 yaml 的键就是 Config 字段,整体透传;--left/right-port 只覆盖 port
    params = {k: v for k, v in cfg.items() if k not in ("kind", "name", "enabled")}
    if port is not None:
        params["port"] = port
    rec = build_recorder("weili_emg", params, Path(tmp) / slot)
    try:
        ok = rec._open()
    except Exception as exc:  # noqa: BLE001 — 串口占用/不存在等,给可读报错
        rec._open_error = f"{type(exc).__name__}: {exc}"
        ok = False
    if not ok:
        print(f"[{slot}] 打开失败 — {rec._open_error or '未知原因'}")
        return None
    rec._ser.timeout = 0    # 显示循环里非阻塞读 + 按 in_waiting 抽干,不背 5ms 固定块
    print(f"[{slot}] 已打开 {rec.config.port or '自动探测'} "
          f"@ {rec.config.baud} ({cfg.get('name', slot)})")
    return rec


def emg_view(rec):
    """最近的波形窗口:np.ndarray (n, 8) 或 None(还没数据)。"""
    buf = rec._arr_buf.get("emg_data")
    if not buf:
        return None
    return np.asarray(buf[-WINDOW_SAMPLES:], dtype=np.float64)


def imu_view(rec, window_s: float = IMU_WINDOW_S):
    """最近 window_s 秒的原始 IMU:(accel, gyro),各 (n, 3) 或 None。"""
    ts = rec._frame_ts.get("imu_gyro") or []
    now = time.perf_counter()
    n = sum(1 for t in ts[-8000:] if now - t <= window_s)
    if n < 2:
        return None, None
    out = []
    for key in ("imu_accel", "imu_gyro"):
        buf = rec._arr_buf.get(key)
        out.append(np.asarray(buf[-n:], dtype=np.float64)
                   if buf and len(buf) >= n else None)
    return out[0], out[1]


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


MAX_TRACE_PTS = 384                   # 单条折线点数上限 —— 画线是每帧耗时大头


def _trace(surf, x0: int, width: int, y0: int, row_h: int, seg, peak: float,
           color) -> None:
    """一段原始值铺成一行折线 —— 仅取样 + 幅度归一,不改样本值。

    取点/归一全程 numpy 向量化:逐点 Python 循环每条折线约 2 ms,
    14 条就是 28 ms/帧 —— 正是之前「延迟很严重」的元凶。
    """
    import pygame
    n = len(seg)
    step = max(1, n // min(width, MAX_TRACE_PTS))
    idx = np.arange(0, n, step)
    xs = x0 + idx * width // max(1, n - 1)
    ys = y0 - np.clip(seg[idx] / peak, -0.9, 0.9) * row_h * 0.45
    pts = list(zip(xs.tolist(), ys.astype(int).tolist()))
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, 1)


def _auto_peak(data, peaks: dict, key: str, floor: float) -> float:
    """同面板跨通道共享的显示幅度:自适应峰值带 0.5% 衰减回缩。"""
    peak = max(float(np.abs(data).max()), peaks.get(key, 0.0) * 0.995, floor)
    peaks[key] = peak
    return peak


_TEXT_CACHE = {}


def _text(fnt, s: str, color):
    """按 (字体, 文本, 颜色) 缓存渲染面 —— 静态文字零每帧开销。"""
    key = (id(fnt), s, color)
    sf = _TEXT_CACHE.get(key)
    if sf is None:
        sf = fnt.render(s, True, color)
        _TEXT_CACHE[key] = sf
    return sf


def draw_panel(surf, rect, rec, title: str, color, peaks: dict, font, font_s,
               fps: float = 0.0):
    """一侧面板:标题 + 左半 8 通道 EMG 波形 + 右半 6 路 IMU 波形(全原始值)。"""
    import pygame
    pygame.draw.rect(surf, (28, 28, 32), rect)
    pygame.draw.rect(surf, (70, 70, 78), rect, 1)
    surf.blit(_text(font, title, COLOR_TEXT), (rect.x + 12, rect.y + 8))

    body = pygame.Rect(rect.x, rect.y + 30, rect.width, rect.height - 30)
    w_emg = int(rect.width * EMG_SPLIT)
    emg_rect = pygame.Rect(body.x, body.y, w_emg, body.height)
    imu_rect = pygame.Rect(body.x + w_emg + 12, body.y,
                           body.width - w_emg - 12, body.height)
    pygame.draw.line(surf, (55, 55, 62), (imu_rect.x - 6, body.y),
                     (imu_rect.x - 6, body.bottom))
    rate = _draw_emg(surf, emg_rect, rec, title, color, peaks, font, font_s,
                     fps)
    _draw_imu(surf, imu_rect, rec, title, peaks, font, font_s)
    return rate


def _draw_emg(surf, rect, rec, title: str, color, peaks: dict, font, font_s,
              fps: float):
    """面板左半:8 通道 EMG 原始波形 + 活动量/延迟读数。返回 EMG 帧率。"""
    import pygame
    data = emg_view(rec)
    rate = 0.0
    if data is not None and data.shape[0] >= 2:
        row_h = rect.height // SHOW_ROWS
        peak = _auto_peak(data, peaks, f"{title}|emg", 500.0)
        for ch in range(SHOW_ROWS):
            y0 = rect.y + ch * row_h + row_h // 2
            _trace(surf, rect.x, rect.width, y0, row_h, data[:, ch], peak, color)
        act = float(np.std(data, axis=0).mean())
        rate = _rate(rec)
        age = _data_age(rec, "emg_data")
        col = COLOR_BAD if age > 0.3 else COLOR_HINT
        _blit_readout(
            surf, peaks, f"{title}|emg_ro", rect.x + 2, rect.bottom - 24,
            lambda: (f"EMG 8ch 原始  活动 {act:.0f}  {rate:.0f} 帧/s  "
                     f"延迟 {age * 1000:.0f} ms  界面 {fps:.0f} fps", col),
            font_s)
    else:
        surf.blit(font_s.render("EMG 等待数据流 …", True, COLOR_HINT),
                  (rect.x + 2, rect.y + rect.height // 2))
    return rate


def _draw_imu(surf, rect, rec, title: str, peaks: dict, font, font_s) -> None:
    """面板右半:上 3 行加速度计、下 3 行陀螺仪,原始解码值直接成线。"""
    import pygame
    accel, gyro = imu_view(rec)
    row_h = rect.height // 6
    for k, (data, color, floor, tag) in enumerate(
            ((accel, COLOR_ACCEL, 20.0, "accel"),
             (gyro, COLOR_GYRO, 1.0, "gyro"))):
        if data is None or data.shape[0] < 2:
            continue
        peak = _auto_peak(data, peaks, f"{title}|{tag}", floor)
        for j in range(3):
            y0 = rect.y + (k * 3 + j) * row_h + row_h // 2
            _trace(surf, rect.x, rect.width, y0, row_h, data[:, j], peak, color)
    if accel is None and gyro is None:
        surf.blit(font_s.render("IMU 等待数据流 …", True, COLOR_HINT),
                  (rect.x + 2, rect.y + rect.height // 2))
        return
    a_acc = float(np.std(accel)) if accel is not None else 0.0
    a_gyro = float(np.std(gyro)) if gyro is not None else 0.0
    r_imu = _rate(rec, "imu_gyro")
    age = _data_age(rec, "imu_gyro")
    col = COLOR_BAD if age > 0.3 else COLOR_HINT
    _blit_readout(
        surf, peaks, f"{title}|imu_ro", rect.x + 2, rect.bottom - 24,
        lambda: (f"IMU 原始(上3=加速度计/下3=陀螺仪)  活动 "
                 f"{a_acc:.2f}/{a_gyro:.2f}  {r_imu:.0f} 帧/s  "
                 f"延迟 {age * 1000:.0f} ms", col),
        font_s)


_RATE = {}


def _rate(rec, key: str = "emg_data") -> float:
    """近 2 秒的帧率,顺带证明数据流健康。"""
    ts = rec._frame_ts.get(key) or []
    now = time.perf_counter()
    recent = [t for t in ts[-8000:] if now - t <= 2.0]
    k = (id(rec), key)
    _RATE[k] = len(recent) / 2.0 if len(recent) >= 2 else _RATE.get(k, 0.0)
    return _RATE[k]


def _data_age(rec, key: str) -> float:
    """最新已解析帧的年龄(秒)= 波形真正的新鲜度;积压越大它越大。"""
    ts = rec._frame_ts.get(key)
    return time.perf_counter() - ts[-1] if ts else float("inf")


def _blit_readout(surf, peaks: dict, key: str, x: int, y: int, make_text,
                  font_s) -> None:
    """读数行文字每 0.1 s 才重排一次 —— CJK 渲染贵,不值得每帧做。"""
    now = time.perf_counter()
    sk, tk = f"{key}|surf", f"{key}|t"
    if now - peaks.get(tk, 0.0) >= 0.1 or sk not in peaks:
        text, color = make_text()
        peaks[tk] = now
        peaks[sk] = font_s.render(text, True, color)
    s = peaks.get(sk)
    if s is not None:
        surf.blit(s, (x, y))


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

        # 打开到此刻设备一直在发,先清掉积压的旧帧再进显示循环
        for s in sides.values():
            s["rec"]._ser.reset_input_buffer()
            s["rec"]._raw_buf.clear()
            s["rec"]._last_sn = None

        state = "idle"                    # idle / test_L / done_L / test_R / done_LR
        t0 = n0L = n0R = 0
        resL = resR = None                # (文本, 颜色, a_exp, a_other)
        peaks: dict = {}
        exit_code = 1
        clock = pygame.time.Clock()
        running = True

        def banner(text, color=COLOR_TEXT):
            screen.fill((16, 16, 18))
            screen.blit(_text(font, text, color), (16, PANEL_H * 2 + 40))

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
                rec = sides[key]["rec"]
                for _ in range(4):        # 一帧内把驱动缓冲抽干(上限防极端占用)
                    if rec._ser.in_waiting <= 0:
                        break
                    rec._poll(time.time())

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
            fps = clock.get_fps()
            draw_panel(screen, pygame.Rect(0, 0, PANEL_W, PANEL_H),
                       sides["left"]["rec"], tL, COLOR_LEFT, peaks, font,
                       font_s, fps)
            draw_panel(screen, pygame.Rect(0, PANEL_H + 20, PANEL_W, PANEL_H),
                       sides["right"]["rec"], tR, COLOR_RIGHT, peaks, font,
                       font_s, fps)

            if resL is not None:
                screen.blit(_text(font, f"左手测试: {resL[0]}  ({resL[2]:.0f} vs {resL[3]:.0f})",
                                  resL[1]), (16, PANEL_H * 2 + 80))
            if resR is not None:
                screen.blit(_text(font, f"右手测试: {resR[0]}  ({resR[3]:.0f} vs {resR[2]:.0f})",
                                  resR[1]), (680, PANEL_H * 2 + 80))

            pygame.display.flip()
            clock.tick(60)

        for s in sides.values():
            s["rec"]._close()
        pygame.quit()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
