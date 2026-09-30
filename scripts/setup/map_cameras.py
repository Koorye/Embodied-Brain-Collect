#!/usr/bin/env python3
"""USB 相机指认 —— 枚举全部 OpenCV 相机并全部开窗,人工确认 idx ↔ 实物对应。

    python scripts/setup/map_cameras.py               # 枚举 + 全部开窗,画面烙大字 idx=N
    python scripts/setup/map_cameras.py --list        # 只列相机清单,不开画面
    python scripts/setup/map_cameras.py --seconds 20  # 显示 20 秒后自动退出
    python scripts/setup/map_cameras.py --backend msmf   # 换枚举/打开后端(排查用)

为什么需要它:OpenCV 相机只有数字索引没有身份 —— idx 是枚举顺序,换 USB 口、
换个顺序插、系统更新都可能变;recorders.yaml 里 idx 配错就录错相机。本脚本把
枚举到的每台相机全部打开,画面正中烙上大号 "idx=N",对着镜头摆一摆/挡一挡,
马上知道哪台实物是哪个 idx,再抄进 yaml 对应槽位(窗口顶栏同时显示设备名、
USB VID:PID 和 yaml 里已绑该 idx 的槽位,重绑了一眼可见)。

枚举用 cv2-enumerate-cameras(pip install cv2-enumerate-cameras),能拿到设备名
与 VID:PID;没装则退化为盲扫 idx 0..9(无设备名)。默认用与采集 recorder 相同的
后端枚举与打开(Windows=DSHOW, Linux=V4L2,见 opencv_camera_recorder.
preferred_backend),保证这里的 idx 与采集时的 idx 同一套;--backend 可换。

注意:RealSense 若在线,其彩色头在 Windows 上就是一个可枚举的 UVC 相机,可能
占掉一个 idx 甚至被本脚本打开 —— 录 D455f 请走 realsense_camera 槽位按 serial
绑定(scripts/setup/map_realsense.py 指认),不要用 idx 录 RealSense。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from embodied_brain_collect.recorders.camera.opencv_camera_recorder import (  # noqa: E402
    preferred_backend,
)

_BLIND_SCAN = 10          # 没装 cv2-enumerate-cameras 时盲扫 0..9
#: --backend 名称 → cv2 能力常量(名字小写;缺省用 preferred_backend)
_BACKEND_OF = {"msmf": "CAP_MSMF", "dshow": "CAP_DSHOW",
               "v4l2": "CAP_V4L2", "gstreamer": "CAP_GSTREAMER",
               "avfoundation": "CAP_AVFOUNDATION", "any": "CAP_ANY"}


# =============================================================================
# 探测:枚举 → 逐台打开
# =============================================================================

def _pick_backend(cv2, name: str | None) -> int:
    """--backend 指定则用之,否则与采集 recorder 同款。"""
    if name:
        const = _BACKEND_OF.get(name.lower())
        if const is None or not hasattr(cv2, const):
            raise SystemExit(f"[x] 未知/不支持的 --backend {name},"
                             f"可选:{', '.join(_BACKEND_OF)}")
        return getattr(cv2, const)
    return preferred_backend(cv2)


def _enum_cameras(cv2, backend: int) -> list[dict]:
    """[{idx, name, vid, pid, backend}];没装枚举包就盲扫,名字留空。"""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        print(f"[i] 未安装 cv2-enumerate-cameras,退化为盲扫 idx 0..{_BLIND_SCAN - 1}"
              "(无设备名;pip install cv2-enumerate-cameras 可看名字)")
        return [{"idx": i, "name": "", "vid": 0, "pid": 0, "backend": backend}
                for i in range(_BLIND_SCAN)]
    cams = [{"idx": c.index, "name": str(c.name or ""),
             "vid": int(c.vid or 0), "pid": int(c.pid or 0),
             "backend": int(c.backend)}
            for c in enumerate_cameras(backend)]
    if not cams:
        print("[i] 没枚举到任何相机 —— 确认已接好/装好驱动;"
              "或换 --backend 再试(Windows 还可试 msmf)")
    return cams


def _open_all(cv2, cams: list[dict]):
    """逐台 VideoCapture,读到首帧才算数;返回 [(cam, cap, 首帧)]。"""
    out = []
    for c in cams:
        cap = cv2.VideoCapture(c["idx"], c["backend"])
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            cap.release()
            print(f"  idx={c['idx']}: 打不开或读不到帧 — 没接 / 被占用 / "
                  "被专用 SDK(RealSense 彩色头等)拿走,跳过")
            continue
        out.append((c, cap, frame))
    return out


# =============================================================================
# 与 recorders.yaml 槽位对应
# =============================================================================

def _camera_slots() -> dict:
    """recorders.yaml 里启用的 opencv_camera 槽位 {slot: 配置};读不到返回空。"""
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        return {}
    return {k: v for k, v in cfg.items()
            if v.get("enabled", True) and v.get("kind") == "opencv_camera"}


def _slots_of_idx(slots: dict, idx: int) -> str:
    """绑了这个 idx 的槽位名;a+b = 配重了(错误配置,当场暴露);'' = 没绑。"""
    hits = [s for s, c in slots.items() if int(c.get("idx", -1)) == idx]
    return "+".join(hits)


# =============================================================================
# 展示
# =============================================================================

def _tag_of(c: dict, res: tuple[int, int], fps: float, slot: str) -> str:
    ident = f" ({c['vid']:04X}:{c['pid']:04X})" if c["vid"] or c["pid"] else ""
    return (f"idx={c['idx']}  {c['name']}{ident}"
            f"  {res[0]}x{res[1]}@{fps:.0f}"
            + (f"  [{slot}]" if slot else "  [no-slot]"))


def show(cv2, feeds, seconds: float, slots: dict) -> None:
    """所有相机拼成一张网格大图(单窗口),每格画面烙大号 idx=N。"""
    import math
    import numpy as np
    n = len(feeds)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cell_w, cell_h = min(640, 1600 // cols), min(480, 900 // rows)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    print(f"\n拼屏显示 {n} 路(单窗口)— 对着镜头摆一摆,看哪个画面在动即哪台;"
          "q / Esc 退出" + (f"(约 {seconds:g}s 后自动关闭)" if seconds else ""))
    t0 = time.time()
    while True:
        if seconds and time.time() - t0 >= seconds:
            break
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        for i, (c, cap, _) in enumerate(feeds):
            r, k = divmod(i, cols)
            x0, y0 = k * cell_w, r * cell_h
            ok, img = cap.read()
            res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            slot = _slots_of_idx(slots, c["idx"])
            tag = _tag_of(c, res, fps, slot)
            if ok:
                h, w = img.shape[:2]
                scale = min(cell_w / w, (cell_h - 26) / h)
                img = cv2.resize(img, (max(1, int(w * scale)),
                                       max(1, int(h * scale))))
                hh, ww = img.shape[:2]
                canvas[y0 + 26 + (cell_h - 26 - hh) // 2:
                       y0 + 26 + (cell_h - 26 - hh) // 2 + hh,
                       x0 + (cell_w - ww) // 2:
                       x0 + (cell_w - ww) // 2 + ww] = img
            else:
                cv2.putText(canvas, "no frame", (x0 + 16, y0 + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 大号 idx 烙在画面中心:录屏/拍照/肉眼都一眼可辨
            cv2.putText(canvas, f"idx={c['idx']}",
                        (x0 + 12, y0 + cell_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.rectangle(canvas, (x0 + 1, y0 + 1),
                          (x0 + cell_w - 2, y0 + 25), (40, 40, 44), -1)
            cv2.putText(canvas, tag, (x0 + 10, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1,
                                             y0 + cell_h - 1), (70, 70, 78), 1)
        cv2.imshow("map_cameras  [q/Esc quit]", canvas)
        time.sleep(0.005)
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="教程见 docs/opencv_camera.md")
    ap.add_argument("--list", action="store_true",
                    help="只列相机清单,不开画面")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="显示 N 秒后自动退出(0 = 直到按 q)")
    ap.add_argument("--backend", default=None,
                    help="枚举/打开后端(msmf/dshow/v4l2/gstreamer/"
                         "avfoundation/any);缺省与采集 recorder 同款")
    args = ap.parse_args(argv)

    try:
        import cv2
    except ImportError:
        raise SystemExit("[x] opencv-python 未安装 (pip install opencv-python)")
    backend = _pick_backend(cv2, args.backend)
    print(f"正在枚举相机...")
    cams = _enum_cameras(cv2, backend)
    if not cams:
        return 1
    print("正在逐台打开(读到首帧才算数)...")
    feeds = _open_all(cv2, cams)

    slots = _camera_slots()
    print(f"\n{'idx':>4}  {'设备名':<28} {'VID:PID':<10} 分辨率@帧率      yaml槽位")
    for c, cap, _ in feeds:
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ident = f"{c['vid']:04X}:{c['pid']:04X}" if c["vid"] or c["pid"] else "-"
        print(f"{c['idx']:>4}  {c['name'][:28]:<28} {ident:<10} "
              f"{res[0]}x{res[1]}@{fps:<5.0f} [{_slots_of_idx(slots, c['idx']) or 'no-slot'}]")
    opened = {c["idx"] for c, _, _ in feeds}
    for slot, c in slots.items():
        if int(c.get("idx", -1)) not in opened:
            print(f"  [yaml] {slot} 配置的 idx={c.get('idx')} 本次没打开"
                  "(没接 / 被占用 / idx 漂移了 —— 重新指认)")
    if args.list:
        return 0
    if not feeds:
        print("\n没有任何能出图的相机,无法开窗指认。")
        return 1
    try:
        show(cv2, feeds, args.seconds, slots)
    finally:
        for _, cap, _ in feeds:
            cap.release()
    print("\n确认对应关系后,把 idx 抄进 configs/recorders.yaml 对应槽位,"
          "并在注释里记下 USB 口位;然后 python scripts/check_cameras.py 复核。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
