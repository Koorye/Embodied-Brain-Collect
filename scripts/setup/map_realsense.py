#!/usr/bin/env python3
"""RealSense 指认 —— 枚举全部 RealSense 并全部开窗,人工确认 serial ↔ 实物对应。

    python scripts/setup/map_realsense.py               # 枚举 + 全部开窗,画面烙 serial
    python scripts/setup/map_realsense.py --list        # 只列设备清单,不开画面
    python scripts/setup/map_realsense.py --seconds 20  # 显示 20 秒后自动退出

recorders.yaml 的 realsense_camera 槽位用 **serial** 绑定设备(跨端口、跨重插
稳定),多台同型号也靠它区分;不配 serial = 默认第一台,多台时等于抓阄。本脚本
把枚举到的每台 RealSense 都打开(640x480 彩色流,与 recorder 同款),画面正中
烙上大号 serial 尾号,顶栏显示完整 serial、固件版本和 yaml 已绑槽位 —— 对着
镜头摆一摆/挡一挡,就知道哪台实物是哪个 serial,再抄进 yaml 对应槽位。

打开的只有彩色流(realsense_camera_recorder 采集的也是彩色流);裸 config 会把
depth/IR 全部默认打开抢带宽导致不出帧,这里与 recorder 一致只显式使能 color。

注意:Windows 上 RealSense 的彩色头同时是一个 UVC 相机,librealsense 先拿走后
OpenCV 打到该索引会读不到帧(正常现象);反之若 OpenCV/其他程序先占了 UVC 流,
这里会 start 成功却永远等不到帧 —— 关掉占用它的程序再试。多台 RealSense 同时
开流吃 USB 带宽,窗口偶有掉帧不影响指认;正式采集请分接不同 USB 控制器。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# =============================================================================
# 探测:枚举 → 逐台打开
# =============================================================================

def _enum_devices(rs) -> list[dict]:
    """[{serial, name, fw}];按 serial 排序,多台时窗口顺序稳定。"""
    out = []
    ctx = rs.context()
    for d in ctx.query_devices():
        try:
            out.append({"serial": str(d.get_info(rs.camera_info.serial_number)),
                        "name": str(d.get_info(rs.camera_info.name)),
                        "fw": str(d.get_info(rs.camera_info.firmware_version))})
        except Exception:
            continue
    return sorted(out, key=lambda x: x["serial"])


def _open_all(rs, devices: list[dict]):
    """逐台起彩色流(与 realsense_camera_recorder 同款配置),读首帧确认。"""
    out = []
    for d in devices:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(d["serial"])
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profile = pipe.start(cfg)
            s = profile.get_stream(rs.stream.color).as_video_stream_profile()
            out.append((d, pipe, (s.get_intrinsics().width,
                                  s.get_intrinsics().height), float(s.fps())))
        except Exception as exc:
            print(f"  serial={d['serial']}: 打开失败 — {exc}")
    return out


# =============================================================================
# 与 recorders.yaml 槽位对应
# =============================================================================

def _camera_slots() -> dict:
    """recorders.yaml 里启用的 realsense_camera 槽位 {slot: 配置};读不到返回空。"""
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        return {}
    return {k: v for k, v in cfg.items()
            if v.get("enabled", True) and v.get("kind") == "realsense_camera"}


def _slots_of_serial(slots: dict, serial: str) -> str:
    """绑了这台 serial 的槽位名;a+b = 配重了(错误配置,当场暴露)。"""
    hits = [s for s, c in slots.items()
            if str(c.get("serial") or "") == serial]
    return "+".join(hits)


# =============================================================================
# 展示
# =============================================================================

def _tail(serial: str) -> str:
    return serial[-6:] if len(serial) > 6 else serial


def show(cv2, feeds, seconds: float, slots: dict) -> None:
    """所有 RealSense 拼成一张网格大图(单窗口),每格画面烙 serial 尾号。"""
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
    misses: dict[str, int] = {}
    while True:
        if seconds and time.time() - t0 >= seconds:
            break
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        for i, (d, pipe, res, fps) in enumerate(feeds):
            r, k = divmod(i, cols)
            x0, y0 = k * cell_w, r * cell_h
            try:
                frames = pipe.poll_for_frames()   # 非阻塞:单路挂了不拖死拼屏
            except Exception:
                frames = None
            f = frames.get_color_frame() if frames else None
            img = np.asanyarray(f.get_data()) if f else None
            slot = _slots_of_serial(slots, d["serial"])
            tag = (f"serial={d['serial']}  {d['name']} {d['fw']}"
                   f"  {res[0]}x{res[1]}@{fps:.0f}"
                   + (f"  [{slot}]" if slot else "  [no-slot]"))
            if img is not None:
                misses[d["serial"]] = 0
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
                misses[d["serial"]] = misses.get(d["serial"], 0) + 1
                if misses[d["serial"]] == 60:    # ~1s 全空拍才喊一次
                    print(f"  serial={d['serial']}: 暂无帧(被占用?等热身?)...")
                cv2.putText(canvas, "no frame", (x0 + 16, y0 + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 大号尾号烙在画面左下:完整 serial 太长,尾号足够区分多台
            cv2.putText(canvas, f"...{_tail(d['serial'])}",
                        (x0 + 12, y0 + cell_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.rectangle(canvas, (x0 + 1, y0 + 1),
                          (x0 + cell_w - 2, y0 + 25), (40, 40, 44), -1)
            cv2.putText(canvas, tag[:110], (x0 + 10, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1,
                                             y0 + cell_h - 1), (70, 70, 78), 1)
        cv2.imshow("map_realsense  [q/Esc quit]", canvas)
        time.sleep(0.005)
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="教程见 docs/realsense_camera.md")
    ap.add_argument("--list", action="store_true", help="只列设备清单,不开画面")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="显示 N 秒后自动退出(0 = 直到按 q)")
    args = ap.parse_args(argv)

    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("[x] pyrealsense2 未安装 (pip install pyrealsense2)")
    try:
        import cv2
    except ImportError:
        raise SystemExit("[x] opencv-python 未安装 (pip install opencv-python)")

    print("正在枚举 RealSense ...")
    devices = _enum_devices(rs)
    if not devices:
        print("  (空)—— 确认已接 USB3 口、线材与驱动正常"
              "(可用 realsense-viewer 交叉验证)")
        return 1
    print("正在逐台打开彩色流(640x480@30)...")
    feeds = _open_all(rs, devices)

    slots = _camera_slots()
    print(f"\n{'serial':<20} {'设备名':<26} {'固件':<12} 分辨率@帧率   yaml槽位")
    for d, _, res, fps in feeds:
        print(f"{d['serial']:<20} {d['name'][:26]:<26} {d['fw']:<12} "
              f"{res[0]}x{res[1]}@{fps:<4.0f} "
              f"[{_slots_of_serial(slots, d['serial']) or 'no-slot'}]")
    opened = {d["serial"] for d, _, _, _ in feeds}
    for slot, c in slots.items():
        sn = str(c.get("serial") or "")
        if sn and sn not in opened:
            print(f"  [yaml] {slot} 配置的 serial={sn} 本次没打开"
                  "(没接 / 被占用)")
        elif not sn:
            print(f"  [yaml] {slot} 未配 serial(默认抓第一台)—— "
                  "多台时务必用本脚本指认后补上")
    if args.list:
        return 0
    if not feeds:
        print("\n没有任何能出流的 RealSense,无法开窗指认。")
        return 1
    try:
        show(cv2, feeds, args.seconds, slots)
    finally:
        for _, pipe, _, _ in feeds:
            try:
                pipe.stop()
            except Exception:
                pass
    print("\n确认对应关系后,把 serial 抄进 configs/recorders.yaml 对应槽位;"
          "然后 python scripts/check_cameras.py 复核。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
