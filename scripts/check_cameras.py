#!/usr/bin/env python3
"""相机体检 —— 枚举并实时显示所有相机画面,不经过 recorder。

    python scripts/check_cameras.py                 # 枚举并开窗显示所有相机
    python scripts/check_cameras.py --list          # 只列出,不开画面
    python scripts/check_cameras.py --seconds 15    # 显示 15 秒后自动退出
    python scripts/check_cameras.py --idx 2         # 只看 USB 索引 2

三类相机分别探测,与 recorder 的打开方式保持一致:

  * OpenCV USB 相机 —— ``VideoCapture(idx, CAP_DSHOW)``,索引与
    configs/recorders.yaml 里的 ``idx`` 一一对应(3=左腕 2=右腕 1=第三视角)
  * RealSense —— ``pyrealsense2`` 枚举全部串号
  * DepthAI(OAK) —— ``depthai`` 枚举 USB 设备

显示时每个相机一个窗口,左上角标注来源与分辨率。按 q / Esc 退出;
开不开画面对采集程序毫无影响,放心开着。

无法导入的 SDK 会标注「未安装」——那类相机在此机器上本来就录不了。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.recorders.camera.opencv_camera_recorder import (  # noqa: E402
    preferred_backend,
)


@dataclass
class Source:
    """一个打开成功的相机流。"""
    kind: str          # opencv / realsense / depthai
    label: str
    res: tuple[int, int] | None = None
    fps: float = 0.0
    handle: object = None
    pipeline: object = None
    q: object = None
    extra: dict = field(default_factory=dict)


# =============================================================================
# 探测
# =============================================================================

def _probe_opencv(idx: int, cv2) -> Source | None:
    cap = cv2.VideoCapture(idx, preferred_backend(cv2))
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()          # 后端可能虚报 isOpened,读到一帧才算数
    if not ok:
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return Source("opencv", f"USB idx={idx}", (w, h), fps, handle=cap)


def _probe_realsense(rs, ctx) -> list[Source]:
    out = []
    try:
        devices = ctx.query_devices()
    except Exception:
        devices = []
    for d in devices:
        try:
            sn = d.get_info(rs.camera_info.serial_number)
            pipe = rs.pipeline(ctx)
            cfg = rs.config()
            cfg.enable_device(sn)
            profile = pipe.start(cfg)
            s = profile.get_stream(rs.stream.color)
            intr = s.as_video_stream_profile().get_intrinsics()
            fps = float(s.as_video_stream_profile().fps())
            out.append(Source("realsense", f"RealSense {sn}",
                              (intr.width, intr.height), fps,
                              pipeline=pipe))
        except Exception as exc:
            print(f"  [realsense] {d} 打开失败: {exc}")
    return out


def _probe_depthai(dai) -> list[Source]:
    out = []
    try:
        infos = dai.Device.getAllAvailableDevices()
    except Exception:
        infos = []
    for info in infos:
        try:
            name = str(info.getMxId() or info.name)
            pipe = dai.Pipeline()
            cam = pipe.createColorCamera()
            xout = pipe.createXLinkOut()
            xout.setStreamName("preview")
            cam.preview.link(xout.input)
            device = dai.Device(pipe, info)
            q = device.getOutputQueue("preview", 4, False)
            frame = q.get()
            img = frame.getCvFrame()
            out.append(Source("depthai", f"DepthAI {name}",
                              (img.shape[1], img.shape[0]), 30.0,
                              device=device, q=q))
        except Exception as exc:
            print(f"  [depthai] {info} 打开失败: {exc}")
    return out


def probe_all(max_usb: int) -> list[Source]:
    """三类相机全扫一遍,返回所有打开成功的流。"""
    found: list[Source] = []
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is None:
        print("[opencv] 未安装 (pip install opencv-python)")
    else:
        for idx in range(max_usb):
            s = _probe_opencv(idx, cv2)
            if s:
                found.append(s)

    try:
        import pyrealsense2 as rs
        found += _probe_realsense(rs, rs.context())
    except Exception:
        print("[realsense] SDK 未安装")

    try:
        import depthai as dai
        found += _probe_depthai(dai)
    except Exception:
        print("[depthai] SDK 未安装")

    return found


# =============================================================================
# 显示
# =============================================================================

def _read(src: Source, cv2) -> tuple[bool, object]:
    if src.kind == "opencv":
        return src.handle.read()
    if src.kind == "realsense":
        import pyrealsense2 as rs
        frames = src.pipeline.wait_for_frames(timeout_ms=1000)
        f = frames.get_color_frame()
        if not f:
            return False, None
        import numpy as np
        return True, np.asanyarray(f.get_data())
    if src.kind == "depthai":
        try:
            frame = src.q.get()
        except Exception:
            return False, None
        return True, frame.getCvFrame()
    return False, None


def show(sources: list[Source], seconds: float, cv2) -> None:
    if not sources:
        print("\n没有找到任何相机。")
        return
    print(f"\n开窗显示 {len(sources)} 路画面 — q / Esc 退出" +
          (f"(约 {seconds:g}s 后自动关闭)" if seconds else ""))
    t0 = time.time()
    while True:
        if seconds and time.time() - t0 >= seconds:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        for src in sources:
            ok, img = _read(src, cv2)
            title = f"{src.label}  {src.res[0]}x{src.res[1]}" if src.res \
                else src.label
            if ok:
                cv2.imshow(title, img)
            else:
                print(f"  {src.label}: 读帧失败")
        time.sleep(0.005)
    cv2.destroyAllWindows()


def release(sources: list[Source]) -> None:
    for src in sources:
        try:
            if src.kind == "opencv":
                src.handle.release()
            elif src.kind == "realsense":
                src.pipeline.stop()
            elif src.kind == "depthai":
                src.device.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="只列出相机,不开画面")
    ap.add_argument("--idx", type=int, default=None,
                    help="只探测这一个 USB 索引(不扫 0..N)")
    ap.add_argument("--max-usb", type=int, default=6,
                    help="USB 索引扫描上限(默认 6)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="显示 N 秒后自动退出(0 = 直到按 q)")
    args = ap.parse_args(argv)

    print("正在探测相机 ...")
    if args.idx is not None:
        import cv2
        sources = [s for s in [_probe_opencv(args.idx, cv2)] if s]
        print(f"  idx={args.idx}: " + ("无" if not sources else
              f"{sources[0].res[0]}x{sources[0].res[1]} @ {sources[0].fps:.0f}fps"))
    else:
        sources = probe_all(args.max_usb)
        if sources:
            for s in sources:
                print(f"  {s.kind:<9} {s.label:<22} "
                      f"{s.res[0]}x{s.res[1]} @ {s.fps:.0f}fps")
        else:
            print("  (空)")

    if args.list:
        return 0

    import cv2
    try:
        show(sources, args.seconds, cv2)
    finally:
        release(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
