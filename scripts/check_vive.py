#!/usr/bin/env python3
"""Vive tracker 体检 —— 实时显示每台 tracker 的位置与轨迹,核对左右手对齐。

    python scripts/check_vive.py                 # 开窗实时显示
    python scripts/check_vive.py --list          # 只列出当前连接的 tracker
    python scripts/check_vive.py --seconds 30    # 30 秒后自动退出
    python scripts/check_vive.py --trail 400     # 加长轨迹
    python scripts/check_vive.py --no-yaml       # 不读 role_serial_map,全按序列号显示

对齐核对方法:依次拿起每台 tracker 走一小圈,看窗口里**哪个标签的轨迹跟着动**
—— 标签就是 recorders.yaml ``role_serial_map`` 里绑定的角色名(left_hand 等)。
标签不动 / 动的是 "unbound" 那台,说明 yaml 绑错了或没绑,当场暴露。
未进 yaml 的设备标 unbound;yaml 里绑了但没连上的,红字列在顶部。

左图 = 俯视(SteamVR X-Z,上=场景深处);右图 = 侧视(X-Y,上=上)。
带 0.5m 网格与等比例缩放,移动的几何关系不失真。运行中开关 tracker 都能
自动跟上(约 2s 重扫一次)。q / Esc 退出;只读位姿,对采集程序毫无影响。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.recorders.position.openvr_position_recorder import (  # noqa: E402
    _init_openvr,
    _select_devices,
)

# tracker 开机/关机会改变 OpenVR 设备索引,轨迹按序列号记才跨重扫存活
_PALETTE = [(80, 255, 80), (0, 165, 255), (255, 80, 255), (255, 255, 80),
            (80, 200, 255), (80, 80, 255), (255, 150, 150), (200, 200, 200)]
_W, _H = 1280, 720          # 窗口
_HEAD = 64                  # 顶部说明/警示条
_FOOT = 34                  # 底部状态条
_RESCAN_S = 2.0             # 设备重扫周期(开机新 tracker 自动加入)


# =============================================================================
# 设备与角色
# =============================================================================

def _role_map_from_yaml() -> dict[str, str]:
    """recorders.yaml position 槽位的 {标识: 角色};读不到返回空。

    标识可以是 OpenVR 序列号(61-BH…)或 VIVE Hub 角色(vive_tracker_*)。
    """
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
        pos = next(v for k, v in cfg.items() if k.startswith("position"))
        return {str(ident): str(role)
                for role, ident in (pos.get("role_serial_map") or {}).items()}
    except Exception:
        return {}


def _label(dev: dict, id_role: dict[str, str]) -> str:
    """设备标签:yaml 绑定的角色名优先(serial 或 vive_role 命中皆可),
    否则 unbound+serial。"""
    role = (id_role.get(dev["serial"])
            or id_role.get(dev.get("vive_role") or ""))
    return role if role else f"unbound:{dev['serial']}"


def list_devices(vr, id_role: dict[str, str]) -> list[dict]:
    devices = _select_devices(vr, {"tracker"})
    for d in devices:
        d["label"] = _label(d, id_role)
    return devices


# =============================================================================
# 位姿采样
# =============================================================================

def sample_poses(vr, devices: list[dict]) -> dict[str, tuple | None]:
    """{serial: (x, y, z) | None};None = 已连接但位姿无效。"""
    import openvr
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    out: dict[str, tuple | None] = {}
    for d in devices:
        pose = poses[d["index"]]
        if not pose.bPoseIsValid:
            out[d["serial"]] = None
            continue
        m = pose.mDeviceToAbsoluteTracking
        out[d["serial"]] = (m[0][3], m[1][3], m[2][3])
    return out


# =============================================================================
# 绘制
# =============================================================================

class _Panel:
    """一个视图平面:world(axis_h, axis_v) -> 画布矩形,等比例自适应。"""

    def __init__(self, canvas, x0, y0, w, h, title, axis_h, axis_v):
        self.cv, self.x0, self.y0, self.w, self.h = canvas, x0, y0, w, h
        self.title, self.axis_h, self.axis_v = title, axis_h, axis_v
        self.scale, self.cx, self.cy = 100.0, 0.0, 0.0   # px/m 与世界中心

    def fit(self, pts: list[tuple]) -> None:
        """按全部轨迹点自适应(等比例,留 15% 边距;单点时给 1m 视野)。"""
        if not pts:
            return
        hs = [p[self.axis_h] for p in pts]
        vs = [p[self.axis_v] for p in pts]
        lo_h, hi_h = min(hs) - 0.3, max(hs) + 0.3
        lo_v, hi_v = min(vs) - 0.3, max(vs) + 0.3
        if hi_h - lo_h < 0.2:
            lo_h, hi_h = lo_h - 1.0, hi_h + 1.0
        if hi_v - lo_v < 0.2:
            lo_v, hi_v = lo_v - 1.0, hi_v + 1.0
        self.scale = min(self.w / (hi_h - lo_h), self.h / (hi_v - lo_v))
        self.cx, self.cy = (lo_h + hi_h) / 2, (lo_v + hi_v) / 2

    def px(self, p: tuple) -> tuple[int, int]:
        u = self.x0 + self.w / 2 + (p[self.axis_h] - self.cx) * self.scale
        v = self.y0 + self.h / 2 - (p[self.axis_v] - self.cy) * self.scale
        return int(u), int(v)

    def draw_frame(self) -> None:
        cv2 = _cv2()
        # 只清自己的矩形:两幅视图共用同一画布,整幅清会把先画的抹掉
        self.cv[self.y0:self.y0 + self.h,
                self.x0:self.x0 + self.w] = 0
        # 0.5m 网格,锚在世界坐标整数倍上(等比例,缩放时格距随之变化)
        step = 0.5
        gx = step * self.scale
        if 8 <= gx <= max(self.w, self.h):
            for k in range(-40, 41):            # ±20m,远超任何房间
                u = (self.x0 + self.w / 2
                     + (k * step - self.cx) * self.scale)
                if self.x0 <= u <= self.x0 + self.w:
                    cv2.line(self.cv, (int(u), self.y0),
                             (int(u), self.y0 + self.h), (38, 38, 38), 1)
                v = (self.y0 + self.h / 2
                     - (k * step - self.cy) * self.scale)
                if self.y0 <= v <= self.y0 + self.h:
                    cv2.line(self.cv, (self.x0, int(v)),
                             (self.x0 + self.w, int(v)), (38, 38, 38), 1)
        # 原点十字(SteamVR play space 零点)
        ou, ov = self.px((0.0, 0.0, 0.0))
        cv2.line(self.cv, (ou - 14, ov), (ou + 14, ov), (90, 90, 90), 1)
        cv2.line(self.cv, (ou, ov - 14), (ou, ov + 14), (90, 90, 90), 1)
        cv2.putText(self.cv, self.title, (self.x0 + 8, self.y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)

    def polyline(self, pts: list[tuple], color) -> None:
        cv2 = _cv2()
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(self.cv, self.px(a), self.px(b), color, 1, cv2.LINE_AA)

    def dot(self, p: tuple, color, label: str, sub: str, valid: bool) -> None:
        cv2 = _cv2()
        u, v = self.px(p)
        cv2.circle(self.cv, (u, v), 7 if valid else 5,
                   color if valid else (110, 110, 110), -1, cv2.LINE_AA)
        cv2.putText(self.cv, label, (u + 10, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color if valid else (110, 110, 110), 1, cv2.LINE_AA)
        if sub:
            cv2.putText(self.cv, sub, (u + 10, v + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1,
                        cv2.LINE_AA)


def _cv2():
    import cv2
    return cv2


def draw(services: dict, panels: tuple[_Panel, _Panel], warn: str,
         rate: float) -> None:
    """重画整窗:顶部警示/说明,两幅视图,底部状态。"""
    cv2 = _cv2()
    canvas = services["canvas"]
    devices, trails, latest = (services["devices"], services["trails"],
                               services["latest"])
    colors = services["colors"]
    canvas[:] = 0
    head = ("move each tracker in turn -> its trail must move under the "
            "matching label   [q/Esc quit]")
    cv2.putText(canvas, head, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    if warn:
        cv2.putText(canvas, warn[:150], (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 80, 255), 1, cv2.LINE_AA)
    for p in panels:
        p.draw_frame()
    for d in devices:
        color = colors.setdefault(
            d["serial"], _PALETTE[len(colors) % len(_PALETTE)])
        trail = trails.get(d["serial"])
        cur = latest.get(d["serial"])
        for p in panels:
            if trail:
                p.polyline(list(trail), color)
        if cur is None:
            continue                     # 无位姿:底部状态行里说明
        for p in panels:
            p.dot(cur, color, d["label"], d["serial"], valid=True)
    # 底部状态:每台一行压缩成一条 —— role serial (x, y, z) m
    parts = []
    for d in devices:
        cur = latest.get(d["serial"])
        pos = (f"({cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f})" if cur
               else "no-pose")
        parts.append(f"{d['label']}:{pos}")
    cv2.rectangle(canvas, (0, _H - _FOOT), (_W, _H), (30, 30, 34), -1)
    cv2.putText(canvas, f"{rate:.0f} fps  " + "   ".join(parts),
                (12, _H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)


# =============================================================================
# 主循环
# =============================================================================

def show(vr, seconds: float, trail_len: int,
         serial_role: dict[str, str]) -> None:
    cv2 = _cv2()
    cv2.namedWindow("check_vive  [q/Esc quit]", cv2.WINDOW_AUTOSIZE)
    panels = (_Panel(None, 0, _HEAD, _W // 2, _H - _HEAD - _FOOT,
                     "top view  X / Z (0.5m grid)", 0, 2),
              _Panel(None, _W // 2, _HEAD, _W // 2, _H - _HEAD - _FOOT,
                     "side view  X / Y (0.5m grid)", 0, 1))
    trails: dict[str, deque] = {}
    latest: dict[str, tuple | None] = {}
    colors: dict[str, tuple] = {}       # serial -> 颜色,首次出现时分配并固定
    rate = 0.0                          # (重扫改变设备顺序也不跳色)
    services = {"canvas": np.zeros((_H, _W, 3), dtype=np.uint8),
                "devices": [], "trails": trails, "latest": latest,
                "colors": colors}
    devices = services["devices"]
    last_scan, reported = 0.0, ""
    t0, n_frames, t_fps = time.time(), 0, time.time()
    while True:
        now = time.time()
        if seconds and now - t0 >= seconds:
            break
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        if now - last_scan >= _RESCAN_S:          # 周期重扫:开关机自动跟上
            last_scan = now
            devices[:] = list_devices(vr, serial_role)
            gone = [s for s in trails if s not in {d["serial"] for d in devices}]
            for s in gone:                        # 离线设备的轨迹一并清掉,
                trails.pop(s, None)               # 不再把视野边界撑到旧位置
                latest.pop(s, None)
            warn_bits = []
            for ident, role in serial_role.items():
                if not any(d["serial"] == ident or d.get("vive_role") == ident
                           for d in devices):
                    warn_bits.append(f"yaml role not connected: "
                                     f"{role}={ident}")
            warn = "; ".join(warn_bits)
            if warn != reported:
                print(("[vive] " + warn) if warn else "[vive] tracker 全部在线")
                reported = warn
        poses = sample_poses(vr, devices)          # 采样 + 轨迹
        for d in devices:
            p = poses.get(d["serial"])
            latest[d["serial"]] = p
            if p is not None:
                trails.setdefault(d["serial"], deque(maxlen=trail_len)) \
                    .append(p)
        n_frames += 1
        if time.time() - t_fps >= 1.0:
            rate = n_frames / (time.time() - t_fps)
            n_frames, t_fps = 0, time.time()
        all_pts = [pt for q in trails.values() for pt in q]
        for p in panels:
            p.cv = services["canvas"]
            p.fit(all_pts)
        draw(services, panels, reported, rate)
        cv2.imshow("check_vive  [q/Esc quit]", services["canvas"])
        time.sleep(0.005)
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="只列出 tracker,不开窗")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="显示 N 秒后自动退出(0 = 直到按 q)")
    ap.add_argument("--trail", type=int, default=200,
                    help="每台 tracker 的轨迹点数(默认 200)")
    ap.add_argument("--no-yaml", action="store_true",
                    help="忽略 role_serial_map,全部按序列号显示")
    args = ap.parse_args(argv)

    print("正在连接 SteamVR / OpenVR ...")
    vr = _init_openvr()
    try:
        serial_role = {} if args.no_yaml else _role_map_from_yaml()
        devices = list_devices(vr, serial_role)
        if serial_role:
            print(f"yaml 绑定: {{{', '.join(f'{r}: {s}' for s, r in serial_role.items())}}}")
        if devices:
            for d in devices:
                vr_role = f"  vive_role={d['vive_role']}" if d.get("vive_role") else ""
                print(f"  [{d['index']:02}] {d['label']:<18} "
                      f"serial={d['serial']}  model={d['model']}{vr_role}")
            print("  (注意:序列号以本列表 serial= 为准;VIVE Hub 里的设备 ID"
                  "如 FA61… 不是 OpenVR 序列号,匹配不到)")
        else:
            print("  (没有 tracker 在线 — 确认已开机、VIVE Hub/SteamVR 已连接)")
        for ident, role in serial_role.items():
            if not any(d["serial"] == ident or d.get("vive_role") == ident
                       for d in devices):
                print(f"  [yaml] 绑定的 {role}={ident} 本次未连上")
        if args.list:
            return 0
        if not devices:
            print("先开机至少一台 tracker 再运行本脚本。")
            return 1
        print("\n依次拿起每台 tracker 移动:窗口里哪个标签的轨迹跟着动,"
              "那个角色就是这台 —— 与 role_serial_map 对不上当场暴露。")
        show(vr, args.seconds, args.trail, serial_role)
    finally:
        import openvr
        openvr.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
