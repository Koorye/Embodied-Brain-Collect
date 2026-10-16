#!/usr/bin/env python3
"""Vive tracker 体检 —— 实时显示每台 tracker 的 3D 位姿与轨迹,核对左右手对齐。

    python scripts/check_vive.py                 # 开窗实时显示
    python scripts/check_vive.py --list          # 只列出当前连接的 tracker
    python scripts/check_vive.py --seconds 30    # 30 秒后自动退出
    python scripts/check_vive.py --trail 400     # 加长轨迹
    python scripts/check_vive.py --no-yaml       # 不读 role_serial_map,全按序列号显示

对齐核对方法:依次拿起每台 tracker 走一小圈,看窗口里**哪个标签的轨迹跟着动**
—— 标签就是 recorders.yaml ``role_serial_map`` 里绑定的角色名(left_wrist 等)。
标签不动 / 动的是 "unbound" 那台,说明 yaml 绑错了或没绑,当场暴露。
未进 yaml 的设备标 unbound;yaml 里绑了但没连上的,红字列在顶部。

单窗口 3D 视图为**真实透视(无镜像)**:+X 画在屏幕左、+Y 向上、+Z 向屏幕
里(深度)。SteamVR/OpenVR 位姿是右手系(+Y 上,米制),轴向物理朝向由 VIVE
建图决定 —— 本套场地实测:面朝场景时 +X 在操作员左手侧、+Z 指向场景深处,
即屏幕画面与面朝场景所见左右一致(左手边的 tracker 显示在屏幕左侧)。地面网格
(0.5m 起,按场景自适应) + 原点 XYZ 轴(R/G/B);每台 tracker 画彩色轨迹、
到地面的垂线(高度参考)、姿态 gizmo(R/G/B 三段短线 = 设备坐标系 XYZ 轴
旋转到世界后的方向)与标签;底部状态栏显示 (x,y,z|r,p,y)。鼠标左键拖拽
旋转视角,滚轮或 +/- 缩放,r 复位取景(自动包围全部轨迹),q / Esc 退出。

rpy 读数与 position.npz 的 euler_rpy_deg 同一换算(recorder 的
_matrix_to_quat + _quat_to_euler);转动 tracker 时 gizmo 应跟着转 ——
佩戴方向对不对一眼可见。运行中开关 tracker 都能自动跟上(约 2s 重扫一次)。
只读位姿,对采集程序毫无影响。
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
    _matrix_to_quat,
    _quat_to_euler,
    _select_devices,
)

# tracker 开机/关机会改变 OpenVR 设备索引,轨迹按序列号记才跨重扫存活
_PALETTE = [(80, 255, 80), (0, 165, 255), (255, 80, 255), (255, 255, 80),
            (80, 200, 255), (80, 80, 255), (255, 150, 150), (200, 200, 200)]
_W, _H = 1280, 720          # 窗口
_HEAD = 64                  # 顶部说明/警示条
_FOOT = 34                  # 底部状态条
_RESCAN_S = 2.0             # 设备重扫周期(开机新 tracker 自动加入)
_DEFAULT_YAW = 0.7          # 默认/复位视角的偏航(~40°)。yaw=0 时 +Z(场景深处)
                            # 与 +Y(上)在屏幕上都投影成竖直方向,前后/上下分不
                            # 开;偏航后 +Z 带上水平分量,与竖直的 +Y 分开 ~55°


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
    """{serial: (pos, rpy, axes) | None};None = 已连接但位姿无效。

    pos = (x, y, z) m;rpy = (roll, pitch, yaw) 度(与 position.npz 的
    euler_rpy_deg 同一换算);axes = 设备三根轴的世界方向(旋转矩阵的
    三个列向量),供 3D 视图画姿态 gizmo。
    """
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
        pos = (m[0][3], m[1][3], m[2][3])
        rpy = _quat_to_euler(*_matrix_to_quat(m))
        axes = tuple((m[0][j], m[1][j], m[2][j]) for j in range(3))
        out[d["serial"]] = (pos, rpy, axes)
    return out


# =============================================================================
# 3D 视图(真实透视:+X 在屏幕左 = 面朝场景者的左手侧,屏幕上 = +Y,屏幕里 = +Z)
# =============================================================================

def _cv2():
    import cv2
    return cv2


class _View3D:
    """软件 3D 投影:透视小孔模型,深度 = 世界 Z(向屏幕里),+X 画在屏幕左
    (真实透视,非镜像),拖拽 orbit。

    取景默认自动包围全部轨迹点;用户一旦拖拽/缩放(user=True)就只跟随
    中心、不再夺走视角与距离,r 可复位。
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.cv = None                       # 每帧由 draw3d 赋值(整幅画布)
        self.yaw, self.pitch = _DEFAULT_YAW, 0.62
        self.dist, self.radius = 5.0, 1.0
        self.center = np.zeros(3)
        self.user = False

    @property
    def f(self) -> float:                    # 焦距(px):视口短边 × 1.2
        return 1.2 * min(self.w, self.h)

    # ---- 交互 ----

    def orbit(self, dx: int, dy: int) -> None:
        self.user = True
        self.yaw -= dx * 0.008   # u 已取负(镜像回真实透视),反转 dx 保住拖拽手感
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -0.35, 1.45))

    def zoom(self, k: float) -> None:
        self.user = True
        self.dist = float(np.clip(self.dist * k, 0.4, 80.0))

    def reset(self) -> None:
        self.user = False
        self.yaw, self.pitch = _DEFAULT_YAW, 0.62

    # ---- 取景与投影 ----

    def fit(self, pts: list[tuple]) -> None:
        """按全部轨迹/当前位置取景(包围球半径留 ~40% 边)。"""
        if not pts:
            return
        arr = np.asarray(pts, dtype=np.float64)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        self.center = (lo + hi) / 2
        self.radius = max(0.8, float(np.linalg.norm(hi - lo) / 2))
        if not self.user:
            self.dist = 2.9 * self.radius

    def proj(self, p) -> tuple[float, float, float] | None:
        """世界点 → (u, v, depth);相机后方(depth<0.1)返回 None。"""
        v = np.asarray(p, dtype=np.float64) - self.center
        ca, sa = np.cos(self.yaw), np.sin(self.yaw)
        cb, sb = np.cos(self.pitch), np.sin(self.pitch)
        x1 = v[0] * ca - v[2] * sa            # 偏航:绕 Y(上)转
        z1 = v[0] * sa + v[2] * ca
        y2 = v[1] * cb + z1 * sb              # 俯仰:绕屏幕水平轴转
        z2 = -v[1] * sb + z1 * cb
        d = z2 + self.dist                    # 真实透视:+Z 远离观察者
        if d < 0.1:
            return None
        # u 取 -x1:相机面朝 +Z、+Y 上时右手系的 +X 落在画面左侧 —— 画面与
        # 面朝场景的物理左右一致(实测场地 +X 在操作员左手侧),不再镜像。
        return (self.x0 + self.w / 2 - self.f * x1 / d,
                self.y0 + self.h / 2 - self.f * y2 / d, d)

    # ---- 图元(全部经 _seg,深度越远越暗) ----

    def _dim(self, color, d: float):
        t = float(np.clip((d - (self.dist - self.radius))
                          / max(2 * self.radius, 1e-6), 0.0, 1.0))
        return tuple(int(c * (1.0 - 0.55 * t)) for c in color)

    def _seg(self, a, b, color, thickness: int = 1) -> None:
        cv2 = _cv2()
        pa, pb = self.proj(a), self.proj(b)
        if pa is None or pb is None:
            return
        cv2.line(self.cv, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 self._dim(color, (pa[2] + pb[2]) / 2), thickness, cv2.LINE_AA)

    def _step(self) -> float:                # 网格步长随场景缩放
        return (0.5 if self.radius <= 3 else
                1.0 if self.radius <= 7 else 2.0)

    def grid(self) -> None:
        half = np.ceil(self.radius * 1.15 / self._step()) * self._step()
        cx, cz = self.center[0], self.center[2]
        for k in np.arange(-half, half + self._step() / 2, self._step()):
            self._seg((cx + k, 0.0, cz - half), (cx + k, 0.0, cz + half),
                      (46, 46, 46))
            self._seg((cx - half, 0.0, cz + k), (cx + half, 0.0, cz + k),
                      (46, 46, 46))

    def axes(self) -> None:
        """原点世界轴 X/Y/Z = R/G/B;真实透视:X 红轴在屏幕左,Z 蓝轴指向屏幕里。"""
        cv2 = _cv2()
        L = 2 * self._step()
        for vec, color, name in (((1, 0, 0), (60, 60, 255), "X"),
                                 ((0, 1, 0), (60, 220, 60), "Y"),
                                 ((0, 0, 1), (255, 90, 40), "Z")):
            tip = np.asarray(vec) * L
            self._seg((0, 0, 0), tip, color, 2)
            pt = self.proj(tip)
            if pt is not None:
                cv2.putText(self.cv, name, (int(pt[0]) + 5, int(pt[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)

    def dropline(self, p, color) -> None:
        """点到地面的垂线 + 落点小圆:高度一眼可读。"""
        cv2 = _cv2()
        if abs(p[1]) < 0.03:
            return
        foot = (p[0], 0.0, p[2])
        self._seg(p, foot, color, 1)
        pf = self.proj(foot)
        if pf is not None:
            cv2.circle(self.cv, (int(pf[0]), int(pf[1])), 2,
                       self._dim(color, pf[2]), 1, cv2.LINE_AA)

    def trail(self, pts, color) -> None:
        for a, b in zip(pts[:-1], pts[1:]):
            self._seg(a, b, color)

    def gizmo(self, pos, axes, length: float = 0.18) -> None:
        """姿态:设备 XYZ 轴(R/G/B)旋转到世界后的三段短线。"""
        for vec, color in zip(axes, ((0, 0, 255), (0, 200, 0), (255, 80, 0))):
            tip = (pos[0] + vec[0] * length, pos[1] + vec[1] * length,
                   pos[2] + vec[2] * length)
            self._seg(pos, tip, color, 2)

    def dot(self, p, color, label: str, sub: str) -> None:
        cv2 = _cv2()
        pt = self.proj(p)
        if pt is None:
            return
        u, v, d = int(pt[0]), int(pt[1]), pt[2]
        c = self._dim(color, d)
        cv2.circle(self.cv, (u, v), 6, c, -1, cv2.LINE_AA)
        cv2.circle(self.cv, (u, v), 7, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(self.cv, label, (u + 10, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
        if sub:
            cv2.putText(self.cv, sub, (u + 10, v + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1,
                        cv2.LINE_AA)


def draw3d(canvas, view: _View3D, devices: list[dict], trails: dict,
           latest: dict, colors: dict, warn: str, rate: float) -> None:
    """整帧重画:顶部说明/警示,3D 场景,底部状态。"""
    cv2 = _cv2()
    canvas[:] = 0
    head = ("3D real-perspective: +X screen-left, +Y up, +Z into screen   "
            "drag=orbit  wheel/+/-=zoom  r=reset  [q/Esc quit]")
    cv2.putText(canvas, head, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    if warn:
        cv2.putText(canvas, warn[:150], (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 80, 255), 1, cv2.LINE_AA)

    view.cv = canvas
    pts = [pt for q in trails.values() for pt in q] \
        + [c[0] for c in latest.values() if c]
    view.fit(pts)
    view.grid()
    view.axes()
    for d in devices:
        color = colors.setdefault(
            d["serial"], _PALETTE[len(colors) % len(_PALETTE)])
        tr = trails.get(d["serial"])
        if tr:
            view.trail(list(tr), color)

    # 点与标签按深度远→近画,近处不被远处遮挡
    items = []
    for d in devices:
        cur = latest.get(d["serial"])
        if cur is not None:
            items.append((cur, d, colors[d["serial"]]))
    def _depth(it):
        pt = view.proj(it[0][0])
        return -pt[2] if pt else 1e9
    items.sort(key=_depth)
    for cur, d, color in items:
        view.dropline(cur[0], color)
        view.gizmo(cur[0], cur[2])
        view.dot(cur[0], color, d["label"], d["serial"])

    # 底部状态:每台压缩成一条 —— role (x,y,z| r,p,y) m/deg
    parts = []
    for d in devices:
        cur = latest.get(d["serial"])
        pos = (f"({cur[0][0]:.2f},{cur[0][1]:.2f},{cur[0][2]:.2f}"
               f"|{cur[1][0]:.0f},{cur[1][1]:.0f},{cur[1][2]:.0f})" if cur
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
    win = "check_vive 3D"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    view = _View3D(0, _HEAD, _W, _H - _HEAD - _FOOT)
    drag = {"on": False, "x": 0, "y": 0}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(on=True, x=x, y=y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["on"] = False
        elif event == cv2.EVENT_MOUSEMOVE and drag["on"]:
            view.orbit(x - drag["x"], y - drag["y"])
            drag["x"], drag["y"] = x, y
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", -1):
            view.zoom(0.88 if flags > 0 else 1.14)

    cv2.setMouseCallback(win, on_mouse)

    trails: dict[str, deque] = {}
    latest: dict[str, tuple | None] = {}
    colors: dict[str, tuple] = {}       # serial -> 颜色,首次出现时分配并固定
    rate = 0.0                          # (重扫改变设备顺序也不跳色)
    canvas = np.zeros((_H, _W, 3), dtype=np.uint8)
    devices: list[dict] = []
    last_scan, reported = 0.0, ""
    t0, n_frames, t_fps = time.time(), 0, time.time()
    while True:
        now = time.time()
        if seconds and now - t0 >= seconds:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            view.zoom(0.88)
        elif key == ord("-"):
            view.zoom(1.14)
        elif key == ord("r"):
            view.reset()
        if now - last_scan >= _RESCAN_S:          # 周期重扫:开关机自动跟上
            last_scan = now
            devices[:] = list_devices(vr, serial_role)
            gone = [s for s in trails if s not in {d["serial"] for d in devices}]
            for s in gone:                        # 离线设备的轨迹一并清掉,
                trails.pop(s, None)               # 不把取景半径撑到旧位置
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
                    .append(p[0])                  # 轨迹只留位置
        n_frames += 1
        if time.time() - t_fps >= 1.0:
            rate = n_frames / (time.time() - t_fps)
            n_frames, t_fps = 0, time.time()
        draw3d(canvas, view, devices, trails, latest, colors, reported, rate)
        cv2.imshow(win, canvas)
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
              "那个角色就是这台 —— 与 role_serial_map 对不上当场暴露。\n"
              "3D 视图为真实透视,画面左右与面朝场景一致"
              "(X 在屏幕左、Y 上、Z 向屏幕里),可拖拽旋转观察。")
        show(vr, args.seconds, args.trail, serial_role)
    finally:
        import openvr
        openvr.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
