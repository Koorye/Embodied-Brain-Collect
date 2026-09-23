"""SteamVR / OpenVR 6-DOF position tracker.

Captures pose (position + quaternion + euler) for each tracked device
at a fixed polling rate.  Handles transient OpenVR init failures with
retry + Windows session diagnostics.

Output (under <session>/position/position.npz)::

    timestamps_s        (T,)      float64  PC clock
    perf_counter_s      (T,)      float64  monotonic perf counter
    roles               (D,)      str      role_serial_map 的角色名(未绑定为空串)
    device_indices      (D,)      int32
    device_classes      (D,)      str
    serials             (D,)      str
    models              (D,)      str
    positions_m         (T, D, 3) float64
    quaternions_wxyz    (T, D, 4) float64
    euler_rpy_deg       (T, D, 3) float64
    valid               (T, D)    bool
    marker_*

列序 D 由 ``role_serial_map`` 钉死(序列号绑定);不配时按 OpenVR 枚举
顺序排列,随开机先后漂移。
"""

import math, sys, time
import numpy as np
from .base_position_recorder import BasePositionRecorder
from .position_recorder_config import PositionRecorderConfig

# ---------------------------------------------------------------------------
# OpenVR helpers (moved here from tests/vive/ to keep recorder self-contained)
# ---------------------------------------------------------------------------

_TRANSIENT_ERRORS = ("InitError_Init_Internal", "InitError_Init_NoServerForBackgroundApp")
_INIT_MAX_RETRY = 4
_INIT_RETRY_DELAY = 0.8


def _device_class_name(cls_id: int) -> str:
    from openvr import (
        TrackedDeviceClass_Invalid, TrackedDeviceClass_HMD,
        TrackedDeviceClass_Controller, TrackedDeviceClass_GenericTracker,
        TrackedDeviceClass_TrackingReference, TrackedDeviceClass_DisplayRedirect,
    )
    return {
        TrackedDeviceClass_Invalid: "invalid",
        TrackedDeviceClass_HMD: "hmd",
        TrackedDeviceClass_Controller: "controller",
        TrackedDeviceClass_GenericTracker: "tracker",
        TrackedDeviceClass_TrackingReference: "tracking_reference",
        TrackedDeviceClass_DisplayRedirect: "display_redirect",
    }.get(cls_id, str(cls_id))


def _matrix_to_quat(m) -> tuple[float, float, float, float]:
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]
    trace = r00 + r11 + r22

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        return ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)


def _quat_to_euler(qw, qx, qy, qz) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _select_devices(vr_system, wanted_classes: set[str]) -> list[dict]:
    import openvr
    devices = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        if not vr_system.isTrackedDeviceConnected(idx):
            continue
        cls = _device_class_name(vr_system.getTrackedDeviceClass(idx))
        if cls not in wanted_classes:
            continue
        try:   # VIVE Hub 里设置的角色(vive_tracker_chest / …_left_elbow …)
            vive_role = vr_system.getStringTrackedDeviceProperty(
                idx, openvr.Prop_ControllerType_String)
        except Exception:
            vive_role = ""
        devices.append({
            "index": idx, "device_class": cls,
            "serial": vr_system.getStringTrackedDeviceProperty(idx, openvr.Prop_SerialNumber_String),
            "model": vr_system.getStringTrackedDeviceProperty(idx, openvr.Prop_ModelNumber_String),
            "vive_role": vive_role,
        })
    return devices


def _bind_roles(devices: list[dict], role_map: dict) -> tuple[list[dict] | None, str]:
    """按 role_map 把物理设备钉到固定角色上(纯函数,便于离线测试)。

    匹配值两种写法任选:OpenVR 序列号(``61-BH…``,推荐 —— 永不变化)或
    VIVE Hub 里设置的角色(``vive_tracker_chest`` 等,在 Hub 里改角色会
    跟着变)。VIVE Hub 的设备 ID(``FA61…``)不是 OpenVR 序列号,匹配不到。

    返回 (绑定后的设备列表, "") 或 (None, 操作员可读的原因)。列表顺序 =
    ``role_map`` 的书写顺序,即 npz 的列序 —— 会话间稳定,不随开机顺序漂移。
    查得到就逐台校验:缺一台、多一台未绑定、两角色配同一设备、多台设备
    上报同一序列号,都是配置/设备问题,拒绝开录而不是静默错位。
    """
    role_map = {str(k): str(v) for k, v in (role_map or {}).items()}
    if not role_map:
        return [dict(d, role="") for d in devices], ""
    if len(set(role_map.values())) != len(role_map):
        return None, (f"role_serial_map 有重复匹配值: {role_map} — "
                      "一个序列号/角色只能绑一个位置")

    def _ids(d: dict) -> tuple[str, ...]:
        return (d["serial"], d.get("vive_role") or f"§{d['serial']}")

    by_id: dict[str, list[dict]] = {}
    for d in devices:
        for ident in _ids(d):
            by_id.setdefault(ident, []).append(d)
    dup = sorted(s for s, ds in by_id.items() if len(ds) > 1)
    if dup:
        return None, (f"多台设备上报同一标识 {dup} — 无法按 "
                      "role_serial_map 绑定")
    missing = [f"{role}={value}" for role, value in role_map.items()
               if value not in by_id]
    if missing:
        known = sorted({d["serial"] for d in devices})
        return None, (f"role_serial_map 里未连接: {', '.join(missing)} — "
                      f"已连接的 serial: {known};注意要用 open 日志设备清单"
                      "里的 serial(VIVE Hub 显示的设备 ID 不是序列号),"
                      "或改用 VIVE 角色 vive_tracker_*")
    extra = sorted({d["serial"] for d in devices
                    if not (set(role_map.values()) & set(_ids(d)))})
    if extra:
        return None, (f"已连接但未绑定的 tracker: {extra} — 关掉备用设备,"
                      "或把它加进 role_serial_map(绑定的列序才稳定)")
    return ([dict(by_id[value][0], role=role)
             for role, value in role_map.items()], "")


def _read_frame(vr_system, devices: list[dict]):
    import openvr
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    nd = len(devices)
    pos = np.full((nd, 3), np.nan, dtype=np.float64)
    quat = np.full((nd, 4), np.nan, dtype=np.float64)
    euler = np.full((nd, 3), np.nan, dtype=np.float64)
    valid = np.zeros(nd, dtype=bool)

    for col, d in enumerate(devices):
        pose = poses[d["index"]]
        if not pose.bPoseIsValid:
            continue
        m = pose.mDeviceToAbsoluteTracking
        qw, qx, qy, qz = _matrix_to_quat(m)
        pos[col] = (m[0][3], m[1][3], m[2][3])
        quat[col] = (qw, qx, qy, qz)
        euler[col] = _quat_to_euler(qw, qx, qy, qz)
        valid[col] = True
    return pos, quat, euler, valid


# ---------------------------------------------------------------------------
# OpenVR init with retry
# ---------------------------------------------------------------------------

def _init_openvr():
    import openvr
    last_exc = None
    for attempt in range(1, _INIT_MAX_RETRY + 1):
        try:
            return openvr.init(openvr.VRApplication_Background)
        except openvr.OpenVRError as exc:
            last_exc = exc
            if not any(t in repr(exc) for t in _TRANSIENT_ERRORS) or attempt == _INIT_MAX_RETRY:
                raise
            print(f"[position:openvr] transient init error "
                  f"(attempt {attempt}/{_INIT_MAX_RETRY}): {exc}", file=sys.stderr)
            try: openvr.shutdown()
            except Exception: pass
            time.sleep(_INIT_RETRY_DELAY)
    raise last_exc  # type: ignore


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class OpenvrPositionRecorder(BasePositionRecorder):
    """Real SteamVR / OpenVR 6-DOF pose tracker."""

    name = "position"
    output_dir = "position"
    config: PositionRecorderConfig

    def __init__(self, config: PositionRecorderConfig):
        super().__init__(config)
        self._vr = None
        self._devices: list[dict] = []

    # ---- lifecycle ----------------------------------------------------------

    def _open(self) -> bool:
        cfg = self.config
        self._wanted_classes = {c.strip() for c in cfg.device_classes.split(",") if c.strip()}

        self._vr = _init_openvr()
        self._devices = _select_devices(self._vr, self._wanted_classes)
        self._log("[position:openvr] connected devices: " + (
            ", ".join(f"[{d['index']:02}] {d['device_class']} "
                      f"serial={d['serial'] or '-'}"
                      + (f" vive_role={d['vive_role']}" if d.get("vive_role") else "")
                      for d in self._devices)
            or "(none matched)"))
        if not self._devices:
            self._open_error = (f"no devices matched --classes={cfg.device_classes}")
            self._log(f"[position:openvr] open failed — {self._open_error}")
            return False

        # 角色绑定:按序列号把物理设备钉到固定角色,npz 列序 = role_serial_map
        # 的书写顺序。不配则按枚举顺序排列 —— 那个顺序随开机先后走,会话间
        # 会漂移(左右手互换)。
        bound, err = _bind_roles(self._devices, cfg.role_serial_map)
        if bound is None:
            self._open_error = err
            self._log(f"[position:openvr] open failed — {err}")
            return False
        self._devices = bound
        if any(d["role"] for d in self._devices):
            self._log("[position:openvr] role bound: " + ", ".join(
                f"{d['role']}→{d['serial']}[idx {d['index']}]"
                for d in self._devices))

        # 设备数闸门:少连一台 tracker 时 SteamVR 不会报错,只有台数本身
        # 能暴露 —— 数量不足直接拒绝开录,而不是录一份缺轴的数据
        if len(self._devices) < cfg.expected_devices:
            self._open_error = (f"found {len(self._devices)} "
                                f"{cfg.device_classes} device(s), expected "
                                f"{cfg.expected_devices} — tracker 未全部"
                                "连接/未亮灯")
            self._log(f"[position:openvr] open failed — {self._open_error}")
            return False

        for d in self._devices:
            self._acc("roles", d["role"])
            self._acc("device_indices", d["index"])
            self._acc("device_classes", d["device_class"])
            self._acc("serials", d["serial"])
            self._acc("models", d["model"])

        self._log(f"[position:openvr] {len(self._devices)} devices — "
                  f"streaming starts with the record loop")
        # 不读首帧:tracker 是否真的在出 pose 由 launcher 的确认阶段判断
        return True

    def _poll(self, ts):
        pos, quat, euler, valid = _read_frame(self._vr, self._devices)
        self._acc("timestamps_s", ts)
        self._acc("perf_counter_s", time.perf_counter())
        self._acc_arr("positions_m", pos)
        self._acc_arr("quaternions_wxyz", quat)
        self._acc_arr("euler_rpy_deg", euler)
        self._acc_arr("valid", valid)

    def _close(self) -> None:
        if self._vr is not None:
            import openvr
            openvr.shutdown()
            self._vr = None

    def _heartbeat_stats(self, elapsed: float) -> str:
        vc = "-"
        if self._arr_buf.get("valid"):
            vc = int(np.sum(self._arr_buf["valid"][-1]))
        return (
            super()._heartbeat_stats(elapsed)
            + f"  valid={vc}/{len(self._devices)}"
        )

