"""SteamVR / OpenVR 6-DOF position tracker.

Captures pose (position + quaternion + euler) for each tracked device
at a fixed polling rate.  Handles transient OpenVR init failures with
retry + Windows session diagnostics.

Output (under <session>/position/position.npz)::

    timestamps_s        (T,)      float64  PC clock
    perf_counter_s      (T,)      float64  monotonic perf counter
    device_indices      (D,)      int32
    device_classes      (D,)      str
    serials             (D,)      str
    models              (D,)      str
    positions_m         (T, D, 3) float64
    quaternions_wxyz    (T, D, 4) float64
    euler_rpy_deg       (T, D, 3) float64
    valid               (T, D)    bool
    marker_*
"""

import math, sys, time
import numpy as np
from .base_position_recorder import BasePositionRecorder
from .position_recorder_config import PositionRecorderConfig

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# OpenVR helpers (moved here from tests/vive/ to keep recorder self-contained)
# ---------------------------------------------------------------------------

import openvr

_DEVICE_CLASS_NAMES = {
    "invalid": "invalid",
    "hmd": "hmd",
    "controller": "controller",
    "tracker": "tracker",
    "tracking_reference": "tracking_reference",
    "display_redirect": "display_redirect",
}

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
    devices = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        if not vr_system.isTrackedDeviceConnected(idx):
            continue
        cls = _device_class_name(vr_system.getTrackedDeviceClass(idx))
        if cls not in wanted_classes:
            continue
        devices.append({
            "index": idx, "device_class": cls,
            "serial": vr_system.getStringTrackedDeviceProperty(idx, openvr.Prop_SerialNumber_String),
            "model": vr_system.getStringTrackedDeviceProperty(idx, openvr.Prop_ModelNumber_String),
        })
    return devices


def _print_device_table(devices: list[dict]) -> None:
    print("[position:openvr] connected devices:")
    if not devices:
        print("  (none matched)")
        return
    for d in devices:
        print(f"  [{d['index']:02}] {d['device_class']:<18} "
              f"serial={d['serial'] or '-'}  model={d['model'] or '-'}")


def _read_frame(vr_system, devices: list[dict]):
    from openvr import TrackingUniverseStanding, k_unMaxTrackedDeviceCount
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        TrackingUniverseStanding, 0, k_unMaxTrackedDeviceCount)
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
        _print_device_table(self._devices)
        if not self._devices:
            self._open_error = (f"no devices matched --classes={cfg.device_classes}")
            self._log(f"[position:openvr] open failed — {self._open_error}")
            return False

        for d in self._devices:
            self._acc("device_indices", d["index"])
            self._acc("device_classes", d["device_class"])
            self._acc("serials", d["serial"])
            self._acc("models", d["model"])

        self._log(f"[position:openvr] {len(self._devices)} devices — "
                  f"waiting for first pose ...")

        def _try_poll() -> bool:
            self._poll(time.time())
            valid = self._arr_buf.get("valid")
            return bool(valid and valid[-1].any())

        if not self._wait_first_sample(_try_poll, "pose", timeout=10.0):
            return False
        # Gate samples used wall-clock ts; clear the poll keys (keep the
        # device metadata rows) so the timeline starts from the launcher's t0.
        for k in ("timestamps_s", "perf_counter_s"):
            self._buf.pop(k, None)
        self._arr_buf.clear()
        self._log("[position:openvr] first pose received — ready")
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

