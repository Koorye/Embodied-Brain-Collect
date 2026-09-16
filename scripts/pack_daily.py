#!/usr/bin/env python
"""每日数据打包 — 把当天采集的会话打包成 mf-lerobot 多频率数据集。

Usage::

    python scripts/pack_daily.py                          # 打包今天(data/ 下)
    python scripts/pack_daily.py --date 2026-09-14        # 指定日期
    python scripts/pack_daily.py --date 2026-09-14 --source data/session-test \
        --out data/lerobot/daily-2026-09-14 --force

在 ``--source``(默认 ``data``)下查找 ``<日期>-*`` 会话目录(兼容
``<批次根>/<日期>-*`` 一层嵌套),每个会话打包成一个 episode,全部进同一
个数据集。会话目录里有 ``qc_report.json``(``scripts/qc.py`` 产出)时,
整体等级为 ERROR 的会话直接排除;特征集取各会话全部模态的并集,缺任一
模态的会话整体剔除(而不是把该模态从数据集中去掉)。依赖 conda 环境
collect 里已安装的 ``mf_lerobot`` 包(Multi-Frequency
LeRobot 扩展,每传感器独立 parquet)。

流 → 特征映射(在 Multi-Frequency-LeRobot 的 convert_session_night 基础上
新增 wristband,并按槽位目录自动发现视频流):

============================  ==================================  =====================
source                        feature                              notes
============================  ==================================  =====================
cam_*/frames.mp4              observation.images.<槽位>_rgb        30 fps,截段对齐主时间轴
eye/eye.mp4                   observation.images.eye_scene_rgb
wristband                     observation.wristband_pressure       3 路,150 Hz(设备钟)
wristband                     observation.wristband_ppg            3 路,100 Hz
wristband                     observation.wristband_imu            6 路,50 Hz
wristband                     observation.wristband_vitals         温度/血氧,1 Hz 值
eeg                           observation.eeg                      按通道数
emg_left/right                observation.emg_left/right           8 通道
emg_*(imu)                    observation.imu_left/right           6 路
eye(gaze)                     observation.gaze                     2 路
eye(imu)                      observation.eye_imu                  6 路
hand_pose                     observation.hand_pose/_skeleton_*    40d / 50x3 / 50x4
position                      observation.state                    追踪器位姿
marker                        episode 窗口(RUN_START..RUN_END)
                              + 独立 30Hz 主时间轴(RUN_START 为 0 点)
============================  ==================================  =====================
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:                      # 无 tqdm 时退化为无进度条
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mf_lerobot import MultiFrequencyLeRobotDataset  # noqa: E402
from mf_lerobot.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH  # noqa: E402
from embodied_brain_collect.utils.media import ffprobe_count, media_tool  # noqa: E402


def _progress(iterable, **kw):
    """tqdm 包装;未安装 tqdm 时原样返回(不阻塞打包)。"""
    return tqdm(iterable, **kw) if tqdm is not None else iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data"


MASTER_FPS = 30.0
WINDOW = (-0.033, 0.0)   # 读取滑窗 (-33ms, 0ms],与 demo 数据一致
IMU_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]

# 视频流的 npz/文件名特例(其余视频槽位按 <slot>.npz + frames.mp4 自动发现)
_EYE_NPZ, _EYE_TS, _EYE_MP4 = "eye.npz", "scene_timestamps", "eye.mp4"

# 旧版 session meta 没有 recorders 字段时的默认设备显示名
# (与 configs/recorders.yaml 的 name 注释一致;新版 meta 优先)
DEFAULT_RECORDER_NAMES = {
    "cam_head": "OAK-D-LITE",
    "cam_left_wrist": "USB2.0 Camera",
    "cam_right_wrist": "USB2.0 Camera",
    "cam_third": "Realsense D435I",
    "eeg": "NeuroScan",
    "emg_left": "Weili EMG",
    "emg_right": "Weili EMG",
    "eye": "Pupil Labs Neon Eye",
    "hand_pose": "Manus Quantum",
    "position": "HTC VIVE",
    "marker": "UDP Marker",
    "wristband": "MaiYuan",
    "ego_headband": "Ego Headband V1.5",
}

# 读取时加滑窗的特征(高频流对齐主时钟时取时间窗);
# observation.device<i>_pose 由 _is_windowed 的前缀规则覆盖
WINDOWED_FEATURES = {
    "observation.eeg",
    "observation.left_wrist_emg", "observation.right_wrist_emg",
    "observation.left_wrist_imu", "observation.right_wrist_imu",
    "observation.eye_imu", "observation.eye_gaze",
    "observation.hand_pose", "observation.hand_skeleton",
    "observation.marker",
    "observation.wristband_pressure", "observation.wristband_ppg",
    "observation.wristband_imu",
}


def _is_windowed(key: str) -> bool:
    return key in WINDOWED_FEATURES or key.startswith("observation.device")


# ── 通用加载小工具 ─────────────────────────────────────────────────────────

def _spread_run_timestamps(ts: np.ndarray) -> np.ndarray:
    """把同一次串口读入的等值时间戳在段内均匀铺开(weili 臂环的到达时刻)。"""
    n = len(ts)
    out = np.empty_like(ts)
    dt_global = (ts[-1] - ts[0]) / max(n - 1, 1) if n > 1 else 0.0
    prev_t = None
    i = 0
    while i < n:
        j = i + 1
        while j < n and ts[j] == ts[i]:
            j += 1
        k = j - i
        if prev_t is None:
            t0 = ts[i] - k * dt_global
        else:
            t0 = prev_t
        out[i:j] = t0 + (np.arange(k) + 1) * (ts[i] - t0) / k
        prev_t = ts[i]
        i = j
    return out


def _load_imu_stream(z, ts_key):
    """(accel, gyro) → (时间戳, (N,6) f32, 通道名)。"""
    ts = _spread_run_timestamps(z[ts_key].astype(np.float64))
    vals = np.hstack([z["imu_accel"], z["imu_gyro"]]).astype(np.float32)
    return ts, vals, IMU_NAMES


def _window_mask(ts: np.ndarray, win: tuple[float | None, float | None]) -> np.ndarray:
    mask = np.ones(len(ts), dtype=bool)
    if win[0] is not None:
        mask &= ts >= win[0]
    if win[1] is not None:
        mask &= ts <= win[1]
    return mask


def _stream_rate(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 0.0
    duration = ts[-1] - ts[0]
    return round(len(ts) / duration, 1) if duration > 0 else 0.0


# ── 流加载(每个会话的全部非视频流) ──────────────────────────────────────

def load_wristband(session_dir: Path):
    """腕带:压力/PPG/IMU 用设备钟(0x20 UTC 同步),体感值随帧存。

    npz 里时间戳已按采样率展开(pressure 927 = 309 帧 × 3 采样),直接配
    同长的原始值列。
    """
    p = session_dir / "wristband" / "wristband.npz"
    if not p.exists():
        return
    z = np.load(p, allow_pickle=False)
    ft = z["frame_timestamps"].astype(np.float64)
    if not len(ft):
        return
    out = []
    out.append(("observation.wristband_pressure",
                z["pressure_timestamps"].astype(np.float64),
                z["pressure_raw"].astype(np.float32).reshape(-1, 1), ["p"]))
    out.append(("observation.wristband_ppg",
                z["ppg_timestamps"].astype(np.float64),
                z["ppg_raw"].astype(np.float32), ["green", "red", "infrared"]))
    out.append(("observation.wristband_imu", ft,
                np.hstack([z["accel_m_s2"], z["gyro_rad_s"]]).astype(np.float32),
                IMU_NAMES))
    vit = np.hstack([
        z["temperature_c"].astype(np.float32).reshape(-1, 1),
        z["spo2_percent"].astype(np.float32).reshape(-1, 1)])
    out.append(("observation.wristband_vitals", ft, vit,
                ["temp_c", "spo2_pct"]))
    yield from out


def load_parquet_streams(session_dir: Path):
    """一个会话的全部非视频流 → (特征键, 时间戳, (N,通道) 值, 通道名)。"""
    yield from load_wristband(session_dir)
    out_extra: list = []

    # EEG — 通道数 = eeg_n_eeg_channels(数据里可能带 Trigger 列)
    p = session_dir / "eeg" / "eeg.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "eeg_timestamps_pc" in z:
            ts = z["eeg_timestamps_pc"].astype(np.float64)
        elif bool(z["eeg_fit_fitted"]) and "eeg_fit_slope_pc_per_eeg" in z:
            rate = float(z["eeg_sample_rate"])
            ts = (float(z["eeg_fit_pc_t0_s"])
                  + np.arange(len(z["eeg_data"])) * float(z["eeg_fit_slope_pc_per_eeg"]) / rate)
            print(f"[eeg] '{session_dir.name}': 从拟合重建 PC 时间戳")
        else:
            print(f"[eeg] '{session_dir.name}': 无 PC 对齐时间戳 — 跳过 eeg")
            ts = None
        if ts is not None:
            n_ch = int(z["eeg_n_eeg_channels"])
            names = [str(s) for s in z["eeg_channel_names"][:n_ch]]
            out_extra.append(("observation.eeg", ts,
                              z["eeg_data"][:, :n_ch].astype(np.float32), names))

    # EMG 臂环 ×2(左/右腕)+ 板载 IMU
    for side in ("left", "right"):
        p = session_dir / f"emg_{side}" / f"emg_{side}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            out_extra.append((
                f"observation.{side}_wrist_emg",
                _spread_run_timestamps(z["emg_timestamps"].astype(np.float64)),
                z["emg_data"].astype(np.float32), None))
            if "imu_timestamps" in z:
                out_extra.append(
                    (f"observation.{side}_wrist_imu",) + _load_imu_stream(z, "imu_timestamps"))

    # 眼动 — gaze + IMU(scene 视频走视频流)
    p = session_dir / "eye" / "eye.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "gaze_timestamps" in z:
            out_extra.append(("observation.eye_gaze",
                              z["gaze_timestamps"].astype(np.float64),
                              z["gaze_xy"].astype(np.float32), ["x", "y"]))
        if "imu_timestamps" in z:
            out_extra.append(("observation.eye_imu",) + _load_imu_stream(z, "imu_timestamps"))

    # 手部姿态 — 40d 人机工学(列名 = 手位_关节) + 50 节点骨架
    # (位置 + 四元数转欧拉,拼接为每节点 [x, y, z, r, p, y] 共 300 维)
    p = session_dir / "hand_pose" / "hand_pose.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "ergo_timestamps" in z and "ergo_data" in z:
            out_extra.append(("observation.hand_pose",
                              z["ergo_timestamps"].astype(np.float64),
                              z["ergo_data"].astype(np.float32),
                              _ergo_channel_names()))
        if {"skeleton_timestamps", "skeleton_positions",
            "skeleton_rotations"} <= set(z.files):
            pos = z["skeleton_positions"].reshape(-1, 50, 3).astype(np.float32)
            r, pt, yw = _quat_xyzw_to_rpy_deg(
                z["skeleton_rotations"].reshape(-1, 50, 4).astype(np.float32))
            vals = np.concatenate(
                [pos, r[..., None], pt[..., None], yw[..., None]],
                axis=2).reshape(len(pos), -1).astype(np.float32)
            out_extra.append(("observation.hand_skeleton",
                              z["skeleton_timestamps"].astype(np.float64),
                              vals, _skeleton_channel_names()))

    # 位置追踪 — 每台设备独立特征 observation.device<i>_pose (x, y, z, r, p, y);
    # 容忍历史/dummy 变体:pos (T,3)|(T,D,3)、rpy 可缺失(NaN 占位)
    p = session_dir / "position" / "position.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if not {"timestamps_s", "positions_m"} <= set(z.files):
            print(f"[position] '{session_dir.name}': 字段不全 "
                  f"({z.files}) — 跳过 position")
        else:
            pos = z["positions_m"]
            if pos.ndim == 2:
                pos = pos[:, None, :]
            n_dev = pos.shape[1]
            rpy = z.get("euler_rpy_deg")
            if rpy is None:
                rpy = np.full(pos.shape[:2] + (3,), np.nan, dtype=np.float32)
            if rpy.ndim == 2:
                rpy = rpy[:, None, :]
            ts = z["timestamps_s"].astype(np.float64)
            for i in range(n_dev):
                vals = np.concatenate(
                    [pos[:, i, :], rpy[:, i, :]], axis=1).astype(np.float32)
                out_extra.append((f"observation.device{i}_pose", ts, vals,
                                  ["x", "y", "z", "roll", "pitch", "yaw"]))

    # state / action —— 统一由 device0..2_pose 与 hand_pose 拼接:
    # 3 台设备 × (x, y, z, roll, pitch, yaw) + 40 手部关节 = 58 维。
    # 时间轴取 hand_pose(组内最高频),device 位姿按最近邻取样对齐。
    pose_streams = sorted(
        ((k, ts, v) for k, ts, v, _ in out_extra
         if k.startswith("observation.device")),
        key=lambda kv: int(kv[0].rsplit("_pose", 1)[0]
                           .replace("observation.device", "")))
    hand = next((s for s in out_extra if s[0] == "observation.hand_pose"), None)
    if hand and pose_streams:
        _, ts_h, v_h, hand_names = hand
        cols, names = [], []
        for k, ts_p, v_p in pose_streams:
            idx = np.searchsorted(ts_p, ts_h).clip(1, len(ts_p) - 1)
            prev = np.clip(idx - 1, 0, len(ts_p) - 1)
            use_prev = np.abs(ts_p[prev] - ts_h) < np.abs(ts_p[idx] - ts_h)
            cols.append(v_p[np.where(use_prev, prev, idx)])
            dev = k.replace("observation.device", "").replace("_pose", "")
            names += [f"device{dev}_{c}"
                      for c in ("x", "y", "z", "roll", "pitch", "yaw")]
        cols.append(v_h)
        names += (hand_names or [f"joint_{i}" for i in range(v_h.shape[1])])
        state_vals = np.concatenate(cols, axis=1).astype(np.float32)
        out_extra.append(("observation.state", ts_h, state_vals, names))
        out_extra.append(("action", ts_h, state_vals.copy(), names))

    # marker — 事件码流(时间戳 = 发送端 PC 钟,缺失回退接收时刻)
    p = session_dir / "marker" / "marker.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        t_sent = z.get("marker_t_sent_pc")
        t_recv = z.get("marker_t_local_recv")
        code = z.get("marker_code")
        if code is not None:
            ts = (t_sent if t_sent is not None else t_recv)
            if ts is not None and len(ts) == len(code):
                out_extra.append(("observation.marker",
                                  np.asarray(ts, dtype=np.float64),
                                  np.asarray(code, dtype=np.float32).reshape(-1, 1),
                                  ["code"]))

    yield from out_extra


def load_stream_index(session_dir: Path):
    """轻量索引(预扫/特征规格用):只读时间戳与列名,不装载数据值。

    产出 (特征键, 时间戳, 通道名, 宽度);特征键与 load_parquet_streams
    保持一致 —— 两处需同步维护。
    """
    out: list[tuple[str, np.ndarray, list[str], int]] = []

    # 腕带
    p = session_dir / "wristband" / "wristband.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        ft = z["frame_timestamps"].astype(np.float64)
        if len(ft):
            out.append(("observation.wristband_pressure",
                        z["pressure_timestamps"].astype(np.float64), ["p"], 1))
            out.append(("observation.wristband_ppg",
                        z["ppg_timestamps"].astype(np.float64),
                        ["green", "red", "infrared"], 3))
            out.append(("observation.wristband_imu", ft, IMU_NAMES, 6))
            out.append(("observation.wristband_vitals", ft,
                        ["temp_c", "spo2_pct"], 2))

    # EEG
    p = session_dir / "eeg" / "eeg.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "eeg_timestamps_pc" in z.files:
            ts = z["eeg_timestamps_pc"].astype(np.float64)
        elif bool(z["eeg_fit_fitted"]) and "eeg_fit_slope_pc_per_eeg" in z.files:
            rate = float(z["eeg_sample_rate"])
            n = int(z["eeg_n_samples"]) if "eeg_n_samples" in z.files else 0
            ts = (float(z["eeg_fit_pc_t0_s"])
                  + np.arange(n) * float(z["eeg_fit_slope_pc_per_eeg"]) / rate)
        else:
            ts = None
        if ts is not None and len(ts):
            n_ch = int(z["eeg_n_eeg_channels"])
            names = [str(s) for s in z["eeg_channel_names"][:n_ch]]
            out.append(("observation.eeg", ts, names, n_ch))

    # EMG ×2 + 板载 IMU
    for side in ("left", "right"):
        p = session_dir / f"emg_{side}" / f"emg_{side}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            emg_ts = _spread_run_timestamps(z["emg_timestamps"].astype(np.float64))
            out.append((f"observation.{side}_wrist_emg", emg_ts,
                        [f"ch_{i}" for i in range(8)], 8))
            if "imu_timestamps" in z.files:
                out.append((f"observation.{side}_wrist_imu",
                            z["imu_timestamps"].astype(np.float64),
                            IMU_NAMES, 6))

    # 眼动
    p = session_dir / "eye" / "eye.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "gaze_timestamps" in z.files:
            out.append(("observation.eye_gaze",
                        z["gaze_timestamps"].astype(np.float64), ["x", "y"], 2))
        if "imu_timestamps" in z.files:
            out.append(("observation.eye_imu",
                        z["imu_timestamps"].astype(np.float64), IMU_NAMES, 6))

    # 手部
    p = session_dir / "hand_pose" / "hand_pose.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "ergo_timestamps" in z.files:
            out.append(("observation.hand_pose",
                        z["ergo_timestamps"].astype(np.float64),
                        _ergo_channel_names(), 40))
        if "skeleton_timestamps" in z.files:
            out.append(("observation.hand_skeleton",
                        z["skeleton_timestamps"].astype(np.float64),
                        _skeleton_channel_names(), 300))

    # 位置
    p = session_dir / "position" / "position.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "timestamps_s" in z.files:
            ts = z["timestamps_s"].astype(np.float64)
            pos = z["positions_m"]
            n_dev = pos.shape[1] if pos.ndim == 3 else 1
            for i in range(n_dev):
                out.append((f"observation.device{i}_pose", ts,
                            ["x", "y", "z", "roll", "pitch", "yaw"], 6))

    # marker
    p = session_dir / "marker" / "marker.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        ts = z["marker_t_sent_pc"] if "marker_t_sent_pc" in z.files \
            else z.get("marker_t_local_recv")
        if ts is not None and len(np.asarray(ts)):
            out.append(("observation.marker",
                        np.asarray(ts, dtype=np.float64), ["code"], 1))

    # state / action
    pose_ts = [(s[0], s[1]) for s in out
               if s[0].startswith("observation.device")]
    hand = next((s for s in out if s[0] == "observation.hand_pose"), None)
    if hand and pose_ts:
        names = _state_channel_names([k for k, _ in pose_ts], hand[2])
        out.append(("observation.state", hand[1], names, 58))
        out.append(("action", hand[1], names, 58))

    yield from out


def _ergo_channel_names() -> list[str]:
    """MANUS 人机工学 40 通道的列名:左 hand 占 0-19,右 20-39。"""
    from embodied_brain_collect.recorders.hand_pose.manus_hand_pose_recorder import (
        _ERGO_INDEX)

    by_pos = {v: k for k, v in _ERGO_INDEX.items()}
    return ([f"left_{by_pos[i]}" for i in range(20)]
            + [f"right_{by_pos[i]}" for i in range(20)])


def _state_channel_names(device_keys: list[str],
                         hand_names: list[str] | None) -> list[str]:
    """state/action 的列名:设备位姿 ×3 + 40 手部关节,共 58。"""
    names = []
    for k in sorted(device_keys, key=lambda s: int(
            s.replace("observation.device", "").replace("_pose", ""))):
        dev = k.replace("observation.device", "").replace("_pose", "")
        names += [f"device{dev}_{c}"
                  for c in ("x", "y", "z", "roll", "pitch", "yaw")]
    names += hand_names or [f"joint_{i}" for i in range(40)]
    return names


def _skeleton_channel_names() -> list[str]:
    """MANUS 骨架 50 节点 × [x, y, z, roll, pitch, yaw] 的列名(节点按 id;
    旋转后缀不用 y,避免与位置的 y 同名冲突)。"""
    names = []
    for i in range(50):
        names += [f"node{i:02d}_{ax}" for ax in ("x", "y", "z")]
        names += [f"node{i:02d}_{rot}" for rot in ("roll", "pitch", "yaw")]
    return names


def _quat_xyzw_to_rpy_deg(rot: np.ndarray):
    """(N, 4) xyzw 四元数 → (roll, pitch, yaw) 角度制(ZYX 约定)。"""
    x, y, z, w = rot[..., 0], rot[..., 1], rot[..., 2], rot[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0,
                     np.copysign(np.pi / 2.0, sinp), np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ── 视频流发现 ─────────────────────────────────────────────────────────────

def discover_video_slots(session_dirs: list[Path]) -> dict[str, tuple[str, str, str]]:
    """扫描会话目录 → {槽位目录: (npz 名, 时间戳键, mp4 名)}。

    约定:``<slot>/<slot>.npz`` 含 ``frames_timestamps``、同目录 ``frames.mp4``;
    eye 槽位特例为 ``eye.npz`` / ``scene_timestamps`` / ``eye.mp4``。
    """
    slots: dict[str, tuple[str, str, str]] = {}
    for sd in session_dirs:
        for d in sorted(sd.iterdir()) if sd.is_dir() else []:
            if not d.is_dir():
                continue
            slot = d.name
            if slot in slots:
                continue
            is_eye = slot == "eye"
            npz_name = _EYE_NPZ if is_eye else f"{slot}.npz"
            ts_key = _EYE_TS if is_eye else "frames_timestamps"
            mp4_name = _EYE_MP4 if is_eye else "frames.mp4"
            npz, mp4 = d / npz_name, d / mp4_name
            if not (npz.exists() and mp4.exists()):
                continue
            try:
                z = np.load(npz, allow_pickle=True)
                if ts_key not in z:
                    continue
            except Exception as exc:
                print(f"[video] 跳过 '{slot}': npz 不可读 ({exc})")
                continue
            slots[slot] = (npz_name, ts_key, mp4_name)
    # 主时钟优先 cam_head(dict 顺序 = info.json 的 master_feature 顺序)
    return dict(sorted(slots.items(), key=lambda kv: kv[0] != "cam_head"))


# ── 特征规格与会话窗口 ─────────────────────────────────────────────────────

def video_feature_key(slot: str) -> str:
    """视频槽位 → 特征键:cam_head → observation.images.head_rgb。"""
    suffix = slot[4:] if slot.startswith("cam_") else slot
    return f"observation.images.{suffix}_rgb"


def probe_video_shape(mp4: Path) -> tuple[int, int, int]:
    import av
    with av.open(str(mp4)) as container:
        frame = next(container.decode(video=0))
        return frame.height, frame.width, 3


def build_feature_specs(sessions: list[Path],
                        video_slots: dict[str, tuple[str, str, str]],
                        common: set[str]) -> dict[str, dict]:
    """跨会话的流并集(限定在预扫交集 ``common`` 内);shape/rate 从第一个
    含该流的会话探测。"""
    specs: dict[str, dict] = {}

    def _first_session_with(stream_dir: str) -> Path | None:
        for sd in sessions:
            if (sd / stream_dir).is_dir():
                return sd
        return None

    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        if video_feature_key(slot) not in common:
            continue
        sd = _first_session_with(slot)
        if sd is None:
            continue
        mp4, npz = sd / slot / mp4_name, sd / slot / npz_name
        if not (mp4.exists() and npz.exists()):
            print(f"[spec] '{slot}' 有目录但 npz/mp4 缺失 — 跳过该特征")
            continue
        try:
            h, w, c = probe_video_shape(mp4)
        except Exception as exc:
            print(f"[spec] '{slot}': 无法解码 {mp4.name} ({exc}) — 跳过该特征")
            continue
        specs[video_feature_key(slot)] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["h", "w", "c"],
            # av 只接受整数 fps
            "fps": int(MASTER_FPS),
            "tolerance_s": 0.001,
        }

    for sd in sessions:
        for key, ts, vals, names in load_parquet_streams(sd):
            if key in specs or key not in common or not len(ts):
                continue
            rate = _stream_rate(ts)
            specs[key] = {
                "dtype": "float32",
                "shape": (vals.shape[1],),
                "names": names,
                "fps": rate,
                "tolerance_s": max(0.005, 3.0 / max(rate, 1.0)),
            }
            if _is_windowed(key):
                specs[key]["window"] = WINDOW
    return specs


def load_marker_window(session_dir: Path) -> tuple[float | None, float | None]:
    """(RUN_START, RUN_END) 的 PC 时刻;没有 marker 则 (None, None)。"""
    p = session_dir / "marker" / "marker.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    t0 = t1 = None
    for tag, t in zip(z["marker_tag"], z["marker_t_sent_pc"]):
        if tag == "RUN_START":
            t0 = float(t)
        elif tag == "RUN_END":
            t1 = float(t)
    return t0, t1


def _hardware_names(session_dirs: list[Path]) -> dict[str, str]:
    """每槽位的设备显示名,只列实际出现的槽位。

    槽位集合 = 各会话里真实存在的槽位目录 ∪ meta.yaml(recorders 字段)
    显式声明的槽位;显示名 meta 声明优先,缺失回退 DEFAULT_RECORDER_NAMES。
    默认表只作名字回退、不再整体预填 —— 某天没采的模态(如 wristband)
    不会带着默认名混进 info.hardware。
    """
    import yaml
    declared: dict[str, str] = {}
    present: set[str] = set()
    for sd in session_dirs:
        for d in sd.iterdir():
            if d.is_dir():
                present.add(d.name)
        p = sd / "meta.yaml"
        if not p.exists():
            continue
        try:
            meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for slot, label in (meta.get("recorders") or {}).items():
            declared[str(slot)] = str(label)
    names: dict[str, str] = {}
    for slot in sorted(present | set(declared)):
        label = declared.get(slot) or DEFAULT_RECORDER_NAMES.get(slot)
        if label:
            names[slot] = label
    return names


def load_task_label(session_dir: Path, env_mod) -> str:
    """任务标签:meta 的 task_name;图纸模式回退 environment 的场景任务。"""
    p = session_dir / "meta.yaml"
    if p.exists():
        meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if meta.get("task_name"):
            return str(meta["task_name"])
        environment = meta.get("environment")
        if environment:
            try:
                return (f"{env_mod.scene_task(environment)} · "
                        f"{env_mod.scene_title(environment)}")
            except Exception:
                pass
    return f"session_{session_dir.name}"


def make_master_timeline(win: tuple[float | None, float | None],
                         session_dir: Path, video_slots) -> np.ndarray | None:
    """独立主时间轴:RUN_START 为 0 点、固定 30Hz 网格、覆盖 RUN_END。

    主时钟不取任何 recorder 的时间戳,由 marker 窗口直接合成:
    ``t_k = RUN_START + k/30``。末帧取 ``ceil((RUN_END-RUN_START)*30)``,
    保证网格不早于 RUN_END。没有 marker 窗口(--full 或缺 RUN_START/
    RUN_END)时,回退用相机首末帧界定时长,同样合成 30Hz 网格。
    """
    if win[0] is not None and win[1] is not None:
        t0, t1 = float(win[0]), float(win[1])
    else:
        span = _camera_span(session_dir, video_slots)
        if span is None:
            return None
        t0, t1 = span
    n = int(np.ceil((t1 - t0) * MASTER_FPS)) + 1
    return t0 + np.arange(n, dtype=np.float64) / MASTER_FPS


def _camera_span(session_dir: Path, video_slots) -> tuple[float, float] | None:
    """回退用相机界定时长:cam_head 优先,取首个有帧槽位的 (首帧, 末帧)。"""
    order = (["cam_head"] if "cam_head" in video_slots else []) + [
        s for s in video_slots if s != "cam_head"]
    for slot in order:
        npz_name, ts_key, _ = video_slots[slot]
        npz = session_dir / slot / npz_name
        if not npz.exists():
            continue
        try:
            ts = np.load(npz, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            continue
        if len(ts):
            return float(ts[0]), float(ts[-1])
    return None


# ── episode 写入 ───────────────────────────────────────────────────────────
# ffmpeg/ffprobe 解析与容器帧数统计统一走 embodied_brain_collect.utils.media
# (Windows 用 third_party exe,其余平台用系统命令),与录制端、reqc 同一套。


def write_video_ffmpeg(src_mp4: Path, out_mp4: Path, seek_s: float,
                       dur_s: float, fps: int) -> int:
    """ffmpeg 截段 + fps 归一化重编码;返回输出帧数。

    跳过逐帧解码/PNG 中间态:直接从源 mp4 截出 episode 对应的时间段。
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    cmd = [media_tool("ffmpeg"), "-y", "-loglevel", "error",
           "-ss", f"{max(0.0, seek_s):.3f}", "-i", str(src_mp4),
           "-t", f"{max(0.0, dur_s):.3f}",
           "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    n = ffprobe_count(tmp)
    tmp.replace(out_mp4)
    return n


def write_video_timestamps(ds, key: str, ep_idx: int, rel_ts: np.ndarray) -> None:
    """写视频特征的时间戳 parquet(与 VideoFeature.save 同路径同格式)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ep_chunk = ep_idx // 1000
    fpath = ds.root / DEFAULT_DATA_PATH.format(
        episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=key)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "timestamp": pa.array(rel_ts, type=pa.float64()),
        "episode_index": pa.array(
            np.full(len(rel_ts), ep_idx, dtype=np.int64), type=pa.int64()),
    })
    pq.write_table(table, fpath, compression="snappy")


def _decode_frames(path: Path, start: int, count: int) -> list[np.ndarray]:
    """容器 [start, start+count) 的灰度帧;不足则截断。"""
    import av
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= start:
                out.append(frame.to_ndarray(format="gray").astype(np.float32))
                if len(out) >= count:
                    break
    return out


def _decode_frame_at(path: Path, index: int) -> np.ndarray:
    """解码容器中第 ``index`` 帧(灰度 float32)。顺序解码到该位置。"""
    import av
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= index:
                return frame.to_ndarray(format="gray").astype(np.float32)
    raise IndexError(f"{path}: 只有不到 {index + 1} 帧")


def cut_video_aligned(src_mp4: Path, out_mp4: Path, all_ts: np.ndarray,
                      win: tuple, t0: float, key: str,
                      fps: int = 30, guard: int = 8) -> int:
    """按到达序号精确截段:返回写出的帧数(时间戳 = all_ts[i0:i1+1] - t0)。

    从源 mp4 的 i0/fps 处截出窗口段(CFR 容器下按序号 -ss 即精确落帧),
    再用源视频内容自验:在 [i0-guard, i0+guard] 的容器帧里找与截段首帧
    最匹配的帧。实测录制严格 CFR 且容器帧序号与 npz 到达序号 1:1,匹配
    落在 i0±1 内;候选窗画面静止时 argmin 不可辨,但窗口内任何一帧内容
    都等价,同样放行。匹配明确落在别处说明容器序号与到达序号错位(编码
    器插入重复帧的旧录制)——内容匹配拿截段首帧自己去比,只能发现错位、
    无法定位正确落点,原样保留并告警。
    """
    mask = _window_mask(all_ts, win) & (all_ts >= t0)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return 0
    i0, i1 = int(idx[0]), int(idx[-1])
    n = i1 - i0 + 1
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    write_video_ffmpeg(src_mp4, out_mp4, i0 / fps, n / fps, fps)

    try:
        naive0 = _decode_frame_at(out_mp4, 0)
    except Exception as exc:   # IndexError/解码错误:截段为空或损坏
        print(f"[video] {key}: 截段无有效视频(源视频可能被截断或长度异常)"
              f"— 跳过该视频流,其余模态继续打包({exc})")
        out_mp4.unlink(missing_ok=True)
        return 0
    lo = max(0, i0 - guard)
    cand = _decode_frames(src_mp4, lo, 2 * guard + 1)
    mses = {lo + j: float(((naive0 - f) ** 2).mean())
            for j, f in enumerate(cand) if f.shape == naive0.shape}
    if mses:
        c_star = min(mses, key=mses.get)
        # 候选窗各帧与截段首帧的 mse 几乎同高 ⇒ 画面静止,argmin 只是噪声
        flat = max(mses.values()) - min(mses.values()) <= 2.0 * mses[c_star]
        if abs(c_star - i0) > 1 and not flat:
            print(f"[video] {key}: 容器帧序号与到达序号疑似错位(内容匹配 "
                  f"{c_star},期望 {i0})— 截段最多偏 {abs(c_star - i0)} 帧,"
                  "请核对该路录制")

    rel_ts = all_ts[i0:i1 + 1].astype(np.float64) - t0
    return len(rel_ts)


def write_episode(ds, session_dir: Path, task_label: str, master_abs: np.ndarray,
                  streams, win, video_slots) -> None:
    # master_abs 是独立 30Hz 时间轴(make_master_timeline),窗口模式
    # 下 master_abs[0] 恰为 RUN_START,故 `ts >= t0` 不会切掉起始事件
    # (RUN_START/FIX_ON 与 RUN_END 一样保留在 episode 内)。
    t0 = float(master_abs[0])
    master_rel = (master_abs - t0).astype(np.float64)
    ep_idx = ds.meta.total_episodes

    for t in master_rel:
        ds.add_frame("task", task_label, float(t))

    present: set[str] = set()
    for key, ts, vals, _names in streams:
        mask = _window_mask(ts, win) & (ts >= t0)
        ts_w, vals_w = ts[mask], vals[mask]
        if not len(ts_w):
            continue
        rel = (ts_w - t0).astype(np.float64)
        samples = _progress(zip(rel, vals_w), total=len(rel),
                            desc=f"  {key}", unit="sample", leave=False)
        for t, v in samples:
            ds.add_frame(key, v, float(t))
        present.add(key)

    # 视频:ffmpeg 按序号精确截段(不经逐帧解码/PNG),时间戳 parquet
    # 自行写出;特征从 _features 移除,save_episode 不再处理它们
    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        npz_p = session_dir / slot / npz_name
        mp4_p = session_dir / slot / mp4_name
        if not (npz_p.exists() and mp4_p.exists()):
            continue
        try:
            all_ts = np.load(npz_p, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            continue
        key = video_feature_key(slot)
        n = cut_video_aligned(
            mp4_p, ds.root / DEFAULT_VIDEO_PATH.format(
                episode_chunk=ep_idx // 1000, video_key=key,
                episode_index=ep_idx),
            all_ts, win, t0, key, int(MASTER_FPS))
        if not n:
            continue
        mask = _window_mask(all_ts, win) & (all_ts >= t0)
        kept = np.where(mask)[0]
        rel_ts = all_ts[kept[0]:kept[-1] + 1].astype(np.float64) - t0
        write_video_timestamps(ds, key, ep_idx, rel_ts)
        ds._features.pop(key, None)
        present.add(key)
        print(f"  {key}: 精确截段 {n} 帧 @{MASTER_FPS:.0f}fps")

    dropped = {k: ds._features.pop(k) for k in list(ds._features) if k not in present}
    ds.save_episode()
    for k, f in dropped.items():
        f.next_episode()
        ds._features[k] = f

    try:
        ds.meta.update_video_info()
    except Exception:
        pass  # 个别 episode 缺视频文件时不阻塞打包

    if dropped:
        print(f"[episode] '{session_dir.name}': 本会话缺少以下特征,已跳过 "
              f"{sorted(dropped)}")


def _cleanup_images(root: Path, ep_idx: int, feature_keys: list[str]) -> None:
    for key in feature_keys:
        d = root / "images" / key / f"episode_{ep_idx:06d}"
        if d.is_dir():
            shutil.rmtree(d)
        try:
            d.parent.rmdir()
        except OSError:
            pass
    try:
        (root / "images").rmdir()
    except OSError:
        pass


# ── CLI ────────────────────────────────────────────────────────────────────

def find_sessions(source: Path, date: str) -> list[Path]:
    """``<source>/<日期>-*`` 与 ``<source>/<班次根>/<日期>-*`` 的会话目录。"""
    found: dict[str, Path] = {}
    for pattern in (str(date) + "-*", "*/" + str(date) + "-*"):
        for p in sorted(source.glob(pattern)):
            if p.is_dir() and p.name not in found:
                found[p.name] = p
    return [found[k] for k in sorted(found)]


def _qc_error_messages(report: dict) -> list[str]:
    """报告里全部 ERROR 摘要(session 级 + 各流级),供打包日志展示。"""
    out = [f.get("message", "") for f in report.get("findings", [])
           if f.get("level") == "ERROR"]
    for name, stream in report.get("streams", {}).items():
        out += [f"{name}: {f.get('message', '')}"
                for f in stream.get("findings", [])
                if f.get("level") == "ERROR"]
    return out


def filter_qc_errors(sessions: list[Path]) -> list[Path]:
    """按会话目录里的 ``qc_report.json`` 过滤:整体等级 ERROR 的直接排除。

    等级取报告的 session 级 ``level``(checkers 已把它算成各流里最严重的),
    所以任一模态出 ERROR 都会排除整个会话。没有报告或报告不可读的会话视为
    未跑 QC,保留并提示。
    """
    kept: list[Path] = []
    for sd in sessions:
        p = sd / "qc_report.json"
        report = None
        if p.exists():
            try:
                report = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[qc] '{sd.name}': qc_report.json 不可读 ({exc}) — 视为未跑 QC")
        if report is None:
            print(f"[qc] '{sd.name}': 未跑 QC(无 qc_report.json)— 保留"
                  "(建议先跑 scripts/qc.py)")
            kept.append(sd)
        elif str(report.get("level", "INFO")).upper() == "ERROR":
            msgs = _qc_error_messages(report)
            print(f"[qc] 排除 '{sd.name}': QC 等级 ERROR")
            for m in msgs[:5]:
                print(f"       · {m}")
            if len(msgs) > 5:
                print(f"       · … 共 {len(msgs)} 处 ERROR")
        else:
            kept.append(sd)
    return kept


def write_qc_meta(out: Path, sessions: list[dict]) -> None:
    """把每个 episode 对应源会话的 qc_report.json 汇总进 meta/qc_reports.jsonl。

    一行一个 episode,episode_index 与数据集一致:``session`` 为源会话
    目录名,``level`` 为报告整体等级(无报告为 null),``qc_report`` 是
    报告原文(findings/streams 明细全保留)。数据集的 QC 结论随数据走,
    不用再回源目录查。
    """
    lines = []
    for i, info in enumerate(sessions):
        p = info["dir"] / "qc_report.json"
        report = None
        if p.exists():
            try:
                report = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
        lines.append(json.dumps({
            "episode_index": i,
            "session": info["dir"].name,
            "level": str(report["level"]).upper() if report else None,
            "qc_report": report,
        }, ensure_ascii=False))
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "qc_reports.jsonl").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    n_with = sum(1 for l in lines if json.loads(l)["qc_report"] is not None)
    print(f"[meta] qc_reports.jsonl: {n_with}/{len(lines)} 个 episode 有报告")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python scripts/pack_daily.py --date 2026-09-14 --force",
    )
    p.add_argument("shift", nargs="?", choices=["day", "night"], default=None,
                   help="可选预设:day → --source data/session-day;"
                        "night → --source data/session-night(默认 data/)")
    p.add_argument("--date", default=None,
                   help="日期 YYYY-MM-DD(默认当天)")
    p.add_argument("--session-dir", type=Path, default=None,
                   help="批次根目录(等价 --source;--out 未指定时自动为 "
                        "data/lerobot/<目录名>/<日期>-起-止)")
    p.add_argument("--source", type=Path, default=None,
                   help="批次根目录覆盖(默认 data/;day/night 预设则 "
                        "data/session-<shift>)")
    p.add_argument("--out", type=Path, default=None,
                   help="输出数据集目录(默认 data/lerobot/<日期>;预设为 "
                        "data/lerobot/session-<shift>/<日期>);最终数据集"
                        "自动落在 <out>/<日期>-起-止 子目录")
    p.add_argument("--force", action="store_true",
                   help="输出目录已存在时先删除")
    p.add_argument("--full", action="store_true",
                   help="使用整段录制,忽略 RUN_START..RUN_END 标记窗口")
    p.add_argument("--keep-images", action="store_true",
                   help="保留中间 PNG 帧(默认编码后删除)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="只转换前 N 个会话(调试用)")
    args = p.parse_args(argv)
    args.out_auto = args.out is None     # 自动命名 → 原地追加数据起止时刻;
                                         # 指定 --out → 进一层 <out>/<日期>-起-止

    if args.date is None:
        args.date = dt.date.today().isoformat()
    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        p.error(f"无效日期: {args.date!r} (应为 YYYY-MM-DD)")
    if args.session_dir is not None:
        if args.shift:
            p.error("--session-dir 与 day/night 预设只能二选一")
        if args.source is not None:
            p.error("--source 与 --session-dir 只能二选一")
        args.source = args.session_dir
    if args.source is None:
        args.source = (PROJECT_ROOT / "data" / f"session-{args.shift}"
                       if args.shift else PROJECT_ROOT / "data")
    if args.out is None:
        if args.shift:
            args.out = (PROJECT_ROOT / "data" / "lerobot"
                        / f"session-{args.shift}" / args.date)
        elif args.source != PROJECT_ROOT / "data":
            # 指定了批次根(--source/--session-dir):输出按其目录名归档
            args.out = (PROJECT_ROOT / "data" / "lerobot"
                        / args.source.name / args.date)
        else:
            args.out = PROJECT_ROOT / "data" / "lerobot" / args.date
    return args


def _session_span(sd: Path, date: str) -> tuple[float, float] | None:
    """会话的数据时间范围(PC 时钟秒)。

    优先 qc_report.json 的 marker 窗口(真实起止);没有报告/窗口则回退
    目录名 ``<日期>-HH-MM-SS`` 里的开始时刻(起止相同)。
    """
    try:
        w = json.loads((sd / "qc_report.json").read_text(encoding="utf-8")
                       ).get("window")
        if w and w.get("t0") and w.get("t1"):
            return float(w["t0"]), float(w["t1"])
    except Exception:
        pass
    try:
        hh, mm, ss = (int(x) for x in sd.name.split("-")[-3:])
        t = dt.datetime.strptime(
            f"{date} {hh:02d}:{mm:02d}:{ss:02d}",
            "%Y-%m-%d %H:%M:%S").timestamp()
        return t, t
    except Exception:
        return None


def rename_out_by_span(args, sessions: list[dict]) -> None:
    """输出目录自动带数据起止标记 ``<日期>-<起>-<止>``(取全部打包会话
    的数据时间范围,HH-MM-SS)。

    自动命名(--out 未指定):``.../<日期>`` 原地改名为
    ``.../<日期>-<起>-<止>``;显式指定 ``--out`` 时,在给定目录下再进
    一层,数据集落在 ``<out>/<日期>-<起>-<止>``。没有可解析的起止
    (无 qc 窗口且目录名不含时刻)时保持原输出目录。
    """
    spans = [s for s in (_session_span(i["dir"], args.date)
                         for i in sessions) if s]
    if not spans:
        return
    t0 = min(s[0] for s in spans)
    t1 = max(s[1] for s in spans)
    stamp = (f"{dt.datetime.fromtimestamp(t0).strftime('%H-%M-%S')}"
             f"-{dt.datetime.fromtimestamp(t1).strftime('%H-%M-%S')}")
    if args.out_auto:
        args.out = args.out.parent / f"{args.out.name}-{stamp}"
    else:
        args.out = args.out / f"{args.date}-{stamp}"
    print(f"[out] 输出目录(按数据起止): {args.out}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sessions = find_sessions(args.source, args.date)
    if not sessions:
        print(f"[error] '{args.source}' 下没有匹配 {args.date}-* 的会话目录")
        return 1
    sessions = filter_qc_errors(sessions)
    if not sessions:
        print(f"[error] {args.date} 的会话全部被 QC 判为 ERROR — 没有可打包数据")
        return 1
    if args.max_episodes:
        sessions = sessions[: args.max_episodes]
    print(f"[input] {len(sessions)} 个会话: "
          + ", ".join(s.name for s in sessions[:5])
          + (" ..." if len(sessions) > 5 else ""))

    video_slots = discover_video_slots(sessions)
    if not video_slots:
        print("[error] 所有会话都没有可用的视频流")
        return 1
    print(f"[video] 视频槽位: {list(video_slots)}")

    if not args.full:
        missing_win = [s.name for s in sessions if load_marker_window(s) == (None, None)]
        if missing_win:
            print(f"[warn] 以下会话没有 marker 窗口,将使用整段录制: {missing_win}")

    # ---- 预扫:每个会话窗口过滤后真正有数据的特征 → 并集即全模态特征集 ----
    print(f"[probe] 预扫 {len(sessions)} 个会话的可用特征(只读时间戳,不装载数据值)")
    probed: list[dict] = []
    for i, sd in enumerate(sessions, 1):
        print(f"[probe] {i}/{len(sessions)}  {sd.name} ...", flush=True)
        info = _probe_session(sd, video_slots, args)
        if info is not None:
            probed.append(info)
    sessions = probed
    if not sessions:
        print("[error] 没有可用的会话(合成不出主时间轴或无数据流)")
        return 1

    # ---- 特征集 = 全部模态(跨会话并集),缺任一模态的会话整体剔除 ----
    # (不取交集,避免稀缺席位被个别会话拖没)。一轮剔除即收敛:留下的会话
    # 本就覆盖旧并集,并集剔除后只会缩小。这样注册进 meta 的特征在每个
    # episode 都写得出文件(save_episode 的 update_video_info 不会去找缺失
    # 的视频),数据集也不丢模态。
    required: set[str] = set()
    for info in sessions:
        required |= info["present"]
    kept: list[dict] = []
    for info in sessions:
        missing = sorted(required - info["present"])
        if missing:
            print(f"[spec] 剔除 '{info['dir'].name}': 缺少模态 {missing}")
        else:
            kept.append(info)
    sessions = kept
    if not sessions:
        print("[error] 没有会话同时具备全部模态 — 无法打包")
        return 1
    common = required

    specs = build_feature_specs([info["dir"] for info in sessions],
                                video_slots, common)
    if not specs:
        print("[error] 所有会话都没有可用数据流")
        return 1
    print(f"[spec] {len(specs)} 个特征: " + ", ".join(specs))

    # 输出目录:按这段数据的起止时刻落名(--out 未指定时改名,指定时
    # 进一层 <out>/<日期>-起-止),再查重
    rename_out_by_span(args, sessions)
    if args.out.exists():
        if not args.force:
            print(f"[error] 输出目录已存在: {args.out} (用 --force 覆盖)")
            return 1
        shutil.rmtree(args.out)

    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=args.out.name, fps=MASTER_FPS,
        features=specs, root=args.out, use_videos=True,
    )

    # 数据集 meta:采集程序版本 + 每槽位硬件显示名
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import embodied_brain_collect
    from mf_lerobot.utils import write_info
    ds.meta.info["collect_version"] = embodied_brain_collect.__version__
    ds.meta.info["hardware"] = _hardware_names([i["dir"] for i in sessions])
    write_info(ds.meta.info, ds.root)

    video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]
    from embodied_brain_collect.session import environment as env_mod

    for i, info in enumerate(sessions):
        session_dir = info["dir"]
        task_label = load_task_label(session_dir, env_mod)
        streams = [s for s in load_parquet_streams(session_dir)
                   if s[0] in common]
        write_episode(ds, session_dir, task_label, info["master"], streams,
                      info["win"], video_slots)
        if not args.keep_images:
            _cleanup_images(args.out, i, video_keys)
        print(f"[episode {i}] '{session_dir.name}' task='{task_label}' "
              f"{len(info['master'])} frames")

    # meta 附上每个 episode 的 qc report(源会话 qc_report.json)
    write_qc_meta(args.out, sessions)

    print(f"[done] {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames → {args.out}")
    return 0


def _probe_session(sd: Path, video_slots, args) -> dict | None:
    """预扫一个会话:窗口、主时间轴、窗口过滤后真正有数据的特征集合。

    返回 None = 该会话不可用(合成不出主时间轴或一个特征都没有)。
    """
    win = load_marker_window(sd) if not args.full else (None, None)
    master_abs = make_master_timeline(win, sd, video_slots)
    if master_abs is None or not len(master_abs):
        print(f"[warn] 跳过 '{sd.name}': 无 marker 窗口且无相机帧,"
              "无法合成主时间轴")
        return None

    t0 = float(master_abs[0])
    present: set[str] = set()
    for key, ts, vals, _names in load_parquet_streams(sd):
        if len(ts) and (_window_mask(ts, win) & (ts >= t0)).any():
            present.add(key)

    # 视频完整性:每路视频的容器帧数必须覆盖窗口内最后一条时间戳,
    # 否则视为视频截断/长度异常 —— 该会话整体跳过
    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        npz = sd / slot / npz_name
        mp4 = sd / slot / mp4_name
        if not (npz.exists() and mp4.exists()):
            continue
        try:
            cam_ts = np.load(npz, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            continue
        in_win = np.where((_window_mask(cam_ts, win)) & (cam_ts >= t0))[0]
        if not len(in_win):
            continue
        nb = ffprobe_count(mp4)
        need = int(in_win[-1]) + 1                     # 窗口末帧的容器序号(1:1)
        if nb < need:
            print(f"[video] 跳过 '{sd.name}': {slot} 视频只有 {nb} 帧,"
                  f"窗口需要到第 {need} 帧 — 视频被截断/长度异常,整条数据跳过")
            return None
        present.add(video_feature_key(slot))

    if not present:
        print(f"[warn] 跳过 '{sd.name}': 窗口内没有任何特征数据")
        return None
    return {"dir": sd, "win": win, "master": master_abs, "present": present}


if __name__ == "__main__":
    sys.exit(main())
