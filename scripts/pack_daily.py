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
ego_headband/{name}.mp4       observation.images.headband_<name>_rgb
                              4 路鱼眼,截段对齐主时间轴
wristband                     observation.wristband_pressure       3 路,150 Hz(设备钟)
wristband                     observation.wristband_ppg            3 路,100 Hz
wristband                     observation.wristband_imu            6 路,50 Hz
wristband                     observation.wristband_vitals         温度/血氧,1 Hz 值
eeg                           observation.eeg                      按通道数
eeg(阻抗门禁)                observation.eeg_impedance            单行,timestamp=0;
                                                                   每通道均值 Ω + 门禁汇总
emg_left/right                observation.emg_left/right           8 通道
emg_*(imu)                    observation.imu_left/right           6 路
eye(gaze)                     observation.gaze                     2 路
eye(imu)                      observation.eye_imu                  6 路
ego_headband(imu)             observation.headband_imu<j>          6 路
ego_headband/microphone.wav   observation.headband_microphone      16kHz PCM,按块对齐
hand_pose                     observation.hand_pose/_skeleton_*    40d / 50x3 / 50x4
position                      observation.<role|deviceN>_pose      每台 6 维,角色名 = role_serial_map
position                      observation.state                    追踪器位姿
marker                        episode 窗口(RUN_START..RUN_END)
                              + 独立 30Hz 主时间轴(RUN_START 为 0 点)
============================  ==================================  =====================
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:                      # 无 tqdm 时退化为无进度条
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# mf_lerobot(带 torch 全家桶,冷启动 10s+)不在模块级导入 —— 预扫阶段
# 只需要 numpy;重依赖推迟到真正创建数据集时(write_* 内按需取路径常量),
# 让打包一启动就先出 [probe]/[warn] 日志,而不是静默卡十几秒。
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

# 头环麦克风(dtype=audio):PCM 本体只在 ego_headband/microphone.wav,npz
# 里是逐块索引元数据(设备侧 sample_index、wav 内采样位置、read_complete_ns)
MICROPHONE_KEY = "observation.headband_microphone"

# EEG 阻抗门禁结果(Curry open 时触发一次,随 eeg.npz 落盘的 eeg_impedance_*):
# 单行常量流 —— 每 episode 一个独立 parquet、一行,列 = 各通道均值(Ω,
# 顺序同 eeg_channel_names)+ 门禁汇总(pass_rate / gate_pass / n_snapshots)。
# 时间戳与时间轴无关,固定 0(CONSTANT_TS 哨兵):窗口掩码直接放行,
# 写帧不做平移。BrainCo / 关门禁的会话 npz 没有这些字段,不产出该流。
EEG_IMPEDANCE_KEY = "observation.eeg_impedance"
_EEG_IMP_SUMMARY = ["pass_rate", "gate_pass", "n_snapshots"]
CONSTANT_TS = 0.0

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
# 全部位姿特征 observation.<role|deviceN>_pose(角色名来自
# role_serial_map,历史数据 device0..2)/ observation.headband_imu<j>
# 由 _is_windowed 的后缀/前缀规则覆盖
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
    return (key in WINDOWED_FEATURES
            or key.endswith("_pose")     # <role|deviceN>_pose,含 hand_pose
            or key.startswith("observation.headband_imu"))


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


def _headband_camera_names(ego_dir: Path) -> list[str]:
    """头环槽位的相机名:``ego_headband.npz`` 里每个 ``{name}_timestamps``
    对应一路 ``{name}.mp4``。npz 里的键序取决于录制端各路解码线程的先来后到,
    这里排序保证特征顺序跨会话稳定。"""
    p = ego_dir / "ego_headband.npz"
    if not p.exists():
        return []
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as exc:
        print(f"[video] 跳过头环: ego_headband.npz 不可读 ({exc})")
        return []
    return sorted(str(k)[:-len("_timestamps")] for k in z.files
                  if k.endswith("_timestamps"))


def _headband_imu_ids(z) -> list[int]:
    """头环 npz 里的 IMU 序号(键 ``imu{j}_ts/gyro/accel``,升序)。"""
    ids = []
    for k in z.files:
        if k.startswith("imu") and k.endswith("_ts"):
            try:
                ids.append(int(k[len("imu"):-len("_ts")]))
            except ValueError:
                continue
    return sorted(ids)


def load_microphone_index(session_dir: Path) -> dict | None:
    """头环麦克风的逐块索引(只读 npz 元数据,不碰 wav — 预扫用)。

    返回 None = 该会话没有可用音频(未启用 / 无数据 / 录制报错 / 无采集
    时间戳)。块起始时刻只取 ``microphone_stamp_ns`` —— 录制端按 ALSA 采样
    位置校验过的首采样 CLOCK_REALTIME 采集时刻;不回退 read_complete 估计:
    缺字段、或有任何一块没有有效时间戳(旧数据)的会话直接跳过音频。
    设备侧 sample_index 不连续只告警不阻塞 —— 块时间戳逐一真实,缺口只是
    内容缺失,不影响对齐。
    """
    p = session_dir / "ego_headband" / "ego_headband.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
    except Exception as exc:
        print(f"[audio] 跳过麦克风: ego_headband.npz 不可读 ({exc})")
        return None
    if "microphone_error" in z.files and str(z["microphone_error"]):
        print(f"[audio] 跳过麦克风: {str(z['microphone_error'])}")
        return None
    need = {"microphone_sample_index", "microphone_wav_sample_offset",
            "microphone_samples", "microphone_sample_rate",
            "microphone_channels", "microphone_sample_width",
            "microphone_stamp_ns", "microphone_has_capture_timestamp"}
    if not need <= set(z.files):
        print(f"[audio] 跳过麦克风: npz 无逐块采集时间戳字段(旧数据)— "
              "音频只按 microphone_stamp_ns 打包,不做 read_complete 估计")
        return None
    width = int(z["microphone_sample_width"])
    if width != 2:
        print(f"[audio] 跳过麦克风: 采样宽度 {width}B 非 PCM16")
        return None
    lens = z["microphone_samples"].astype(np.int64)
    index = z["microphone_sample_index"].astype(np.int64)
    offsets = z["microphone_wav_sample_offset"].astype(np.int64)
    rate = int(z["microphone_sample_rate"])
    st = z["microphone_stamp_ns"].astype(np.int64)
    has = np.asarray(z["microphone_has_capture_timestamp"]).astype(bool)
    if rate <= 0 or not len(lens):
        return None
    if (st.size != len(lens) or has.size != len(lens)
            or not has.all() or not (st > 0).all()
            or not (np.diff(st) > 0).all()):
        n_ok = int((has & (st > 0)).sum())
        print(f"[audio] 跳过麦克风: 采集时间戳缺失/无效"
              f"({n_ok}/{len(lens)} 块有效)— 不用 read_complete 估计")
        return None
    n_break = int(np.sum(np.diff(index) != lens[:-1])) if len(index) > 1 else 0
    if n_break:
        print(f"[audio] 麦克风块不连续:{n_break} 处缺口 — 照常打包,"
              "缺口处内容缺失")
    return {
        "rate": rate,
        "channels": int(z["microphone_channels"]),
        "lens": lens,
        "offsets": offsets,
        # 块首采样时刻 = 采集时间戳(硬件时刻),严格不估计
        "start": st / 1e9,
    }


def load_microphone_chunks(session_dir: Path, mic: dict) -> list[np.ndarray] | None:
    """按索引从 microphone.wav 切出每块 PCM → float32 [-1,1]。

    mono 每块 ``(n_i,)``;多声道 ``(n_i, channels)``(wav 交错存储)。
    切片以 npz 的 ``wav_sample_offset`` 为准,所以旧会话 wav 头部的
    无元数据预热段(门控修复前)天然被跳过。读不出 wav 返回 None。
    """
    wav_p = session_dir / "ego_headband" / "microphone.wav"
    if not wav_p.exists():
        print("[audio] microphone.wav 缺失 — 跳过音频")
        return None
    try:
        with wave.open(str(wav_p), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    except Exception as exc:
        print(f"[audio] microphone.wav 读不出 — 跳过音频 ({exc})")
        return None
    ch = max(mic["channels"], 1)
    total = pcm.size // ch
    chunks = []
    for off, n in zip(mic["offsets"], mic["lens"]):
        lo, hi = int(off) * ch, (int(off) + int(n)) * ch
        if hi > total:
            print(f"[audio] 索引越界(块尾 {hi} > wav {total} 采样)— 截到末尾")
            hi = min(hi, total)
        block = pcm[lo:hi].astype(np.float32)
        block = block.reshape(-1, ch) if ch > 1 else block
        chunks.append(block / 32768.0)
    return chunks


def _window_mask(ts: np.ndarray, win: tuple[float | None, float | None]) -> np.ndarray:
    mask = np.ones(len(ts), dtype=bool)
    if win[0] is not None:
        mask &= ts >= win[0]
    if win[1] is not None:
        mask &= ts <= win[1]
    return mask


def _is_constant_stream(ts: np.ndarray) -> bool:
    """单行常量流(eeg 阻抗门禁):timestamp 为 0 哨兵,与时间轴无关。"""
    return len(ts) == 1 and ts[0] == CONSTANT_TS


def _stream_in_window(ts: np.ndarray, win: tuple[float | None, float | None],
                      t0: float) -> bool:
    """预扫用:该流在窗口内是否有数据;常量单行流恒在。"""
    if not len(ts):
        return False
    if _is_constant_stream(ts):
        return True
    return bool((_window_mask(ts, win) & (ts >= t0)).any())


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

        # 阻抗门禁结果与波形对齐无关(开录时刻的一次性检测),没有 PC
        # 时间戳的会话也照样带出来
        if "eeg_impedance_ohm" in z.files:
            imp = np.asarray(z["eeg_impedance_ohm"],
                             dtype=np.float32).reshape(1, -1)
            names = [str(s) for s in z["eeg_channel_names"][: imp.shape[1]]]
            summary = np.asarray(
                [[float(z["eeg_impedance_pass_rate"]),
                  float(bool(z["eeg_impedance_check_pass"])),
                  float(z["eeg_impedance_n_snapshots"])]], dtype=np.float32)
            out_extra.append((
                EEG_IMPEDANCE_KEY,
                np.asarray([CONSTANT_TS], dtype=np.float64),
                np.hstack([imp, summary]).astype(np.float32),
                names + _EEG_IMP_SUMMARY))

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

    # 头环 — 板载 IMU(四路鱼眼视频走视频流);时间戳为设备同步钟(UTC 秒),
    # 本身严格递增,不用像串口流那样铺开
    p = session_dir / "ego_headband" / "ego_headband.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        for j in _headband_imu_ids(z):
            out_extra.append((
                f"observation.headband_imu{j}",
                z[f"imu{j}_ts"].astype(np.float64),
                np.hstack([z[f"imu{j}_accel"],
                           z[f"imu{j}_gyro"]]).astype(np.float32),
                IMU_NAMES))

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

    # 位置追踪 — 每台设备独立特征 observation.<role>_pose (x, y, z, r, p, y);
    # 角色名来自 npz 的 roles 字段(recorder 开录时按 recorders.yaml 的
    # role_serial_map 写入),如 left_wrist / right_wrist / chest;历史/dummy
    # 数据没有 roles 时回退 device<i>。容忍变体:pos (T,3)|(T,D,3)、rpy 可缺
    p = session_dir / "position" / "position.npz"
    pose_streams = []          # (特征名, ts, 值, 列前缀) —— npz 列序
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
            pose_names = _pose_names(z.get("roles"), n_dev)
            for i, (feat, prefix) in enumerate(pose_names):
                vals = np.concatenate(
                    [pos[:, i, :], rpy[:, i, :]], axis=1).astype(np.float32)
                pose_streams.append((feat, ts, vals, prefix))
                out_extra.append((feat, ts, vals,
                                  ["x", "y", "z", "roll", "pitch", "yaw"]))

    # state / action —— 统一由各 tracker 位姿(left_wrist/right_wrist/chest
    # …,历史数据 device0..2)与 hand_pose 拼接:
    # 3 台设备 × (x, y, z, roll, pitch, yaw) + 40 手部关节 = 58 维。
    # 时间轴取 hand_pose(组内最高频),device 位姿按最近邻取样对齐。
    hand = next((s for s in out_extra if s[0] == "observation.hand_pose"), None)
    if hand and pose_streams:
        _, ts_h, v_h, hand_names = hand
        cols, names = [], []
        for feat, ts_p, v_p, prefix in pose_streams:   # 追加序 = npz 列序
            idx = np.searchsorted(ts_p, ts_h).clip(1, len(ts_p) - 1)
            prev = np.clip(idx - 1, 0, len(ts_p) - 1)
            use_prev = np.abs(ts_p[prev] - ts_h) < np.abs(ts_p[idx] - ts_h)
            cols.append(v_p[np.where(use_prev, prev, idx)])
            names += [f"{prefix}_{c}"
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
        if "eeg_impedance_ohm" in z.files:
            n_imp = int(np.asarray(z["eeg_impedance_ohm"]).reshape(-1).size)
            imp_names = ([str(s) for s in z["eeg_channel_names"][:n_imp]]
                         + _EEG_IMP_SUMMARY)
            out.append((EEG_IMPEDANCE_KEY,
                        np.asarray([CONSTANT_TS], dtype=np.float64),
                        imp_names, len(imp_names)))

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

    # 头环 IMU(视频流走视频槽位发现)
    p = session_dir / "ego_headband" / "ego_headband.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        for j in _headband_imu_ids(z):
            out.append((f"observation.headband_imu{j}",
                        z[f"imu{j}_ts"].astype(np.float64), IMU_NAMES, 6))

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

    # 位置 — 特征名与 load_parquet_streams 同口径(角色优先,回退 device<i>)
    p = session_dir / "position" / "position.npz"
    pose_feats: list[tuple[str, str]] = []       # (特征名, 列前缀) —— npz 列序
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "timestamps_s" in z.files:
            ts = z["timestamps_s"].astype(np.float64)
            pos = z["positions_m"]
            n_dev = pos.shape[1] if pos.ndim == 3 else 1
            for feat, prefix in _pose_names(z.get("roles"), n_dev):
                pose_feats.append((feat, prefix))
                out.append((feat, ts,
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
    hand = next((s for s in out if s[0] == "observation.hand_pose"), None)
    if hand and pose_feats:
        names = _state_channel_names([prefix for _, prefix in pose_feats],
                                     hand[2])
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


def _pose_names(roles, n_dev: int) -> list[tuple[str, str]]:
    """每台 tracker -> (特征名, state 列前缀);纯函数,便于离线测试。

    roles 是 position.npz 里的角色数组(recorder 按 recorders.yaml 的
    role_serial_map 写入,如 left_wrist / right_wrist / chest):特征 =
    ``observation.<role>_pose``,state 通道 = ``<role>_x`` …。历史 npz 没有
    roles、长度与设备数不符、角色名含非法字符/重复时,该台回退
    ``device<i>`` —— 与旧命名兼容,且永不产生重复特征名。
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    usable = (roles is not None and len(np.asarray(roles).reshape(-1)) == n_dev)
    for i in range(n_dev):
        role = ""
        if usable:
            role = str(np.asarray(roles).reshape(-1)[i])
        role = "".join(c if (c.isalnum() or c == "_") else "_"
                       for c in role).strip("_")
        if not role or role in seen:
            role = f"device{i}"
        seen.add(role)
        out.append((f"observation.{role}_pose", role))
    return out


def _state_channel_names(prefixes: list[str],
                         hand_names: list[str] | None) -> list[str]:
    """state/action 的列名:设备位姿 ×3 + 40 手部关节,共 58。

    ``prefixes`` 按设备列序给出(left_wrist / device0 …),与位姿特征的
    追加顺序一致 —— 不再从特征名反解设备号。
    """
    names = []
    for prefix in prefixes:
        names += [f"{prefix}_{c}"
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

def discover_video_slots(session_dirs: list[Path]) -> dict[str, list[tuple[str, str, str, str]]]:
    """扫描会话目录 → {槽位目录: [(npz 名, 时间戳键, mp4 名, 特征后缀)]}。

    约定:``<slot>/<slot>.npz`` 含 ``frames_timestamps``、同目录 ``frames.mp4``;
    eye 槽位特例为 ``eye.npz`` / ``scene_timestamps`` / ``eye.mp4``;
    ego_headband 一槽多路 —— ``ego_headband.npz`` 里每个 ``{name}_timestamps``
    对应同目录 ``{name}.mp4``,特征后缀 ``headband_<name>``。
    """
    slots: dict[str, list[tuple[str, str, str, str]]] = {}
    for sd in session_dirs:
        for d in sorted(sd.iterdir()) if sd.is_dir() else []:
            if not d.is_dir():
                continue
            slot = d.name
            if slot in slots:
                continue
            if slot == "ego_headband":
                entries = [( "ego_headband.npz", f"{name}_timestamps",
                             f"{name}.mp4", f"headband_{name}")
                           for name in _headband_camera_names(d)
                           if (d / f"{name}.mp4").exists()]
                if entries:
                    slots[slot] = entries
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
            suffix = slot[4:] if slot.startswith("cam_") else slot
            slots[slot] = [(npz_name, ts_key, mp4_name, suffix)]
    # 主时钟优先 cam_head(dict 顺序 = info.json 的 master_feature 顺序)
    return dict(sorted(slots.items(), key=lambda kv: kv[0] != "cam_head"))


# ── 特征规格与会话窗口 ─────────────────────────────────────────────────────

def video_feature_key(suffix: str) -> str:
    """视频特征后缀 → 特征键:head → observation.images.head_rgb。"""
    return f"observation.images.{suffix}_rgb"


def probe_video_shape(mp4: Path) -> tuple[int, int, int]:
    import av
    with av.open(str(mp4)) as container:
        frame = next(container.decode(video=0))
        return frame.height, frame.width, 3


def build_feature_specs(sessions: list[Path],
                        video_slots: dict[str, list[tuple[str, str, str, str]]],
                        common: set[str]) -> dict[str, dict]:
    """跨会话的流并集(限定在预扫交集 ``common`` 内);shape/rate 从第一个
    含该流的会话探测。"""
    specs: dict[str, dict] = {}

    def _first_session_with(stream_dir: str) -> Path | None:
        for sd in sessions:
            if (sd / stream_dir).is_dir():
                return sd
        return None

    for slot, entries in video_slots.items():
        sd = _first_session_with(slot)
        if sd is None:
            continue
        for npz_name, ts_key, mp4_name, suffix in entries:
            if video_feature_key(suffix) not in common:
                continue
            mp4, npz = sd / slot / mp4_name, sd / slot / npz_name
            if not (mp4.exists() and npz.exists()):
                print(f"[spec] '{slot}/{mp4_name}' 有目录但 npz/mp4 缺失 — 跳过该特征")
                continue
            try:
                h, w, c = probe_video_shape(mp4)
            except Exception as exc:
                print(f"[spec] '{slot}/{mp4_name}': 无法解码 ({exc}) — 跳过该特征")
                continue
            specs[video_feature_key(suffix)] = {
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
            # 常量单行流没有速率可言,spec 的 fps 取主轴帧率(避免 0)
            rate = (MASTER_FPS if _is_constant_stream(ts)
                    else _stream_rate(ts))
            specs[key] = {
                "dtype": "float32",
                "shape": (vals.shape[1],),
                "names": names,
                "fps": rate,
                "tolerance_s": max(0.005, 3.0 / max(rate, 1.0)),
            }
            if _is_windowed(key):
                specs[key]["window"] = WINDOW

    # 麦克风(dtype=audio):取第一个有音频的会话探测块长/采样率。
    # validate_features 要求 dtype/shape/names/sample_rate。
    if MICROPHONE_KEY in common and MICROPHONE_KEY not in specs:
        for sd in sessions:
            mic = load_microphone_index(sd)
            if mic is None:
                continue
            specs[MICROPHONE_KEY] = {
                "dtype": "audio",
                "sample_rate": mic["rate"],
                "shape": (int(mic["lens"][0]),),
                "names": None,
            }
            break
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
        for npz_name, ts_key, _mp4_name, _suffix in video_slots[slot]:
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


# 单个编码任务的 ffmpeg 线程上限:并发 4 路 × 2 线程恰好铺满 4 核 8 线程
# 的采集机。此前每路 libx264 默认开约 1.5×逻辑核的线程,4 路并发互相踩踏
# 把整机打满,是"打包跑很久且系统卡死"的直接原因。帧线程模式下线程数
# 不改变编码结果。
_FFMPEG_THREADS = max(1, (os.cpu_count() or 4) // 4)


def _smart_cut_enabled() -> bool:
    """EB_PACK_SMART_CUT=0/false/no/off 关闭智能截段,强制全量重编码。"""
    return os.environ.get("EB_PACK_SMART_CUT", "").strip().lower() not in (
        "0", "false", "no", "off")


def _run_ffmpeg(args: list[str]) -> None:
    """跑一条 ffmpeg 命令;失败时抛 RuntimeError 携带 stderr 尾部。"""
    p = subprocess.run(
        [media_tool("ffmpeg"), "-y", "-loglevel", "error", *args],
        capture_output=True)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError("; ".join(tail[-4:]) or f"rc={p.returncode}")


def _probe_stream(mp4: Path) -> dict:
    """视频流容器元数据:codec、重排帧数、帧率、时基、像素格式。"""
    p = subprocess.run(
        [media_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=codec_name,has_b_frames,r_frame_rate,time_base,pix_fmt",
         "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    st = (json.loads(p.stdout).get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "").partition("/")
    tb_num, _, tb_den = (st.get("time_base") or "").partition("/")
    try:
        st["fps"] = float(num) / float(den)
        st["tb"] = (int(tb_num), int(tb_den))
    except (ValueError, ZeroDivisionError):
        st["fps"], st["tb"] = 0.0, (0, 0)
    return st


def _packet_pts(mp4: Path) -> list[tuple[int, bool]]:
    """容器包序的 (pts, 是否关键帧);有无 pts 的包(损坏容器)时抛错。"""
    p = subprocess.run(
        [media_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_packets", "-show_entries", "packet=pts,flags",
         "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    out: list[tuple[int, bool]] = []
    for pkt in json.loads(p.stdout).get("packets") or []:
        pts = pkt.get("pts")
        if pts is None:
            raise RuntimeError(f"{mp4.name}: 存在无 pts 的包")
        out.append((int(pts), "K" in (pkt.get("flags") or "")))
    return out


def _smart_cut(src_mp4: Path, out_mp4: Path, i0: int, n: int, fps: int) -> None:
    """智能截段:头部边角重编码 + 其余整 GOP 逐比特流拷贝。

    录制端契约(FFmpegWriter/FFmpegJpegWriter:CFR、bframes=0、
    keyint=fps、timescale 90000)成立时,窗口 [i0, i1] 的截段不必整段
    重编码:

    - 头段 [i0, k):k 为 i0 起第一个关键帧,不足一个 GOP,libx265 按
      录制端同款参数重编码(bframes=0);
    - 中尾段 [k, i1]:自关键帧起的整 GOP 流拷贝,尾部按包数精确截断
      (闭合 GOP + 无 B 帧,截断前缀自洽可解码)。

    两段经 MPEG-TS 中转拼接:TS 在关键帧前重发带内参数集,最终 mp4 以
    hev1 tag 落盘(允许带内参数集)—— 头段重编码与源码流参数集不同
    (编码器版本差异)也能各自可解码。输出与全量重编码同契约:CFR、
    PTS = i/fps、无 B 帧,中段无二次编码损失,编码量从整段降到一个
    GOP 以内。任何前提不成立或校验不过都抛异常,由调用方回退全量
    重编码。
    """
    st = _probe_stream(src_mp4)
    tb_num, tb_den = st["tb"]
    try:
        n_bf = int(st.get("has_b_frames"))
    except (TypeError, ValueError):
        n_bf = -1
    if (st.get("codec_name") != "hevc" or n_bf != 0
            or st["fps"] != float(fps) or not tb_den
            or tb_den % (tb_num * fps)):
        raise RuntimeError("容器不满足录制端契约(非 hevc/有 B 帧/非整帧时基)")
    ticks = tb_den // (tb_num * fps)        # 每帧 pts 步进,90000/30 = 3000
    pkts = _packet_pts(src_mp4)
    i1 = i0 + n - 1
    if not (0 <= i0 <= i1 < len(pkts)):
        raise RuntimeError("窗口超出容器范围")
    if any(pts != j * ticks for j, (pts, _k) in enumerate(pkts)):
        raise RuntimeError("包序 pts 不在 CFR 整数网格上")
    kf = [j for j, (_pts, k) in enumerate(pkts[i0:], start=i0) if k]
    if not kf:
        raise RuntimeError("窗口内没有关键帧")
    k = min(kf)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    head = out_mp4.with_suffix(".head.ts")
    rest = out_mp4.with_suffix(".rest.ts")
    lst = out_mp4.with_suffix(".concat.txt")
    try:
        if k > i0:
            # 头段重编码:镜像录制端输出参数(no-info 免去 5KB 编码器
            # 信息 SEI),像素格式沿用源流;TS 封装自带带内参数集
            _run_ffmpeg([
                "-ss", f"{i0 / fps:.6f}", "-i", str(src_mp4),
                "-frames:v", str(k - i0),
                "-c:v", "libx265", "-crf", "23", "-preset", "medium",
                "-x265-params", "bframes=0:no-info=1",
                "-pix_fmt", str(st["pix_fmt"]),
                "-r", str(fps), "-fps_mode", "cfr",
                "-g", str(fps), "-keyint_min", str(fps),
                "-threads", str(_FFMPEG_THREADS),
                "-f", "mpegts", str(head)])
        _run_ffmpeg([
            "-ss", f"{k / fps:.6f}", "-i", str(src_mp4),
            "-frames:v", str(i1 - k + 1),
            "-map", "0:v:0", "-c", "copy", "-f", "mpegts", str(rest)])
        # concat demuxer 相对清单文件解析条目路径 → 必须写绝对路径
        lst.write_text(
            "".join(f"file '{p.resolve().as_posix()}'\n"
                    for p in (head, rest) if p.exists()),
            encoding="utf-8")
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                     "-map", "0:v:0", "-c", "copy", "-tag:v", "hev1",
                     "-video_track_timescale", str(tb_den), str(out_mp4)])
        # 校验与全量重编码同口径:帧数精确、pts 严格 CFR 整数网格。
        # 解码自验聚焦拼接风险区(解码错误不改 ffmpeg 返回码,以进度行
        # 的实际输出帧数为准):
        # - 头段重编码与拷贝段的参数集切换点:从头解 [0, 头段+2个GOP),
        #   精确数帧 —— SPS 不衔接在这里表现为解码中断在头段末尾;
        # - 尾段截断点:输入 seek 到窗口末尾前 3s,精确解出 90 帧。
        # 中段是源码流的逐比特拷贝,且源已通过上面的包序网格校验;
        # 整段解码自验在 4 路并发时占 ~70s,聚焦后每路 <3s
        _q = _packet_pts(out_mp4)
        if len(_q) != n or any(pts != j * ticks
                               for j, (pts, _k) in enumerate(_q)):
            raise RuntimeError(f"截段校验失败({len(_q)}/{n} 帧,网格不齐)")
        jn = min(n, (k - i0) + 60)            # 头段 + 拷贝段头两个 GOP
        p = subprocess.run(
            [media_tool("ffmpeg"), "-threads", str(_FFMPEG_THREADS),
             "-i", str(out_mp4), "-map", "0:v:0",
             "-frames:v", str(jn), "-f", "null", "-"], capture_output=True)
        frames = re.findall(rb"frame=\s*(\d+)", p.stderr)
        n_dec = int(frames[-1]) if frames else 0
        if p.returncode != 0 or n_dec != jn:
            raise RuntimeError(f"拼接点解码自验失败({n_dec}/{jn} 帧)")
        if jn < n:
            tail = 90
            p2 = subprocess.run(
                [media_tool("ffmpeg"), "-threads", str(_FFMPEG_THREADS),
                 "-ss", f"{(n - tail) / fps:.6f}", "-i", str(out_mp4),
                 "-map", "0:v:0", "-frames:v", str(tail),
                 "-f", "null", "-"], capture_output=True)
            f2 = re.findall(rb"frame=\s*(\d+)", p2.stderr)
            n_tail = int(f2[-1]) if f2 else 0
            if p2.returncode != 0 or n_tail != tail:
                raise RuntimeError(f"尾段解码自验失败({n_tail}/{tail} 帧)")
    finally:
        for p in (head, rest, lst):
            p.unlink(missing_ok=True)


def write_video_ffmpeg(src_mp4: Path, out_mp4: Path, seek_s: float,
                       dur_s: float, fps: int) -> int:
    """ffmpeg 截段 + fps 归一化;返回输出帧数。

    默认智能截段(见 ``_smart_cut``):录制端契约成立时只重编码头部不足
    一个 GOP 的边角,其余整 GOP 流拷贝。回退路径为全量重编码(原行为,
    仅新增 ``-bf 0`` 与线程上限),两者输出同契约:CFR、PTS = i/fps、
    无 B 帧,保持录制端"解码第 i 帧 == 第 i 个时间戳"的帧序不变量。
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n, i0 = round(dur_s * fps), round(seek_s * fps)
    if n > 0 and _smart_cut_enabled():
        try:
            _smart_cut(src_mp4, out_mp4, i0, n, fps)
            return n
        except Exception as exc:
            print(f"[video] 智能截段回退全量重编码: {exc}")
            out_mp4.unlink(missing_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    cmd = [media_tool("ffmpeg"), "-y", "-loglevel", "error",
           "-ss", f"{max(0.0, seek_s):.3f}", "-i", str(src_mp4),
           "-t", f"{max(0.0, dur_s):.3f}",
           "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
           "-bf", "0",
           "-crf", "18", "-pix_fmt", "yuv420p", "-threads", str(_FFMPEG_THREADS),
           str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    n_out = ffprobe_count(tmp)
    tmp.replace(out_mp4)
    return n_out


def write_video_timestamps(ds, key: str, ep_idx: int, rel_ts: np.ndarray) -> None:
    """写视频特征的时间戳 parquet(与 VideoFeature.save 同路径同格式)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mf_lerobot.utils import DEFAULT_DATA_PATH

    ep_chunk = ep_idx // 1000
    fpath = ds.root / DEFAULT_DATA_PATH.format(
        episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=key)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "timestamp": pa.array(rel_ts, type=pa.float64()),
        "episode_index": pa.array(
            np.full(len(rel_ts), ep_idx, dtype=np.int64), type=np.int64()),
    })
    pq.write_table(table, fpath, compression="snappy")


def _decode_frames_seq(path: Path, start: int, count: int) -> list[np.ndarray]:
    """顺序解码版(帧号 = 解码序):兼容任何容器,量大的容器较慢。"""
    import av
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= start:
                out.append(frame.to_ndarray(format="gray").astype(np.float32))
                if len(out) >= count:
                    break
    return out


def _decode_frames(path: Path, start: int, count: int) -> list[np.ndarray]:
    """容器 [start, start+count) 的灰度帧;不足则截断。

    先 seek 到 start 附近的关键帧再向前解码(录制契约下 pts×fps 即帧号),
    不再从第 0 帧解起:长录制里窗口靠后时,自验解码量从整段降到 GOP 量级。
    帧号推算出现不连续(丢包/非 CFR 容器)时退回顺序解码,行为与原实现
    一致。
    """
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0)
        tb = float(stream.time_base)
        if rate > 0 and tb > 0:
            target = max(0.0, start / rate - 1.5)   # keyint ≤1s,留 1.5s 余量
            container.seek(int(target / tb), stream=stream, backward=True)
            out: list[np.ndarray] = []
            expect: int | None = None
            for frame in container.decode(video=0):
                j = round(frame.pts * tb * rate)
                if expect is not None and j != expect:
                    return _decode_frames_seq(path, start, count)
                expect = j + 1
                if j >= start:
                    out.append(
                        frame.to_ndarray(format="gray").astype(np.float32))
                    if len(out) >= count:
                        return out
            return out
    return _decode_frames_seq(path, start, count)


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

    截段走 write_video_ffmpeg(默认智能截段:整 GOP 流拷贝 + 头部边角
    重编码,不满足录制端契约时全量重编码),输出统一 CFR、无 B 帧。
    实际写出的帧数与窗口帧数不一致时丢弃该视频流并告警 —— 时间戳
    parquet 与容器帧必须逐帧 1:1,宁缺毋错。

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

    n_out = write_video_ffmpeg(src_mp4, out_mp4, i0 / fps, n / fps, fps)

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
    if n_out != len(rel_ts):
        print(f"[video] {key}: 截段实际写出 {n_out} 帧,与窗口帧数 "
              f"{len(rel_ts)} 不一致 — 时间戳无法逐帧对齐,丢弃该视频流,"
              "其余模态继续打包")
        out_mp4.unlink(missing_ok=True)
        return 0
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
        if _is_constant_stream(ts):
            # 单行常量流(如 eeg 阻抗门禁):不过窗口、不平移,
            # rel timestamp 固定 0 原样落一行
            ts_w, vals_w = ts, vals
            rel = ts.astype(np.float64)
        else:
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
    from mf_lerobot.utils import DEFAULT_VIDEO_PATH
    for slot, entries in video_slots.items():
        for npz_name, ts_key, mp4_name, suffix in entries:
            npz_p = session_dir / slot / npz_name
            mp4_p = session_dir / slot / mp4_name
            if not (npz_p.exists() and mp4_p.exists()):
                continue
            try:
                all_ts = np.load(npz_p, allow_pickle=True)[ts_key].astype(np.float64)
            except Exception:
                continue
            key = video_feature_key(suffix)
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

    # 音频:按块投喂 AudioFeature(时间戳 = 块首采样),窗口裁剪按块起始。
    # mf_lerobot 在 save_episode 时统一写 wav + 索引 parquet 并校验一致性。
    # 本会话没有音频时不喂 —— 走下面的 dropped 机制跳过该特征。
    mic = load_microphone_index(session_dir)
    if mic is not None and MICROPHONE_KEY in ds._features:
        chunks = load_microphone_chunks(session_dir, mic)
        if chunks is not None:
            mask = _window_mask(mic["start"], win) & (mic["start"] >= t0)
            sel = np.where(mask)[0]
            if len(sel):
                for i in sel:
                    ds.add_frame(MICROPHONE_KEY, chunks[i],
                                 float(mic["start"][i] - t0))
                present.add(MICROPHONE_KEY)
                print(f"  {MICROPHONE_KEY}: {len(sel)} 块 @{mic['rate']}Hz "
                      f"{mic['channels']}ch")

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


def filter_meta_status(sessions: list[Path]) -> list[Path]:
    """按会话 meta.yaml 顶层 ``status`` 过滤:非 ``success`` 的整条排除。

    status 是录制端收尾时写的录制结局(success/failed/error…),操作员
    判失败或录制中途出错的会话不进数据集。没有 meta.yaml 或没有 status
    字段的会话视为元数据缺失,保留并提示(与 QC 过滤同一保守口径)。
    """
    kept: list[Path] = []
    for sd in sessions:
        p = sd / "meta.yaml"
        status = None
        if p.exists():
            try:
                import yaml
                meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                status = meta.get("status")
            except yaml.YAMLError as exc:
                print(f"[meta] '{sd.name}': meta.yaml 不可读 ({exc}) — 保留")
                kept.append(sd)
                continue
        if status is None:
            print(f"[meta] '{sd.name}': 无 meta.yaml / 无 status 字段 — 保留")
            kept.append(sd)
        elif str(status).strip().lower() != "success":
            print(f"[meta] 排除 '{sd.name}': status={status} (≠ success)")
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


def write_collect_meta(out: Path, sessions: list[dict]) -> None:
    """数据集的元数据边车:所有额外信息逐 episode 写 meta/collect_info.jsonl。

    一行一个 episode,顶层平铺:

    - ``episode_index`` / ``session``:与数据集 episode 对齐,session 为
      源会话目录名;
    - ``collect_version``:采集程序版本;
    - ``hardware``:该会话实际采集的槽位 → 设备显示名(逐会话,不再是
      跨会话并集);
    - ``collector_id`` 等操作员维护键(session.yaml 顶层 + run_session CLI
      覆盖的解析结果,框架保留键除外;旧会话包在 ``collect:`` 块下的键
      自动解包);
    - ``status``:录制结局(success/failed,来自 session meta.yaml);
    - ``task_name``:两种模式都有(图纸模式取图纸 config.yaml 的 task);
    - ``notes``:图纸 config.yaml 的 ``scene.notes`` 版权备注(随开录固化
      在 session meta.yaml 顶层;旧会话按 environment 路径重建;无则 null);
    - ``scene`` / ``objects``:图纸模式的物体摆放(scene = config.yaml 的
      ``scene.name``;objects = 物体列表,name/color/shape/dims 来自
      config.yaml,cx/cy/ang 来自 placements.csv。优先用开录时固化在
      session meta.yaml 顶层的快照;旧会话的包壳快照自动拆包,再旧按
      environment 路径重建);tasks 模式为 null。

    info.json 保持 mf_lerobot 写下的 LeRobot 标准字段,不再携带任何额外
    信息 —— 读数据集时以本文件为元数据入口。
    """
    from embodied_brain_collect.session import environment as env_mod
    from embodied_brain_collect.session.config import FRAMEWORK_KEYS

    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import embodied_brain_collect

    # 单独处理、不进"其余操作员键"循环的 meta 键
    _special = {"collect", "objects", "status", "notes"}

    lines = []
    for i, info in enumerate(sessions):
        meta: dict = {}
        p = info["dir"] / "meta.yaml"
        if p.exists():
            try:
                meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        line = {
            "episode_index": i,
            "session": info["dir"].name,
            "collect_version": embodied_brain_collect.__version__,
            "hardware": _hardware_names([info["dir"]]),
        }
        for k, v in meta.items():
            if k not in FRAMEWORK_KEYS and k not in _special and v is not None:
                line[k] = v
        legacy = meta.get("collect")         # 旧方案:包在 collect 块下的键解包并入
        if isinstance(legacy, dict):
            for k, v in legacy.items():
                line.setdefault(k, v)
        line["status"] = meta.get("status")

        # 场景与物体摆放:新快照 = meta 顶层 scene/objects(list)/task_name;
        # 旧快照 = objects 包了一层(task/scene/num/combo/rep/objects);
        # 都没有时按 environment 路径按当前 configs 原地重建。
        # notes(图纸 scene.notes)同路:优先 meta 快照,缺失走重建
        scene = meta.get("scene")
        task_name = meta.get("task_name")
        objects = meta.get("objects")
        notes = meta.get("notes")
        if isinstance(objects, dict):        # 旧快照格式
            scene = scene or objects.get("scene")
            task_name = task_name or objects.get("task")
            objects = objects.get("objects")
        if not isinstance(objects, list):
            objects = None
        if (objects is None or not notes) and meta.get("environment"):
            layout = env_mod.scene_layout(meta["environment"]) or None
            if layout:
                scene = scene or layout["scene"]
                task_name = task_name or layout["task_name"]
                notes = notes or layout.get("notes")
                if objects is None:
                    objects = layout["objects"]
        line["scene"] = scene or None
        line["objects"] = objects
        line["task_name"] = task_name or None
        line["notes"] = notes or None
        lines.append(json.dumps(line, ensure_ascii=False))

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "collect_info.jsonl").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    n_status = sum(1 for l in lines if json.loads(l)["status"])
    print(f"[meta] collect_info.jsonl: {len(lines)} 个 episode "
          f"(status 已标 {n_status})")


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
    p.add_argument("--save-threads", type=int, default=8,
                   help="save_episode 阶段特征级并行线程数(1=串行;"
                        "每个特征写各自的 parquet/wav,默认 8)")
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
    sessions = filter_meta_status(sessions)
    if not sessions:
        print(f"[error] {args.date} 的会话全部被 QC / meta status 判为不可打包"
              " — 没有可打包数据")
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
        # 阻抗门禁单行流是附属信息而非模态:装备不支持(BrainCo)或关门禁
        # 的会话不因别人有它而被整体剔除 —— 缺它自己的 episode 走
        # write_episode 的 dropped 机制跳过该特征,其余模态照常打包
        missing = sorted((required - info["present"]) - {EEG_IMPEDANCE_KEY})
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

    if "mf_lerobot" not in sys.modules:    # 已被预热/上一条加载过则免提示
        print("[pack] 加载打包依赖(mf_lerobot / torch,冷启动约 10s)…",
              flush=True)
    from mf_lerobot import MultiFrequencyLeRobotDataset
    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=args.out.name, fps=MASTER_FPS,
        features=specs, root=args.out, use_videos=True,
        save_threads=args.save_threads,
    )

    # info.json 保持 mf_lerobot 写下的 LeRobot 标准字段 —— 采集版本/硬件/
    # 采集信息等额外信息一律进 meta/collect_info.jsonl(见 write_collect_meta)

    video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]
    from embodied_brain_collect.session import environment as env_mod

    for i, info in enumerate(sessions):
        session_dir = info["dir"]
        task_label = load_task_label(session_dir, env_mod)
        streams = [s for s in load_parquet_streams(session_dir)
                   if s[0] in common]
        # 写侧异常直接中止(不是重复检查):视频完整性已在 QC 把关,能到这里
        # 的会话都过了门槛;若仍写崩(典型:未跑 QC 的历史数据带坏视频),
        # 半写的 episode 继续追加会产出损坏数据集 —— 中止并保留现场比
        # 硬跳过更安全,重跑前先补 QC。
        try:
            write_episode(ds, session_dir, task_label, info["master"], streams,
                          info["win"], video_slots)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] '{session_dir.name}' 写入失败: "
                  f"{type(exc).__name__}: {exc} — 中止打包(该会话建议先跑 "
                  "scripts/qc.py 复核;数据目录未动)")
            return 1
        if not args.keep_images:
            _cleanup_images(args.out, i, video_keys)
        print(f"[episode {i}] '{session_dir.name}' task='{task_label}' "
              f"{len(info['master'])} frames")

    # meta 附上每个 episode 的 qc report(源会话 qc_report.json)
    write_qc_meta(args.out, sessions)
    # meta 附上每个 episode 的采集信息(session.yaml 顶层键 + CLI 覆盖快照)
    write_collect_meta(args.out, sessions)

    print(f"[done] {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames → {args.out}")
    return 0


def _probe_session(sd: Path, video_slots, args) -> dict | None:
    """预扫一个会话:窗口、主时间轴、窗口过滤后真正有数据的特征集合。

    返回 None = 该会话不可用(合成不出主时间轴或一个特征都没有)。
    """
    win = load_marker_window(sd) if not args.full else (None, None)
    if not args.full and win[0] is None:
        # 与 QC 同一硬门槛:没有 RUN_START/RUN_END 的会话不可对齐,绝不
        # 静默回退相机区间打包(那会产出看似正常、实则无法对齐的数据)。
        # --full 是显式全量模式,保持相机区间回退。
        print(f"[error] 跳过 '{sd.name}': marker 无 RUN_START/RUN_END 窗口"
              " — 无法对齐(数据需重采;确要全量打包用 --full)")
        return None
    master_abs = make_master_timeline(win, sd, video_slots)
    if master_abs is None or not len(master_abs):
        print(f"[warn] 跳过 '{sd.name}': 无 marker 窗口且无相机帧,"
              "无法合成主时间轴")
        return None

    t0 = float(master_abs[0])
    present: set[str] = set()
    for key, ts, vals, _names in load_parquet_streams(sd):
        if _stream_in_window(ts, win, t0):
            present.add(key)

    mic = load_microphone_index(sd)
    if mic is not None:
        m = _window_mask(mic["start"], win) & (mic["start"] >= t0)
        if m.any():
            present.add(MICROPHONE_KEY)

    # 视频流发现:窗口内有时间戳的视频槽位进特征集合。容器完整性与帧数
    # 覆盖**只在 QC 检查一次**(ContainerIntegrity / FrameCountMatch,
    # ERROR 即被 filter_qc_errors 挡在打包之外),probe 不重复确认 ——
    # 能走到这里的会话都过了 QC 门槛。
    for slot, entries in video_slots.items():
        for npz_name, ts_key, mp4_name, suffix in entries:
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
            present.add(video_feature_key(suffix))

    if not present:
        print(f"[warn] 跳过 '{sd.name}': 窗口内没有任何特征数据")
        return None
    return {"dir": sd, "win": win, "master": master_abs, "present": present}


if __name__ == "__main__":
    sys.exit(main())
