"""
将所有传感器数据按统一时间轴组合成 ML-ready 数据集。

本脚本演示两种方式：
  A. 用 resample_aligned.py 重采样到统一频率 → 直接 np.load 即用
  B. 保留原始采样率 → 按时间窗口查找最近帧

运行:
  python record/tools/assemble_dataset.py data/2026-07-27_14-00-36_subj01_t36_run18_p1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ============================================================
# 方式 A: 重采样后的统一网格 (最直接, 推荐)
# ============================================================

def load_resampled_grid(session_dir: str, hz: int = 120):
    """
    加载 resample_aligned.py 生成的统一网格。

    返回的数组每一行对应一个时间点, 所有模态已对齐到同一时间轴。
    """
    path = Path(session_dir) / "aligned" / "resampled" / f"grid_{hz}hz.npz"
    return dict(np.load(path, allow_pickle=False))


def grid_to_ml_array(session_dir: str):
    """
    把 120Hz 网格中的各模态拼成一个大数组, 直接用于 ML。

    返回: (X, columns)
      X: (N_frames, total_features) 的 float64 数组
      columns: 每个特征的名称列表
    """
    g120 = load_resampled_grid(session_dir, 120)
    t = g120["timestamps_pc"]

    blocks = []   # (名称, 数据) 的列表
    col_names = []

    # 眼动凝视 (2D)
    if "gaze_xy" in g120:
        blocks.append(g120["gaze_xy"])
        col_names += ["gaze_x", "gaze_y"]

    # 眼动 IMU (6D)
    if "imu_gyro" in g120:
        blocks.append(g120["imu_gyro"])
        col_names += ["eye_gyro_x", "eye_gyro_y", "eye_gyro_z"]
    if "imu_accel" in g120:
        blocks.append(g120["imu_accel"])
        col_names += ["eye_accel_x", "eye_accel_y", "eye_accel_z"]

    # EMG (8D)
    if "emg" in g120:
        blocks.append(g120["emg"])
        col_names += [f"emg_ch{c+1}" for c in range(g120["emg"].shape[1])]

    # EMG IMU (6D)
    if "emg_imu_gyro" in g120:
        blocks.append(g120["emg_imu_gyro"])
        col_names += ["emg_gyro_x", "emg_gyro_y", "emg_gyro_z"]
    if "emg_imu_accel" in g120:
        blocks.append(g120["emg_imu_accel"])
        col_names += ["emg_accel_x", "emg_accel_y", "emg_accel_z"]

    # VIVE (来自 60Hz, 需要上采样到 120Hz)
    g60 = load_resampled_grid(session_dir, 60)
    if "positions_m" in g60:
        # 上采样到 120Hz: np.interp
        t60 = g60["timestamps_pc"]
        pos_up = np.empty((len(t), 3, 3))
        for tracker in range(3):
            for axis in range(3):
                pos_up[:, tracker, axis] = np.interp(t, t60,
                    g60["positions_m"][:, tracker, axis])
        # flatten: 3 trackers x 3 axes = 9
        pos_flat = pos_up.reshape(len(t), -1)
        blocks.append(pos_flat)
        for trk in range(3):
            for ax in ["x", "y", "z"]:
                col_names.append(f"vive_t{trk}_{ax}")

    if blocks:
        X = np.column_stack(blocks)
        return X, col_names, t
    return None, None, None


# ============================================================
# 方式 B: 原始采样率 + 按 trial 切窗口 (保留最高精度)
# ============================================================

def extract_trial_data(session_dir: str):
    """
    从 aligned.npz 按 trial 切出各模态的原始数据。

    不同于方式 A, 这里保留每个模态的原始采样率,
    通过 PC 时钟对齐。适合需要高精度时间序列分析的场景。
    """
    d = dict(np.load(Path(session_dir) / "aligned" / "aligned.npz"))

    # 找到 trial 边界
    codes = d["marker_code"]
    t_pc  = d["marker_t_pc"]

    trials = []
    for i in range(len(codes)):
        if codes[i] == 0x51:   # EXEC_START
            t0 = t_pc[i]
        elif codes[i] == 0x52: # EXEC_END
            t1 = t_pc[i]
            trials.append({"exec_start": t0, "exec_end": t1})

    if not trials:
        print("  无 trial 数据")
        return

    for idx, trial in enumerate(trials):
        t0, t1 = trial["exec_start"], trial["exec_end"]
        print(f"\n  Trial {idx+1}  EXEC [{t1-t0:.2f}s]:")

        # 各模态在执行窗口内的原始数据
        for name, ts_key, data_key in [
            ("EMG",      "emg_emg_timestamps",      "emg_emg_data"),
            ("Eye gaze", "eye_gaze_timestamps_pc",   "eye_gaze_xy"),
            ("VIVE",     "vive_timestamps_s",        "vive_positions_m"),
            ("Manus",    "manus_ergo_timestamps",     "manus_ergo_data"),
        ]:
            if ts_key not in d or data_key not in d:
                continue
            ts = d[ts_key]
            mask = (ts >= t0) & (ts < t1)
            n = mask.sum()
            if n > 0:
                hz = n / (t1 - t0) if (t1 - t0) > 0 else 0
                print(f"    {name:<12s}: {n:>6d} 帧 @ {hz:.1f} Hz")


# ============================================================
# main
# ============================================================

def main(session_dir: str):
    print("=" * 60)
    print("方式 A: 统一重采样网格 (120Hz)")
    print("=" * 60)

    X, cols, t = grid_to_ml_array(session_dir)
    if X is not None:
        print(f"\n  ML 数组: {X.shape[0]:,} 帧 × {X.shape[1]} 特征")
        print(f"  时间跨度: {t[-1] - t[0]:.1f} 秒")
        print(f"  特征列表: {cols}")
        print(f"\n  使用方式:")
        print(f"    X, cols, t = grid_to_ml_array('{session_dir}')")
        print(f"    # X[trial_mask] 取某个 trial 的数据")
        print(f"    # X 可以直接输入 sklearn / pytorch")

        # ---- 按 trial 切分示例 ----
        g120 = load_resampled_grid(session_dir, 120)
        mc = g120.get("marker_code", np.array([]))
        mt = g120.get("marker_t_pc", np.array([]))

        exec_starts = mt[mc == 0x51]
        exec_ends   = mt[mc == 0x52]

        print(f"\n  按 trial 切分:")
        for i, (t0, t1) in enumerate(zip(exec_starts, exec_ends)):
            mask = (t >= t0) & (t < t1)
            n = mask.sum()
            print(f"    Trial {i+1}: {n} 帧 ({n/120:.1f}s) @ 120Hz")

        # ---- 保存为单个 .npy ----
        out = Path(session_dir) / "aligned" / "resampled" / "ml_features.npy"
        np.save(out, X)
        with open(out.with_suffix(".columns.txt"), "w") as f:
            f.write("\n".join(cols))
        print(f"\n  已保存: {out}")
        print(f"  列名: {out.with_suffix('.columns.txt')}")

    print("\n" + "=" * 60)
    print("方式 B: 原始采样率 (保留最高精度)")
    print("=" * 60)
    extract_trial_data(session_dir)

    print("\n" + "=" * 60)
    print("文件清单")
    print("=" * 60)
    p = Path(session_dir) / "aligned" / "resampled"
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file():
                print(f"  {f.name:<30s}  {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python record/tools/assemble_dataset.py <session_dir>")
        sys.exit(1)
    main(sys.argv[1])
