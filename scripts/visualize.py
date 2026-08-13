"""Visualize session NPZ files in one big figure."""

import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/session1")
    files = sorted(session.glob("*/*.npz"))
    names = [f.parent.name for f in files]
    data = {n: dict(np.load(p)) for n, p in zip(names, files)}

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f"Session: {session}", fontsize=16)
    gs = GridSpec(6, 6, figure=fig, hspace=0.5, wspace=0.4)

    # ---- row 0: camera frames + eye scene ----
    frames = data.get("camera", {}).get("frames")
    if frames is not None:
        n = len(frames)
        for j, i in enumerate([0, n//3, 2*n//3, n-1]):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(frames[i])
            ax.set_title(f"camera #{i}", fontsize=9)
            ax.axis('off')

    frames = data.get("eye", {}).get("scene_frames")
    if frames is not None:
        n = len(frames)
        for j, i in enumerate([0, n//2, n-1]):
            ax = fig.add_subplot(gs[0, 3+j])
            ax.imshow(frames[i])
            ax.set_title(f"eye scene #{i}", fontsize=9)
            ax.axis('off')

    # ---- row 1-2: EMG 8ch + IMU ----
    emg = data.get("emg", {}).get("emg_data")
    if emg is not None:
        for ch in range(8):
            ax = fig.add_subplot(gs[1 + ch//4, ch%4])
            ax.plot(emg[:500, ch], linewidth=0.3)
            ax.set_ylabel(f"EMG{ch}", fontsize=6)
            ax.tick_params(labelsize=5)

    gyro = data.get("emg", {}).get("imu_gyro")
    if gyro is not None:
        ax = fig.add_subplot(gs[1, 4])
        ax.plot(gyro[:500], linewidth=0.5)
        ax.legend(["gx","gy","gz"], fontsize=5)
        ax.set_title("EMG IMU gyro", fontsize=9)
    accel = data.get("emg", {}).get("imu_accel")
    if accel is not None:
        ax = fig.add_subplot(gs[1, 5])
        ax.plot(accel[:500], linewidth=0.5)
        ax.legend(["ax","ay","az"], fontsize=5)
        ax.set_title("EMG IMU accel", fontsize=9)

    # ---- row 2: eye gaze + IMU ----
    gaze = data.get("eye", {}).get("gaze_xy")
    if gaze is not None:
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(gaze[:2000, 0], gaze[:2000, 1], linewidth=0.2)
        ax.set_title(f"eye gaze XY ({gaze.shape[0]})", fontsize=9)

    gyro = data.get("eye", {}).get("imu_gyro")
    if gyro is not None:
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(gyro[:500], linewidth=0.5)
        ax.legend(["gx","gy","gz"], fontsize=5)
        ax.set_title("eye IMU gyro", fontsize=9)
    accel = data.get("eye", {}).get("imu_accel")
    if accel is not None:
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(accel[:500], linewidth=0.5)
        ax.legend(["ax","ay","az"], fontsize=5)
        ax.set_title("eye IMU accel", fontsize=9)

    # ---- row 2: position ----
    pos = data.get("position", {}).get("positions_m")
    if pos is not None:
        ax = fig.add_subplot(gs[2, 3])
        ax.plot(pos[:, 0], pos[:, 1], linewidth=0.5)
        ax.set_title(f"position XY ({pos.shape[0]})", fontsize=9)
    quat = data.get("position", {}).get("quaternions_wxyz")
    if quat is not None:
        ax = fig.add_subplot(gs[2, 4])
        ax.plot(quat[:500], linewidth=0.5)
        ax.legend(["w","x","y","z"], fontsize=5)
        ax.set_title("quaternion", fontsize=9)

    # ---- row 2: tactile ----
    tac = data.get("tactile_glove", {}).get("glove_data")
    if tac is not None:
        ax = fig.add_subplot(gs[2, 5])
        ax.plot(tac[:500, :5], linewidth=0.3)
        ax.set_title(f"tactile ({tac.shape[0]}×{tac.shape[1]})", fontsize=9)

    # ---- row 3-4: hand_pose skeleton joint positions ----
    skel = data.get("hand_pose", {}).get("skeleton_positions")
    if skel is not None and skel.ndim == 3:
        T, N, _ = skel.shape
        for j in range(min(N, 12)):
            row = 3 + j // 6
            ax = fig.add_subplot(gs[row, j % 6])
            n_pts = min(T, 3000)
            ax.plot(skel[:n_pts, j, 0], linewidth=0.3, label='X')
            ax.plot(skel[:n_pts, j, 1], linewidth=0.3, label='Y')
            ax.plot(skel[:n_pts, j, 2], linewidth=0.3, label='Z')
            ax.set_title(f"joint {j}", fontsize=7)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.legend(fontsize=4, loc='upper right')

    # ---- row 5: hand_pose ergo + markers ----
    hp = data.get("hand_pose", {}).get("ergo_data")
    if hp is not None:
        n_chan = hp.shape[1]
        for ch in range(min(n_chan, 6)):
            ax = fig.add_subplot(gs[5, ch])
            ax.plot(hp[:1000, ch], linewidth=0.3)
            ax.set_title(f"ergo ch{ch}", fontsize=7)
            ax.tick_params(labelsize=5)

    codes = data.get("markers", {}).get("marker_code")
    if codes is not None:
        start_col = 0 if hp is None else 3
        ax = fig.add_subplot(gs[5, start_col:])
        ax.vlines(range(len(codes)), 0, codes)
        ax.set_title(f"markers ({len(codes)})", fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
