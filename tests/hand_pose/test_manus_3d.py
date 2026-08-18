"""3D hand skeleton visualization for MANUS gloves via matplotlib.

Usage:  python -m tests.hand_pose.test_manus_3d

Press Q in the plot window to stop.
"""

import threading
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from src.recorders.hand_pose import ManusHandPoseRecorder, HandPoseRecorderConfig
from src.recorders.hand_pose.manus_hand_pose_recorder import (
    _ERGO_INDEX, _ERGO_SIDE_OFFSET,
)
from tests.base import SESSION_DIR


def main():
    cfg = HandPoseRecorderConfig(session_dir=f"{SESSION_DIR}/hand_pose")
    rec = ManusHandPoseRecorder(cfg)
    if not rec._open():
        # _open() blocks until BOTH gloves are connected and fails otherwise
        print(f"[3d] open failed — {rec._open_error}")
        return

    fig = plt.figure(figsize=(10, 9))
    fig.canvas.manager.set_window_title("MANUS Hand Skeleton 3D — Q to stop")

    # One 3D subplot per glove, plus a 40-ch ergo panel across the bottom.
    n_gloves = len(rec._glove_ids)
    cols = min(n_gloves, 2)
    rows = (n_gloves + cols - 1) // cols
    gs = GridSpec(rows + 1, cols, figure=fig,
                  height_ratios=[3.0] * rows + [1.2])
    axes = {}
    for i, gid in enumerate(rec._glove_ids):
        ax = fig.add_subplot(gs[i // cols, i % cols], projection='3d')
        side = rec._glove_sides.get(gid, "?")
        ax.set_title(f"Glove {gid} ({side})", fontsize=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        axes[gid] = ax

    ax_ergo = fig.add_subplot(gs[rows, :])
    ax_ergo.set_title("ergonomics 40-ch (L 0-19 | R 20-39)", fontsize=10)
    ax_ergo.set_xticks(range(0, 40, 5))
    ax_ergo.set_ylim(-100, 100)
    ax_ergo.tick_params(labelsize=6)

    running = True

    def on_key(e):
        nonlocal running
        if e.key == 'q':
            running = False
    fig.canvas.mpl_connect('key_press_event', on_key)

    # ---- background poll ----
    stop_event = threading.Event()
    latest_data: dict[int, dict] = {}
    data_lock = threading.Lock()

    def poll_loop():
        t0 = time.time()
        while not stop_event.is_set():
            rec._poll(time.time() - t0)
            if rec._pub is not None:
                for gid in rec._glove_ids:
                    data = rec._pub.GetGloveData(gid)
                    if data is not None:
                        with data_lock:
                            latest_data[gid] = data
            time.sleep(0.005)

    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    print("[3d] Rendering... (press Q in plot window to stop)")

    try:
        while running:
            with data_lock:
                snapshot = dict(latest_data)

            for gid, ax in axes.items():
                data = snapshot.get(gid)
                ax.clear()
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

                if data is None:
                    continue

                nodes = data.get("raw_nodes")
                if not nodes:
                    continue

                positions = {n["id"]: np.array(n["position"]) for n in nodes}
                parent_map = {n.get("id"): n.get("parentId") for n in nodes}

                # Parent IDs = leaf nodes (nobody's parent)
                all_pids = {n.get("parentId") for n in nodes
                            if n.get("parentId") is not None}
                leaf_ids = set(positions.keys()) - all_pids

                # ---- nodes ----
                for nid, pos in positions.items():
                    is_leaf = nid in leaf_ids
                    ax.scatter(*pos, c='#e8a020' if is_leaf else '#5577bb',
                              s=30 if is_leaf else 18, edgecolors='none')

                # ---- connections ----
                for nid, pos in positions.items():
                    pid = parent_map.get(nid)
                    if pid is not None and pid in positions:
                        ppos = positions[pid]
                        ax.plot([ppos[0], pos[0]],
                                [ppos[1], pos[1]],
                                [ppos[2], pos[2]],
                                c='#888888', linewidth=0.8)

                # ---- auto-camera ----
                all_pos = np.stack(list(positions.values()))
                center = all_pos.mean(axis=0)
                radius = float(np.percentile(
                    np.linalg.norm(all_pos - center, axis=1), 95)) * 1.3
                for dim, lim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                    lim(center[dim] - radius, center[dim] + radius)

            # ---- 40-dim ergo angles, straight from the snapshots ----
            flat = np.zeros(40, dtype=np.float32)
            have_ergo = False
            for data in snapshot.values():
                if data is None:
                    continue
                ergo = data.get("ergonomics")
                if not ergo:
                    continue
                have_ergo = True
                # entry["type"] is side-agnostic; the snapshot only carries
                # this glove's own side — offset left to 0-19, right to 20-39.
                offset = _ERGO_SIDE_OFFSET.get(data.get("side", "Left"), 0)
                for entry in ergo:
                    idx = _ERGO_INDEX.get(entry["type"], -1)
                    if idx >= 0:
                        flat[offset + idx] = entry["value"]
            if have_ergo:
                ax_ergo.clear()
                ax_ergo.set_ylim(-100, 100)
                ax_ergo.set_xticks(range(0, 40, 5))
                ax_ergo.bar(range(40), flat, width=0.85,
                            color=["#5577bb"] * 20 + ["#e8a020"] * 20)
                ax_ergo.set_title("ergonomics 40-ch (L 0-19 | R 20-39)",
                                  fontsize=10)

            plt.pause(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t.join(timeout=1.0)
        rec._close()
        plt.close()
        print("[3d] stopped.")


if __name__ == "__main__":
    main()
