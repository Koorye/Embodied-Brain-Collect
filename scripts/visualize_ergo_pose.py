"""Render recorded hand_pose data as a 3D pose video.

The 3D view matches the live ``tests.hand_pose.test_manus_3d`` style:
skeleton nodes (blue joints, orange leaf nodes) with gray parent-child
bones and a per-frame auto-follow camera, plus the 40-ch ergonomics bar
panel beneath.

Two data sources, chosen automatically:

  * skeleton mode (default when present): ``skeleton_positions`` +
    ``skeleton_node_ids`` + ``skeleton_parent_ids`` from the npz;
  * FK mode (fallback, or ``--fk``): a forward-kinematics hand model
    built from ``ergo_data`` (0-19 left, 20-39 right).  Used for npz files
    recorded before the recorder saved skeleton topology.

Usage:
  python scripts/visualize_ergo_pose.py data/session1/hand_pose/hand_pose.npz
      [-o pose.mp4] [--fps 30] [--start 0] [--end -1] [--every 1] [--fk]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # offscreen render
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---- FK hand model (fallback for old npz files) ----------------------------

# Per-finger channel offsets within one hand's 20 channels:
# (spread, mcp stretch, pip stretch, dip stretch)
_FINGERS = {
    "thumb":  (0, 1, 2, 3),
    "index":  (4, 5, 6, 7),
    "middle": (8, 9, 10, 11),
    "ring":   (12, 13, 14, 15),
    "pinky":  (16, 17, 18, 19),
}
_BASES = {
    "thumb":  np.array([-0.055, 0.0, 0.015]),
    "index":  np.array([-0.032, 0.0, 0.0]),
    "middle": np.array([0.0, 0.0, 0.0]),
    "ring":   np.array([0.032, 0.0, 0.0]),
    "pinky":  np.array([0.058, 0.0, 0.0]),
}
_THUMB_YAW = -50.0
_SEGMENTS = (0.050, 0.030, 0.020)  # mcp->pip, pip->dip, dip->tip (meters)

_JOINT = "#5577bb"   # internal nodes
_LEAF = "#e8a020"    # leaf nodes (same palette as test_manus_3d)
_BONE = "#888888"

# === BEGIN GENERATED TOPOLOGY (python scripts/dump_skeleton_topology.py --write) ===
# Metaglove raw-skeleton topology, one glove, in CoreSdk_GetRawSkeletonData
# order.  Identical for every glove of the same model, so it is hardcoded
# instead of stored per session.
_NODES_PER_GLOVE = 25  # 0 = not dumped yet; skeleton mode falls back to FK
_NODE_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=np.int32)
_PARENT_IDS = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23], dtype=np.int32)  # -1 = root
# === END GENERATED TOPOLOGY ===


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def hand_pose_nodes(ch: np.ndarray, side: str) -> np.ndarray | None:
    """Forward-kinematics node positions for one hand's 20 channels."""
    if ch.shape[0] < 20 or np.isnan(ch[:20]).any():
        return None
    sgn = -1.0 if side == "left" else 1.0
    nodes: list[np.ndarray] = []
    for finger, (sp, mc, pi, di) in _FINGERS.items():
        base = _BASES[finger] * np.array([sgn, 1.0, 1.0])
        yaw = (_THUMB_YAW + ch[sp]) * sgn if finger == "thumb" else sgn * ch[sp]
        pts = [base]
        d = _rot_y(yaw) @ np.array([0.0, 0.0, 1.0])
        p = base
        for bend, length in zip((ch[mc], ch[pi], ch[di]), _SEGMENTS):
            d = _rot_x(bend) @ d
            p = p + length * d
            pts.append(p)
        nodes.extend(pts)
    return np.array(nodes)


# ---- drawing (same visual language as test_manus_3d) -----------------------

def _auto_camera(ax, positions: np.ndarray) -> None:
    """Per-frame auto-follow camera: center + 95th-percentile radius."""
    if len(positions) < 2:
        return
    center = positions.mean(axis=0)
    radius = float(np.percentile(
        np.linalg.norm(positions - center, axis=1), 95)) * 1.3
    radius = max(radius, 0.02)
    for dim, lim in enumerate((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d)):
        lim(center[dim] - radius, center[dim] + radius)


def _draw_nodes(ax, pos: np.ndarray, colors, sizes, segs) -> None:
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=sizes,
               edgecolors="none", depthshade=False)
    if segs:
        ax.add_collection(Line3DCollection(segs, colors=_BONE,
                                           linewidths=0.8))


def draw_skeleton(ax, positions: np.ndarray,
                  ids: np.ndarray, parents: np.ndarray
                  ) -> np.ndarray | None:
    """Draw one glove's recorded skeleton nodes + bones.

    ``positions`` is this glove's slice of a frame (NaN rows = padded).
    ``ids``/``parents`` is the hardcoded per-glove topology.
    Returns the drawn node positions (for the auto-follow camera) or None.
    """
    valid = ~np.isnan(positions).any(axis=1)
    pos = positions[valid]
    if len(pos) < 2:
        return None
    k = len(pos)
    ids_v = ids[:k]
    parents_v = parents[:k]

    parent_set = {int(p) for p in parents_v if 0 <= p < 2**31}
    colors = [_LEAF if int(i) not in parent_set else _JOINT for i in ids_v]
    sizes = [30 if c == _LEAF else 18 for c in colors]
    by_id = {int(ids_v[j]): pos[j] for j in range(k)}
    segs = []
    for j in range(k):
        pid = int(parents_v[j])
        if pid in by_id:
            segs.append((by_id[pid], pos[j]))
    _draw_nodes(ax, pos, colors, sizes, segs)
    return pos


def draw_fk(ax, ch: np.ndarray, side: str) -> np.ndarray | None:
    """Draw the FK hand model for one hand (4-node chains per finger)."""
    nodes = hand_pose_nodes(ch, side)
    if nodes is None:
        return None
    pos_all, colors, sizes, segs = [], [], [], []
    for f in range(5):
        seg = nodes[f * 4:(f + 1) * 4]
        pos_all.append(seg)
        colors += [_JOINT, _JOINT, _JOINT, _LEAF]
        sizes += [18, 18, 18, 30]
        segs += [(seg[j], seg[j + 1]) for j in range(3)]
    pos = np.concatenate(pos_all)
    _draw_nodes(ax, pos, colors, sizes, segs)
    return pos


# ---- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="hand_pose/hand_pose.npz path")
    ap.add_argument("-o", "--out", default="", help="output mp4 (default: <npz>.pose.mp4)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1 = last frame")
    ap.add_argument("--every", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--fk", action="store_true",
                    help="force the forward-kinematics model instead of the skeleton")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        print(f"not found: {npz_path}")
        return 1
    data = dict(np.load(npz_path))
    ergo = data.get("ergo_data")
    skel = data.get("skeleton_positions")
    ts = data.get("ergo_timestamps", data.get("skeleton_timestamps"))

    use_skeleton = (not args.fk and skel is not None
                    and _NODES_PER_GLOVE > 0
                    and len(_NODE_IDS) == _NODES_PER_GLOVE)
    n = len(skel) if use_skeleton else len(ergo) if ergo is not None else 0
    if n == 0:
        print("no frames in npz")
        return 1
    print("mode:", "skeleton" if use_skeleton else "FK-from-ergo")
    if ergo is not None and ergo.ndim == 2 and ergo.shape[1] >= 40:
        left = float((np.abs(ergo[:, :20]).sum(axis=1) > 1e-3).mean())
        right = float((np.abs(ergo[:, 20:40]).sum(axis=1) > 1e-3).mean())
        print(f"ergo data coverage — left: {left:.0%}, right: {right:.0%}")
        if right < 0.5:
            print("WARNING: right-hand channels (20-39) are mostly zero — "
                  "this npz likely predates the side-mapping fix; re-record.")

    start = max(0, args.start)
    end = n if args.end < 0 else min(args.end, n)
    idx = list(range(start, end, max(1, args.every)))
    if not idx:
        print("empty frame range")
        return 1

    out = Path(args.out) if args.out else npz_path.with_name(npz_path.stem + ".pose.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # One 3D subplot per hand (like the live test_manus_3d view): raw
    # skeletons are per-glove LOCAL coordinates and FK hands are synthetic,
    # so each hand gets its own subplot with its own auto-follow camera.
    fig = plt.figure(figsize=(10, 7), dpi=120)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.2])
    glove_axs = [fig.add_subplot(gs[0, 0], projection="3d"),
                 fig.add_subplot(gs[0, 1], projection="3d")]
    glove_titles = (["glove #1", "glove #2"] if use_skeleton
                    else ["left hand (FK)", "right hand (FK)"])
    for a, title in zip(glove_axs, glove_titles):
        a.set_title(title, fontsize=10)
        a.set_xlabel("X"); a.set_ylabel("Y"); a.set_zlabel("Z")
    ax_ergo = fig.add_subplot(gs[1, :])
    ax_ergo.set_title("ergonomics 40-ch (L 0-19 | R 20-39)", fontsize=10)
    ax_ergo.set_xticks(range(0, 40, 5))
    ax_ergo.set_ylim(-100, 100)
    ax_ergo.tick_params(labelsize=6)

    import cv2
    fig.canvas.draw()
    h, w = np.asarray(fig.canvas.buffer_rgba()).shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        print(f"cannot open video writer for {out}")
        return 1

    try:
        for i in idx:
            if use_skeleton:
                halves = [skel[i][h:h + _NODES_PER_GLOVE]
                          for h in range(0, skel[i].shape[0], _NODES_PER_GLOVE)]
                for a, title, half in zip(glove_axs, glove_titles, halves):
                    a.cla()
                    a.set_title(title, fontsize=10)
                    pos = draw_skeleton(a, half, _NODE_IDS, _PARENT_IDS)
                    if pos is not None:
                        _auto_camera(a, pos)  # per-glove camera
            else:
                for a, title, side in zip(glove_axs, glove_titles,
                                          ("left", "right")):
                    a.cla()
                    a.set_title(title, fontsize=10)
                    pos = None
                    if ergo is not None:
                        ch = ergo[i, :20] if side == "left" else ergo[i, 20:]
                        pos = draw_fk(a, ch, side)
                    if pos is not None:
                        _auto_camera(a, pos)  # per-hand camera

            if ergo is not None and i < len(ergo):
                ax_ergo.clear()
                ax_ergo.set_ylim(-100, 100)
                ax_ergo.set_xticks(range(0, 40, 5))
                ax_ergo.tick_params(labelsize=6)
                ax_ergo.bar(range(40), ergo[i], width=0.85,
                            color=[_JOINT] * 20 + [_LEAF] * 20)
                ax_ergo.set_title("ergonomics 40-ch (L 0-19 | R 20-39)",
                                  fontsize=10)

            t = ts[i] if ts is not None and i < len(ts) else i / args.fps
            fig.suptitle(f"frame {i}  t={t:6.2f}s", fontsize=10)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
            if i % 50 == 0:
                print(f"\r[{i - start + 1}/{len(idx)}] frames rendered", end="",
                      flush=True)
    finally:
        writer.release()
        plt.close(fig)

    print(f"\nsaved {out}  ({len(idx)} frames @ {args.fps}fps = {len(idx)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
