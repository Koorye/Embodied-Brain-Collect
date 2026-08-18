"""Dump the Metaglove raw-skeleton node topology once, for hardcoding.

Connects to Manus Core (same flow as the hand_pose recorder), reads the node
hierarchy of the first glove via ``GetNodeInfo`` and prints it.  With
``--write`` the topology constants block in
``scripts/visualize_ergo_pose.py`` is rewritten, so skeleton-mode video
rendering works from then on without any per-session topology.

The raw skeleton is deterministic for a given glove model, so this only ever
needs to be run once per model.

Usage:
  python scripts/dump_skeleton_topology.py           # print the table
  python scripts/dump_skeleton_topology.py --write   # + hardcode into the viz script
"""

import argparse
import re
import sys
import time
from pathlib import Path


def fetch_topology(lib_path: str = "") -> list[dict]:
    """Connect to Manus Core and return the first glove's node list."""
    from manus_glove import ManusDataPublisher

    kwargs = {"debug": False}
    if lib_path:
        kwargs["lib_path"] = lib_path
    pub = ManusDataPublisher(**kwargs)
    try:
        pub.Initialize()
        pub.Connect()
        gloves: list[dict] = []
        t0 = time.time()
        while not gloves and time.time() - t0 < 15:
            landscape = pub.GetLandscape()
            if landscape:
                gloves = landscape.get("gloves", [])
            time.sleep(0.5)
        if not gloves:
            raise RuntimeError("no gloves found after 15 s — gloves paired?")
        gid = gloves[0]["id"]
        nodes = pub.GetNodeInfo(gid)
        if not nodes:
            raise RuntimeError("GetNodeInfo returned nothing — "
                               "is skeleton data flowing?")
        return nodes
    finally:
        pub.ShutDown()


def write_topology(target: Path, ids: list[int], parents: list[int]) -> None:
    """Rewrite the generated-topology constants block in *target*."""
    block = (
        "# === BEGIN GENERATED TOPOLOGY (python scripts/dump_skeleton_topology.py --write) ===\n"
        "# Metaglove raw-skeleton topology, one glove, in CoreSdk_GetRawSkeletonData\n"
        "# order.  Identical for every glove of the same model, so it is hardcoded\n"
        "# instead of stored per session.\n"
        f"_NODES_PER_GLOVE = {len(ids)}  "
        "# 0 = not dumped yet; skeleton mode falls back to FK\n"
        f"_NODE_IDS = np.array([{', '.join(map(str, ids))}], dtype=np.int32)\n"
        f"_PARENT_IDS = np.array([{', '.join(map(str, parents))}], dtype=np.int32)"
        "  # -1 = root\n"
        "# === END GENERATED TOPOLOGY ==="
    )
    src = target.read_text(encoding="utf-8")
    pattern = re.compile(
        r"# === BEGIN GENERATED TOPOLOGY.*?# === END GENERATED TOPOLOGY ===",
        re.DOTALL,
    )
    if not pattern.search(src):
        raise RuntimeError(f"generated-topology marker block not found in {target}")
    target.write_text(pattern.sub(block, src), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the topology constants in "
                         "scripts/visualize_ergo_pose.py")
    ap.add_argument("--lib-path", default="",
                    help="optional path to the Manus SDK library")
    args = ap.parse_args()

    nodes = fetch_topology(args.lib_path)

    print(f"{'idx':>3}  {'nodeId':>6}  {'parentId':>8}  "
          f"{'jointType':>14}  chainType")
    ids, parents = [], []
    for i, n in enumerate(nodes):
        nid = int(n["nodeId"])
        pid = int(n["parentId"])
        if pid >= 2**31:  # uint32 "invalid" marker -> root
            pid = -1
        ids.append(nid)
        parents.append(pid)
        print(f"{i:>3}  {nid:>6}  {pid:>8}  "
              f"{str(n.get('fingerJointType', '')):>14}  "
              f"{n.get('chainType', '')}")

    if not args.write:
        print(f"\n{len(nodes)} nodes per glove — re-run with --write to "
              "hardcode into scripts/visualize_ergo_pose.py")
        return 0

    target = Path(__file__).resolve().parent / "visualize_ergo_pose.py"
    write_topology(target, ids, parents)
    print(f"\nwrote {len(nodes)}-node topology into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
