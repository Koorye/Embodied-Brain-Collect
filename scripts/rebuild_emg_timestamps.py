#!/usr/bin/env python3
"""Backfill per-frame EMG timestamps into sessions recorded before the fit.

    python scripts/rebuild_emg_timestamps.py data/session-night/*   # preview
    python scripts/rebuild_emg_timestamps.py data/... --write       # apply

Recordings made before the recorder gained its close-time fit hold read-arrival
timestamps: ~140 frames share the timestamp of the ``Serial.read()`` that
carried them, so ~99% of the series are duplicates and every frame sits up to a
batch-period late.  This refits those files in place using the same code the
recorder now runs (``recorders.emg.timestamp_rebuild``), so old and new
sessions come out identical in structure.

Dry-run by default — it prints what each file would become and touches nothing.
``--write`` rewrites the ``.npz`` via a temporary file next to it, keeping the
arrival series as ``*_arrival_timestamps``.  Files that already carry
``emg_arrival_timestamps`` are skipped: the fit has run there, and re-fitting
a fitted series would regress the timeline onto itself.

Exit code: 0 all good, 1 if any file was skipped or refused.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.append('.')
from scripts.timestamp_rebuild import (  # noqa: E402
    rebuild)


_ALREADY = "emg_arrival_timestamps"


def _emg_npz(roots: list[Path]) -> list[Path]:
    """Every emg*.npz under the given session dirs (or the files themselves)."""
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".npz":
            out.append(root)
        elif root.is_dir():
            out.extend(sorted(root.glob("**/emg*/emg*.npz")))
    return sorted(set(out))


def _describe(ts: np.ndarray) -> str:
    if ts.size < 2:
        return f"n={ts.size}"
    dt = np.diff(ts)
    dup = float((dt == 0).mean())
    return (f"n={ts.size} dup={dup:6.2%} "
            f"unique={np.unique(ts).size}")


def process(path: Path, write: bool, force: bool = False) -> bool:
    """Refit one file.  Returns True when it is (or already was) fitted.

    ``force`` re-runs a file that already carries the arrival fields (e.g.
    after the fit itself changed): it refits from the preserved arrival
    series, never from the previous fit.
    """
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}

    if _ALREADY in data and not force:
        print(f"  {path}\n    skip — already fitted")
        return True
    if "emg_sn" not in data or "emg_timestamps" not in data:
        print(f"  {path}\n    skip — no emg_sn/emg_timestamps to fit from")
        return False
    if _ALREADY in data and force:
        data["emg_timestamps"] = data[_ALREADY]
        if "imu_arrival_timestamps" in data:
            data["imu_timestamps"] = data["imu_arrival_timestamps"]
        print(f"  {path}\n    re-fitting from preserved arrival timestamps")

    r = rebuild(data["emg_timestamps"], data["emg_sn"],
                data.get("imu_timestamps", np.zeros(0)),
                data.get("imu_sn", np.zeros(0, dtype=np.int64)))
    print(f"  {path}")
    print(f"    before: {_describe(np.asarray(data['emg_timestamps'], float))}")
    if not r.ok:
        print(f"    REFUSED — {r.note}")
        return False
    print(f"    after : {_describe(r.emg_timestamps)}")
    print(f"    {r.summary()}")
    if not write:
        return True

    data[_ALREADY] = data["emg_timestamps"]
    data["emg_timestamps"] = r.emg_timestamps
    if "imu_timestamps" in data and data["imu_timestamps"].size:
        data["imu_arrival_timestamps"] = data["imu_timestamps"]
        data["imu_timestamps"] = r.imu_timestamps

    # Write beside the original then replace: a crash mid-save must not leave
    # a truncated npz where the only copy of a recording used to be.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp)
    try:
        np.savez(tmp, **data)   # savez keeps the name when it ends in .npz
        os.replace(tmp, path)
        print(f"    written ({path.stat().st_size / 1e6:.1f} MB)")
    finally:
        if tmp.exists():
            tmp.unlink()
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, nargs="+",
                    help="session 目录或 emg*.npz 文件")
    ap.add_argument("--write", action="store_true",
                    help="真正写回;默认只预览")
    ap.add_argument("--force", action="store_true",
                    help="已拟合过的文件也重做(从保留的到达时间戳重新拟合)")
    args = ap.parse_args(argv)

    files = _emg_npz(args.path)
    if not files:
        print("no emg*.npz found", file=sys.stderr)
        return 1

    print(f"{len(files)} file(s), mode={'WRITE' if args.write else 'dry-run'}"
          + (" force" if args.force else ""))
    ok = all([process(p, args.write, args.force) for p in files])
    if not args.write:
        print("\ndry-run — nothing written; re-run with --write to apply")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
