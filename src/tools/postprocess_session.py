"""One-shot post-session QA: Curry triggers, markers.npz, and aligner.

Example:
    python tools/postprocess_session.py \\
        --session-dir sessions/2026-05-25_14-30-00_subj01_run1_p1 \\
        --eeg-dap C:/Users/31454/Desktop/Acquisition/subj01_run1_p1.dap
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run(cmd: list[str]) -> int:
    print("[postprocess] " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(SRC))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--eeg-dap", type=Path, required=True,
                    help="Curry acquisition: .cdt.dpo, .cdt, .dap, or .dat pair")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--skip-inspect", action="store_true")
    args = ap.parse_args(argv)

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"[postprocess] ERROR: session dir not found: {session_dir}", file=sys.stderr)
        return 1

    from tools.curry_io import resolve_curry_paths

    eeg_src = args.eeg_dap.resolve()
    try:
        meta_src, data_src = resolve_curry_paths(eeg_src)
    except Exception as exc:
        print(f"[postprocess] ERROR: {exc}", file=sys.stderr)
        return 1
    if not meta_src.is_file():
        print(f"[postprocess] ERROR: Curry metadata not found: {meta_src}", file=sys.stderr)
        return 1

    sessions = ROOT / "sessions"
    sessions.mkdir(exist_ok=True)
    local_stem = meta_src.stem.replace(" ", "_")
    if meta_src.name.endswith(".cdt.dpo"):
        local_meta = sessions / meta_src.name.replace(" ", "_")
        local_data = local_meta.with_name(local_meta.name.replace(".cdt.dpo", ".cdt"))
    else:
        local_meta = sessions / f"{local_stem}.dap"
        local_data = sessions / f"{local_stem}.dat"
    local_csv = sessions / f"{local_stem}_triggers.csv"
    local_npz = sessions / f"{local_stem}_triggers.npz"

    if meta_src != local_meta:
        shutil.copy2(meta_src, local_meta)
        if data_src.is_file():
            shutil.copy2(data_src, local_data)
        print(f"[postprocess] copied EEG -> {local_meta}")
    eeg_dap = local_meta

    py = sys.executable
    rc = 0

    if not args.skip_inspect:
        markers = session_dir / "markers.npz"
        rc = _run([
            py, "-m", "record.tools.inspect_curry_triggers",
            str(eeg_dap),
            "--npz", str(local_npz),
            "--csv", str(local_csv),
        ])
        if rc != 0:
            return rc
        if markers.is_file():
            rc = _run([py, "-m", "record.tools.inspect_markers", str(markers)])
            if rc != 0:
                return rc
        else:
            print(f"[postprocess] WARN: no markers.npz at {markers}")

    if not args.skip_align:
        rc = _run([
            py, "-m", "record.session.aligner",
            str(session_dir),
            "--eeg-dap", str(eeg_dap),
        ])
        if rc == 0:
            report = session_dir / "aligned" / "align_report.json"
            aligned = session_dir / "aligned" / "aligned.npz"
            print(f"[postprocess] done.")
            print(f"  aligned.npz       -> {aligned}")
            print(f"  align_report.json -> {report}")
        return rc

    print("[postprocess] inspect-only done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
