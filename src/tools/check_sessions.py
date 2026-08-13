"""Quick post-hoc sanity check across many sessions.

Walks ``record/sessions/`` and prints a one-line summary per session,
showing which modalities produced data and which didn't.  Useful for
bulk-auditing a day's collection at a glance.

For a deeper per-session report use ``record.session.sanity``.

Examples
--------
    # Summary of today's sessions for subj01
    python -m record.tools.check_sessions --prefix 2026-05-27_ --contains subj01

    # All sessions, including smoke tests
    python -m record.tools.check_sessions

    # Output JSON for programmatic processing
    python -m record.tools.check_sessions --prefix 2026-05-25_ --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "sessions"
MODALITIES = ["emg", "eye", "manus", "tactile", "vive", "wrist_cam"]


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _folder_summary(folder: Path) -> tuple[int, int, datetime | None, datetime | None]:
    if not folder.exists():
        return 0, 0, None, None
    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        return 0, 0, None, None
    total = sum(f.stat().st_size for f in files)
    mtimes = [datetime.fromtimestamp(f.stat().st_mtime) for f in files]
    return len(files), total, min(mtimes), max(mtimes)


def _load_enabled(session: Path) -> set[str]:
    sj = session / "session.json"
    if not sj.exists():
        return set()
    try:
        data = json.loads(sj.read_text(encoding="utf-8"))
    except Exception:
        return set()
    recs = data.get("recorders") or []
    return {r["name"] for r in recs if isinstance(r, dict) and "name" in r}


def _row_for(session: Path) -> dict:
    enabled = _load_enabled(session)
    row: dict = {"session": session.name, "enabled": sorted(enabled)}

    sj = session / "session.json"
    try:
        data = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
        row["end_reason"] = data.get("end_reason")
        row["degraded_modalities"] = data.get("degraded_modalities", [])
    except Exception:
        pass

    modal_status: dict[str, str] = {}
    for m in MODALITIES:
        sub = session / m
        n, size, *_ = _folder_summary(sub)
        if not enabled or m in enabled:
            if n == 0:
                modal_status[m] = "MISSING" if (enabled and m in enabled) else "n/a"
            else:
                modal_status[m] = f"{n}f / {_fmt_size(size)}"
        else:
            modal_status[m] = "off"
    row["modalities"] = modal_status
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=None,
                    help="Only inspect sessions whose name starts with this "
                         "(e.g. '2026-05-27_').")
    ap.add_argument("--contains", default=None,
                    help="Only inspect sessions whose name contains this "
                         "(e.g. 'subj01').")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the table.")
    args = ap.parse_args(argv)

    if not ROOT.is_dir():
        print(f"[check_sessions] no sessions dir: {ROOT}", file=sys.stderr)
        return 2

    sessions = sorted(
        s for s in ROOT.iterdir()
        if s.is_dir()
        and (args.prefix is None or s.name.startswith(args.prefix))
        and (args.contains is None or args.contains in s.name)
    )
    if not sessions:
        print("[check_sessions] no sessions match the filter.", file=sys.stderr)
        return 0

    if args.json:
        out = [_row_for(s) for s in sessions]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    # Plain-text table.  Width is wide; use a wide terminal.
    header = (
        f"{'session':<48} "
        + " ".join(f"{m:^14}" for m in MODALITIES)
        + "  reason"
    )
    print(header)
    print("-" * len(header))
    for s in sessions:
        row = _row_for(s)
        cells = [f"{row['modalities'][m]:^14}" for m in MODALITIES]
        reason = row.get("end_reason") or "-"
        print(f"{s.name:<48} " + " ".join(cells) + f"  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
