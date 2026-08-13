"""Run a full multimodal collection from SSH while Vive/OpenVR runs interactively.

This helper creates a temporary Windows scheduled task that runs
``session/launcher.py`` in the logged-in desktop session, waits for the marker
hub to start, then sends one marker sequence to both ParallelBox COM and the UDP
sync_hub.  It is intended for real acquisition after Curry7 has already started
recording.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_PYTHON = Path(r"C:\Users\31454\miniconda3\Scripts\conda.exe")
DEFAULT_TASK = "RecordFullModalCollection"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[run] " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True)


def _write_launcher_bat(args: argparse.Namespace, bat_path: Path, log_path: Path) -> None:
    launcher_cmd = [
        str(DEFAULT_PYTHON), "run", "-n", args.conda_env,
        "python", "session\\launcher.py",
        "--subject", args.subject,
        "--run", str(args.run),
        "--paradigm", str(args.paradigm),
        "--duration", str(args.duration),
        "--flush-timeout", str(args.flush_timeout),
        "--neon-ip", args.neon_ip,
        "--recorders", *args.recorders,
    ]
    line = " ".join(launcher_cmd) + f" > {log_path} 2>&1"
    bat_path.write_text(
        "@echo off\n"
        f"cd /d {ROOT_DIR}\n"
        f"{line}\n",
        encoding="ascii",
    )


def _create_or_update_task(task_name: str, bat_path: Path) -> None:
    # /ST is required for ONCE tasks even though we start it immediately with /Run.
    _run([
        "schtasks", "/Create", "/F",
        "/TN", task_name,
        "/SC", "ONCE",
        "/ST", "23:59",
        "/TR", str(bat_path),
        "/IT",
    ])


def _extract_session_dir(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\[launcher\] session dir: (.+)", text)
    if matches:
        return Path(matches[-1].strip())
    matches = re.findall(r"\[launcher\] done\.  session: (.+)", text)
    if matches:
        return Path(matches[-1].strip())
    return None


def _wait_for_session_dir(log_path: Path, timeout_s: float) -> Path | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sd = _extract_session_dir(log_path)
        if sd:
            return sd
        time.sleep(0.5)
    return None


def _send_markers(args: argparse.Namespace, marker_log: Path) -> None:
    cmd = [
        str(DEFAULT_PYTHON), "run", "-n", args.conda_env,
        "python", "-m", "record.tools.send_parallelbox_markers",
        "--port", args.parallelbox_port,
        "--baud", str(args.parallelbox_baud),
        "--codes", args.codes,
        "--hold-s", str(args.hold_s),
        "--isi-s", str(args.isi_s),
        "--pre-clear-s", str(args.pre_clear_s),
        "--trial", str(args.trial),
        "--tag-prefix", args.tag_prefix,
        "--udp-host", args.udp_host,
        "--udp-port", str(args.udp_port),
        "--log", str(marker_log),
    ]
    _run(cmd, cwd=PROJECT_ROOT)


def _wait_for_launcher(log_path: Path, timeout_s: float) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if "[launcher] done." in text:
                return 0
        time.sleep(2.0)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="subject01")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--paradigm", default="1", choices=["1", "2", "3"])
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--flush-timeout", type=float, default=30.0)
    ap.add_argument("--neon-ip", default="172.16.19.213")
    ap.add_argument("--recorders", nargs="+", default=["eye", "tactile", "wrist_cam", "emg", "vive"])
    ap.add_argument("--conda-env", default="record")
    ap.add_argument("--task-name", default=DEFAULT_TASK)
    ap.add_argument("--marker-delay-s", type=float, default=10.0,
                    help="Seconds to wait after launching recorders before sending markers.")
    ap.add_argument("--codes", default="241,31,32,33,34,35,242")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--tag-prefix", default="FULL_RUN")
    ap.add_argument("--parallelbox-port", default="COM14")
    ap.add_argument("--parallelbox-baud", type=int, default=115200)
    ap.add_argument("--hold-s", type=float, default=0.08)
    ap.add_argument("--isi-s", type=float, default=0.35)
    ap.add_argument("--pre-clear-s", type=float, default=1.0)
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=9999)
    ap.add_argument("--no-markers", action="store_true",
                    help="Only run the full-modal launcher; do not send ParallelBox/UDP markers.")
    ap.add_argument("--no-wait", action="store_true",
                    help="Return after launching and optional marker send instead of waiting for completion.")
    args = ap.parse_args(argv)

    sessions_dir = ROOT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    bat_path = sessions_dir / "run_fullmodal_collection.bat"
    launcher_log = sessions_dir / "fullmodal_launcher_latest.log"
    marker_log = sessions_dir / "fullmodal_marker_latest.jsonl"

    for path in (launcher_log, marker_log):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    _write_launcher_bat(args, bat_path, launcher_log)
    _create_or_update_task(args.task_name, bat_path)
    _run(["schtasks", "/Run", "/TN", args.task_name])

    session_dir = _wait_for_session_dir(launcher_log, timeout_s=30.0)
    if session_dir:
        print(f"[fullmodal] session_dir={session_dir}")
    else:
        print(f"[fullmodal] launcher has not written session dir yet; see {launcher_log}")

    if not args.no_markers:
        print(f"[fullmodal] waiting {args.marker_delay_s:.1f}s before marker burst")
        time.sleep(args.marker_delay_s)
        _send_markers(args, marker_log)
        print(f"[fullmodal] marker_log={marker_log}")

    if args.no_wait:
        print(f"[fullmodal] not waiting. launcher_log={launcher_log}")
        return 0

    wait_s = args.duration + args.flush_timeout + 60.0
    rc = _wait_for_launcher(launcher_log, wait_s)
    if rc == 0:
        session_dir = _extract_session_dir(launcher_log) or session_dir
        print(f"[fullmodal] done. session_dir={session_dir}")
        print(f"[fullmodal] launcher_log={launcher_log}")
        return 0

    print(f"[fullmodal] timed out waiting for launcher. Check {launcher_log}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
