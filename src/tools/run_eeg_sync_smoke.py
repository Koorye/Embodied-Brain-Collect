"""Run an EEG marker sync smoke test with ParallelBox + sync_hub.

Curry7 must already be recording before this script starts. The script starts a
short-lived sync_hub, sends the same marker bytes to ParallelBox and UDP, then
optionally decodes a saved Curry .dap and runs the aligner.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _latest_dap(curry_dir: Path) -> Path | None:
    if not curry_dir.exists():
        return None
    candidates = sorted(curry_dir.glob("*.dap"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _run(cmd: list[str], *, cwd: Path, log_path: Path | None = None) -> int:
    print("[run] " + " ".join(str(c) for c in cmd))
    if log_path is None:
        return subprocess.call(cmd, cwd=str(cwd))
    with log_path.open("w", encoding="utf-8", buffering=1) as fh:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
        return p.wait()


def _start_sync_hub(args: argparse.Namespace, session_dir: Path) -> tuple[subprocess.Popen, Path]:
    log_path = session_dir / "sync_hub_smoke.log"
    cmd = [
        sys.executable, "-m", "record.sync.sync_hub",
        "--out", str(session_dir),
        "--bind", args.bind,
        "--udp-port", str(args.udp_port),
        "--zmq-port", str(args.zmq_port),
        "--flush-every-events", "1",
        "--flush-every-seconds", "1",
    ]
    if args.quiet_hub:
        cmd.append("--quiet")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    fh = log_path.open("w", encoding="utf-8", buffering=1)
    p = subprocess.Popen(
        cmd,
        cwd=str(SRC),
        stdout=fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    p._log_fh = fh  # type: ignore[attr-defined]
    time.sleep(args.hub_start_s)
    if p.poll() is not None:
        try:
            fh.close()
        except Exception:
            pass
        raise RuntimeError(f"sync_hub exited early; see {log_path}\n{_tail(log_path)}")
    print(f"[smoke] sync_hub pid={p.pid}, log={log_path}")
    return p, log_path


def _stop_process(p: subprocess.Popen, timeout: float = 5.0) -> None:
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.send_signal(signal.SIGINT)
        p.wait(timeout=timeout)
    except Exception:
        try:
            p.terminate()
            p.wait(timeout=timeout)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    try:
        p._log_fh.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, default=None)
    ap.add_argument("--subject", default="eeg_smoke")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--codes", default="241,17,33,81,82,97,98,113,114,242")
    ap.add_argument("--hold-s", type=float, default=0.05)
    ap.add_argument("--isi-s", type=float, default=0.45)
    ap.add_argument("--pre-clear-s", type=float, default=1.0)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=9999)
    ap.add_argument("--zmq-port", type=int, default=9998)
    ap.add_argument("--hub-start-s", type=float, default=1.0)
    ap.add_argument("--quiet-hub", action="store_true")
    ap.add_argument("--curry-dir", type=Path, default=Path(r"C:\Users\31454\Desktop\Acquisition"))
    ap.add_argument("--eeg-dap", type=Path, default=None)
    ap.add_argument("--eeg-min-duration-s", type=float, default=0.02)
    ap.add_argument("--skip-align", action="store_true")
    args = ap.parse_args(argv)

    if args.session_dir is None:
        args.session_dir = ROOT_DIR / "sessions" / f"{_timestamp()}_{args.subject}_run{args.run}_eeg_smoke"
    session_dir = args.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": _timestamp(),
        "port": args.port,
        "baud": args.baud,
        "codes": args.codes,
        "hold_s": args.hold_s,
        "isi_s": args.isi_s,
        "udp_host": args.udp_host,
        "udp_port": args.udp_port,
        "curry_dir": str(args.curry_dir),
        "eeg_dap_arg": str(args.eeg_dap) if args.eeg_dap else None,
    }
    (session_dir / "eeg_sync_smoke.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[smoke] session_dir={session_dir}")

    hub = None
    try:
        hub, _hub_log = _start_sync_hub(args, session_dir)
        send_log = session_dir / "parallelbox.jsonl"
        send_stdout = session_dir / "parallelbox_stdout.log"
        send_cmd = [
            sys.executable, "-m", "record.tools.send_parallelbox_markers",
            "--port", args.port,
            "--baud", str(args.baud),
            "--codes", args.codes,
            "--hold-s", str(args.hold_s),
            "--isi-s", str(args.isi_s),
            "--pre-clear-s", str(args.pre_clear_s),
            "--trial", "1",
            "--tag-prefix", "EEG_SMOKE",
            "--udp-host", args.udp_host,
            "--udp-port", str(args.udp_port),
            "--log", str(send_log),
        ]
        rc = _run(send_cmd, cwd=PROJECT_ROOT, log_path=send_stdout)
        print(_tail(send_stdout, 80))
        if rc != 0:
            print(f"[smoke] marker sender failed; see {send_stdout}")
            return rc
        time.sleep(1.0)
    finally:
        if hub is not None:
            _stop_process(hub)

    markers = session_dir / "markers.npz"
    if not markers.exists():
        print(f"[smoke] missing {markers}; sync_hub log tail:\n{_tail(session_dir / 'sync_hub_smoke.log')}")
        return 1

    inspect_log = session_dir / "inspect_markers.txt"
    rc = _run([sys.executable, "-m", "record.tools.inspect_markers", str(markers)], cwd=PROJECT_ROOT, log_path=inspect_log)
    print(_tail(inspect_log, 120))
    if rc != 0:
        return rc

    if args.skip_align:
        print("[smoke] skip-align set; stop/save Curry now, then rerun aligner with --eeg-dap <file>.")
        return 0

    eeg_dap = args.eeg_dap or _latest_dap(args.curry_dir)
    if eeg_dap is None:
        print(f"[smoke] no .dap found under {args.curry_dir}; stop/save Curry and rerun aligner manually.")
        return 0

    align_log = session_dir / "aligner_stdout.log"
    rc = _run([
        sys.executable, "-m", "record.session.aligner", str(session_dir),
        "--eeg-dap", str(eeg_dap),
        "--eeg-min-duration-s", str(args.eeg_min_duration_s),
    ], cwd=PROJECT_ROOT, log_path=align_log)
    print(_tail(align_log, 120))
    report = session_dir / "aligned" / "align_report.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            eeg = data.get("eeg", {})
            fit = eeg.get("fit_to_pc", {})
            print("[smoke] EEG fit:", json.dumps(fit, ensure_ascii=False))
        except Exception:
            pass
    print(f"[smoke] done. session={session_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
