"""Launcher — runs recorders in-process via threads, optionally with stim.

Usage::

    # CLI: production hardware + stim
    python -m src.session.launcher --session-dir ./sessions/run1 --with-stim

    # CLI: dummy test without stim
    python -m src.session.launcher --dummy --session-dir ./test --duration 10

    # CLI: only specific modalities with stim
    python -m src.session.launcher --session-dir ./sessions/run1 \\
        --recorders emg hand_pose --with-stim

    # Code: custom setup
    from src.session.launcher import launch
    from src.session.recorder_presets import get_production_recorders

    recs = get_production_recorders(session_dir="./sessions/run1")
    launch(recs, stim_cmd=["python", "-m", "src.stim.paradigm1_pickplace",
                           "--task-id", "0", "--windowed"])
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from src.recorders.base import BaseRecorder

SRC = Path(__file__).resolve().parents[1]


# ---- core ---------------------------------------------------------------


def launch(
    recorders: dict[str, BaseRecorder],
    *,
    stim_cmd: Sequence[str] | None = None,
    duration: float = 0.0,
) -> int:
    """Run *recorders* in threads until stim ends, duration elapses, or Ctrl+C.

    Every recorder is opened first (with a per-recorder timeout); recording
    starts only when **all** of them return True from ``_open()``.  If any
    open returns False, raises, or times out, the specific error is printed
    and the launch aborts with a non-zero exit code — nothing is recorded.

    Args:
        recorders: pre-configured recorder instances (``_open`` not yet called)
        stim_cmd: optional argv to spawn the stim program as a subprocess.
                  When the stim exits, all recorders are shut down gracefully.
        duration: fallback max seconds (0 = no limit, wait for stim/Ctrl+C)
    """
    if not recorders:
        print("[launcher] nothing to run.")
        return 0

    threads: dict[str, threading.Thread] = {}
    stop_event = threading.Event()
    _closed: set[str] = set()

    print(f"[launcher] modalities: {list(recorders.keys())}")

    # ---- open every recorder; ALL must return True before recording ----
    opened: list[tuple[str, BaseRecorder]] = []

    def _open_one(name: str, rec: BaseRecorder) -> tuple[bool, str]:
        """Open one recorder under a watchdog timeout.

        Returns ``(ok, reason)``; ``reason`` is the specific failure message
        ('' when ok): ``_open()`` returned False, raised, or hung past
        ``config.open_timeout``.
        """
        timeout = float(getattr(rec.config, "open_timeout", 30.0) or 30.0)
        result: dict = {"done": False, "ok": False, "exc": None}

        def _do_open():
            try:
                result["ok"] = bool(rec._open())
            except Exception as exc:  # noqa: BLE001 — report, don't crash
                result["exc"] = exc
            finally:
                result["done"] = True

        print(f"[launcher] opening {name} ({type(rec).__name__}) ...")
        t = threading.Thread(target=_do_open, name=f"open:{name}", daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result["done"]:
            return False, f"open TIMEOUT after {timeout:g}s"
        if result["exc"] is not None:
            exc = result["exc"]
            return False, f"open ERROR — {type(exc).__name__}: {exc}"
        if result["ok"]:
            return True, ""
        return False, "open FAILED — " + (
            getattr(rec, "_open_error", "") or "unknown reason")

    for name, rec in recorders.items():
        ok, reason = _open_one(name, rec)
        if ok:
            print(f"[launcher] {name} ({type(rec).__name__}): open OK")
            opened.append((name, rec))
        else:
            print(f"[launcher] {name} ({type(rec).__name__}): {reason}")
            rec._log(f"[launcher] {name}: {reason}", echo=False)
            print("[launcher] ERROR: not all recorders opened — "
                  "aborting, nothing will be recorded.")
            for op_name, op_rec in opened:
                try:
                    op_rec._close()
                except Exception:
                    pass
            return 1

    print("[launcher] all recorders ready — starting.")

    # ---- start stim subprocess (if requested) ----
    stim_proc: subprocess.Popen | None = None
    if stim_cmd:
        stim_argv = list(stim_cmd)
        print(f"[launcher] stim: {' '.join(stim_argv)}")
        stim_proc = subprocess.Popen(
            stim_argv,
            cwd=str(SRC.parent),  # project root
        )

    def _stop_one(name: str, rec: BaseRecorder) -> None:
        if name in _closed:
            return
        _closed.add(name)
        try:
            rec._close()
        except Exception:
            pass
        try:
            rec._save()
        except Exception:
            pass

    # ---- poll worker per recorder ----
    def _worker(name: str, rec: BaseRecorder) -> None:
        t0 = time.time()
        try:
            while not stop_event.is_set():
                rec._poll(time.time() - t0)
                time.sleep(0.001)
        except Exception as exc:
            print(f"\n[launcher] {name} crashed: {type(exc).__name__}: {exc}")
        finally:
            _stop_one(name, rec)
            if not stop_event.is_set():
                stop_event.set()

    for name, rec in recorders.items():
        t = threading.Thread(target=_worker, args=(name, rec), daemon=True)
        t.start()
        threads[name] = t

    # ---- main loop ----
    rc = 0
    t0 = time.time()

    try:
        while threads:
            elapsed = time.time() - t0

            # --- stop conditions ---
            if 0 < duration <= elapsed:
                print(f"\n[launcher] duration {duration}s reached.")
                break

            if stim_proc is not None and stim_proc.poll() is not None:
                rc = stim_proc.returncode
                print(f"\n[launcher] stim exited (code={rc}) — stopping recorders.")
                break

            # --- thread health ---
            for name, t in list(threads.items()):
                if not t.is_alive():
                    print(f"\n[launcher] {name} thread exited")
                    del threads[name]

            # --- status line every 1 s ---
            if int(elapsed) > int(elapsed - 0.5):
                parts = [f"t={elapsed:5.1f}s"]
                for name, rec in recorders.items():
                    extra = rec._heartbeat_stats(elapsed)
                    parts.append(f"{name}:{extra}" if extra else name)
                print("  ".join(parts).ljust(140))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C — stopping ...")
    finally:
        stop_event.set()

        # Kill stim if still running
        if stim_proc is not None and stim_proc.poll() is None:
            stim_proc.terminate()
            try:
                stim_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stim_proc.kill()

        for name, t in threads.items():
            t.join(timeout=2.0)
        for name, rec in recorders.items():
            _stop_one(name, rec)

    print(f"[launcher] done.")
    return rc


# ---- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from src.session.recorder_presets import get_dummy_recorders, get_production_recorders

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Fallback max seconds (0 = wait for stim or Ctrl+C)")
    ap.add_argument("--dummy", action="store_true",
                    help="Use dummy/simulated recorders (default: production)")
    ap.add_argument("--recorders", nargs="*", default=None,
                    help="Modalities to enable (default: all)")
    ap.add_argument("--with-stim", action="store_true",
                    help="Launch paradigm1_pickplace stim alongside recorders")
    ap.add_argument("--stim-task-id", type=int, default=None,
                    help="Task ID for stim (default: from collection config)")
    ap.add_argument("--stim-fullscreen", action="store_true",
                    help="Run stim fullscreen (default: windowed)")
    ap.add_argument("--stim-fast", type=float, default=1.0,
                    help="Stim time-compression for dry-runs")
    ap.add_argument("--stim-serial", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable ParallelBox TTL serial writes in stim "
                         "(default: on; pass --no-stim-serial for dry-runs "
                         "without the ParallelBox)")
    args = ap.parse_args(argv)

    session_dir = str(args.session_dir.resolve())

    # ---- recorders ----
    factory = get_dummy_recorders if args.dummy else get_production_recorders
    recs = factory(session_dir=session_dir, duration=args.duration)
    if args.recorders:
        recs = {k: v for k, v in recs.items() if k in args.recorders}

    # ---- stim command ----
    stim_cmd = None
    if args.with_stim:
        stim_cmd = [
            sys.executable, "-m", "src.stim.paradigm1_pickplace",
            "--once",
            "--windowed" if not args.stim_fullscreen else "--fullscreen",
            "--fast", str(args.stim_fast),
        ]
        if not args.stim_serial:
            stim_cmd.append("--no-serial")
        if args.stim_task_id is not None:
            stim_cmd.extend(["--task-id", str(args.stim_task_id)])

    return launch(recs, stim_cmd=stim_cmd, duration=args.duration)


if __name__ == "__main__":
    sys.exit(main())
