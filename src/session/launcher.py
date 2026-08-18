"""Launcher — runs each recorder in its OWN PROCESS, optionally with stim.

Threads would serialize CPU-bound work (PNG/mp4 encoding, numpy, serial
parsing) on the GIL; a process per recorder gives real parallelism.

Every recorder child first runs its ``_open()`` first-data gate and reports
over a control queue.  Recording starts only when **all** children report
ready; anything else (open failed / raised / timed out / died) aborts the
launch with the specific reason — nothing is recorded.

Usage::

    # CLI: production hardware + stim
    python -m src.session.launcher --session-dir ./sessions/run1 --with-stim

    # CLI: dummy test without stim
    python -m src.session.launcher --dummy --session-dir ./test --duration 10

    # Code: custom setup
    from src.session.launcher import launch
    from src.session.recorder_presets import get_production_recorders

    recs = get_production_recorders(session_dir="./sessions/run1")
    launch(recs, stim_cmd=["python", "-m", "src.stim.paradigm1_pickplace",
                           "--task-id", "0", "--windowed"])
"""

import argparse
import multiprocessing as mp
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from src.recorders.base import BaseRecorder

SRC = Path(__file__).resolve().parents[1]

# fork: children inherit the already-constructed recorder objects (they are
# not picklable — loguru loggers, open handles — but fork needs no pickling).
_CTX = mp.get_context("fork")


# ---- core ---------------------------------------------------------------


def _recorder_main(name: str, rec: BaseRecorder, ctrl_q, go_evt) -> None:
    """Child process: open (first-data gate) -> report -> wait for go ->
    record until stop_event/duration, then teardown + save."""
    try:
        ok = rec._open()
    except Exception as exc:
        ctrl_q.put(("ready", name, f"open ERROR — {type(exc).__name__}: {exc}"))
        rec.logger.opt(exception=True).error(f"[{name}] open crashed")
        return
    if not ok:
        ctrl_q.put(("ready", name,
                    "open FAILED — " + (rec._open_error or "unknown reason")))
        return
    ctrl_q.put(("ready", name, ""))
    if not go_evt.wait(timeout=600):  # parent aborted the launch
        try:
            rec._close()
        except Exception:
            pass
        return
    # One flow for every recorder: setup → loop → teardown (asyncio-style
    # recorders override _record with their own event loop).
    try:
        rec._record()
    except Exception as exc:
        rec.logger.opt(exception=True).error(
            f"[{name}] crashed: {type(exc).__name__}: {exc}")


def launch(
    recorders: dict[str, BaseRecorder],
    *,
    stim_cmd: Sequence[str] | None = None,
    duration: float = 0.0,
) -> int:
    """Run each *recorders* entry in its own process until stim ends,
    duration elapses, or Ctrl+C.

    Args:
        recorders: pre-configured recorder instances (``_open`` not yet called)
        stim_cmd: optional argv to spawn the stim program as a subprocess.
                  When the stim exits, all recorders are shut down gracefully.
        duration: fallback max seconds (0 = no limit, wait for stim/Ctrl+C)
    """
    if not recorders:
        print("[launcher] nothing to run.")
        return 0

    print(f"[launcher] modalities: {list(recorders.keys())}")

    ctrl_q = _CTX.Queue()
    go_evt = _CTX.Event()
    procs: dict[str, mp.Process] = {}
    for name, rec in recorders.items():
        rec.stop_event = _CTX.Event()   # parent signals graceful stop
        rec._hb_queue = ctrl_q          # heartbeats route to the parent
        procs[name] = _CTX.Process(
            target=_recorder_main, args=(name, rec, ctrl_q, go_evt),
            name=f"rec:{name}", daemon=True)
    all_procs = dict(procs)

    # ---- open phase: all children run their first-data gate in parallel ----
    for name, p in procs.items():
        print(f"[launcher] opening {name} ({type(recorders[name]).__name__}) ...")
        p.start()

    t0 = time.time()
    results: dict[str, str] = {}   # name -> "" (ready) or failure reason
    while len(results) < len(procs):
        # per-recorder open watchdog
        for name, p in procs.items():
            if name in results:
                continue
            timeout = float(getattr(recorders[name].config,
                                    "open_timeout", 30.0) or 30.0)
            if time.time() - t0 > timeout:
                results[name] = f"open TIMEOUT after {timeout:g}s"
        # children that died without reporting
        for name, p in procs.items():
            if name in results:
                continue
            if not p.is_alive():
                results[name] = f"process exited early (code={p.exitcode})"
        # control messages
        try:
            msg = ctrl_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg[0] == "ready":
            _, name, reason = msg
            if name in procs and name not in results:
                results[name] = reason
        # ("hb", ...) messages are ignored during the open phase

    failed = [(n, r) for n, r in results.items() if r]
    for n, reason in failed:
        print(f"[launcher] {n} ({type(recorders[n]).__name__}): {reason}")
        recorders[n]._log(f"[launcher] {n}: {reason}", echo=False)
    if failed:
        print("[launcher] ERROR: not all recorders opened — "
              "aborting, nothing will be recorded.")
        go_evt.set()  # release the ready children from their wait
        for name, p in all_procs.items():
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        return 1

    for name in recorders:
        print(f"[launcher] {name}: open OK")
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

    go_evt.set()

    # ---- recording phase: monitor children + stim + duration ----
    rc = 0
    t0 = time.time()
    stats: dict[str, str] = {}
    try:
        while procs:
            elapsed = time.time() - t0

            # --- stop conditions ---
            if 0 < duration <= elapsed:
                print(f"\n[launcher] duration {duration}s reached.")
                break

            if stim_proc is not None and stim_proc.poll() is not None:
                rc = stim_proc.returncode
                print(f"\n[launcher] stim exited (code={rc}) — "
                      f"stopping recorders.")
                break

            # --- child health ---
            for name, p in list(procs.items()):
                if not p.is_alive():
                    print(f"\n[launcher] {name} process exited "
                          f"(code={p.exitcode})")
                    del procs[name]

            # --- heartbeat aggregation ---
            while True:
                try:
                    msg = ctrl_q.get_nowait()
                except queue.Empty:
                    break
                if msg[0] == "hb":
                    _, name, line = msg
                    stats[name] = line

            # --- status line every 1 s ---
            if int(elapsed) > int(elapsed - 0.5):
                parts = [f"t={elapsed:5.1f}s"]
                for name in recorders:
                    parts.append(f"{name}:{stats[name]}" if name in stats
                                 else name)
                print("  ".join(parts).ljust(140))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C — stopping ...")
    finally:
        # Ask every child to stop gracefully (their _loop / signal tasks
        # poll the stop_event), then hard-kill stragglers.
        for name, rec in recorders.items():
            rec.stop_event.set()

        if stim_proc is not None and stim_proc.poll() is None:
            stim_proc.terminate()
            try:
                stim_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stim_proc.kill()

        for name, p in all_procs.items():
            p.join(timeout=15.0)   # children need time to flush mp4/npz
        for name, p in all_procs.items():
            if p.is_alive():
                print(f"[launcher] {name} did not stop — killing.")
                p.terminate()
                p.join(timeout=5.0)

    print("[launcher] done.")
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
