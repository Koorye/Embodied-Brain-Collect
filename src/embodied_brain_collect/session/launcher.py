"""Launcher — runs each recorder in its OWN PROCESS, optionally with stim.

Threads would serialize CPU-bound work (PNG/mp4 encoding, numpy, serial
parsing) on the GIL; a process per recorder gives real parallelism.

Every recorder child first runs its ``_open()`` first-data gate and reports
over a control queue.  Recording starts only when **all** children report
ready; anything else (open failed / raised / timed out / died) aborts the
launch with the specific reason — nothing is recorded.

Usage::

    # CLI: production hardware + stim
    python -m embodied_brain_collect.session.launcher --session-dir ./sessions/run1 --with-stim

    # CLI: dummy test without stim
    python -m embodied_brain_collect.session.launcher --dummy --session-dir ./test --duration 10

    # Code: custom setup
    from embodied_brain_collect.session.launcher import launch
    from embodied_brain_collect.session.recorder_presets import get_production_recorders

    recs = get_production_recorders(session_dir="./sessions/run1")
    launch(recs, stim_cmd=["python", "-m", "embodied_brain_collect.stim.paradigm1_pickplace",
                           "--task-id", "0", "--windowed"])

On Windows the children are spawned, which re-imports the caller's main
module — call ``launch`` from a script guarded by the usual
``if __name__ == "__main__":`` (running the CLI via ``python -m`` is
already safe).
"""

import argparse
import multiprocessing as mp
import queue
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

from embodied_brain_collect.recorders.base import BaseRecorder

SRC = Path(__file__).resolve().parents[3]  # repo root

# fork: children inherit the already-constructed recorder objects (they are
# not picklable — loguru loggers, open handles — but fork needs no pickling).
# Windows has no fork, so use spawn there: each child re-imports this module
# and the recorders are pickled.  BaseRecorder.__getstate__/__setstate__ strip
# and rebuild the unpicklable bits, and stop_event/_hb_queue are replaced with
# picklable multiprocessing primitives before the children start.
_CTX = (mp.get_context("spawn") if sys.platform == "win32"
        else mp.get_context("fork"))


# ---- core ---------------------------------------------------------------


class LaunchResult(int):
    """``launch()`` 的返回码 —— int 兼容(0 成功),附带失败明细。

    ``open_failures``: 启动阶段 open 失败的 slot → 原因(本次没有数据)。
    ``runtime_errors``: 录制/保存阶段异常退出的 slot → 原因(数据可能不完整)。
    编排层(run_session/preflight)据此给出分设备的排查指引。
    """

    open_failures: dict[str, str]
    runtime_errors: dict[str, str]

    def __new__(cls, rc: int, open_failures: dict | None = None,
                runtime_errors: dict | None = None):
        obj = super().__new__(cls, rc)
        obj.open_failures = dict(open_failures or {})
        obj.runtime_errors = dict(runtime_errors or {})
        return obj

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return (f"LaunchResult({int(self)}, open_failures={self.open_failures!r}, "
                f"runtime_errors={self.runtime_errors!r})")


def _recorder_main(name: str, rec: BaseRecorder, ctrl_q, go_evt, abort_evt) -> None:
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
    # Wait for go; close quietly if the parent aborts the launch (another
    # recorder failed to open) or disappears without sending anything.
    deadline = time.time() + 600
    while not go_evt.wait(timeout=0.5):
        if abort_evt.is_set() or time.time() > deadline:
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
        # 非零退出:父进程把"录制中异常退出"记为运行期错误,而不是无声结束
        sys.exit(1)


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
    abort_evt = _CTX.Event()
    procs: dict[str, mp.Process] = {}
    for name, rec in recorders.items():
        rec.stop_event = _CTX.Event()   # parent signals graceful stop
        rec._hb_queue = ctrl_q          # heartbeats route to the parent
        procs[name] = _CTX.Process(
            target=_recorder_main, args=(name, rec, ctrl_q, go_evt, abort_evt),
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
    open_failures = dict(failed)
    if failed:
        print("[launcher] ERROR: not all recorders opened — "
              "aborting, nothing will be recorded.")
        abort_evt.set()  # ready children close WITHOUT recording
        for name, p in all_procs.items():
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        return LaunchResult(1, open_failures=open_failures)

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
            cwd=str(SRC),  # repo root
        )

    go_evt.set()

    # ---- recording phase: monitor children + stim + duration ----
    rc = 0
    t0 = time.time()
    stats: dict[str, str] = {}
    runtime_errors: dict[str, str] = {}
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
                    if p.exitcode:
                        runtime_errors[name] = (
                            f"Recording process exited abnormally (code={p.exitcode}) — "
                            f"see {name}/{name}.log for traceback")
                    else:
                        runtime_errors[name] = (
                            "Recording process exited unexpectedly before receiving stop "
                            f"(code=0) — see {name}/{name}.log for details")
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
    return LaunchResult(rc, open_failures=open_failures,
                        runtime_errors=runtime_errors)


# ---- post-run: session metadata + automatic QC ---------------------------


def _write_session_meta(run_dir: Path, *, task_id: int | None = None) -> None:
    """Stamp the session dir with what produced it and what it contains."""
    from embodied_brain_collect.config.load import load_meta, task_name

    try:
        meta = dict(load_meta())
    except FileNotFoundError:
        meta = {}
    meta.update({
        "session_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
    if task_id is not None:
        meta["task_id"] = task_id
        name = task_name(task_id)
        if name:
            meta["task_name"] = name
    import yaml
    (run_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def run_qc(session_dir: Path) -> int:
    """QC the just-saved session: console report + JSON + HTML page.

    Runs after ``launch()`` returns — every child has flushed its NPZ by
    then.  The QC verdict is reported but deliberately does not change the
    launcher's exit code: recording succeeded or failed on its own terms.
    """
    import json
    from embodied_brain_collect.checkers import print_report, qc_session
    from embodied_brain_collect.visualizers.qc_page import Options, build_page

    print(f"\n[launcher] running QC on {session_dir} ...")
    try:
        from embodied_brain_collect.config.load import load_checker
        try:
            checker_cfg = load_checker()
        except FileNotFoundError:
            checker_cfg = {}
        report = qc_session(session_dir, checker_cfg=checker_cfg)
    except Exception as exc:      # noqa: BLE001
        print(f"[launcher] QC failed: {type(exc).__name__}: {exc}")
        return 1
    print_report(report)

    json_path = session_dir / "qc_report.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                    indent=2, default=str), encoding="utf-8")
    print(f"[launcher] QC 报告 -> {json_path}")

    try:
        html = build_page(report.to_dict(), session_dir, Options())
        html_path = session_dir / "qc.html"
        html_path.write_text(html, encoding="utf-8")
        mb = html_path.stat().st_size / 1e6
        print(f"[launcher] QC 页面 -> {html_path} ({mb:.1f} MB)")
    except Exception as exc:      # noqa: BLE001
        print(f"[launcher] QC 页面生成失败: {type(exc).__name__}: {exc}")
    return 0


# ---- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from embodied_brain_collect.session.recorder_presets import get_dummy_recorders, get_production_recorders

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session-dir", type=Path, required=True,
                    help="班次根目录(如 data/session-day)。实际录制目录为其下的 "
                         "yyyy-MM-dd-HH-mm-ss 子目录")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Fallback max seconds (0 = wait for stim or Ctrl+C)")
    ap.add_argument("--dummy", action="store_true",
                    help="Use dummy/simulated recorders (default: production)")
    ap.add_argument("--recorders", nargs="*", default=None,
                    help="Modalities to enable (default: all)")
    from embodied_brain_collect.stim.factory import STIM_KINDS
    ap.add_argument("--stim", choices=sorted(STIM_KINDS), default=None,
                    help="指定刺激程序(paradigm1 / sync_test),不传则不启动;"
                         "stim 的全部参数来自 configs/stim.yaml")
    ap.add_argument("--with-stim", action="store_true",
                    help="等价 --stim paradigm1(向后兼容)")
    ap.add_argument("--skip-qc", action="store_true",
                    help="采集后跳过自动 QC(默认跑)")
    args = ap.parse_args(argv)

    # ---- stim:--stim <kind> 指定,其余参数全部来自 configs/stim.yaml ----
    from embodied_brain_collect.config.load import load_tasks
    from embodied_brain_collect.stim.factory import build_stim_cmd

    kind = args.stim or ("paradigm1" if args.with_stim else None)
    task_id = None
    if kind == "paradigm1":
        tasks = load_tasks()
        if not tasks:
            print("[launcher] configs/tasks.yaml 没有任务 — 无法带 paradigm1 启动",
                  file=sys.stderr)
            return 2
        task_id = int(tasks[0].get("task_id"))
        print(f"[launcher] 本次任务: #{task_id} {tasks[0].get('task_name', '')}")

    # The recording lives in a timestamped subdirectory of the shift root,
    # computed BEFORE the factory runs: recorders create their output dirs
    # in __init__, so the final path must exist by then.
    shift_root = args.session_dir.resolve()
    run_dir = shift_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] session dir: {run_dir}")

    # ---- meta.yaml: version + what this run is ----
    _write_session_meta(run_dir, task_id=task_id)

    # ---- recorders (slots selected BEFORE construction — see presets) ----
    if args.dummy:
        # dummy 模式:EEG 事件节奏跟随所选 stim,保证对齐链路可全绿
        recs = get_dummy_recorders(session_dir=str(run_dir),
                                   duration=args.duration,
                                   slots=args.recorders, stim=kind)
    else:
        recs = get_production_recorders(session_dir=str(run_dir),
                                        duration=args.duration,
                                        slots=args.recorders)

    # ---- stim command ----
    stim_cmd = build_stim_cmd(kind, task_id=task_id) if kind else None

    rc = launch(recs, stim_cmd=stim_cmd, duration=args.duration)
    if not args.skip_qc:
        run_qc(run_dir)
    # 任务队列由 run_session 在内存中管理;这里不再改写 tasks.yaml。
    # 失败/中断的排查提示看 launch 返回的 open_failures / runtime_errors。
    return rc


if __name__ == "__main__":
    sys.exit(main())
