"""Launcher — runs each recorder in its OWN PROCESS, optionally with stim.

Threads would serialize CPU-bound work (PNG/mp4 encoding, numpy, serial
parsing) on the GIL; a process per recorder gives real parallelism.

Startup is two-phase.  Each child opens the device (connect + configure),
reports ready, and starts recording IMMEDIATELY — connecting proves nothing
about throughput, and poll cadence and driver batching only settle once the
loop has been running.  The child then passively confirms data is actually
flowing and reports; only when **all** slots confirm does the parent launch
stim and broadcast commit, which drops the pre-roll and restarts every
session clock (duration included).  Anything failing along the way (open /
confirmation / timeout / death) aborts the launch with the specific reason —
pre-roll is discarded, nothing is written.

Usage::

    # CLI: production hardware + stim
    python -m embodied_brain_collect.session.launcher --session-dir ./sessions/run1 --with-stim

    # CLI: dummy test without stim
    python -m embodied_brain_collect.session.launcher --dummy --session-dir ./test --duration 10

    # Code: custom setup
    from embodied_brain_collect.session.launcher import launch
    from embodied_brain_collect.recorders.factory import get_production_recorders

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
import threading
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


def _record_worker(rec: BaseRecorder, exc_box: list) -> None:
    """Run the record loop in a child-side thread; exceptions land in exc_box
    so the child main can exit non-zero after a clean join."""
    try:
        rec._record()
    except Exception as exc:
        rec.logger.opt(exception=True).error(
            f"[{rec.name}] recording crashed: {type(exc).__name__}: {exc}")
        exc_box.append(exc)


def _wait_data_flowing(rec: BaseRecorder, abort_evt,
                       timeout: float, window: float = 1.0) -> str:
    """Passively confirm the record loop is producing: the recorder's
    production signal grows.  Never touches the device — the loop already
    owns it.  Empty string = confirmed."""
    n_prev = rec._recording_signal()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if abort_evt.is_set():
            return "launch aborted during confirmation"
        time.sleep(window)
        n_now = rec._recording_signal()
        if n_now > n_prev:
            return ""
        n_prev = n_now
    return (f"open 后 {timeout:g}s 内缓冲无新增样本 — "
            "poll 循环没有在产数(设备静默或驱动卡死)")


def _recorder_main(name: str, rec: BaseRecorder, ctrl_q, go_evt, abort_evt) -> None:
    """Child process lifecycle:

    open (connect + configure) -> report ready -> record immediately
    (pre-roll) -> report once data is confirmed flowing -> wait for the
    commit broadcast (parent sends it right before stim launches) ->
    pre-roll dropped, session clock resets -> record until
    stop_event/duration -> teardown + save.
    """
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
    rec._launch_mode = True   # 预热期 duration 兜底不生效,commit 后才起算

    # 录制立即开始,不等其他 slot:first frame 通过不代表数据已经稳态
    #(驱动批量、设备缓冲、poll 节奏都要跑起来才算),预热期正好消化这些。
    # 预热数据在 commit 时统一丢弃。
    exc_box: list = []
    worker = threading.Thread(target=_record_worker, args=(rec, exc_box),
                              name=f"{name}:record", daemon=True)
    worker.start()

    # 确认数据在持续写入缓冲(stim 驱动的 slot 除外 —— marker 在 stim
    # 启动前天然无数据,直接放行)
    reason = ""
    if type(rec).data_before_stim:
        reason = _wait_data_flowing(rec, abort_evt, timeout=30.0)
    ctrl_q.put(("recording", name, reason))
    if reason:
        rec.discard_recording()
        rec.stop_event.set()
        worker.join(timeout=30.0)
        return

    # 等 commit 广播(parent 在 stim 启动前发出);期间被 abort 或 parent
    # 消失则丢弃预热数据退出,不落盘。
    deadline = time.time() + 600
    while not go_evt.wait(timeout=0.5):
        if abort_evt.is_set() or time.time() > deadline:
            rec.discard_recording()
            rec.stop_event.set()
            worker.join(timeout=30.0)
            return

    rec.request_commit()
    worker.join(timeout=300.0)   # teardown + 整段 npz/mp4 落盘可能较慢
    if worker.is_alive() or exc_box:
        # 非零退出:父进程把"录制中异常退出"记为运行期错误,而不是无声结束
        sys.exit(1)


def launch(
    recorders: dict[str, BaseRecorder],
    *,
    stim_cmd: Sequence[str] | None = None,
    duration: float = 0.0,
    confirm_timeout: float = 30.0,
) -> int:
    """Run each *recorders* entry in its own process until stim ends,
    duration elapses, or Ctrl+C.

    Startup is two-phase: children open, report ready, and start recording
    immediately (pre-roll) — being connected proves nothing about throughput,
    and poll cadence/driver batching only settle once the loop has run.  Each child
    then confirms data is actually flowing; only when **all** slots confirm
    does the parent launch stim and broadcast commit, which drops the
    pre-roll and restarts every session clock.  Duration anchors on the
    commit as well.

    Args:
        recorders: pre-configured recorder instances (``_open`` not yet called)
        stim_cmd: optional argv to spawn the stim program as a subprocess.
                  When the stim exits, all recorders are shut down gracefully.
        duration: fallback max seconds (0 = no limit, wait for stim/Ctrl+C)
        confirm_timeout: max seconds between "open OK" and "data confirmed
                         flowing" before the slot (and the launch) fails
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

    # ---- open phase: all children connect + configure in parallel ----
    for name, p in procs.items():
        print(f"[launcher] opening {name} ({type(recorders[name]).__name__}) ...")
        p.start()

    t0 = time.time()
    results: dict[str, str] = {}   # name -> "" (ready) or failure reason
    recording: dict[str, str] = {}  # name -> "" (data flowing) or failure
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
        elif msg[0] == "recording":
            # 快 slot 的确认先到 —— 存下,等 phase B 收齐
            _, name, reason = msg
            if name in procs and name not in recording:
                recording[name] = reason
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
    print("[launcher] all recorders ready — recording (pre-roll) ...")

    # ---- recording-confirm phase: children are already recording; wait
    # until every slot reports data actually flowing, then commit + stim.
    # The commit drops the pre-roll, so what gets saved anchors here. ----
    t0c = time.time()
    while len(recording) < len(procs):
        for name, p in procs.items():
            if name in recording:
                continue
            if not p.is_alive():
                recording[name] = f"process exited early (code={p.exitcode})"
            elif time.time() - t0c > confirm_timeout:
                recording[name] = (f"recording TIMEOUT after "
                                   f"{confirm_timeout:g}s — 无持续数据")
        try:
            msg = ctrl_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg[0] == "recording":
            _, name, reason = msg
            if name in procs and name not in recording:
                recording[name] = reason

    failed_rec = [(n, r) for n, r in recording.items() if r]
    for n, reason in failed_rec:
        print(f"[launcher] {n} ({type(recorders[n]).__name__}): {reason}")
        recorders[n]._log(f"[launcher] {n}: {reason}", echo=False)
    if failed_rec:
        print("[launcher] ERROR: 有 slot 确认失败(数据未在流动)— "
              "中止本次采集,预热数据不落盘。")
        abort_evt.set()
        for name, p in all_procs.items():
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        return LaunchResult(1, open_failures={**open_failures,
                                              **dict(failed_rec)})

    for name in recorders:
        print(f"[launcher] {name}: recording confirmed")

    # ---- start stim subprocess (if requested) ----
    stim_proc: subprocess.Popen | None = None
    if stim_cmd:
        stim_argv = list(stim_cmd)
        print(f"[launcher] stim: {' '.join(stim_argv)}")
        stim_proc = subprocess.Popen(
            stim_argv,
            cwd=str(SRC),  # repo root
        )

    go_evt.set()   # commit:所有 slot 丢弃预热数据,会话时钟从现在起算

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


def _write_session_meta(run_dir: Path, *, task_id: int | None = None,
                        environment: str | None = None,
                        recorders: dict[str, str] | None = None,
                        collect_info: dict | None = None,
                        layout: dict | None = None) -> None:
    """Stamp the session dir with what produced it and what it contains."""
    from embodied_brain_collect.session.config import (
        load_collect, load_meta, task_name)

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
    if environment is not None:
        meta["environment"] = environment
    if recorders:
        meta["recorders"] = recorders
    # 采集信息(collector_id 等操作员维护键,session.yaml 顶层 + run_session
    # CLI 覆盖的解析结果)随开录固化为 meta.yaml 顶层字段 —— 打包时以这份
    # 快照为准;未显式给时回退读 yaml(launcher 独立 CLI 路径)
    if collect_info is None:
        collect_info = load_collect()
    if collect_info:
        meta.update(collect_info)
    # 图纸模式(config.yaml 属性 + placements.csv 位姿)随开录固化:
    # task_name 两种模式都有(此处取图纸 config 的 task),scene/objects
    # 仅图纸模式;打包时以这份快照为准
    if layout:
        if layout.get("task_name"):
            meta.setdefault("task_name", layout["task_name"])
        meta["scene"] = layout["scene"]
        meta["objects"] = layout["objects"]
    import yaml
    (run_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def _recorder_names(recs: dict) -> dict[str, str]:
    """{槽位名: 设备显示名} — 写进 meta 的 recorders 字段;没配 display
    name 的槽位回退到 recorder 类名,保证每路都有可读的名字。"""
    return {name: (rec.display_name or type(rec).__name__)
            for name, rec in recs.items()}


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
    from embodied_brain_collect.session.config import load_checker
    try:
        report = qc_session(session_dir, checker_cfg=load_checker())
    except Exception as exc:      # noqa: BLE001
        print(f"[launcher] QC failed: {type(exc).__name__}: {exc}")
        return 1
    print_report(report)

    json_path = session_dir / "qc_report.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                    indent=2, default=str), encoding="utf-8")
    print(f"[launcher] QC 报告 -> {json_path}")

    # qc.html 是可选项(checker.yaml `html:`,默认开):渲染要解码视频抽
    # 缩略图、嵌入滤波副本,耗时且占体积;关掉后只留必选的 qc_report.json,
    # 需要时用 scripts/qc_report.py <session> 手动补渲染。
    if not checker_cfg.get("html", True):
        print("[launcher] qc.html 已关闭(checker.yaml `html: false`)— 跳过渲染")
        return 0

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
    from embodied_brain_collect.recorders.factory import get_dummy_recorders, get_production_recorders

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
    ap.add_argument("--no-env", action="store_true",
                    help="跳过图纸抽取/摆放阶段(默认:非 dummy 启动时先抽一张"
                         "未用图纸全屏显示,摆放完成按 n + Enter 才开录)")
    args = ap.parse_args(argv)

    # ---- 环境图纸:抽一张未用过的,摆放完成按 n + Enter 后才正式开录 ----
    env_rel = None
    if not args.dummy and not args.no_env:
        from embodied_brain_collect.session import environment as env
        try:
            env_rel = env.draw_unused()
        except RuntimeError as exc:
            print(f"[launcher] {exc}", file=sys.stderr)
            return 2
        print(f"[launcher] 本次环境图纸: {env_rel} — 场景任务: "
              f"{env.scene_task(env_rel)};全屏打开中,照图摆放完成后"
              f"按 n + Enter 开始采集")
        if env.show(env_rel) == "abort":
            print("[launcher] 已在图纸阶段取消,未开始任何录制。")
            return 2

    # ---- stim:--stim <kind> 指定,其余参数全部来自 configs/stim.yaml ----
    from embodied_brain_collect.session.config import load_tasks, task_name
    from embodied_brain_collect.stim.factory import build_stim_cmd

    kind = args.stim or ("paradigm1" if args.with_stim else None)
    if kind == "paradigm1" and task_id is None and env_rel is None:
        # 非图纸模式才回退 tasks.yaml;图纸模式由 config.yaml 的场景/任务驱动
        tasks = load_tasks()
        if not tasks:
            print("[launcher] configs/tasks.yaml 没有任务 — 无法带 paradigm1 启动",
                  file=sys.stderr)
            return 2
        task_id = int(tasks[0].get("task_id"))
    if env_rel is not None:
        from embodied_brain_collect.session import environment as env
        print(f"[launcher] 图纸模式 — 场景任务: {env.scene_task(env_rel)}")
    elif task_id is not None:
        print(f"[launcher] 本次任务: #{task_id} {task_name(task_id) or ''}")

    # The recording lives in a timestamped subdirectory of the shift root,
    # computed BEFORE the factory runs: recorders create their output dirs
    # in __init__, so the final path must exist by then.
    shift_root = args.session_dir.resolve()
    run_dir = shift_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] session dir: {run_dir}")

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

    # ---- meta.yaml: version + what this run is ----
    from embodied_brain_collect.session.environment import scene_layout
    _write_session_meta(
        run_dir, task_id=task_id, environment=env_rel,
        recorders=_recorder_names(recs),
        layout=(scene_layout(env_rel) or None) if env_rel else None)

    # ---- stim command ----
    stim_cmd = (build_stim_cmd(kind, task_id=task_id, environment=env_rel)
                if kind else None)
    if args.dummy and stim_cmd:
        # dummy = 无硬件试跑:串口强制关闭(机器上没有 ParallelBox 时 stim
        # 会在打开串口时直接崩掉)
        stim_cmd = stim_cmd + ["--no-serial"]
        print("[launcher] dummy 模式 — stim 串口已强制关闭(--no-serial)")

    rc = launch(recs, stim_cmd=stim_cmd, duration=args.duration)
    if env_rel is not None:
        if rc == 0:
            from embodied_brain_collect.session.environment import mark_used
            mark_used(env_rel, session=str(run_dir))
            print(f"[launcher] 采集成功 — 图纸 {env_rel} 已记入台账,"
                  "后续抽取不再出现")
        else:
            print(f"[launcher] 采集未成功(rc={rc})— 图纸 {env_rel} "
                  "留在抽取池")
    if not args.skip_qc:
        run_qc(run_dir)
    # 任务队列由 run_session 在内存中管理;这里不再改写 tasks.yaml。
    # 失败/中断的排查提示看 launch 返回的 open_failures / runtime_errors。
    return rc


if __name__ == "__main__":
    sys.exit(main())
