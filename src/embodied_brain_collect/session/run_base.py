"""采集会话公共引擎 —— 三个入口脚本(图纸/任务/视频)共享的编排逻辑。

每个入口只负责自己模式特有的部分,通过 :class:`ModeHooks` 注入四个缝:

* 开录前准备     :meth:`ModeHooks.preroll`   图纸全屏摆放 / 提示按 Enter /
                                              视频无操作
* meta 字段      :meth:`ModeHooks.meta_kwargs` environment / task_id /
                                              task_name+scene
* stim 命令      :meth:`ModeHooks.stim_cmd`    --environment / --task-id /
                                              --trials-json
* 成功记账       :meth:`ModeHooks.on_result`   图纸台账 / 视频台账 / 无

队列构建、录制(record_one)、n/r/q 确认与 meta 状态合成、失败排查提示、
会话汇总(collect_runs/print_summary)都在这里,三个入口行为保持一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from embodied_brain_collect.recorders.factory import (
    get_dummy_recorders, get_production_recorders)
from embodied_brain_collect.session import environment as env
from embodied_brain_collect.session.config import (
    FRAMEWORK_KEYS, SESSION_RUN_KEYS, task_name)
from embodied_brain_collect.session.launcher import (
    _recorder_names, _write_session_meta, launch, run_qc)
from embodied_brain_collect.session.troubleshooting import format_failure_help


# =============================================================================
# 汇总报告
# =============================================================================

def load_qc(run_dir: Path) -> dict | None:
    p = run_dir / "qc_report.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def has_any_data(run_dir: Path) -> bool:
    """该 session 是否有任何一个 recorder 落了数据文件(npz/mp4)。

    启动失败时 recorder 的目录仍然会建出来(构造时就建),里面只有 .log —
    这样的"空录制"不算一次有效采集,也不计入无误数据比例的分母。
    """
    try:
        for d in run_dir.iterdir():
            if d.is_dir() and (any(d.glob("*.npz")) or any(d.glob("*.mp4"))):
                return True
    except OSError:
        return False
    return False


def _iter_findings(qc: dict):
    """(流名, finding) —— 流级 findings 带所属 slot;会话级 findings 的
    ``subject`` 是它针对的 slot(如 StreamPresent 的缺失目录名)。"""
    for st_name, st in (qc.get("streams") or {}).items():
        for f in st.get("findings", []):
            yield st_name, f
    for f in qc.get("findings", []):
        yield (f.get("subject") or ""), f


def _count_by_check(qc: dict, level: str, out: dict) -> tuple[dict[str, int], set[str]]:
    """把一个 QC 报告里该级别的 findings 按 检查项 -> {流: 条数} 计入
    ``out``;返回 (该 session 的 检查项 -> 条数, 出现过的检查项集合)。
    集合供调用方按"每 session 每检查项只计 1 次"更新涉及 session 数。"""
    per_check: dict[str, Counter] = {}
    bucket = out["errors" if level == "ERROR" else "warnings"]
    for stream, f in _iter_findings(qc):
        if f.get("level") != level:
            continue
        check = f.get("check", "?")
        key = stream or "会话级"
        bucket.setdefault(check, Counter())[key] += 1
        per_check.setdefault(check, Counter())[key] += 1
    return ({c: int(sum(s.values())) for c, s in per_check.items()},
            set(per_check))


def meta_task_label(run_dir: Path) -> str:
    """meta 里的任务标识:视频模式取 task_name;图纸模式取 environment 对应
    的场景任务;旧模式(有 task_id)取 tasks.yaml 的任务名。"""
    try:
        import yaml
        meta = yaml.safe_load(
            (run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    if meta.get("task_name"):
        return str(meta["task_name"])
    if meta.get("task_id") is not None:
        return task_name(int(meta["task_id"])) or ""
    environment = meta.get("environment")
    return env.scene_task(environment) if environment else ""


def _session_span(qc: dict) -> float:
    w = qc.get("window") or {}
    t0, t1 = w.get("t0"), w.get("t1")
    return float(t1 - t0) if t0 is not None and t1 is not None else 0.0


def collect_runs(runs: list[Path],
                 outcomes: dict[str, str] | None = None) -> dict:
    """把一组 session 目录的 QC 报告压成一张汇总表。

    ``outcomes`` 是 dir 名 -> 结局标签(kept/rerun/quit);保留、重跑、
    退出产生的目录一律计入统计,结局只影响明细列的标注。

    分层计数(无误比例的分母只含"有数据且跑过 QC"的录制):

    * ``n_empty``  空录制 —— 没有任何数据文件(典型:启动失败),标 ∅;
    * ``n_no_qc``  有数据但没有 qc_report.json(--skip-qc 等),标 ?;
    * ``n_judged`` 有数据且有 QC,只有这些计入 ``n_clean/n_judged``。

    ``errors``/``warnings`` 是 检查项 -> {流名: 条数}(流名空串 = 会话级),
    ``*_sessions`` 是 检查项 -> 涉及的 session 数。
    """
    outcomes = outcomes or {}
    out = {
        "sessions": [], "n_sessions": 0, "n_clean": 0,
        "n_empty": 0, "n_no_qc": 0, "n_judged": 0,
        "errors": {}, "warnings": {},
        "error_sessions": Counter(), "warning_sessions": Counter(),
    }
    for run_dir in sorted(runs):
        base = {
            "dir": run_dir.name,
            "status": outcomes.get(run_dir.name, "-"),
            "task": meta_task_label(run_dir) or "-",
        }
        qc = load_qc(run_dir)

        if not has_any_data(run_dir):
            # 空录制:没有数据可判,不计入无误比例;QC 的 StreamPresent
            # 照样统计 —— 明细里的 9 ERROR 就是它,正好解释"为什么空"。
            base["empty"] = True
            out["n_sessions"] += 1
            out["n_empty"] += 1
            if qc is not None:
                errs, err_checks = _count_by_check(qc, "ERROR", out)
                warns, warn_checks = _count_by_check(qc, "WARN", out)
                base["errors"] = errs
                base["warnings"] = warns
                base["span_s"] = _session_span(qc)
                out["error_sessions"].update(err_checks)
                out["warning_sessions"].update(warn_checks)
            out["sessions"].append(base)
            continue

        if qc is None:
            base["qc"] = None
            out["n_sessions"] += 1
            out["n_no_qc"] += 1
            out["sessions"].append(base)
            continue

        errs, err_checks = _count_by_check(qc, "ERROR", out)
        warns, warn_checks = _count_by_check(qc, "WARN", out)
        out["error_sessions"].update(err_checks)
        out["warning_sessions"].update(warn_checks)
        clean = not errs
        out["n_sessions"] += 1
        out["n_judged"] += 1
        out["n_clean"] += int(clean)
        out["sessions"].append({
            **base,
            "clean": clean,
            "errors": errs, "warnings": warns,
            "span_s": _session_span(qc),
        })
    out["errors"] = {c: dict(s) for c, s in out["errors"].items()}
    out["warnings"] = {c: dict(s) for c, s in out["warnings"].items()}
    return out


_STATUS_LABELS = {"kept": "保留", "rerun": "重跑", "quit": "退出",
                  "fail_quit": "失败退出",
                  "success": "成功", "failed": "失败", "error": "QC错误"}


def _print_check_counts(title: str, by_check: dict[str, dict[str, int]],
                        sess_counts: Counter, n: int, limit: int | None = None,
                        ) -> None:
    """错误/警告小节:检查项一行 + 流明细一行(计数降序)。"""
    items = sorted(by_check.items(),
                   key=lambda kv: -sum(kv[1].values()))
    if limit:
        items = items[:limit]
    if not items:
        return
    print(f"\n  {title}:")
    for check, streams in items:
        total = sum(streams.values())
        ns = sess_counts.get(check, 0)
        print(f"    {check:<24} {total:>4} 条 · {ns}/{n} 个 session"
              f" ({ns / n:.1%})")
        parts = [f"{s} {c}" for s, c in
                 sorted(streams.items(), key=lambda kv: -kv[1])]
        print(f"        流: {' · '.join(parts)}")


def print_summary(summary: dict, *, title: str) -> None:
    n = summary["n_sessions"]
    clean = summary["n_clean"]
    judged = summary["n_judged"]
    frac = clean / judged if judged else 0.0
    print("\n" + "=" * 68)
    print(f"  会话汇总 — {title}")
    print("=" * 68)
    status = Counter(s.get("status", "-") for s in summary["sessions"])
    parts = [f"录制 {n} 条"]
    for key, label in (("kept", "保留"), ("rerun", "重跑"), ("quit", "退出"),
                       ("fail_quit", "失败退出")):
        if status.get(key):
            parts.append(f"{label} {status[key]}")
    print("  " + " · ".join(parts))
    if summary["n_empty"]:
        print(f"  其中空录制 {summary['n_empty']} 条(未产生任何数据文件,"
              "多为启动失败 — 明细标 ∅,不计入无误比例)")
    if summary["n_no_qc"]:
        print(f"  有数据但缺 QC 报告 {summary['n_no_qc']} 条(明细标 ?,"
              "不计入无误比例)")
    done, total = summary.get("tasks_done"), summary.get("tasks_total")
    if judged:
        clean_line = (f"无误数据: {clean}/{judged} ({frac:.1%})"
                      f"  · 分母 = 有数据且跑过 QC 的 {judged} 条")
    else:
        clean_line = ("无误数据: 无可判定的录制(本次全部为空录制"
                      "或没有 QC 报告)")
    if done is not None and total is not None:
        print(f"  完成: {len(done)}/{len(total)} 项      {clean_line}")
    else:
        print(f"  {clean_line}")
    total_span = sum(s.get("span_s", 0.0) for s in summary["sessions"])
    print(f"  有效录制时长合计: {total_span / 60:.1f} 分钟")

    _print_check_counts("错误(按检查项 → 流)",
                        summary["errors"], summary["error_sessions"], n)
    if "StreamPresent" in summary["errors"]:
        print("\n  注: StreamPresent = 该 recorder 的目录在、但没有任何数据"
              "文件(未启动成功或没保存)。")
        print("      空录制的目录里只有 .log,对应明细行标 ∅;"
              "具体原因看 <slot>/<slot>.log 与启动错误提示。")
    _print_check_counts("警告(按检查项 → 流,前 10)",
                        summary["warnings"], summary["warning_sessions"],
                        n, limit=10)

    print("\n  明细(全部录制,按时间;结局 = 保留/重跑/退出):")
    for s in summary["sessions"]:
        # ∅ 空录制 > ✓/✗ QC 判定 > ? 缺 QC 报告
        if s.get("empty"):
            mark = "∅"
        elif s.get("clean"):
            mark = "✓"
        elif "clean" in s:
            mark = "✗"
        else:
            mark = "?"
        span = f"{s.get('span_s', 0):.0f}s" if s.get("span_s") else "-"
        n_err = sum(s.get("errors", {}).values())
        err = f"  {n_err} ERROR" if n_err else ""
        label = _STATUS_LABELS.get(s.get("status"), s.get("status", "-"))
        if s.get("empty"):
            label += "(空录制)"
        print(f"    {mark} {s['dir']}  任务[{s['task']}]  "
              f"{span:>6}  {label}{err}")
    print()


def save_summary(summary: dict, session_root: Path) -> Path:
    p = session_root / "run_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p


# =============================================================================
# 交互与状态
# =============================================================================

def mark_meta(run_dir: Path, status: str) -> None:
    """把本次采集的结局写进该 session 的 meta.yaml(status 字段)。

    三档合成:QC 有 ERROR → error(无论 n/r/q);QC 无错且 r → failed
    (操作员判失败重采);QC 无错且 n/q → success。仅 success 记台账。
    标记随数据目录走 —— 打包、汇总不依赖班次根的 run_summary.json
    也能识别单条数据的有效性。
    """
    import yaml
    p = run_dir / "meta.yaml"
    meta: dict = {}
    if p.is_file():
        try:
            meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}            # meta 坏了也要把结局记上
    meta["status"] = status
    p.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")


def ask_next(qc_error: bool = False) -> str:
    """录制结束后的确认输入,防误触:必须输入 n/r/f/q 之一再回车。

    只回车、输错字母都会要求重输 —— 录制现场经常双手忙着摘设备,
    误触一下 Enter 不能直接吞掉一条录制。
    ``qc_error`` 只留作签名兼容:QC 出错时的说明已由 run_queue 统一打印
    (hooks.qc_error_note + 保留提示),这里不再重复一遍。
    """
    while True:
        ans = input("  输入 n(当前采集成功,下一条) / "
                    "r(当前采集失败,重跑) / "
                    "f(当前采集失败,退出) / q(成功并退出),"
                    "回车确认: ").strip().lower()
        if ans == "n":
            return "next"
        if ans == "r":
            return "rerun"
        if ans == "f":
            return "fail_quit"
        if ans == "q":
            return "quit"
        print(f"  无效输入 {ans!r} —— 只接受 n(next) / r(retry) / "
              "f(fail & quit) / q(quit),请重新输入")


def qc_slot_errors(qc: dict | None) -> dict[str, str]:
    """QC 报告 → slot 级错误摘要(stream 键即 slot 名)。"""
    out: dict[str, str] = {}
    if not qc:
        return out
    for stream, st in (qc.get("streams") or {}).items():
        errs = [f for f in st.get("findings", [])
                if f.get("level") == "ERROR"]
        if errs:
            checks = ", ".join(sorted({str(f.get("check", "?"))
                                       for f in errs})[:4])
            out[stream] = f"QC 报 {len(errs)} 条 ERROR({checks})"
    return out


# =============================================================================
# 模式钩子与录制引擎
# =============================================================================

@dataclass
class ModeHooks:
    """一个入口 = 一组模式行为。三个缝 + 记账,见模块 docstring。"""

    label: str                                            # 模式名(汇总显示)
    # 开录前准备(打印横幅 + 图纸摆放/Enter 确认);False = 取消,不开录
    preroll: Callable[[dict, Path, argparse.Namespace], bool]
    # _write_session_meta 的模式字段(environment / task_id / task_name+scene)
    meta_kwargs: Callable[[dict], dict]
    # 完整 stim 子进程 argv。run_dir = 本次录制目录 —— 需要把产物写进录制
    # 目录的范式用它(如 video_rate 的 --result-dir <run_dir>)
    stim_cmd: Callable[[dict, str, Path], list]
    # 队列项的稳定标识(汇总 tasks_done/total 用)
    job_id: Callable[[dict], object]
    # 录制保留/退出后的模式记账(台账/拷贝等);rc = launcher 返回码
    on_result: Callable[[dict, Path, str, str, int], None] = field(
        default=lambda *a: None)
    # QC 有 ERROR 时、n/r/q 询问前的模式提示(如"图纸不记台账")
    qc_error_note: str = ""


def new_run_dir(session_root: Path) -> Path:
    """时间戳录制目录;同一秒连续录制加后缀避让 —— 绝不覆盖旧数据。"""
    run_dir = session_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = run_dir.with_name(f"{run_dir.name}-{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def record_one(session_root: Path, job: dict, *, stim: str,
               args, collect: dict | None, hooks: ModeHooks,
               ) -> tuple[Path, int]:
    """录一个队列项,返回 (run_dir, launcher 返回码)。

    取消(preroll 返回 False)时返回码 2,未开录;目录留档。
    """
    run_dir = new_run_dir(session_root)
    if not hooks.preroll(job, run_dir, args):
        print("[run_session] 开录前取消 — 未开录,目录留档")
        return run_dir, 2

    if args.dummy:
        recs = get_dummy_recorders(session_dir=str(run_dir),
                                   duration=args.duration,
                                   slots=args.recorders, stim=stim)
    else:
        recs = get_production_recorders(session_dir=str(run_dir),
                                        duration=args.duration,
                                        slots=args.recorders)

    _write_session_meta(run_dir, recorders=_recorder_names(recs),
                        collect_info=collect, **hooks.meta_kwargs(job))
    stim_cmd = hooks.stim_cmd(job, stim, run_dir)
    if args.dummy and stim_cmd:
        # dummy = 无硬件试跑:串口强制关闭 —— 机器上没有 ParallelBox 时
        # stim 会在打开串口时直接崩掉(serial: true 也一样)
        stim_cmd = stim_cmd + ["--no-serial"]
        print("[run_session] dummy 模式 — stim 串口已强制关闭(--no-serial),"
              "marker 走 UDP 通路")

    rc = 1
    try:
        rc = launch(recs, stim_cmd=stim_cmd, duration=args.duration)
        if not args.skip_qc:
            run_qc(run_dir)
    except KeyboardInterrupt:
        print("\n[run_session] Ctrl+C — 录制中止,目录保留")
    return run_dir, rc


def run_queue(session_root: Path, queue: list[dict], *, stim: str,
              args, collect: dict | None, hooks: ModeHooks,
              seed: int, strict_rc: bool = False) -> int:
    """按队列逐项录制 → n/r/f/q 确认与状态合成 → 失败排查 → 会话汇总。

    ``strict_rc=True``(视频模式用):launcher 返回码非 0(刺激中止/崩溃)
    一律记 failed,不再吃"QC 干净"的亏 —— 没录完整就是没录成,哪怕
    已落盘的流看起来没问题。

    逐条打包(--pack-episode / session.yaml pack_episode: true):每条
    **QC 无错且被保留**(n/q)后自动打成独立 episode 数据集;error/
    failed 的目录留档但不出数据集。
    """
    from embodied_brain_collect.session.config import (
        load_recorders, load_session_run)
    from embodied_brain_collect.session.launcher import (
        run_pack_episode, warm_pack_dependencies)

    pack_on = (getattr(args, "pack_episode", False)
               or bool(load_session_run().get("pack_episode")))
    if pack_on:
        # 趁操作员看队列/摆放的工夫,后台预载打包依赖(torch ~11s),
        # 之后每条录完的打包零导入等待
        warm_pack_dependencies()
    disabled = [s for s, c in load_recorders().items()
                if not c.get("enabled", True)]
    if disabled:
        print(f"[run_session] 已停用槽位(recorders.yaml enabled: false,本次"
              f"不启动也不采集): {disabled}")

    kept_jobs: list[dict] = []
    runs: list[Path] = []           # 全部录制目录(保留/重采/退出都计入汇总)
    outcomes: dict[str, str] = {}   # dir 名 -> kept / rerun / quit / fail_quit
    slot_fail: Counter = Counter()  # slot -> 本次会话累计出错次数(排查提示用)
    queue_pos = 0
    task_no = 0
    interrupted = False
    try:
        while queue_pos < len(queue):
            job = queue[queue_pos]
            task_no += 1
            remaining = len(queue) - queue_pos
            print(f"\n▶ 第 {task_no} 次录制 · 队列剩余 {remaining} · "
                  f"已完成 {len(kept_jobs)}/{len(queue)}")

            run_dir, rc = record_one(session_root, job, stim=stim, args=args,
                                     collect=collect, hooks=hooks)
            runs.append(run_dir)
            outcomes[run_dir.name] = "-"   # 结局由下面的选择更新
            print(f"\n[run_session] launcher 返回码 {int(rc)}"
                  + ("(Esc 中止)" if rc == 2 else ""))

            # 启动错误(设备 open 失败)与录制/保存错误分别给排查指引;
            # 累计次数决定提示力度:首次失败只建议重采,多次失败展开
            # 分设备排查方案(见 troubleshooting.self_check_flow)。
            open_failures = dict(getattr(rc, "open_failures", {}) or {})
            runtime_errors = dict(getattr(rc, "runtime_errors", {}) or {})
            for slot in list(open_failures) + list(runtime_errors):
                slot_fail[slot] += 1
            if open_failures:
                print(format_failure_help(
                    open_failures, phase="startup",
                    fail_counts=slot_fail, log_root=str(run_dir)))
            elif runtime_errors:
                print(format_failure_help(
                    runtime_errors, phase="data",
                    fail_counts=slot_fail, log_root=str(run_dir)))

            qc_error = False
            qc = load_qc(run_dir)
            if qc:
                findings = [f for st in qc.get("streams", {}).values()
                            for f in st.get("findings", [])]
                findings += qc.get("findings", [])   # 会话级(StreamPresent 等)
                n_err = sum(1 for f in findings if f.get("level") == "ERROR")
                verdict = "无 ERROR" if n_err == 0 else f"{n_err} 条 ERROR"
                # qc.html 是 checker.yaml 的可选项(html:,默认开);关掉时
                # 细节只在 qc_report.json 里
                from embodied_brain_collect.session.config import load_checker
                html_on = (load_checker() or {}).get("html", True)
                detail = run_dir / ("qc.html" if html_on else "qc_report.json")
                print(f"  QC 判定: {qc.get('level')} ({verdict}) — 细节见 {detail}")
                if n_err:
                    qc_error = True
                    if hooks.qc_error_note:
                        print(f"  ⚠ {hooks.qc_error_note}")
                    print("  确认无碍可按 n 保留数据")
                if n_err and not open_failures:
                    # 启动都没成功时 QC 缺流是必然,不再重复提示
                    qc_errs = qc_slot_errors(qc)
                    if qc_errs:
                        for slot in qc_errs:
                            if slot not in runtime_errors:
                                slot_fail[slot] += 1
                        print(format_failure_help(
                            qc_errs, phase="data",
                            fail_counts=slot_fail, log_root=str(run_dir)))

            choice = "next" if args.auto_keep else ask_next(qc_error=qc_error)

            def _status() -> str:
                # 状态合成:strict_rc 下没录完整(rc!=0)一律 failed;
                # 否则 QC 有 ERROR → error,n/q 保留且无错 = success
                if strict_rc and rc != 0:
                    return "failed"
                return "error" if qc_error else "success"

            if choice == "rerun":
                outcomes[run_dir.name] = "rerun"
                # 状态合成:QC 有 ERROR 一律 error;否则操作员判失败 = failed
                status = "error" if qc_error else "failed"
                mark_meta(run_dir, status)
                print(f"  当前采集失败(meta 标记 {status})— {run_dir.name} "
                      "留档不删除,马上重新录制 ...")
                continue
            if choice == "quit":
                outcomes[run_dir.name] = "quit"
                status = _status()
                mark_meta(run_dir, status)
                print(f"  退出本次会话 — {run_dir.name} 标记 {status}")
                hooks.on_result(job, run_dir, "quit", status, rc)
                if pack_on and status == "success":
                    # q 保留的这条同样有效,逐条打包一并执行
                    run_pack_episode(run_dir)
                interrupted = True
                break
            if choice == "fail_quit":
                outcomes[run_dir.name] = "fail_quit"
                # 状态合成与重跑一致:QC 有 ERROR 一律 error;否则操作员
                # 判失败 = failed。不记台账,素材/图纸留在池里重采
                status = "error" if qc_error else "failed"
                mark_meta(run_dir, status)
                print(f"  失败退出本次会话 — {run_dir.name} 标记 {status},"
                      "不记台账")
                interrupted = True
                break

            outcomes[run_dir.name] = "kept"
            status = _status()
            mark_meta(run_dir, status)
            print(f"  已保留 {run_dir.name}(标记 {status})")
            hooks.on_result(job, run_dir, "kept", status, rc)
            if pack_on and status == "success":
                # 逐条打包只对 success 生效 —— 与 pack 的 meta status
                # 过滤同一口径:error/failed 的目录留档但不出数据集
                run_pack_episode(run_dir)
            kept_jobs.append(job)
            queue_pos += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_session] Ctrl+C — 结束会话")

    # ---- 汇总 ----
    title = ("全部完成" if queue_pos >= len(queue)
             else f"中断于 {len(kept_jobs)}/{len(queue)} 项")
    summary = collect_runs(runs, outcomes)
    summary["tasks_done"] = sorted(hooks.job_id(j) for j in kept_jobs)
    summary["tasks_total"] = sorted(hooks.job_id(j) for j in queue)
    summary["queue_seed"] = seed
    summary["interrupted"] = interrupted
    print_summary(summary, title=title)
    p = save_summary(summary, session_root)
    print(f"汇总已写入 {p}")
    return 0


# =============================================================================
# CLI 公共件
# =============================================================================

def add_common_cli(ap: argparse.ArgumentParser, *,
                   session_dir_default: str = "data/session-night") -> None:
    """三个入口共享的 CLI 参数(模式特有参数由各入口自己加)。"""
    ap.add_argument("--session-dir", type=Path, default=Path(session_dir_default),
                    help=f"会话根目录(默认 {session_dir_default}),"
                         "录制目录在其下")
    ap.add_argument("--seed", type=int, default=None,
                    help="队列/抽取随机种子(默认取当前时间戳,启动时打印)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="录制兜底最长秒数(0 = 等刺激程序结束)")
    ap.add_argument("--dummy", action="store_true",
                    help="用假设备试跑整条链路(不开硬件)")
    ap.add_argument("--recorders", nargs="*", default=None,
                    help="只启用这些模态(默认全部)")
    ap.add_argument("--skip-qc", action="store_true",
                    help="录完不跑自动 QC(汇总将缺明细)")
    ap.add_argument("--pack-episode", action="store_true",
                    help="每条录完并保留(n 或 q)后,自动把该条打成独立 "
                         "episode 数据集,输出镜像保存路径(data/lerobot/"
                         "<班次根>/<会话名>;默认关,也可在 configs/"
                         "session.yaml 设 pack_episode: true)")
    ap.add_argument("--collector-id", default=None,
                    help="采集编号(覆盖 configs/session.yaml 的 collector_id)")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="覆盖/追加任意采集信息键,可多次(如 --set subject=XX);"
                         "session.yaml 顶层除 mode/stim 外的键都是采集信息")


def resolve_collect(args) -> dict:
    """采集信息:session.yaml 顶层除运行期开关/框架保留键外都是;
    CLI 逐键覆盖,优先级 yaml < --collector-id < --set(最后写 wins)。
    场景名不在此列 —— 图纸模式以图纸 config.yaml 的 scene.name 为准。"""
    from embodied_brain_collect.session.config import load_session_run
    run_cfg = load_session_run()
    collect: dict = {k: v for k, v in run_cfg.items()
                     if k not in SESSION_RUN_KEYS
                     and k not in FRAMEWORK_KEYS and v is not None}
    if getattr(args, "collector_id", None) is not None:
        collect["collector_id"] = args.collector_id
    for kv in getattr(args, "set", []):
        k, _, v = kv.partition("=")
        k = k.strip()
        if not k:
            continue
        if k in FRAMEWORK_KEYS or k in SESSION_RUN_KEYS:
            print(f"[run_session] 忽略 --set {k}=… — 框架保留键不可覆盖",
                  file=sys.stderr)
            continue
        collect[k] = v.strip()
    return collect


def seed_or_now(args) -> int:
    return args.seed if args.seed is not None else int(time.time())
