#!/usr/bin/env python3
"""主控制脚本 —— 一次采集会话的完整编排:随机队列 → 逐任务录制+QC → 汇总。

    python scripts/run_session.py                     # 完整会话(交互式)
    python scripts/run_session.py --seed 42           # 指定队列种子(可复现)
    python scripts/run_session.py --auto-keep         # 不询问,录完即留
    python scripts/run_session.py --dummy             # 假设备试跑整条链路
    python scripts/run_session.py --skip-qc           # 录完不跑 QC

流程:

  1. 以当前时间戳为 seed 在内存中随机采样本次任务队列(tasks.yaml 文件
     本身不再被改写),打印完整任务列表,按 Enter 才开始采集
  2. 按队列顺序逐任务录制:launcher(所有 recorder 多进程 + paradigm1
     刺激程序)-> 自动 QC
  3. 每个任务录完需输入 字母+Enter 确认,防止误触:
     n = 保留下一条 / r = 重采本条 / q = 退出;输错字母要求重输
     * 重采不删除本次录制目录(留档备查),任务留在队首,马上重新开始
  4. 全部任务完成(或退出)后打印汇总(保留/重采/退出的录制都计入):
     无误数据的比例、每种 QC 错误/警告的数量与占比、各 session 时长与结局
  5. 录制前后出现错误时按设备给出排查指引(首次失败直接重采;多次失败
     按 eeg/emg/eye/hand_pose/position/cam 各自的方案排查),完整
     traceback 在 <录制目录>/<slot>/<slot>.log

QC 的判定只供参考,不替人做决定:某次录制即使有 ERROR 也可以保留
(留待后处理),完全由你在第 3 步拍板。

Windows 注意:recorder 子进程以 spawn 启动,本脚本必须作为主入口
(``python scripts/run_session.py``),不要在交互式 shell 里直接调用内部函数。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.config.load import load_tasks, task_name  # noqa: E402
from embodied_brain_collect.session.launcher import (  # noqa: E402
    _write_session_meta, launch, run_qc)
from embodied_brain_collect.session.recorder_presets import (  # noqa: E402
    get_dummy_recorders, get_production_recorders)
from embodied_brain_collect.session.troubleshooting import (  # noqa: E402
    format_failure_help)
from embodied_brain_collect.stim.factory import build_stim_cmd  # noqa: E402


# =============================================================================
# 汇总报告
# =============================================================================

def _load_qc(run_dir: Path) -> dict | None:
    p = run_dir / "qc_report.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has_any_data(run_dir: Path) -> bool:
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


def _collect(runs: list[Path],
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
            "task": task_name(_meta_task(run_dir)) or "-",
        }
        qc = _load_qc(run_dir)

        if not _has_any_data(run_dir):
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


def _meta_task(run_dir: Path) -> int | None:
    try:
        import yaml
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
        return meta.get("task_id")
    except Exception:
        return None


def _session_span(qc: dict) -> float:
    w = qc.get("window") or {}
    t0, t1 = w.get("t0"), w.get("t1")
    return float(t1 - t0) if t0 is not None and t1 is not None else 0.0


_STATUS_LABELS = {"kept": "保留", "rerun": "重跑", "quit": "退出"}


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
    for key, label in (("kept", "保留"), ("rerun", "重跑"), ("quit", "退出")):
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
        print(f"  任务完成: {len(done)}/{len(total)} 个      {clean_line}")
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


def _save_summary(summary: dict, session_root: Path) -> Path:
    p = session_root / "run_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p


# =============================================================================
# 交互
# =============================================================================

def _ask_next() -> str:
    """录制结束后的确认输入,防误触:必须输入 n/r/q 之一再回车。

    只回车、输错字母都会要求重输 —— 录制现场经常双手忙着摘设备,
    误触一下 Enter 不能直接吞掉一条录制。
    """
    while True:
        ans = input("  输入 n(保留下一条) / r(重采本条) / q(退出),"
                    "回车确认: ").strip().lower()
        if ans == "n":
            return "next"
        if ans == "r":
            return "rerun"
        if ans == "q":
            return "quit"
        print(f"  无效输入 {ans!r} —— 只接受 n(next) / r(retry) / q(quit),"
              "请重新输入")


def _qc_slot_errors(qc: dict | None) -> dict[str, str]:
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
# 单任务录制
# =============================================================================

def record_one(session_root: Path, task: dict, args) -> tuple[Path, int]:
    """录一个任务,返回 (run_dir, launcher 返回码)。"""
    task_id = int(task["task_id"])
    run_dir = session_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 重跑不再删除旧目录,同一秒内连续录制会撞名 —— 加后缀避让,绝不覆盖旧数据
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = run_dir.with_name(f"{run_dir.name}-{n}")
    run_dir.mkdir(parents=True)

    _write_session_meta(run_dir, task_id=task_id)

    print(f"\n{'─' * 68}")
    print(f"  任务 #{task_id}  {task.get('task_name', '')}")
    print(f"  录制目录: {run_dir}")
    print(f"{'─' * 68}")
    input("按任意键开始采集...")

    if args.dummy:
        recs = get_dummy_recorders(session_dir=str(run_dir),
                                   duration=args.duration,
                                   slots=args.recorders, stim="paradigm1")
    else:
        recs = get_production_recorders(session_dir=str(run_dir),
                                        duration=args.duration,
                                        slots=args.recorders)

    stim_cmd = build_stim_cmd("paradigm1", task_id=task_id)
    rc = 1
    try:
        rc = launch(recs, stim_cmd=stim_cmd, duration=args.duration)
        if not args.skip_qc:
            run_qc(run_dir)
    except KeyboardInterrupt:
        print("\n[run_session] Ctrl+C — 录制中止,目录保留")
    return run_dir, rc


# =============================================================================
# 主流程
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, default=Path("data/session-night"),
                    help="会话根目录(默认 data/session-night),录制目录在其下")
    ap.add_argument("--seed", type=int, default=None,
                    help="任务队列随机种子(默认取当前时间戳,启动时打印)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="录制兜底最长秒数(0 = 等刺激程序结束)")
    ap.add_argument("--dummy", action="store_true",
                    help="用假设备试跑整条链路(不开硬件)")
    ap.add_argument("--recorders", nargs="*", default=None,
                    help="只启用这些模态(默认全部)")
    ap.add_argument("--auto-keep", action="store_true",
                    help="每个任务录完自动保留,不询问")
    ap.add_argument("--skip-qc", action="store_true",
                    help="录完不跑自动 QC(汇总将缺明细)")
    args = ap.parse_args(argv)

    session_root = args.session_dir.resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    try:
        tasks = load_tasks()
    except FileNotFoundError as exc:
        print(f"缺少配置文件: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print("configs/tasks.yaml 没有任务", file=sys.stderr)
        return 2
    all_ids = {int(t["task_id"]) for t in tasks}

    # ---- 1. 随机采样本次队列(内存中,tasks.yaml 不动) ----
    seed = args.seed if args.seed is not None else int(time.time())
    queue = list(tasks)
    random.Random(seed).shuffle(queue)
    print(f"\n任务库共 {len(all_ids)} 个任务;本次随机队列 seed={seed}")
    print(f"{'─' * 68}\n执行顺序:")
    for i, t in enumerate(queue, 1):
        print(f"  {i:>3}. #{t['task_id']:<3} {t.get('task_name', '')}")
    print(f"{'─' * 68}")
    try:
        input("按 Enter 开始采集(Ctrl+C 取消) ...")
    except KeyboardInterrupt:
        print("\n[run_session] 已取消")
        return 0

    # ---- 2-3. 按队列逐任务录制 → n/r/q 确认 ----
    kept_ids: set[int] = set()
    runs: list[Path] = []           # 全部录制目录(保留/重采/退出都计入汇总)
    outcomes: dict[str, str] = {}   # dir 名 -> kept / rerun / quit
    slot_fail: Counter = Counter()  # slot -> 本次会话累计出错次数(排查提示用)
    queue_pos = 0
    task_no = 0
    interrupted = False
    try:
        while queue_pos < len(queue):
            task = queue[queue_pos]
            task_id = int(task["task_id"])
            task_no += 1
            remaining = len(queue) - queue_pos
            print(f"\n▶ 第 {task_no} 次录制 · 队列剩余 {remaining} 个 · "
                  f"已完成 {len(kept_ids)}/{len(queue)}")

            run_dir, rc = record_one(session_root, task, args)
            runs.append(run_dir)
            outcomes[run_dir.name] = "-"   # 结局由下面的选择更新
            print(f"\n[run_session] launcher 返回码 {int(rc)}"
                  + ("(刺激程序被 Esc 中止,数据仍已保存)" if rc == 2 else ""))

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

            qc = _load_qc(run_dir)
            if qc:
                findings = [f for st in qc.get("streams", {}).values()
                            for f in st.get("findings", [])]
                findings += qc.get("findings", [])   # 会话级(如 StreamPresent)
                n_err = sum(1 for f in findings if f.get("level") == "ERROR")
                verdict = "无 ERROR" if n_err == 0 else f"{n_err} 条 ERROR"
                print(f"  QC 判定: {qc.get('level')} ({verdict}) — "
                      f"细节见 {run_dir / 'qc.html'}")
                if n_err and not open_failures:
                    # 启动都没成功时 QC 缺流是必然,不再重复提示
                    qc_errs = _qc_slot_errors(qc)
                    if qc_errs:
                        for slot in qc_errs:
                            if slot not in runtime_errors:
                                slot_fail[slot] += 1
                        print(format_failure_help(
                            qc_errs, phase="data",
                            fail_counts=slot_fail, log_root=str(run_dir)))

            if args.auto_keep:
                choice = "next"
            else:
                choice = _ask_next()
            if choice == "rerun":
                outcomes[run_dir.name] = "rerun"
                print(f"  重采本任务 — {run_dir.name} 留档不删除,马上重新录制 ...")
                continue
            if choice == "quit":
                outcomes[run_dir.name] = "quit"
                interrupted = True
                print(f"  退出本次会话 — {run_dir.name} 留档不删除")
                break

            outcomes[run_dir.name] = "kept"
            kept_ids.add(task_id)
            queue_pos += 1
            print(f"  已保留 {run_dir.name} — 任务 #{task_id} 完成,进入下一个")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_session] Ctrl+C — 结束会话")

    # ---- 4. 汇总 ----
    title = ("全部完成" if queue_pos >= len(queue)
             else f"中断于 {len(kept_ids)}/{len(all_ids)} 个任务")
    summary = _collect(runs, outcomes)
    summary["tasks_done"] = sorted(kept_ids)
    summary["tasks_total"] = sorted(all_ids)
    summary["queue_seed"] = seed
    summary["interrupted"] = interrupted
    print_summary(summary, title=title)
    p = _save_summary(summary, session_root)
    print(f"汇总已写入 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
