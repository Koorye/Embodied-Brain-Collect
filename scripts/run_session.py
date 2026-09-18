#!/usr/bin/env python3
"""主控制脚本 —— 一次采集会话的完整编排:图纸队列 → 逐张摆放+录制+QC → 汇总。

    python scripts/run_session.py                     # 完整会话(交互式)
    python scripts/run_session.py --seed 42           # 指定队列种子(可复现)
    python scripts/run_session.py --auto-keep         # 不询问,录完即留
    python scripts/run_session.py --dummy             # 假设备试跑整条链路
    python scripts/run_session.py --skip-qc           # 录完不跑 QC

流程:

  1. 抽取队列来自 configs/environments 的图纸池(随机排序,seed 可固定),
     已采集成功的图纸自动跳过;打印完整队列,按 Enter 才开始采集
  2. 按队列逐张图纸录制:全屏显示图纸 → 采集员照图摆放实物 → 按 n + Enter
     关闭图纸 → launcher 启动(所有 recorder 多进程 + 刺激程序)→ 自动 QC;
     Esc 可在图纸阶段取消本次(不开录)
  3. 每张录完需输入 字母+Enter 确认,防止误触:
     n = 当前采集成功,下一条 / r = 当前采集失败,重跑 / q = 成功并退出;
     输错字母要求重输
     * 结局写进该条 meta.yaml 的 status 字段(r → failed,n/q → success),
       随数据目录走,打包/汇总据此识别单条数据的有效性
     * 只有"成功"记入图纸台账 configs/environments/<场景目录>/used.yaml,之后不再
       被抽到;重跑/退出不记账,图纸留在池里。重跑不删除本次录制目录
       (留档备查,meta 标记 failed)
  4. 全部图纸完成(或退出)后打印汇总(保留/重采/退出的录制都计入):
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

from embodied_brain_collect.session.config import (  # noqa: E402
    FRAMEWORK_KEYS, SESSION_RUN_KEYS, load_session_run, load_tasks,
    task_name)
from embodied_brain_collect.session import environment as env  # noqa: E402
from embodied_brain_collect.session.launcher import (  # noqa: E402
    _recorder_names, _write_session_meta, launch, run_qc)
from embodied_brain_collect.recorders.factory import (  # noqa: E402
    get_dummy_recorders, get_production_recorders)
from embodied_brain_collect.session.troubleshooting import (  # noqa: E402
    format_failure_help)
from embodied_brain_collect.stim.factory import (  # noqa: E402
    STIM_KINDS, build_stim_cmd)


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
            "task": _meta_task_label(run_dir) or "-",
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


def _meta_task_label(run_dir: Path) -> str:
    """meta 里的任务标识:图纸模式取 environment 对应的场景任务;
    旧模式(有 task_id)取 tasks.yaml 的任务名。"""
    try:
        import yaml
        meta = yaml.safe_load(
            (run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    if meta.get("task_id") is not None:
        return task_name(int(meta["task_id"])) or ""
    environment = meta.get("environment")
    return env.scene_task(environment) if environment else ""


def _session_span(qc: dict) -> float:
    w = qc.get("window") or {}
    t0, t1 = w.get("t0"), w.get("t1")
    return float(t1 - t0) if t0 is not None and t1 is not None else 0.0


_STATUS_LABELS = {"kept": "保留", "rerun": "重跑", "quit": "退出",
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


def _save_summary(summary: dict, session_root: Path) -> Path:
    p = session_root / "run_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p


# =============================================================================
# 交互
# =============================================================================

def _mark_meta(run_dir: Path, status: str) -> None:
    """把本次采集的结局写进该 session 的 meta.yaml(status 字段)。

    三档合成:QC 有 ERROR → error(无论 n/r/q);QC 无错且 r → failed
    (操作员判失败重采);QC 无错且 n/q → success。仅 success 记图纸台账。
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


def _ask_next(qc_error: bool = False) -> str:
    """录制结束后的确认输入,防误触:必须输入 n/r/q 之一再回车。

    只回车、输错字母都会要求重输 —— 录制现场经常双手忙着摘设备,
    误触一下 Enter 不能直接吞掉一条录制。
    """
    if qc_error:
        print("  ⚠ QC 有 ERROR — 此时按 n/q 仍保留数据,但图纸不记台账,"
              "之后会重新抽到重采")
    while True:
        ans = input("  输入 n(当前采集成功,下一条) / "
                    "r(当前采集失败,重跑) / q(成功并退出),"
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

def record_one(session_root: Path, job: dict, stim: str,
               args, collect: dict | None = None) -> tuple[Path, int]:
    """录一个队列项,返回 (run_dir, launcher 返回码)。

    job: env 模式 = {"mode": "env", "rel": 图纸相对路径};
         tasks 模式 = {"mode": "tasks", "task_id": int, "task_name": str}。
    collect: 采集信息(session.yaml + CLI 覆盖的解析结果),随 meta.yaml 固化。

    env 模式:正式开录前全屏显示图纸,采集员照图摆放实物,按 n + Enter
    关闭后才开录(未确认直接关窗 = 取消,返回码 2,未开录)。stim 指令屏
    按图纸 config.yaml 的场景/任务显示(dummy 模式同样显示图纸)。
    """
    from embodied_brain_collect.session import environment as env

    mode = job["mode"]
    run_dir = session_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 重跑不再删除旧目录,同一秒内连续录制会撞名 —— 加后缀避让,绝不覆盖旧数据
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = run_dir.with_name(f"{run_dir.name}-{n}")
    run_dir.mkdir(parents=True)

    if mode == "env":
        rel = job["rel"]
        print(f"\n{'─' * 68}")
        print(f"  图纸 {rel}  场景任务: {env.scene_task(rel)}")
        print(f"  录制目录: {run_dir}")
        print(f"{'─' * 68}")
        print("  图纸已全屏打开 — 照图摆放实物,按 n + Enter 关闭并正式开始"
              "(未确认直接关窗 = 取消)")
        if env.show(rel) == "abort":
            print("[run_session] 图纸阶段取消 — 未开录,目录留档")
            return run_dir, 2
        print("[run_session] 图纸已关闭,正式启动采集")
    else:
        print(f"\n{'─' * 68}")
        print(f"  任务 #{job['task_id']}  {job.get('task_name', '')}")
        print(f"  录制目录: {run_dir}")
        print(f"{'─' * 68}")
        if not args.dummy:
            input("按 Enter 正式开始...")

    if args.dummy:
        recs = get_dummy_recorders(session_dir=str(run_dir),
                                   duration=args.duration,
                                   slots=args.recorders, stim=stim)
    else:
        recs = get_production_recorders(session_dir=str(run_dir),
                                        duration=args.duration,
                                        slots=args.recorders)

    if mode == "env":
        _write_session_meta(run_dir, environment=job["rel"],
                            recorders=_recorder_names(recs),
                            collect_info=collect,
                            layout=env.scene_layout(job["rel"]) or None)
        stim_cmd = build_stim_cmd(stim, environment=job["rel"])
    else:
        _write_session_meta(run_dir, task_id=job["task_id"],
                            recorders=_recorder_names(recs),
                            collect_info=collect)
        stim_cmd = build_stim_cmd(stim, task_id=job["task_id"])

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
    ap.add_argument("--mode", choices=["env", "tasks"], default=None,
                    help="队列模式:env=图纸模式(默认) / tasks=任务列表模式;"
                         "缺省读 configs/session.yaml 的 mode")
    ap.add_argument("--stim", choices=sorted(STIM_KINDS), default=None,
                    help="刺激程序(缺省读 configs/session.yaml 的 stim,"
                         "再缺省 paradigm1);参数见 configs/stim.yaml")
    ap.add_argument("--collector-id", default=None,
                    help="采集编号(覆盖 configs/session.yaml 的 collector_id)")
    ap.add_argument("--max-runs", type=int, default=None, dest="max_runs",
                    help="单次会话最大采集条数(覆盖 configs/session.yaml "
                         "的 max_runs;0 = 不限)")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="覆盖/追加任意采集信息键,可多次(如 --set subject=XX);"
                         "session.yaml 顶层除 mode/stim 外的键都是采集信息")
    args = ap.parse_args(argv)

    session_root = args.session_dir.resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    # ---- 0. 模式/stim:CLI 显式传参优先,否则读 configs/session.yaml ----
    run_cfg = load_session_run()
    mode = args.mode or str(run_cfg.get("mode") or "env")
    stim = args.stim or str(run_cfg.get("stim") or "paradigm1")
    if mode not in ("env", "tasks"):
        print(f"未知 mode: {mode!r} (env=图纸模式 / tasks=任务列表模式)",
              file=sys.stderr)
        return 2
    if stim not in STIM_KINDS:
        print(f"未知 stim: {stim!r} (可用: {sorted(STIM_KINDS)})",
              file=sys.stderr)
        return 2

    max_runs = (args.max_runs if args.max_runs is not None
                else int(run_cfg.get("max_runs") or 0))

    # ---- 0b. 采集信息:session.yaml 顶层除运行期开关/框架保留键外都是;
    # CLI 逐键覆盖,优先级 yaml < --collector-id < --set(最后写 wins)。
    # 场景名不在此列 —— 图纸模式以图纸 config.yaml 的 scene.name 为准
    # (objects.jsonl)。打包后这些键平铺在 meta/collect_info.jsonl 顶层。
    collect: dict = {k: v for k, v in run_cfg.items()
                     if k not in SESSION_RUN_KEYS
                     and k not in FRAMEWORK_KEYS and v is not None}
    if args.collector_id is not None:
        collect["collector_id"] = args.collector_id
    for kv in args.set:
        k, _, v = kv.partition("=")
        k = k.strip()
        if not k:
            continue
        if k in FRAMEWORK_KEYS or k in SESSION_RUN_KEYS:
            print(f"[run_session] 忽略 --set {k}=… — 框架保留键不可覆盖",
                  file=sys.stderr)
            continue
        collect[k] = v.strip()

    # ---- 1. 队列:env = 未采集图纸;tasks = tasks.yaml 随机采样 ----
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    queue: list[dict] = []
    if mode == "env":
        unused = env.pool_unused()
        if not unused:
            print("configs/environments 里没有未采集过的图纸 — 各场景目录"
                  "台账已全部记账,或目录为空", file=sys.stderr)
            return 2
        rng.shuffle(unused)
        queue = [{"mode": "env", "rel": rel} for rel in unused]
        print(f"\n图纸池共 {len(unused)} 张未采集;本次随机队列 seed={seed}")
        print(f"{'─' * 68}\n执行顺序:")
        for i, job in enumerate(queue, 1):
            print(f"  {i:>3}. {job['rel']}  [{env.scene_task(job['rel'])}]")
    else:
        try:
            tasks = load_tasks()
        except FileNotFoundError as exc:
            print(f"缺少配置文件: {exc}", file=sys.stderr)
            return 2
        if not tasks:
            print("configs/tasks.yaml 没有任务", file=sys.stderr)
            return 2
        rng.shuffle(tasks)
        queue = [{"mode": "tasks", "task_id": int(t["task_id"]),
                  "task_name": t.get("task_name", "")} for t in tasks]
        print(f"\n任务库共 {len(queue)} 个任务;本次随机队列 seed={seed}")
        print(f"{'─' * 68}\n执行顺序:")
        for i, job in enumerate(queue, 1):
            print(f"  {i:>3}. #{job['task_id']:<3} {job['task_name']}")
    if max_runs > 0 and len(queue) > max_runs:
        queue = queue[:max_runs]
        print(f"最大采集量 max_runs={max_runs} — 队列只取前 {max_runs} 条")
    print(f"{'─' * 68}")
    print(f"[run_session] 模式={mode}  stim={stim}  计划={len(queue)} 条"
          + (f"  采集信息: {collect}" if collect else ""))
    try:
        input("按 Enter 开始采集(Ctrl+C 取消) ...")
    except KeyboardInterrupt:
        print("\n[run_session] 已取消")
        return 0

    # ---- 2-3. 按队列逐项录制 → n/r/q 确认 ----
    kept_jobs: list[dict] = []
    runs: list[Path] = []           # 全部录制目录(保留/重采/退出都计入汇总)
    outcomes: dict[str, str] = {}   # dir 名 -> kept / rerun / quit
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

            run_dir, rc = record_one(session_root, job, stim, args, collect)
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
            qc = _load_qc(run_dir)
            if qc:
                findings = [f for st in qc.get("streams", {}).values()
                            for f in st.get("findings", [])]
                findings += qc.get("findings", [])   # 会话级(如 StreamPresent)
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
                    print("  ⚠ 图纸不记入台账 — 之后会重新抽到重采;"
                          "确认无碍可按 n 保留数据")
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
                choice = _ask_next(qc_error=qc_error)
            if choice == "rerun":
                outcomes[run_dir.name] = "rerun"
                # 状态合成:QC 有 ERROR 一律 error;否则操作员判失败 = failed
                status = "error" if qc_error else "failed"
                _mark_meta(run_dir, status)
                print(f"  当前采集失败(meta 标记 {status})— {run_dir.name} "
                      "留档不删除,马上重新录制 ...")
                continue
            if choice == "quit":
                outcomes[run_dir.name] = "quit"
                # 状态合成:QC ERROR → error;退出前无错按 q = 成功保留
                status = "error" if qc_error else "success"
                _mark_meta(run_dir, status)
                interrupted = True
                if job["mode"] == "env":
                    if status == "success":
                        env.mark_used(job["rel"], session=str(run_dir))
                        print(f"  退出本次会话 — {run_dir.name} 标记 {status},"
                              f"图纸 {job['rel']} 已记入台账")
                    else:
                        print(f"  退出本次会话 — {run_dir.name} 标记 {status};"
                              "QC 有 ERROR,图纸不记台账(会重新抽到重采)")
                else:
                    print(f"  退出本次会话 — {run_dir.name} 标记 {status}并留档")
                break

            # 状态合成:QC ERROR → error;n/q 保留且无错 = success
            status = "error" if qc_error else "success"
            outcomes[run_dir.name] = "kept"
            _mark_meta(run_dir, status)
            kept_jobs.append(job)
            if job["mode"] == "env":
                if status == "success":
                    env.mark_used(job["rel"], session=str(run_dir))
                    print(f"  已保留 {run_dir.name}(标记 {status})— 图纸"
                          f" {job['rel']} 已记入台账,进入下一张")
                else:
                    print(f"  已保留 {run_dir.name}(标记 {status})— QC 有"
                          " ERROR,图纸不记台账(会重新抽到重采),进入下一张")
            else:
                print(f"  已保留 {run_dir.name} — 任务 #{job['task_id']} 完成,"
                      "进入下一个")
            queue_pos += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_session] Ctrl+C — 结束会话")

    # ---- 4. 汇总 ----
    title = ("全部完成" if queue_pos >= len(queue)
             else f"中断于 {len(kept_jobs)}/{len(queue)} 项")
    summary = _collect(runs, outcomes)
    summary["tasks_done"] = sorted(
        (j["rel"] if j["mode"] == "env" else j["task_id"])
        for j in kept_jobs)
    summary["tasks_total"] = sorted(
        (j["rel"] if j["mode"] == "env" else j["task_id"]) for j in queue)
    summary["queue_seed"] = seed
    summary["interrupted"] = interrupted
    print_summary(summary, title=title)
    p = _save_summary(summary, session_root)
    print(f"汇总已写入 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
