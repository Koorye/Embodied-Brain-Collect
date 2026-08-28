#!/usr/bin/env python3
"""Pre-flight check: every configured recorder opens AND keeps producing data.

    python scripts/preflight.py            # recorders.yaml 里的全部
    python scripts/preflight.py cam_head emg_left
    python scripts/preflight.py --probe-seconds 5

每个 slot 的检查分三步,在**独立子进程**里跑(一个设备卡死不会拖住整个
预检),数据落在临时目录里,检查完即删:

  1. open —— 与 launcher 相同的首帧闸门(打开设备并等到第一份数据);
  2. 数据流 —— 再轮询几秒,确认数据在**持续**到达并给出实测速率
     (自带事件循环的 recorder 用各自的探测实现,如 neon 的 standby 队列);
  3. 收尾 —— 关闭设备并落盘,验证 teardown 路径不报错。

注意:``_open`` 通过本身就证明了"来过数据" —— 相机/EMG/手套在 open 闸门
通过后会清掉探测样本(录制时间线必须从 go 开始),所以判定不能用"缓冲区
里有没有样本",旧版预检正是因此把好设备误判成"首帧数据为空"。

输出:每个 slot 一节详细报告(阶段、耗时、速率、错误、traceback、该
recorder 日志的最后几行、分设备排查建议),末尾汇总表;全部通过退出码 0,
否则 1。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as _queue
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.config.load import configs_dir, load_recorders  # noqa: E402
from embodied_brain_collect.session.troubleshooting import (  # noqa: E402
    guide_for_slot, self_check_flow)

DEFAULT_HARD_TIMEOUT = 240.0   # 子进程硬超时:eye 冷启动(发现+传感器+首帧)
                              # 最坏 ~90s,再留足探测与收尾
LOG_TAIL_LINES = 15


# =============================================================================
# 子进程:单个 slot 的完整检查
# =============================================================================

def _read_log_tail(rec, workdir: str) -> str:
    """该 recorder 会话日志的最后几行 —— open 错误/写盘失败都在这里。"""
    try:
        name = rec.output_dir or rec.name or "recorder"
        path = Path(workdir) / name / f"{name.replace(':', '_')}.log"
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-LOG_TAIL_LINES:])
    except OSError:
        return ""


def check_one(slot: str, cfg: dict, workdir: str,
              probe_seconds: float) -> dict:
    """在当前进程里检查一个 slot(由 ``_child_main`` 在子进程中调用)。

    返回的报告 dict 全部是可 pickle 的标量/字符串,经队列回传父进程。
    """
    report: dict = {
        "slot": slot, "kind": cfg.get("kind", ""),
        "status": "FAIL", "phase": "open", "error": "",
        "traceback": "", "open_s": 0.0, "total_s": 0.0,
        "probe": "", "log_tail": "",
    }
    t_all = time.time()

    # ---- 1. open(首帧闸门) ----
    try:
        from embodied_brain_collect.session.recorder_presets import build_recorder
        rec = build_recorder(slot, cfg, session_dir=workdir, duration=0)
    except Exception:                                   # noqa: BLE001
        report["error"] = "recorder 构造失败(参数/依赖问题)"
        report["traceback"] = traceback.format_exc()
        return report

    t0 = time.time()
    try:
        ok = rec._open()
    except Exception:                                   # noqa: BLE001
        report["error"] = f"open 崩溃: {traceback.format_exc(limit=1).strip().splitlines()[-1]}"
        report["traceback"] = traceback.format_exc()
        report["open_s"] = time.time() - t0
        report["log_tail"] = _read_log_tail(rec, workdir)
        try:
            rec._close()
        except Exception:                               # noqa: BLE001
            pass
        return report
    report["open_s"] = time.time() - t0
    if not ok:
        report["error"] = rec._open_error or "未知原因"
        report["log_tail"] = _read_log_tail(rec, workdir)
        try:
            rec._close()
        except Exception:                               # noqa: BLE001
            pass
        return report

    # ---- 2. 数据流探测(持续进样 + 实测速率) ----
    report["phase"] = "probe"
    try:
        flow_ok, flow_detail = rec.probe_data_flow(probe_seconds)
    except Exception:                                   # noqa: BLE001
        report["error"] = "数据流探测崩溃"
        report["traceback"] = traceback.format_exc()
        flow_ok = False
        flow_detail = ""
    report["probe"] = flow_detail
    if not flow_ok:
        report["error"] = report["error"] or (f"数据未持续到达 — {flow_detail}")
        report["log_tail"] = _read_log_tail(rec, workdir)

    # ---- 3. 收尾(关设备 + 落盘;eye 的常驻事件循环要先停再等) ----
    report["phase"] = "close"
    try:
        rec._close()
    except Exception:                                   # noqa: BLE001
        report["traceback"] = report["traceback"] or traceback.format_exc()
        if not report["error"]:
            report["error"] = "关闭设备时出错"
    loop_done = getattr(rec, "_loop_done", None)
    if loop_done is not None and not loop_done.wait(timeout=10.0):
        if not report["error"]:
            report["error"] = "事件循环线程 10s 内未退出"
    try:
        rec._save()
    except Exception:                                   # noqa: BLE001
        report["traceback"] = report["traceback"] or traceback.format_exc()
        if not report["error"]:
            report["error"] = "保存探测数据时出错(落盘路径异常)"

    if not report["error"]:
        report["status"] = "OK"
        report["phase"] = ""
    report["log_tail"] = report["log_tail"] or _read_log_tail(rec, workdir)
    report["total_s"] = time.time() - t_all
    return report


def _child_main(slot: str, cfg: dict, workdir: str, probe_seconds: float,
                result_q) -> None:
    try:
        result_q.put(check_one(slot, cfg, workdir, probe_seconds))
    except Exception:                                   # noqa: BLE001
        result_q.put({"slot": slot, "kind": cfg.get("kind", ""),
                      "status": "FAIL", "phase": "check",
                      "error": f"检查进程崩溃: {traceback.format_exc()[-800:]}",
                      "traceback": traceback.format_exc(), "open_s": 0.0,
                      "total_s": 0.0, "probe": "", "log_tail": ""})


# =============================================================================
# 父进程:编排 + 报告
# =============================================================================

def _param_summary(cfg: dict) -> str:
    """配置里值得一眼看到的关键参数(端口/索引/串号等)。"""
    parts = []
    for k in ("idx", "serial", "baud", "host", "device_classes", "open_timeout"):
        if cfg.get(k) not in (None, ""):
            parts.append(f"{k}={cfg[k]}")
    if cfg.get("port"):
        parts.append(f"port={cfg['port']}")
    return "  ".join(parts)


def run_checks(slots: list[str], config: dict, probe_seconds: float,
               hard_timeout: float, tmp_root: Path) -> list[dict]:
    ctx = mp.get_context("spawn")
    reports: list[dict] = []
    for i, slot in enumerate(slots, 1):
        cfg = dict(config[slot])
        # EEG 收尾时会对齐 marker(轮询 npz),预检没有 marker 流,缩短等待
        if cfg.get("kind") == "curry_eeg" and "marker_wait_s" not in cfg:
            cfg["marker_wait_s"] = 0.5
        workdir = tempfile.mkdtemp(prefix=f"preflight-{slot}-", dir=str(tmp_root))
        print(f"[{i:>2}/{len(slots)}] {slot} ({cfg.get('kind', '?')}) "
              f"检查中 ...", flush=True)
        q = ctx.Queue()
        p = ctx.Process(target=_child_main,
                        args=(slot, cfg, workdir, probe_seconds, q), daemon=True)
        t0 = time.time()
        p.start()
        p.join(hard_timeout)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
            report = {"slot": slot, "kind": cfg.get("kind", ""),
                      "status": "FAIL", "phase": "hard-timeout",
                      "error": (f"超过硬超时 {hard_timeout:g}s 未完成 — "
                                "设备或驱动卡死,考虑拔插后重试"),
                      "traceback": "", "open_s": hard_timeout,
                      "total_s": time.time() - t0, "probe": "", "log_tail": ""}
        else:
            try:
                # 子进程退出后其队列 feeder 还要一小会儿刷数据,get_nowait
                # 会在这段窗口误报 —— 给 5s 缓冲
                report = q.get(timeout=5.0)
            except _queue.Empty:
                report = {"slot": slot, "kind": cfg.get("kind", ""),
                          "status": "FAIL", "phase": "child",
                          "error": f"子进程退出(code={p.exitcode})但没有回报结果",
                          "traceback": "", "open_s": 0.0,
                          "total_s": time.time() - t0, "probe": "", "log_tail": ""}
        q.close()
        shutil.rmtree(workdir, ignore_errors=True)
        reports.append(report)
    return reports


def print_report(reports: list[dict], *, probe_seconds: float,
                 started_at: datetime) -> None:
    line = "=" * 72
    print("\n" + line)
    print(f"  设备预检报告 — {started_at:%Y-%m-%d %H:%M:%S}"
          f"   (数据流探测 {probe_seconds:g}s/slot)")
    print(line)
    for r in reports:
        mark = "✓ OK  " if r["status"] == "OK" else "✗ FAIL"
        print(f"\n[{mark}] {r['slot']} — {r['kind']}")
        params = _param_summary(_cfg_for_report(r))
        if params:
            print(f"    参数:   {params}")
        print(f"    打开耗时: {r['open_s']:.1f}s    总耗时: {r['total_s']:.1f}s")
        if r["probe"]:
            print(f"    数据流: {r['probe']}")
        if r["status"] == "OK":
            print("    判定:   打开成功且数据持续到达")
            continue
        phase_label = {"open": "启动(打开设备)", "probe": "数据流探测",
                       "close": "关闭/落盘", "check": "检查过程",
                       "hard-timeout": "硬超时", "child": "子进程"}
        print(f"    失败阶段: {phase_label.get(r['phase'], r['phase'])}")
        print(f"    错误:   {r['error']}")
        if r["traceback"]:
            tail = "\n".join(r["traceback"].strip().splitlines()[-8:])
            print("    traceback(末 8 行):")
            for ln in tail.splitlines():
                print(f"      {ln}")
        print(f"    排查:   {guide_for_slot(r['slot'])}")
        if r["log_tail"]:
            print(f"    该 recorder 日志(末 {LOG_TAIL_LINES} 行):")
            for ln in r["log_tail"].splitlines():
                print(f"      {ln}")

    n_ok = sum(1 for r in reports if r["status"] == "OK")
    failed = [r for r in reports if r["status"] != "OK"]
    print("\n" + line)
    print(f"  预检结果: {n_ok}/{len(reports)} 就绪"
          + ("" if not failed else f",{len(failed)} 失败:"))
    for r in failed:
        print(f"    ✗ {r['slot']:<16} {r['error']}")
    if failed:
        print()
        print(self_check_flow([r["slot"] for r in failed]))
    print(line)


def _cfg_for_report(report: dict) -> dict:
    """报告打印用的参数摘要 —— 记录在报告里,避免父进程再查一遍配置。"""
    return report.get("_params", {})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slots", nargs="*", default=None,
                    help="只检查这些 slot(默认 recorders.yaml 里全部启用的)")
    ap.add_argument("--probe-seconds", type=float, default=3.0,
                    help="数据流持续探测时长(默认 3s)")
    ap.add_argument("--hard-timeout", type=float, default=DEFAULT_HARD_TIMEOUT,
                    help="单个 slot 子进程硬超时(默认 240s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="把完整报告同时写入这个文件")
    args = ap.parse_args(argv)

    started_at = datetime.now()
    try:
        config = load_recorders()
    except FileNotFoundError as exc:
        print(f"缺少配置文件: {exc}", file=sys.stderr)
        return 2

    slots = list(args.slots)
    if slots:
        unknown = [s for s in slots if s not in config]
        if unknown:
            print(f"recorders.yaml 没有这些 slot: {unknown}", file=sys.stderr)
            return 2
    else:
        slots = [s for s, c in config.items() if c.get("enabled", True)]
    if not slots:
        print("没有可检查的 slot(全部 enabled: false?)", file=sys.stderr)
        return 2

    print(f"预检 {len(slots)} 个 slot:{', '.join(slots)}")
    print(f"配置目录: {configs_dir()}")
    tmp_root = Path(tempfile.mkdtemp(prefix="preflight-"))
    try:
        reports = run_checks(slots, config, args.probe_seconds,
                             args.hard_timeout, tmp_root)
        for slot, r in zip(slots, reports):
            r["_params"] = config.get(slot, {})
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if args.out:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(reports, probe_seconds=args.probe_seconds,
                         started_at=started_at)
        args.out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"完整报告已写入 {args.out}")
    print_report(reports, probe_seconds=args.probe_seconds, started_at=started_at)
    n_ok = sum(1 for r in reports if r["status"] == "OK")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
