#!/usr/bin/env python3
"""EMG 臂环指认 —— 逐台插入,自动 diff 串口列表,告诉你哪台是新插入的 COM 口。

    python scripts/setup/map_emg.py           # 交互指认:插一台,认一台,最后给 yaml 填法
    python scripts/setup/map_emg.py --list    # 只列出当前串口,标记哪些长得像 EMG 臂环

臂环是同款设备、USB 特征完全一样,系统按插入顺序分配 COM 口,插错顺序左右就
反了。本脚本把 docs/emg.md 第 3 节的"逐一插入"流程自动化:

  1. 开始前建议拔掉所有臂环,脚本先扫一遍串口做基线;
  2. 每轮提示插入一条臂环(先左后右),边插边轮询;
  3. 新出现的 COM 口(连续两次扫描都在才算稳定,防枚举抖动)当场报告,
     串口描述里带 EMG 特征(CP210x / CH343)的会标注出来;
  4. 全部认完后打印指认结果与 recorders.yaml 建议填法(和当前 yaml 比对),
     并提示用 scripts/check_emg.py 做左右手对应验证。

等待插入期间按 Ctrl+C 随时退出,已认到的结果会先打印出来。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

#: 与 weili_emg_recorder._auto_detect_port 同源的 EMG 串口特征
_EMG_TAGS = ("VID:PID=10C4", "CP210", "Silicon Labs",
             "VID:PID=1A86:55D3", "CH343")
_POLL_S = 0.5        # 轮询周期
_STABLE_N = 2        # 新口连续出现这么多次才算稳定(防枚举抖动)


# =============================================================================
# 串口扫描
# =============================================================================

def scan_ports() -> dict[str, tuple[str, str]]:
    """{设备名: (描述, hwid)};Windows 如 COM31,Linux 如 /dev/ttyACM0。"""
    from serial.tools import list_ports
    return {p.device: (p.description or "", p.hwid or "")
            for p in list_ports.comports()}


def looks_like_emg(desc: str, hwid: str) -> bool:
    h, d = (hwid or "").upper(), desc or ""
    return any(t in h or t in d for t in _EMG_TAGS)


def describe(device: str, info: tuple[str, str]) -> str:
    desc, hwid = info
    tag = " [EMG 特征]" if looks_like_emg(desc, hwid) else ""
    return f"{device}  {desc or '?'}{tag}"


# =============================================================================
# 等新口:轮询直到出现稳定的新 COM
# =============================================================================

def wait_new_port(known: set[str]) -> tuple[str, tuple[str, str]] | None:
    """轮询串口表,返回新出现的稳定端口;期间拔掉的口也顺带报告。None = 被打断。"""
    t0 = time.time()
    cand_since: dict[str, int] = {}
    while True:
        now = scan_ports()
        gone = known - set(now)
        if gone:
            print(f"\n[i] 期间拔掉了:{', '.join(sorted(gone))}")
            known -= gone
        fresh = [d for d in now if d not in known]
        for d in fresh:
            cand_since[d] = cand_since.get(d, 0) + 1
            if cand_since[d] >= _STABLE_N:
                return d, now[d]
        print(f"\r  等待新串口接入 ... {time.time() - t0:3.0f}s"
              f"（已看到候选:{', '.join(fresh) or '无'}）   ",
              end="", flush=True)
        time.sleep(_POLL_S)


# =============================================================================
# 指认主流程
# =============================================================================

def identify() -> list[tuple[str, str, tuple[str, str]]]:
    """交互指认;返回 [(槽位 emg_left/emg_right, 设备名, 串口信息)]。"""
    print("开始前请拔掉所有 EMG 臂环（已在位的口会混进基线,认不出新插入的）。")
    input("拔好后按回车扫描基线 ...")
    baseline = scan_ports()
    if baseline:
        print("基线(已在位,不参与指认):")
        for d in sorted(baseline):
            print(f"  {describe(d, baseline[d])}")
    else:
        print("基线:无任何串口。")

    results: list[tuple[str, str, tuple[str, str]]] = []
    taken: set[str] = set()          # 本轮指认已绑走的口,下一轮不再当新口
    while True:
        n = len(results) + 1
        tip = "L" if not any(r[0] == "emg_left" for r in results) else "R"
        ans = input(f"\n第 {n} 条:插入臂环,它绑到哪只手?"
                    f"([L]左 / [R]右 / [Q]完成指认,默认 {tip}):").strip().upper()
        if ans == "Q":
            break
        if ans not in ("L", "R", ""):
            print("  输入 L / R / Q")
            continue
        slot = "emg_right" if ans == "R" else "emg_left"
        if any(r[0] == slot for r in results):
            print(f"  [!] {slot} 已经指认过({dict((r[0], r[1]) for r in results)[slot]}),"
                  "重复指认会覆盖之前的")
        try:
            dev, info = wait_new_port(set(baseline) | taken)
        except KeyboardInterrupt:
            print("\n[i] 手动中断,结束等待")
            break
        taken.add(dev)
        results.append((slot, dev, info))
        like = " ← 带 EMG 串口特征,多半就是臂环" if looks_like_emg(*info) \
            else " ← 注意:不带 EMG 特征,确认插的是臂环而不是别的设备"
        print(f"\n  ✔ 新发现 {describe(dev, info)}{like}"
              f"  → 绑定到 {slot}")
        if {"emg_left", "emg_right"} <= {r[0] for r in results}:
            if input("\n左右都认到了,完成指认?[Y/n]:").strip().upper() != "N":
                break
    return results


# =============================================================================
# 结果与 yaml 比对
# =============================================================================

def yaml_ports() -> dict[str, str]:
    """recorders.yaml 里 emg_left/emg_right 当前配置的 port;读不到返回空。"""
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        return {}
    return {s: str(cfg[s].get("port") or "")
            for s in ("emg_left", "emg_right") if s in cfg}


def report(results: list[tuple[str, str, tuple[str, str]]]) -> None:
    if not results:
        print("\n没有指认到任何设备。")
        return
    print("\n==== 指认结果 ====")
    for slot, dev, info in results:
        print(f"  {slot:<9} = {describe(dev, info)}")
    print("\nrecorders.yaml 建议填法(只列 port,其余字段照模板保留):")
    for slot in ("emg_left", "emg_right"):
        hit = next((r for r in results if r[0] == slot), None)
        if hit:
            print(f"  {slot}:")
            print(f'    port: "{hit[1]}"')
    cur = yaml_ports()
    if cur:
        print("\n与当前 recorders.yaml 比对:")
        for slot, port in cur.items():
            hit = next((r for r in results if r[0] == slot), None)
            if hit is None:
                print(f"  {slot}.port={port or '(空,自动探测)'}  本次未指认")
            elif hit[1] == port:
                print(f"  {slot}.port={port}  ✓ 与指认一致")
            else:
                print(f"  {slot}.port={port or '(空)'}  ✗ 与指认的 {hit[1]} 不一致,建议改")
    print("\n下一步:改好 yaml 后 python scripts/check_emg.py 做左右手对应检查;"
          "口位固定下来不再动(动了 COM 号会漂移,Linux 可改用 /dev/serial/by-id/...)。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="教程见 docs/emg.md")
    ap.add_argument("--list", action="store_true",
                    help="只列出当前串口(标记 EMG 特征),不进入指认")
    args = ap.parse_args(argv)

    if args.list:
        ports = scan_ports()
        if not ports:
            print("(无任何串口)")
            return 0
        for d in sorted(ports):
            print(f"  {describe(d, ports[d])}")
        return 0

    try:
        results = identify()
    except KeyboardInterrupt:
        results = []
        print("\n[i] 中止")
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
