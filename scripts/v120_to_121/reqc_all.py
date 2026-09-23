#!/usr/bin/env python3
"""按视频帧数 vs 时间戳数量核对已保存 session —— 必须严格相等,否则 ERROR。

    python scripts/reqc_all.py data/session-day data/session-night
    python scripts/reqc_all.py data/session-night --date 2026-09-15
    python scripts/reqc_all.py data/session-night --write     # 默认 dry-run

核对 session 里**每一路视频**(与打包脚本 discover_video_slots 同一约定):
相机槽位 ``<slot>/frames.mp4`` vs ``<slot>/<slot>.npz`` 的
``frames_timestamps``;eye 槽位 ``eye/eye.mp4`` vs ``eye/eye.npz`` 的
``scene_timestamps``。录制器逐帧 1:1 写时间戳,帧数与时间戳数**不相等
即 ERROR**——视频录制中断/丢帧,数据不可信。

帧数用 ``ffprobe -count_packets`` 直接数容器里的包(不解码像素,毫秒级;
比逐帧解码快百倍以上),与录制写盘、打包预扫走同一个
``embodied_brain_collect.utils.media`` 工具模块,三处结论必然一致。

* 默认 dry-run:只打印每个 session 的核对结果,不改任何文件
* ``--write``:把 ERROR finding 合并进该 session 的 qc_report.json 的
  对应流(其余 findings 与文件一律不动),该流与会话等级随之变为 ERROR;
  没有 qc_report.json 的 session 只打印结果、不创建报告

Exit code: 0 全部一致,1 存在不一致(可当流水线闸门用)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from embodied_brain_collect.checkers.base import worst_level  # noqa: E402
from embodied_brain_collect.utils.media import ffprobe_count  # noqa: E402


def _video_pairs(session_dir: Path) -> list[dict]:
    """session 里每一路视频 → {slot, mp4, npz, key, n_ts}。

    mp4 找时间戳的顺序:同目录 npz 里的 ``{stem}_timestamps``,eye.mp4
    再退回 ``scene_timestamps``;找不到 = 视频无法核对,也算异常。
    """
    videos: list[dict] = []
    for slot_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        keys: dict[str, tuple[Path, int]] = {}
        for npz in sorted(slot_dir.glob("*.npz")):
            try:
                with np.load(npz, allow_pickle=True) as z:
                    for k in z.files:
                        if k.endswith("_timestamps"):
                            keys[k] = (npz, int(len(z[k])))
            except Exception:
                continue                # npz 坏了:对应视频核对时按缺失处理
        for mp4 in sorted(slot_dir.glob("*.mp4")):
            if mp4.stem.endswith((".orig", ".repair")):
                continue        # repair_videos 的备份/中间产物,不参与核对
            ts_key = f"{mp4.stem}_timestamps"
            if ts_key not in keys and mp4.stem == "eye":
                ts_key = "scene_timestamps"       # eye 槽位特例
            if ts_key not in keys:
                videos.append({"slot": slot_dir.name, "mp4": mp4,
                               "error": "找不到对应时间戳,视频无法核对"})
                continue
            npz, n_ts = keys[ts_key]
            videos.append({"slot": slot_dir.name, "mp4": mp4,
                           "npz": npz, "key": ts_key, "n_ts": n_ts})
    return videos


def check_session(session_dir: Path) -> list[dict]:
    """逐路核对:容器帧数 vs 时间戳数,严格相等才算过。"""
    results: list[dict] = []
    for v in _video_pairs(session_dir):
        if "error" in v:
            results.append(v)
            continue
        try:
            n_frames = ffprobe_count(v["mp4"])
        except Exception as exc:
            results.append({"slot": v["slot"], "mp4": v["mp4"],
                            "error": f"ffprobe 数帧失败({exc})"})
            continue
        results.append({"slot": v["slot"], "mp4": v["mp4"], "npz": v["npz"],
                        "key": v["key"], "n_frames": n_frames,
                        "n_ts": v["n_ts"],
                        "missing": abs(n_frames - v["n_ts"])})
    return results


def _apply_error_to_report(qc_path: Path, slot: str, mp4_name: str,
                           n_frames: int, n_ts: int) -> None:
    """把该流的 FrameCountMatch finding 置为 ERROR 并重算等级。"""
    data = json.loads(qc_path.read_text(encoding="utf-8"))
    streams = data.setdefault("streams", {})
    stream = streams.setdefault(slot, {"level": "ERROR", "files": [],
                                       "findings": [], "series": {},
                                       "stats": {}})
    diff = abs(n_frames - n_ts)
    findings = [f for f in stream.get("findings", [])
                if f.get("check") != "FrameCountMatch"]
    findings.append({
        "level": "ERROR",
        "check": "FrameCountMatch",
        "message": (f"视频帧数({n_frames})与时间戳数({n_ts})不一致"
                    f"(差 {diff} 帧)— 视频录制中断/丢帧"),
        "field": mp4_name,
        "observed": float(diff),
        "threshold": 0.0,
    })
    stream["findings"] = findings
    stream["level"] = worst_level(f.get("level", "INFO") for f in findings)
    data["level"] = worst_level(
        [s.get("level", "INFO") for s in streams.values()]
        + [f.get("level", "INFO") for f in data.get("findings", [])])
    qc_path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="班次根目录(如 data/session-day),其下 <日期>-* "
                         "的目录逐个核对")
    ap.add_argument("--date", default=None,
                    help="只核对 <date>-* 的会话(默认全部)")
    ap.add_argument("--write", action="store_true",
                    help="把 ERROR finding 写回 qc_report.json(默认 dry-run)")
    a = ap.parse_args(argv)

    sessions: list[Path] = []
    for root in a.roots:
        pattern = f"{a.date}-*" if a.date else "*"
        sessions += [p for p in sorted(root.glob(pattern)) if p.is_dir()]
    if not sessions:
        print("没有找到 session 目录", file=sys.stderr)
        return 1

    bad: list[dict] = []
    for i, sd in enumerate(sessions, 1):
        results = check_session(sd)
        if not results:
            print(f"[{i}/{len(sessions)}] {sd.name}: 跳过(无视频)")
            continue
        lines = []
        for r in results:
            if "error" in r:
                lines.append(f"✗ {r['slot']}/{r['mp4'].name}: {r['error']}")
                bad.append({"dir": sd, "slot": r["slot"],
                            "mp4": r["mp4"].name})
                continue
            mark = "✓" if r["missing"] == 0 else "✗"
            lines.append(f"{mark} {r['slot']}/{r['mp4'].name}: "
                         f"帧 {r['n_frames']} vs 时间戳 {r['n_ts']}"
                         + ("" if r["missing"] == 0
                            else f" — 差 {r['missing']}"))
            if r["missing"]:
                bad.append({"dir": sd, "slot": r["slot"],
                            "mp4": r["mp4"].name,
                            "n_frames": r["n_frames"], "n_ts": r["n_ts"]})
        print(f"[{i}/{len(sessions)}] {sd.name}: " + "; ".join(lines))

    if a.write and bad:
        written = 0
        for item in bad:
            qc_path = item["dir"] / "qc_report.json"
            if qc_path.exists() and "n_frames" in item:
                _apply_error_to_report(qc_path, slot=item["slot"],
                                       mp4_name=item["mp4"],
                                       n_frames=item["n_frames"],
                                       n_ts=item["n_ts"])
                written += 1
        print(f"已把 {written} 处 ERROR finding 写回 qc_report.json")

    print(f"\n完成: {len(sessions)} 个 session 核对,{len(bad)} 处视频帧数与"
          f"时间戳不一致" + ("(dry-run,未写回)" if not a.write else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
