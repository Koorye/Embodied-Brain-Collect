#!/usr/bin/env python3
"""把历史数据里多余的预热帧从视频头部截掉,恢复 mp4 与时间戳 1:1。

    python scripts/repair_videos.py data/session-day            # 预览
    python scripts/repair_videos.py data/session-day --write    # 真正重编码
    python scripts/repair_videos.py data/session-day --date 2026-09-16

背景:两段式启动引入 commit 机制之后、录制器源头丢弃预热帧之前的一批
录制,预热帧已经写进 mp4,而预热时间戳在 commit 时被整段丢弃 —— mp4
因此比 ``{key}_timestamps`` 恰好多出 δ = 帧数 − 时间戳数 帧(各路 δ 不同,
等于各自 open→commit 的时长,几十到两百帧)。多出的帧全部位于会话窗口
之前,截掉头部 δ 帧后容器与时间戳重新 1:1 —— 严格版 FrameCountMatch、
reqc_all 与打包截段的序号假设全部恢复成立。

处理方式:ffmpeg 从 δ/fps 处帧精确重编码(与 FFmpegWriter 同参:libx265、
bframes=0、keyint=fps、CFR;像素格式沿用源视频),写临时文件 → ffprobe
验证帧数 == 时间戳数 → 原子替换。**缺帧**(帧数 < 时间戳数,如 eye 录制
中断)无法用截断修复,原样跳过并如实报告。

* 默认 dry-run;``--write`` 才动文件
* ``--keep-original``:替换前把原视频改名为 ``<名字>.orig.mp4``(占双倍空间)
* 重编码有轻微代际损失,crf/preset 可调

Exit code: 0 全部处理完(无需修复或修复成功);1 存在缺帧/验证失败等
未解决项。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # 复用 reqc_all
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.utils.media import ffprobe_count, media_tool  # noqa: E402
from reqc_all import check_session  # noqa: E402


def _probe_stream(mp4: Path) -> tuple[float, str]:
    """(fps, 像素格式) —— 截段的 seek 与重编码参数都要沿用源视频。"""
    out = subprocess.run(
        [media_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,pix_fmt", "-of", "json",
         str(mp4)], capture_output=True, text=True, check=True)
    st = json.loads(out.stdout)["streams"][0]
    num, _, den = st["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return fps, st["pix_fmt"]


def repair_one(mp4: Path, n_ts: int, surplus: int, crf: int, preset: str,
               keep_original: bool) -> tuple[bool, str]:
    """截掉头部 surplus 帧,重编码后验证并原子替换。"""
    fps, pix_fmt = _probe_stream(mp4)
    if fps <= 0:
        fps = 30.0
    seek = surplus / fps
    tmp = mp4.with_name(mp4.stem + ".repair.mp4")
    cmd = [media_tool("ffmpeg"), "-y", "-loglevel", "error",
           "-ss", f"{seek:.6f}", "-i", str(mp4),
           "-r", str(fps), "-fps_mode", "cfr",
           "-video_track_timescale", "90000",
           "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", pix_fmt,
           "-g", str(int(fps)), "-keyint_min", str(int(fps)),
           "-x265-params", "bframes=0", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        n_new = ffprobe_count(tmp)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        return False, f"重编码失败:{(exc.stderr or '').strip()[-200:]}"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, f"验证失败:{exc}"
    if n_new != n_ts:
        tmp.unlink(missing_ok=True)
        return False, f"截断后帧数 {n_new} != 时间戳数 {n_ts},已保留原视频"
    if keep_original:
        orig = mp4.with_name(mp4.stem + ".orig.mp4")
        orig.unlink(missing_ok=True)
        mp4.replace(orig)
    tmp.replace(mp4)
    return True, f"截掉头部 {surplus} 帧({seek:.2f}s),现为 {n_ts} 帧"


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="班次根目录(如 data/session-day)或具体 session 目录")
    ap.add_argument("--date", default=None,
                    help="只处理 <date>-* 的会话(默认全部)")
    ap.add_argument("--write", action="store_true",
                    help="真正重编码替换(默认 dry-run)")
    ap.add_argument("--keep-original", action="store_true",
                    help="替换前把原视频改名为 <名字>.orig.mp4")
    ap.add_argument("--crf", type=int, default=23, help="libx265 CRF(默认 23)")
    ap.add_argument("--preset", default="medium", help="libx265 preset")
    a = ap.parse_args(argv)

    sessions: list[Path] = []
    for root in a.roots:
        pattern = f"{a.date}-*" if a.date else "*"
        cands = sorted(root.glob(pattern)) if root.is_dir() else []
        sessions += [p for p in cands if p.is_dir()] or ([root] if root.is_dir() else [])
    if not sessions:
        print("没有找到 session 目录", file=sys.stderr)
        return 1

    n_ok = n_repaired = n_failed = 0
    for sd in sessions:
        results = check_session(sd)
        if not results:
            continue
        for r in results:
            name = f"{r['slot']}/{r['mp4'].name}"
            if "error" in r:
                print(f"✗ {sd.name} {name}: {r['error']}")
                n_failed += 1
                continue
            n_frames, n_ts = r["n_frames"], r["n_ts"]
            surplus = n_frames - n_ts
            if surplus < 0:
                print(f"✗ {sd.name} {name}: 帧 {n_frames} < 时间戳 {n_ts} "
                      f"(缺 {-surplus})— 缺帧无法截补,保持原样")
                n_failed += 1
            elif surplus == 0:
                n_ok += 1
            else:
                if not a.write:
                    print(f"· {sd.name} {name}: 将截掉头部 {surplus} 帧"
                          f"({surplus / 30.0:.2f}s)→ {n_ts} 帧")
                    n_repaired += 1
                    continue
                ok, note = repair_one(r["mp4"], n_ts, surplus,
                                      a.crf, a.preset, a.keep_original)
                if ok:
                    print(f"✓ {sd.name} {name}: {note}")
                    n_repaired += 1
                else:
                    print(f"✗ {sd.name} {name}: {note}")
                    n_failed += 1

    mode = "已修复" if a.write else "待修复(dry-run)"
    print(f"\n完成: {n_ok} 个无需修复,{n_repaired} 个{mode},"
          f"{n_failed} 个失败/无法修复")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
