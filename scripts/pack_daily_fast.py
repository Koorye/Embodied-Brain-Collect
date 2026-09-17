#!/usr/bin/env python
"""pack_daily 的多进程加速版 — 打包结果与原版一致,过程并行。

与 ``scripts/pack_daily.py`` 的差别(数据集输出相同):

1. 预扫与特征规格不再整段装载 npz(原版每个会话要完整解压三遍:
   预扫一遍、规格一遍、写帧再一遍),改用 ``load_stream_index``
   只读时间戳与列名,并把各会话的预扫放进进程池并行;索引异常时
   自动回退整段装载,语义与原版一致。
2. 视频截段(ffmpeg 重编码 + 首帧自验 + 时间戳 parquet)全部提前
   提交到独立进程池,与主进程的 add_frame 写帧循环完全重叠。
3. 每个会话的 npz 流装载在进程池预取(深度 1),与上一会话的写帧
   重叠;大数组经管道回传,内存峰值约为两个会话的流数据。
4. 去掉逐样本 tqdm(每样本开销可观),改为每个流写完打印样本数。

mf_lerobot 的 add_frame 逐样本 API 决定写帧循环仍在主进程串行;
加速 = 更轻的预扫 + 视频编码/流装载与写帧的流水线重叠。

用法与 pack_daily.py 完全一致,另加::

    --workers N   进程池大小(默认 4;预扫/流装载与视频截段池同宽)

依赖 conda 环境 collect(mf_lerobot)。Windows spawn 子进程有数秒
导入开销,会话/视频很少时收益缩小;要完全复刻原版行为用原脚本。
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPT_DIR), str(_SCRIPT_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_daily as base  # noqa: E402
from pack_daily import (  # noqa: E402
    MASTER_FPS, WINDOW,
    _cleanup_images, _hardware_names, _is_windowed, _stream_rate,
    _window_mask, cut_video_aligned, discover_video_slots,
    filter_qc_errors, find_sessions, load_marker_window,
    load_parquet_streams, load_stream_index, load_task_label,
    make_master_timeline, probe_video_shape, rename_out_by_span,
    video_feature_key, write_collect_meta, write_qc_meta,
)
from mf_lerobot import MultiFrequencyLeRobotDataset  # noqa: E402
from mf_lerobot.utils import (  # noqa: E402
    DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, write_info)
from embodied_brain_collect.utils.media import ffprobe_count  # noqa: E402

# 原版 build_feature_specs 对 emg 两键传 names=None,规格构建时对齐
_EMG_KEYS = ("observation.left_wrist_emg", "observation.right_wrist_emg")


class _VideoResult(NamedTuple):
    slot: str
    key: str
    n: int      # 写出的帧数(0 = 该槽位无输出)
    log: str    # worker 里的告警输出,由主进程按序回放


# ── worker:预扫(轻量索引优先,整段装载回退) ──────────────────────────────

def _probe_session_index(sd: Path, video_slots, full: bool) -> dict | None:
    """轻量预扫:load_stream_index 只读时间戳,不装载数据值。

    返回 info 同 pack_daily._probe_session,另带 stream_meta
    (键 → (估算帧率, 通道宽, 通道名)),供特征规格直接构建,
    省掉原版 build_feature_specs 的第二次整段装载。
    """
    win = load_marker_window(sd) if not full else (None, None)
    master_abs = make_master_timeline(win, sd, video_slots)
    if master_abs is None or not len(master_abs):
        print(f"[warn] 跳过 '{sd.name}': 无 marker 窗口且无相机帧,"
              "无法合成主时间轴")
        return None

    t0 = float(master_abs[0])
    present: set[str] = set()
    stream_meta: dict[str, tuple[float, int, list[str] | None]] = {}
    for key, ts, names, width in load_stream_index(sd):
        if not len(ts):
            continue
        if (_window_mask(ts, win) & (ts >= t0)).any():
            present.add(key)
            stream_meta[key] = (_stream_rate(ts), int(width),
                                list(names) if names is not None else None)

    # state/action 的通道宽在 index 里写死 58,按实际设备数修正
    n_dev = sum(1 for k in present if k.startswith("observation.device"))
    for k in ("observation.state", "action"):
        if k in stream_meta and n_dev:
            rate, _w, names = stream_meta[k]
            stream_meta[k] = (rate, n_dev * 6 + 40, names)

    # 视频完整性:与原版相同的 ffprobe 容器帧数检查
    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        npz = sd / slot / npz_name
        mp4 = sd / slot / mp4_name
        if not (npz.exists() and mp4.exists()):
            continue
        try:
            cam_ts = np.load(npz, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            continue
        in_win = np.where((_window_mask(cam_ts, win)) & (cam_ts >= t0))[0]
        if not len(in_win):
            continue
        nb = ffprobe_count(mp4)
        need = int(in_win[-1]) + 1                     # 窗口末帧的容器序号(1:1)
        if nb < need:
            print(f"[video] 跳过 '{sd.name}': {slot} 视频只有 {nb} 帧,"
                  f"窗口需要到第 {need} 帧 — 视频被截断/长度异常,整条数据跳过")
            return None
        present.add(video_feature_key(slot))

    if not present:
        print(f"[warn] 跳过 '{sd.name}': 窗口内没有任何特征数据")
        return None
    return {"dir": sd, "win": win, "master": master_abs,
            "present": present, "stream_meta": stream_meta}


def _probe_session_full(sd: Path, video_slots, full: bool) -> dict | None:
    """整段装载版预扫(= pack_daily._probe_session + stream_meta)。

    仅当轻量索引抛异常时回退使用;与原版 _probe_session 需同步维护。
    """
    win = load_marker_window(sd) if not full else (None, None)
    master_abs = make_master_timeline(win, sd, video_slots)
    if master_abs is None or not len(master_abs):
        print(f"[warn] 跳过 '{sd.name}': 无 marker 窗口且无相机帧,"
              "无法合成主时间轴")
        return None

    t0 = float(master_abs[0])
    present: set[str] = set()
    stream_meta: dict[str, tuple[float, int, list[str] | None]] = {}
    for key, ts, vals, names in load_parquet_streams(sd):
        if len(ts) and (_window_mask(ts, win) & (ts >= t0)).any():
            present.add(key)
            stream_meta[key] = (_stream_rate(ts), int(vals.shape[1]), names)

    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        npz = sd / slot / npz_name
        mp4 = sd / slot / mp4_name
        if not (npz.exists() and mp4.exists()):
            continue
        try:
            cam_ts = np.load(npz, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            continue
        in_win = np.where((_window_mask(cam_ts, win)) & (cam_ts >= t0))[0]
        if not len(in_win):
            continue
        nb = ffprobe_count(mp4)
        need = int(in_win[-1]) + 1
        if nb < need:
            print(f"[video] 跳过 '{sd.name}': {slot} 视频只有 {nb} 帧,"
                  f"窗口需要到第 {need} 帧 — 视频被截断/长度异常,整条数据跳过")
            return None
        present.add(video_feature_key(slot))

    if not present:
        print(f"[warn] 跳过 '{sd.name}': 窗口内没有任何特征数据")
        return None
    return {"dir": sd, "win": win, "master": master_abs,
            "present": present, "stream_meta": stream_meta}


def _probe_job(session_dir: str, video_slots, full: bool):
    """进程池入口:预扫一个会话,stdout 捕获后交主进程按序回放。"""
    sd = Path(session_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"[probe] {sd.name} ...")
        try:
            info = _probe_session_index(sd, video_slots, full)
        except Exception as exc:
            print(f"[probe-fast] '{sd.name}': 轻量索引失败({exc!r})— 回退整段装载")
            info = _probe_session_full(sd, video_slots, full)
    return buf.getvalue(), info


def _load_streams_job(session_dir: str, common: frozenset):
    """进程池入口:装载一个会话的全部数据流(只保留 common 里的键)。"""
    sd = Path(session_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        streams = [(k, ts, vals, names)
                   for k, ts, vals, names in load_parquet_streams(sd)
                   if k in common]
    return buf.getvalue(), streams


# ── worker:视频截段(文件与 parquet 都在 worker 里落盘) ───────────────────

def _write_timestamps_parquet(root: Path, key: str, ep_idx: int,
                              rel_ts: np.ndarray) -> None:
    """= pack_daily.write_video_timestamps,root 显式传参(worker 无 ds)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fpath = root / DEFAULT_DATA_PATH.format(
        episode_chunk=ep_idx // 1000, episode_index=ep_idx, feature_key=key)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "timestamp": pa.array(rel_ts, type=pa.float64()),
        "episode_index": pa.array(
            np.full(len(rel_ts), ep_idx, dtype=np.int64), type=pa.int64()),
    })
    pq.write_table(table, fpath, compression="snappy")


def _cut_video_job(slot_dir: str, npz_name: str, ts_key: str, mp4_name: str,
                   out_mp4: str, root: str, ep_idx: int, win: tuple,
                   t0: float, key: str, fps: int) -> _VideoResult:
    """进程池入口:一路视频的截段 + 时间戳 parquet(路径与原版一致)。

    对应原版 write_episode 的视频段:文件缺失/npz 不可读 → n=0 静默
    跳过;ffmpeg 失败则异常透传,主进程与原版一样直接崩。
    """
    d = Path(slot_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        npz_p, mp4_p = d / npz_name, d / mp4_name
        if not (npz_p.exists() and mp4_p.exists()):
            return _VideoResult(d.name, key, 0, buf.getvalue())
        try:
            all_ts = np.load(npz_p, allow_pickle=True)[ts_key].astype(np.float64)
        except Exception:
            return _VideoResult(d.name, key, 0, buf.getvalue())
        n = cut_video_aligned(mp4_p, Path(out_mp4), all_ts, win, t0, key, fps)
        if n:
            mask = _window_mask(all_ts, win) & (all_ts >= t0)
            kept = np.where(mask)[0]
            rel_ts = all_ts[kept[0]:kept[-1] + 1].astype(np.float64) - t0
            _write_timestamps_parquet(Path(root), key, ep_idx, rel_ts)
    return _VideoResult(d.name, key, n, buf.getvalue())


# ── 特征规格(读预扫缓存的 stream_meta,免第二次整段装载) ──────────────────

def _build_specs(sessions: list[dict], video_slots, common: set[str]) -> dict:
    """= pack_daily.build_feature_specs,流规格来自预扫的 stream_meta。

    视频部分与原版相同(仍解码首个含该槽位会话的第一帧取 shape)。
    对齐差异:emg 两键原版 spec 的 names 为 None,这里同样置 None。
    """
    specs: dict[str, dict] = {}

    def _first_session_with(stream_dir: str) -> Path | None:
        for info in sessions:
            if (info["dir"] / stream_dir).is_dir():
                return info["dir"]
        return None

    for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
        if video_feature_key(slot) not in common:
            continue
        sd = _first_session_with(slot)
        if sd is None:
            continue
        mp4, npz = sd / slot / mp4_name, sd / slot / npz_name
        if not (mp4.exists() and npz.exists()):
            print(f"[spec] '{slot}' 有目录但 npz/mp4 缺失 — 跳过该特征")
            continue
        try:
            h, w, c = probe_video_shape(mp4)
        except Exception as exc:
            print(f"[spec] '{slot}': 无法解码 {mp4.name} ({exc}) — 跳过该特征")
            continue
        specs[video_feature_key(slot)] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["h", "w", "c"],
            "fps": int(MASTER_FPS),
            "tolerance_s": 0.001,
        }

    for info in sessions:
        meta = info["stream_meta"]
        # marker 放最后,与原版整段装载的产出顺序一致
        keys = [k for k in meta if k != "observation.marker"]
        if "observation.marker" in meta:
            keys.append("observation.marker")
        for key in keys:
            if key in specs or key not in common:
                continue
            rate, width, names = meta[key]
            if key in _EMG_KEYS:
                names = None
            specs[key] = {
                "dtype": "float32",
                "shape": (width,),
                "names": names,
                "fps": rate,
                "tolerance_s": max(0.005, 3.0 / max(rate, 1.0)),
            }
            if _is_windowed(key):
                specs[key]["window"] = WINDOW
    return specs


# ── episode 写入(视频已由 worker 截好,主进程只写帧) ──────────────────────

def _write_episode(ds, session_dir: Path, task_label: str,
                   master_abs: np.ndarray, streams, win,
                   video_results: dict) -> None:
    """= pack_daily.write_episode,视频段换成预截结果(文件已就位)。

    master_abs 是独立 30Hz 时间轴(make_master_timeline),窗口模式下
    master_abs[0] 恰为 RUN_START,`ts >= t0` 不会切掉起始事件。
    """
    t0 = float(master_abs[0])
    master_rel = (master_abs - t0).astype(np.float64)

    for t in master_rel:
        ds.add_frame("task", task_label, float(t))

    present: set[str] = set()
    for key, ts, vals, _names in streams:
        mask = _window_mask(ts, win) & (ts >= t0)
        ts_w, vals_w = ts[mask], vals[mask]
        if not len(ts_w):
            continue
        rel = (ts_w - t0).astype(np.float64)
        for t, v in zip(rel, vals_w):
            ds.add_frame(key, v, float(t))
        present.add(key)
        print(f"  {key}: {len(rel)} samples")

    for slot, res in video_results.items():
        if not res.n:
            continue
        ds._features.pop(res.key, None)
        present.add(res.key)
        print(f"  {res.key}: 精确截段 {res.n} 帧 @{MASTER_FPS:.0f}fps")

    dropped = {k: ds._features.pop(k) for k in list(ds._features) if k not in present}
    ds.save_episode()
    for k, f in dropped.items():
        f.next_episode()
        ds._features[k] = f

    try:
        ds.meta.update_video_info()
    except Exception:
        pass  # 个别 episode 缺视频文件时不阻塞打包

    if dropped:
        print(f"[episode] '{session_dir.name}': 本会话缺少以下特征,已跳过 "
              f"{sorted(dropped)}")


# ── CLI(= pack_daily.parse_args + --workers) ──────────────────────────────

def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def parse_args(argv: list[str] | None = None):
    """先剥出 --workers,其余参数原样交给 pack_daily.parse_args。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    workers: int | None = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--workers":
            if i + 1 >= len(argv):
                raise SystemExit("[error] --workers 需要一个正整数参数")
            try:
                workers = int(argv[i + 1])
            except ValueError:
                raise SystemExit(f"[error] 无效 --workers: {argv[i + 1]!r}")
            i += 1
        elif a.startswith("--workers="):
            try:
                workers = int(a.split("=", 1)[1])
            except ValueError:
                raise SystemExit(f"[error] 无效 {a}")
        else:
            rest.append(a)
        i += 1
    if workers is not None and workers < 1:
        raise SystemExit("[error] --workers 至少为 1")
    if "-h" in rest or "--help" in rest:
        print(__doc__)
    args = base.parse_args(rest)
    args.workers = workers if workers is not None else _default_workers()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sessions = find_sessions(args.source, args.date)
    if not sessions:
        print(f"[error] '{args.source}' 下没有匹配 {args.date}-* 的会话目录")
        return 1
    sessions = filter_qc_errors(sessions)
    if not sessions:
        print(f"[error] {args.date} 的会话全部被 QC 判为 ERROR — 没有可打包数据")
        return 1
    if args.max_episodes:
        sessions = sessions[: args.max_episodes]
    print(f"[input] {len(sessions)} 个会话: "
          + ", ".join(s.name for s in sessions[:5])
          + (" ..." if len(sessions) > 5 else ""))

    video_slots = discover_video_slots(sessions)
    if not video_slots:
        print("[error] 所有会话都没有可用的视频流")
        return 1
    print(f"[video] 视频槽位: {list(video_slots)}")

    if not args.full:
        missing_win = [s.name for s in sessions if load_marker_window(s) == (None, None)]
        if missing_win:
            print(f"[warn] 以下会话没有 marker 窗口,将使用整段录制: {missing_win}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # ---- 并行预扫(轻量索引)→ 特征集并集 → 剔除缺模态会话 ----
        print(f"[probe-fast] 并行预扫 {len(sessions)} 个会话 "
              f"(workers={args.workers},只读时间戳索引)")
        probe_futs = [pool.submit(_probe_job, str(sd), video_slots, bool(args.full))
                      for sd in sessions]
        probed = []
        for sd, fut in zip(sessions, probe_futs):
            log, info = fut.result()
            sys.stdout.write(log)
            if info is not None:
                probed.append(info)
        sessions = probed
        if not sessions:
            print("[error] 没有可用的会话(合成不出主时间轴或无数据流)")
            return 1

        # 特征集 = 全部模态(跨会话并集),缺任一模态的会话整体剔除
        # (剔除逻辑与原版一致,见 pack_daily.main 注释)
        required: set[str] = set()
        for info in sessions:
            required |= info["present"]
        kept = []
        for info in sessions:
            missing = sorted(required - info["present"])
            if missing:
                print(f"[spec] 剔除 '{info['dir'].name}': 缺少模态 {missing}")
            else:
                kept.append(info)
        sessions = kept
        if not sessions:
            print("[error] 没有会话同时具备全部模态 — 无法打包")
            return 1
        common = required

        specs = _build_specs(sessions, video_slots, common)
        if not specs:
            print("[error] 所有会话都没有可用数据流")
            return 1
        print(f"[spec] {len(specs)} 个特征: " + ", ".join(specs))

        # 输出目录:按这段数据的起止时刻落名(--out 未指定时改名,指定时
        # 进一层 <out>/<日期>-起-止),再查重
        rename_out_by_span(args, sessions)
        if args.out.exists():
            if not args.force:
                print(f"[error] 输出目录已存在: {args.out} (用 --force 覆盖)")
                return 1
            shutil.rmtree(args.out)

        ds = MultiFrequencyLeRobotDataset.create(
            repo_id=args.out.name, fps=MASTER_FPS,
            features=specs, root=args.out, use_videos=True,
        )

        # info.json 保持 mf_lerobot 写下的 LeRobot 标准字段 —— 额外信息
        # 一律进 meta/collect_info.jsonl(与原版一致)

        video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]
        from embodied_brain_collect.session import environment as env_mod

        # ---- 视频截段全部提前提交:独立进程池,与写帧循环完全重叠;
        #      ep_idx = 会话序号(episode 按序落盘,与 ds.meta 一致) ----
        n_video_jobs = len(sessions) * len(video_slots)
        with ProcessPoolExecutor(
                max_workers=min(args.workers, max(1, n_video_jobs))) as video_pool:
            vid_futs: dict[tuple, object] = {}
            for i, info in enumerate(sessions):
                ep_idx = i
                t0 = float(info["master"][0])
                for slot, (npz_name, ts_key, mp4_name) in video_slots.items():
                    key = video_feature_key(slot)
                    out_mp4 = ds.root / DEFAULT_VIDEO_PATH.format(
                        episode_chunk=ep_idx // 1000, video_key=key,
                        episode_index=ep_idx)
                    vid_futs[(i, slot)] = video_pool.submit(
                        _cut_video_job, str(info["dir"] / slot),
                        npz_name, ts_key, mp4_name, str(out_mp4),
                        str(ds.root), ep_idx, tuple(info["win"]), t0, key,
                        int(MASTER_FPS))

            # ---- 流装载预取(深度 1):写会话 i 时后台装 i+1 ----
            stream_futs: dict[int, object] = {}

            def _submit_streams(i: int) -> None:
                stream_futs[i] = pool.submit(_load_streams_job,
                                             str(sessions[i]["dir"]),
                                             frozenset(common))

            try:
                _submit_streams(0)
                for i, info in enumerate(sessions):
                    if i + 1 < len(sessions):
                        _submit_streams(i + 1)
                    log, streams = stream_futs[i].result()
                    sys.stdout.write(log)

                    video_results: dict[str, _VideoResult] = {}
                    for slot in video_slots:
                        res = vid_futs[(i, slot)].result()
                        video_results[slot] = res
                        if res.log:
                            sys.stdout.write(res.log)

                    task_label = load_task_label(info["dir"], env_mod)
                    print(f"[episode {i}] '{info['dir'].name}' "
                          f"(master {len(info['master'])} frames) 写帧中 ...")
                    _write_episode(ds, info["dir"], task_label, info["master"],
                                   streams, info["win"], video_results)
                    if not args.keep_images:
                        _cleanup_images(args.out, i, video_keys)
                    print(f"[episode {i}] '{info['dir'].name}' task='{task_label}' "
                          f"{len(info['master'])} frames")
            finally:
                # 中途异常时取消还没开工的后台任务,进程池尽快退出
                for f in stream_futs.values():
                    f.cancel()
                for f in vid_futs.values():
                    f.cancel()

            # meta 附上每个 episode 的 qc report(源会话 qc_report.json)
            write_qc_meta(args.out, sessions)
            # meta 附上每个 episode 的采集信息(源会话 meta.yaml 的 collect)
            write_collect_meta(args.out, sessions)

    print(f"[done] {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
