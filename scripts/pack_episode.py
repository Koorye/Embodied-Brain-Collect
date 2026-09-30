#!/usr/bin/env python3
"""单条(episode)打包 —— 一个会话目录打成一份独立 LeRobot 数据集。

与 pack_daily(一天的所有会话合并成一个数据集、每会话一个 episode)相对,
本脚本以"条"为单位:一条数据(RUN_START..RUN_END 窗口)= 一份独立数据集,
方便单条送训练/回放/质检,不必等凑齐一整天。

    python scripts/pack_episode.py data/session-day/2026-09-18-14-53-54
    python scripts/pack_episode.py <session_dir> --out data/lerobot/episode/xx

输出(默认):镜像会话的保存路径、只多一层 ``lerobot``,日期-时刻形态的
会话名拆成日期/时刻两级,叶子名带采集编号(meta.yaml 快照里的
``collector_id``,即该条 ``collect_info.jsonl`` 单行里的同名字段)——
``data/session-day/2026-09-18-19-16-33`` 打包后落在
``data/lerobot/session-day/2026-09-18/<collector_id>-2026-09-18-19-16-33``;
会话不在仓库 ``data/`` 下时回退 ``data/lerobot/episode/<会话名>``
(同样拆级/加前缀)。``--out`` 指定时数据集直接落在给定目录。数据质量门槛与
pack_daily 完全一致:qc_report / meta status 非法的会话拒绝打包,marker 无
RUN_START/RUN_END 窗口拒绝对齐(--full 显式全量除外),视频容器帧数必须
覆盖窗口末帧。

加速(pack_daily_fast 同款,打包结果与原版一致):

1. 预扫走 ``load_stream_index`` 只读时间戳与列名(原版预扫要整段解压全部
   npz,写帧时再解压一遍 —— 每条流被完整解压两次);索引异常自动回退整段
   装载,语义不变。特征规格直接从预扫缓存的 stream_meta 构建,又省一次。
2. 视频截段(ffmpeg 重编码 + 首帧自验 + 时间戳 parquet)在预扫后立刻
   全部提交独立进程池,与主进程的 npz 装载、add_frame 写帧完全重叠。
3. 去掉逐样本 tqdm(每样本开销可观),改为每个流写完打印样本数。

mf_lerobot 的 add_frame 逐样本 API 决定写帧循环仍在主进程串行。
launcher 侧的 --pack-episode(run_session / launcher)在每条录完保留后
自动调用本脚本;也可随时对历史会话手动执行。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shutil
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPT_DIR), str(_SCRIPT_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pack_daily import (  # noqa: E402
    MASTER_FPS, MICROPHONE_KEY, PROJECT_ROOT, _cleanup_images,
    _is_constant_stream, _window_mask, cut_video_aligned, discover_video_slots,
    filter_meta_status, filter_qc_errors, load_marker_window,
    load_microphone_chunks, load_microphone_index, load_parquet_streams,
    load_task_label, write_collect_meta, write_qc_meta,
)
from pack_daily_fast import (  # noqa: E402
    _build_specs, _default_workers, _probe_session_full, _probe_session_index,
)

_SESSION_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})$")


class _VideoResult(NamedTuple):
    slot: str
    key: str
    n: int      # 写出的帧数(0 = 该槽位无输出)
    log: str    # worker 里的告警输出,由主进程按序回放


def _split_session_dir(p: Path) -> tuple[Path, str | None]:
    """会话名 ``<日期>-<时刻>`` 拆成日期/时刻两级 ``<日期>/<时刻>``。

    返回 (拆级后的路径, 完整会话名);名字不是日期-时刻形态时原样返回
    (完整会话名为 None)。已拆级的名字(纯时刻)不匹配,重复调用安全。
    """
    m = _SESSION_NAME_RE.match(p.name)
    if m is None:
        return p, None
    return p.parent / m.group(1) / m.group(2), p.name


def _load_collector_id(sd: Path) -> str | None:
    """会话 meta.yaml 快照里的 ``collector_id``(打包后即该条的
    ``collect_info.jsonl`` 单行里的同名字段)。

    读取失败或未维护时返回 None —— 输出目录不带采集编号前缀。
    """
    try:
        meta = yaml.safe_load(
            (sd / "meta.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(meta, dict):
        return None
    cid = meta.get("collector_id")
    if cid is None:
        return None
    s = str(cid).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    return s or None


# ── worker:视频截段(不 import mf_lerobot,spawn 子进程免 torch 冷启动)────

def _cut_video_job(slot_dir: str, npz_name: str, ts_key: str, mp4_name: str,
                   out_mp4: str, out_parquet: str, ep_idx: int, win: tuple,
                   t0: float, key: str, fps: int) -> _VideoResult:
    """进程池入口:一路视频的截段 + 时间戳 parquet。

    与 pack_daily_fast._cut_video_job 同一件事,两处不同:parquet 路径由
    主进程算好传入(worker 不必 import mf_lerobot);stdout 捕获回放一致。
    文件缺失/npz 不可读 → n=0 静默跳过;ffmpeg 失败则异常透传,主进程与
    原版一样直接崩。
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
            import pyarrow as pa
            import pyarrow.parquet as pq
            fpath = Path(out_parquet)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table({
                "timestamp": pa.array(rel_ts, type=pa.float64()),
                "episode_index": pa.array(
                    np.full(len(rel_ts), ep_idx, dtype=np.int64),
                    type=pa.int64()),
            })
            pq.write_table(table, fpath, compression="snappy")
    return _VideoResult(d.name, key, n, buf.getvalue())


# ── episode 写入(视频已由 worker 截好,主进程只写帧) ─────────────────────

def _write_episode(ds, session_dir: Path, task_label: str,
                   master_abs: np.ndarray, streams, win,
                   video_results: dict[str, _VideoResult]) -> None:
    """= pack_daily.write_episode,去掉 tqdm、视频段换成预截结果。

    master_abs 是独立 30Hz 时间轴(make_master_timeline),窗口模式下
    master_abs[0] 恰为 RUN_START,`ts >= t0` 不会切掉起始事件。
    """
    t0 = float(master_abs[0])
    master_rel = (master_abs - t0).astype(np.float64)

    for t in master_rel:
        ds.add_frame("task", task_label, float(t))

    present: set[str] = set()
    for key, ts, vals, _names in streams:
        if _is_constant_stream(ts):
            # 单行常量流(如 eeg 阻抗门禁):不过窗口、不平移,
            # rel timestamp 固定 0 原样落一行
            ts_w, vals_w = ts, vals
            rel = ts.astype(np.float64)
        else:
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

    # 音频:按块投喂 AudioFeature(时间戳 = 块首采样),窗口裁剪按块起始。
    # mf_lerobot 在 save_episode 时统一写 wav + 索引 parquet 并校验一致性。
    # 本会话没有音频时不喂 —— 走下面的 dropped 机制跳过该特征。
    mic = load_microphone_index(session_dir)
    if mic is not None and MICROPHONE_KEY in ds._features:
        chunks = load_microphone_chunks(session_dir, mic)
        if chunks is not None:
            mask = _window_mask(mic["start"], win) & (mic["start"] >= t0)
            sel = np.where(mask)[0]
            if len(sel):
                for i in sel:
                    ds.add_frame(MICROPHONE_KEY, chunks[i],
                                 float(mic["start"][i] - t0))
                present.add(MICROPHONE_KEY)
                print(f"  {MICROPHONE_KEY}: {len(sel)} 块 @{mic['rate']}Hz "
                      f"{mic['channels']}ch")

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


def main(argv: list[str] | None = None, *, gated: bool = True) -> int:
    """``gated=False``:跳过 QC / meta status 门槛 —— 仅供采集侧
    (launcher.run_pack_episode,经 run_base.run_queue 以"QC 无错且操作员
    保留"为触发条件)进程内调用,同一信息源不再判第二次;独立 CLI 一律
    gated=True,门槛必须自足。"""
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python scripts/pack_episode.py "
               "data/session-day/2026-09-18-14-53-54",
    )
    ap.add_argument("session_dir", type=Path,
                    help="单个会话目录(一条 episode 的录制数据)")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录(默认镜像会话保存路径,只多一层 "
                         "lerobot,会话名按 <日期>/<collector_id>-<日期>-<时刻>"
                         "组织;指定时数据集直接落在该目录)")
    ap.add_argument("--force", action="store_true",
                    help="输出目录已存在时先删除")
    ap.add_argument("--full", action="store_true",
                    help="使用整段录制,忽略 RUN_START..RUN_END 标记窗口")
    ap.add_argument("--keep-images", action="store_true",
                    help="保留中间 PNG 帧(默认编码后删除)")
    ap.add_argument("--workers", type=int, default=None,
                    help="视频截段进程池大小(默认 min(4, 核数-1))")
    args = ap.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        print("[error] --workers 需要一个正整数")
        return 1

    sd = args.session_dir.resolve()
    if not sd.is_dir():
        print(f"[error] 会话目录不存在: {sd}")
        return 1

    # ---- 与 pack_daily 同一门槛:QC / meta status 不过的会话不打包 ----
    # (采集侧进程内调用已由 run_queue 的触发条件把关,信息源相同,不重判)
    sessions = [sd]                      # gated=False 的基线;下方按需过滤
    if gated:
        sessions = filter_qc_errors(sessions)
        sessions = filter_meta_status(sessions)
        if not sessions:
            print(f"[error] '{sd.name}' 被 QC / meta status 判为不可打包")
            return 1

    video_slots = discover_video_slots(sessions)
    if not video_slots:
        print("[error] 会话没有可用的视频流")
        return 1

    print(f"[probe] 预扫 '{sd.name}' 的可用特征(只读时间戳,不装载数据值)")
    try:
        info = _probe_session_index(sd, video_slots, bool(args.full))
    except Exception as exc:
        print(f"[probe] 轻量索引失败({exc!r})— 回退整段装载")
        info = _probe_session_full(sd, video_slots, bool(args.full))
    if info is None:
        print("[error] 会话不可打包(见上方跳过原因)")
        return 1

    specs = _build_specs([info], video_slots, info["present"])
    if not specs:
        print("[error] 会话没有可用数据流")
        return 1
    print(f"[spec] {len(specs)} 个特征: " + ", ".join(specs))

    # 输出目录:镜像会话的保存路径、只多一层 lerobot,日期-时刻形态的
    # 会话名拆成日期/时刻两级,叶子名带采集编号 ——
    #   data/<班次根>/<日期>-<时刻>
    #   → data/lerobot/<班次根>/<日期>/<collector_id>-<日期>-<时刻>
    # collector_id 读会话 meta.yaml 快照(即该条 collect_info.jsonl 单行
    # 里的同名字段),未维护时不加前缀。会话不在仓库 data/ 下时回退
    # data/lerobot/episode/<会话名>(同样拆级/加前缀);--out 显式指定时
    # 数据集直接落在给定目录,不再改名。
    repo_id: str
    if args.out is not None:
        out = args.out.resolve()
        repo_id = out.name
    else:
        data_root = PROJECT_ROOT / "data"
        try:
            out, repo_id = _split_session_dir(
                data_root / "lerobot" / sd.relative_to(data_root))
        except ValueError:
            out, repo_id = _split_session_dir(
                data_root / "lerobot" / "episode" / sd.name)
        repo_id = repo_id or out.name
        cid = _load_collector_id(sd)
        if cid is not None:
            repo_id = f"{cid}-{repo_id}"
            out = out.parent / repo_id
    if out.exists():
        if not args.force:
            print(f"[error] 输出目录已存在: {out} (用 --force 覆盖)")
            return 1
        shutil.rmtree(out)

    if "mf_lerobot" not in sys.modules:    # 已被预热/上一条加载过则免提示
        print("[pack] 加载打包依赖(mf_lerobot / torch,冷启动约 10s)…",
              flush=True)
    from mf_lerobot import MultiFrequencyLeRobotDataset
    from mf_lerobot.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=repo_id, fps=MASTER_FPS,
        features=specs, root=out, use_videos=True,
    )

    from embodied_brain_collect.session import environment as env_mod
    from pack_daily import video_feature_key
    ep_idx = ds.meta.total_episodes
    t0 = float(info["master"][0])
    video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]

    # ---- 视频截段全部提前提交:独立进程池,与 npz 装载、写帧循环重叠 ----
    # (ego_headband 一槽多路:每个 <name> 是独立 entry/特征,逐 entry 提交)
    jobs = [(slot, npz_name, ts_key, mp4_name, suffix)
            for slot, entries in video_slots.items()
            for npz_name, ts_key, mp4_name, suffix in entries]
    workers = args.workers or _default_workers()
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(jobs)))) as pool:
        futs: dict[str, object] = {}
        for slot, npz_name, ts_key, mp4_name, suffix in jobs:
            key = video_feature_key(suffix)
            futs[suffix] = pool.submit(
                _cut_video_job, str(sd / slot), npz_name, ts_key, mp4_name,
                str(ds.root / DEFAULT_VIDEO_PATH.format(
                    episode_chunk=ep_idx // 1000, video_key=key,
                    episode_index=ep_idx)),
                str(out / DEFAULT_DATA_PATH.format(
                    episode_chunk=ep_idx // 1000, episode_index=ep_idx,
                    feature_key=key)),
                ep_idx, tuple(info["win"]), t0, key, int(MASTER_FPS))
        try:
            task_label = load_task_label(sd, env_mod)
            streams = [s for s in load_parquet_streams(sd)
                       if s[0] in info["present"]]
            video_results: dict[str, _VideoResult] = {}
            for _slot, *_rest, suffix in jobs:
                res: _VideoResult = futs[suffix].result()
                video_results[suffix] = res
                if res.log:
                    sys.stdout.write(res.log)
            _write_episode(ds, sd, task_label, info["master"], streams,
                           info["win"], video_results)
        finally:
            for f in futs.values():
                f.cancel()
    if not args.keep_images:
        _cleanup_images(out, ep_idx, video_keys)

    write_qc_meta(out, [info])
    write_collect_meta(out, [info])

    print(f"[done] 1 episode, {ds.meta.total_frames} frames → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
