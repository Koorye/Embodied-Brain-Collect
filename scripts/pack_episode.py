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

launcher 侧的 --pack-episode(run_session / launcher)在每条录完保留后
自动调用本脚本;也可随时对历史会话手动执行。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import types
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPT_DIR), str(_SCRIPT_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pack_daily import (  # noqa: E402
    MASTER_FPS, PROJECT_ROOT, _cleanup_images, _probe_session,
    build_feature_specs, discover_video_slots, filter_meta_status,
    filter_qc_errors, load_parquet_streams, load_task_label,
    write_collect_meta, write_episode, write_qc_meta,
)

_SESSION_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})$")


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
    args = ap.parse_args(argv)

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
    info = _probe_session(sd, video_slots, types.SimpleNamespace(full=args.full))
    if info is None:
        print("[error] 会话不可打包(见上方跳过原因)")
        return 1

    specs = build_feature_specs([sd], video_slots, info["present"])
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
    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=repo_id, fps=MASTER_FPS,
        features=specs, root=out, use_videos=True,
    )

    from embodied_brain_collect.session import environment as env_mod
    task_label = load_task_label(sd, env_mod)
    streams = [s for s in load_parquet_streams(sd) if s[0] in info["present"]]
    write_episode(ds, sd, task_label, info["master"], streams,
                  info["win"], video_slots)
    video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]
    if not args.keep_images:
        _cleanup_images(out, 0, video_keys)

    write_qc_meta(out, [info])
    write_collect_meta(out, [info])

    print(f"[done] 1 episode, {ds.meta.total_frames} frames → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
