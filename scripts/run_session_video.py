#!/usr/bin/env python3
"""video_rate(RLHF 视频打分)入口 —— 按子集(任务)抽取组合 → 录制 → 记台账。

素材按**子集**组织:``configs/videos/<子集>/``,一个子集 = 一个任务
(与图纸模式的场景目录同构)。每个子集必须有 ``config.yaml`` 给出任务指令
(``task`` 字段,开录时填进 stim.yaml 的 ``instr1_text`` 的 ``{task}``
占位符,作为被试的想象内容),以及一批 ``success_*`` / ``fail_*`` mp4::

    configs/videos/
      盖笔盖/
        config.yaml          # task: 双手分别拿起笔和笔盖、盖上笔盖后把笔放入笔筒
        fail_episode_000019.mp4
        success_episode_000074.mp4
        used.yaml            # 该子集的台账(自动维护)

每次运行:

  1. 选子集(--subset;只有一个子集时自动选,多个时列出交互选)
  2. 读子集 config.yaml 的任务指令 + 台账(每条视频的 uses 次数)
  3. 抽取组合:两池各取"使用次数最少"的视频,fail/success 尽量对半
     (fail 偏多),场内打乱;trial 数上限受 markers.yaml video_rate 码位限制
  4. 试解每条素材的第一帧(fail-fast:解不出的直接拦,别等 stim 中途死)
  5. 逐条录制 —— **每条 trial 一次独立的录制会话**(与图纸/任务入口对齐):
     launcher 启动(EEG + marker 多进程)→ stim 单 trial → 落盘 → 自动 QC
     → 下一条。评分 JSON 经 ``--result-dir`` 直接写进本次录制目录,
     播放的视频也**拷贝**进录制目录 —— 数据自包含,单目录 = 单 episode
  6. 每条录完(rc==0):视频拷贝进录制目录;QC 无错才把该视频记入子集
     台账(uses+1)。没录完整(rc!=0)一律 meta 标 failed,不拷贝不记账

    python scripts/run_session_video.py                  # 自动选子集
    python scripts/run_session_video.py --subset 盖笔盖
    python scripts/run_session_video.py --subset 盖笔盖 --n-trials 4
    python scripts/run_session_video.py --subset 盖笔盖 --dry-run
    python scripts/run_session_video.py --subset 盖笔盖 --dummy

编排逻辑(队列循环/汇总/失败排查)与图纸/任务入口共享,见
session/run_base.py;本入口固定自动保留(无 n/r/q),失败重跑整个脚本即可
—— 台账只记成功的,失败素材下次仍优先被抽到。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.session.config import (  # noqa: E402
    configs_dir, load_stim)
from embodied_brain_collect.session.run_base import (  # noqa: E402
    ModeHooks, add_common_cli, resolve_collect, run_queue, seed_or_now)
from embodied_brain_collect.stim import marker_codes as M  # noqa: E402
from embodied_brain_collect.stim.factory import build_stim_cmd  # noqa: E402

#: 池子前缀 -> 池名;历史素材的 sucess_ 拼写按 success 识别
_POOL_PREFIXES = (("success_", "success"), ("sucess_", "success"),
                  ("fail_", "fail"))


def videos_dir() -> Path:
    return configs_dir() / "videos"


def list_subsets(video_dir: Path | None = None) -> list[Path]:
    """``configs/videos/`` 下含 mp4 的子目录(=子集/任务),按名排序。"""
    root = video_dir or videos_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and any(p.glob("*.mp4")))


def resolve_subset(spec: str | None,
                   video_dir: Path | None = None) -> Path:
    """选一个子集目录:显式名字 > 仅一个时自动 > 列表交互选。"""
    root = video_dir or videos_dir()
    subsets = list_subsets(root)
    if spec:
        cand = Path(spec)
        if not cand.is_dir():
            cand = root / str(spec)
        if cand.is_dir():
            return cand
        raise SystemExit(
            f"找不到视频子集 {spec!r} — "
            f"可用: {[p.name for p in subsets] or ['(无)']}")
    if len(subsets) == 1:
        return subsets[0]
    if not subsets:
        raise SystemExit(
            f"{root} 下没有任何视频子集 — 请建 <子集>/ 子目录,放入 "
            "success_*/fail_*.mp4 与 config.yaml(见该目录 README)")
    print("[video] 多个子集,请选择(--subset <名字> 可跳过交互):")
    for i, p in enumerate(subsets, 1):
        print(f"  [{i}] {p.name}")
    try:
        raw = input("子集编号(回车=1): ").strip() or "1"
    except (OSError, EOFError):     # 非交互环境(管道/CI)没有 stdin
        raise SystemExit(
            "标准输入不可交互 — 请用 --subset 指定;"
            f"可用: {[p.name for p in subsets]}")
    if raw.isdigit() and 1 <= int(raw) <= len(subsets):
        return subsets[int(raw) - 1]
    print("无效编号,请重输")


def load_subset_config(subset_dir: Path) -> dict:
    """子集 config.yaml;``task``(任务指令/想象内容)必须非空。"""
    path = subset_dir / "config.yaml"
    if not path.is_file():
        raise SystemExit(
            f"{path} 不存在 — 每个子集需要 config.yaml 给出任务指令"
            "(task 字段,作为被试的想象内容)")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"{path} 读取失败: {exc}")
    if not isinstance(cfg, dict) or not str(cfg.get("task") or "").strip():
        raise SystemExit(
            f"{path} 缺 task 字段(或为空)— 任务指令(想象内容)必须给定,"
            '例如: task: 双手分别拿起笔和笔盖，盖上笔盖后把笔放入笔筒')
    return cfg


def build_instr(cfg: dict) -> tuple[str, str | None]:
    """子集的任务指令 → (instr1, instr2|None)。

    stim.yaml 的 ``video_rate.instr1_text`` 是模板,``{task}`` 占位符填
    子集 config.yaml 的 ``task``;子集直接给 ``instr1_text`` 时整体覆盖。
    """
    over = (load_stim() or {}).get("video_rate") or {}
    custom1 = str(cfg.get("instr1_text") or "").strip()
    if custom1:
        instr1 = custom1
    else:
        template = str(over.get("instr1_text") or "")
        if "{task}" not in template:
            raise SystemExit(
                "configs/stim.yaml 的 video_rate.instr1_text 缺 {task} "
                "占位符 — 子集的任务指令没有落点,补上占位符或在子集 "
                "config.yaml 里直接给 instr1_text")
        instr1 = template.replace("{task}", str(cfg["task"]).strip())
    custom2 = str(cfg.get("instr2_text") or "").strip()
    instr2 = custom2 or None
    return instr1, instr2


def label_of(stem: str) -> str | None:
    low = stem.lower()
    for prefix, pool in _POOL_PREFIXES:
        if low.startswith(prefix):
            return pool
    return None


def scan_pool(subset_dir: Path) -> dict[str, list[Path]]:
    """子集目录 → {pool: [mp4 路径]}(按文件名排序)。"""
    pool: dict[str, list[Path]] = {"success": [], "fail": []}
    for p in sorted(subset_dir.glob("*.mp4")):
        label = label_of(p.stem)
        if label is None:
            print(f"[video] ⚠ 跳过无法识别前缀的素材(需 success_*/fail_*): "
                  f"{p.name}")
            continue
        pool[label].append(p)
    return pool


def load_ledger(subset_dir: Path) -> dict[str, dict]:
    """子集台账 used.yaml → {stem: {uses, last_session}};缺失/损坏 = 空。"""
    p = subset_dir / "used.yaml"
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 — 台账坏了不该挡住采集
        print(f"[video] ⚠ 台账 {p} 读取失败({exc})— 按空台账处理")
        return {}
    return {str(k): dict(v) for k, v in data.items()
            if isinstance(v, dict)}


def save_ledger(subset_dir: Path, ledger: dict[str, dict]) -> None:
    lines = ["# 本子集(video_rate)素材台账(run_session_video 自动维护;"
             "删除即清零)\n"]
    for stem in sorted(ledger):
        item = ledger[stem]
        lines.append(f"{stem}:\n  uses: {int(item.get('uses', 0))}\n"
                     f"  last_session: {item.get('last_session', '')}\n")
    (subset_dir / "used.yaml").write_text("".join(lines), encoding="utf-8")


def _pick(pool: list[Path], ledger: dict[str, dict], n: int,
          rng: random.Random) -> list[Path]:
    """取 n 条:按 (使用次数, 随机) 排序,使用最少的优先。"""
    def uses(p: Path) -> int:
        return int(ledger.get(p.stem, {}).get("uses", 0))

    keyed = sorted(pool, key=lambda p: (uses(p), rng.random()))
    return keyed[:n]


def compose_trials(pool: dict[str, list[Path]], ledger: dict[str, dict],
                   n_trials: int,
                   rng: random.Random) -> list[dict[str, Path]]:
    """一场的组合:fail/success 尽量对半(fail 偏多),场内打乱。

    某一池不够时从另一池补(打印告警);两池都空则报错。
    """
    if not pool["fail"] and not pool["success"]:
        raise SystemExit(
            "素材池为空 — 请把 success_*.mp4 / fail_*.mp4 放入子集目录")

    n_fail = n_trials - n_trials // 2          # 奇数时 fail 多一个
    n_success = n_trials // 2
    picks: list[tuple[str, Path]] = []
    taken: set[Path] = set()

    def take(label: str, n: int) -> None:
        candidates = [p for p in pool[label] if p not in taken]
        if not candidates:
            return
        got = _pick(candidates, ledger, min(n, len(candidates)), rng)
        taken.update(got)
        picks.extend((label, p) for p in got)

    take("fail", n_fail)
    take("success", n_success)
    # 池子不够对半:用另一池的剩余补齐到 n_trials
    short = n_trials - len(picks)
    if short > 0:
        print(f"[video] ⚠ 某一池素材不足,用另一池补 {short} 条")
        rest = [p for label in ("fail", "success")
                for p in pool[label] if p not in taken]
        got = _pick(rest, ledger, min(short, len(rest)), rng)
        taken.update(got)
        picks.extend((label_of(p.stem), p) for p in got)
    if len(picks) < n_trials:
        print(f"[video] ⚠ 素材总量不足,本场只有 {len(picks)}/{n_trials} trial")

    rng.shuffle(picks)
    return [{"label": label, "video": p} for label, p in picks]


def verify_decodable(trials: list[dict]) -> list[str]:
    """逐条试解第一帧;返回解不出的视频路径列表。

    复用 stim 的解码回退(cv2 优先,ffmpeg 软解兜底)—— 这里能过,
    stim 播放才能过;提前拦住比开录后 stim 中途死掉强。
    """
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource
    bad: list[str] = []
    for t in trials:
        path = str(t["video"].resolve())
        try:
            if _FrameSource(path).first_frame() is None:
                bad.append(path)
        except SystemExit:
            bad.append(path)
    return bad


def _hooks(subset_dir: Path, cfg: dict, ledger: dict,
           instr1: str, instr2: str | None) -> ModeHooks:
    """视频模式行为:每 trial 一次录制会话 —— 评分 JSON 经 --result-dir
    写进录制目录,视频拷贝进录制目录,成功才记子集台账。"""
    task_name = str(cfg.get("name") or "").strip() or subset_dir.name
    scene = str(cfg.get("scene") or "").strip()

    def preroll(job: dict, run_dir: Path, args) -> bool:
        print(f"\n{'─' * 68}")
        print(f"  trial {job['trial']}/{job['total']}  [{job['label']:^7}] "
              f"{job['video'].name}")
        print(f"  录制目录: {run_dir}")
        print(f"{'─' * 68}")
        return True

    def stim_cmd(job: dict, stim: str, run_dir: Path) -> list:
        item = {"video": str(job["video"].resolve()),
                "instr1_text": instr1}
        if instr2:
            item["instr2_text"] = instr2
        return (build_stim_cmd("video_rate")
                + ["--trials-json", json.dumps([item], ensure_ascii=False),
                   # 评分 JSON 直接写进本次录制目录(stim 会以目录名做文件戳)
                   "--result-dir", str(run_dir)])

    def on_result(job: dict, run_dir: Path, choice: str, status: str,
                  rc: int) -> None:
        if rc != 0:
            print(f"  本条未录完(rc={rc})— 视频不拷贝、不记台账,下次抽取仍优先")
            return
        dst = run_dir / job["video"].name
        try:
            shutil.copy2(job["video"], dst)
            print(f"  视频已拷贝 → {dst}")
        except OSError as exc:
            print(f"[video] ⚠ 视频拷贝失败({exc})— 录制目录缺视频副本",
                  file=sys.stderr)
        if status != "success":
            print("  QC 有 ERROR — 视频不记台账(下次抽取仍优先)")
            return
        item = ledger.setdefault(job["video"].stem,
                                 {"uses": 0, "last_session": ""})
        item["uses"] = int(item.get("uses", 0)) + 1
        item["last_session"] = str(run_dir)
        save_ledger(subset_dir, ledger)
        print(f"  台账已更新 → {subset_dir / 'used.yaml'}")

    return ModeHooks(
        label="video",
        preroll=preroll,
        meta_kwargs=lambda job: {"task_name": task_name,
                                 "scene": scene or None},
        stim_cmd=stim_cmd,
        job_id=lambda job: job["video"].stem,
        on_result=on_result,
        qc_error_note="视频不记入台账 — 下次抽取仍优先",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_cli(ap, session_dir_default="data/video-session")
    ap.add_argument("--subset", default=None,
                    help="视频子集 = 一个任务(configs/videos 下的子目录名);"
                         "缺省:仅一个子集时自动选,多个时交互选")
    ap.add_argument("--n-trials", type=int, default=6,
                    help=f"本场 trial 数(1..{M.MAX_VR_TRIALS})")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help=f"子集根目录(默认 {videos_dir()})")
    ap.add_argument("--dry-run", action="store_true",
                    help="抽取组合并校验素材可解码,不启动录制")
    args = ap.parse_args(argv)
    args.auto_keep = True   # 视频模式没有 n/r/q:录完自动保留,失败重跑整个脚本

    if M.MAX_VR_TRIALS is None:
        raise SystemExit("markers.yaml 已禁用 video_rate 码段"
                         "(video_rate_base: null)— 无法发 trial 阶段码")
    if not 1 <= args.n_trials <= M.MAX_VR_TRIALS:
        raise SystemExit(f"--n-trials 须在 1..{M.MAX_VR_TRIALS}"
                         "(markers.yaml video_rate 码位限制)")

    subset_dir = resolve_subset(args.subset, args.video_dir)
    cfg = load_subset_config(subset_dir)
    instr1, instr2 = build_instr(cfg)
    print(f"[video] 子集 {subset_dir.name} — 任务: {cfg['task']}"
          + (f"(场景: {cfg.get('scene')})" if cfg.get("scene") else ""))

    seed = seed_or_now(args)
    rng = random.Random(seed)
    pool = scan_pool(subset_dir)
    ledger = load_ledger(subset_dir)
    trials = compose_trials(pool, ledger, args.n_trials, rng)

    # fail-fast:素材解码不了(AV1 无解码器/文件损坏)直接拦,别等开录后
    # stim 中途死掉;能过这里 stim 才能播
    bad = verify_decodable(trials)
    if bad:
        print("[video] ✗ 以下素材解码失败(cv2/ffmpeg 都读不出帧)— "
              "请重编码为 h264/h265 后再放入:", file=sys.stderr)
        for b in bad:
            print(f"    {b}", file=sys.stderr)
        print("    重编码示例: ffmpeg -i in.mp4 -c:v libx265 -crf 20 out.mp4",
              file=sys.stderr)
        return 2

    print(f"[video] 本次组合({len(trials)} trials, "
          f"fail={sum(1 for t in trials if t['label'] == 'fail')} "
          f"success={sum(1 for t in trials if t['label'] == 'success')}):")
    for i, t in enumerate(trials, 1):
        uses = int(ledger.get(t["video"].stem, {}).get("uses", 0))
        print(f"  {i:2d}. [{t['label']:^7}] {t['video'].name} "
              f"(已用 {uses} 次)")
    if args.dry_run:
        print("[video] dry-run — 未启动录制。")
        return 0

    session_root = args.session_dir.resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    collect = resolve_collect(args)
    hooks = _hooks(subset_dir, cfg, ledger, instr1, instr2)
    # 每条 trial 一个队列项 = 一次独立录制(录→存→下一条,与其他入口对齐)
    queue = [{"trial": i, "total": len(trials), "video": t["video"],
              "label": t["label"]} for i, t in enumerate(trials, 1)]

    return run_queue(session_root, queue, stim="video_rate", args=args,
                     collect=collect, hooks=hooks, seed=seed, strict_rc=True)


if __name__ == "__main__":
    sys.exit(main())
