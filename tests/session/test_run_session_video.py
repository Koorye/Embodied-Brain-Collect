"""run_session_video —— 子集解析/config 校验/抽取配平/台账(不启动录制)。"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_session_video.py"

_STIM_OVER = {"video_rate": {
    "instr1_text": "请观看视频的第一帧，想象自己{task}。",
    "instr2_text": "请观看遥操视频，评估动作执行质量",
}}


def _load():
    spec = importlib.util.spec_from_file_location("run_session_video", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod():
    return _load()


def _make_subset(tmp_path: Path, name: str = "盖笔盖", n_fail: int = 0,
                 n_success: int = 0, config: dict | None = None,
                 prefix_success: str = "success_") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_fail):
        (d / f"fail_episode_{i:06d}.mp4").write_bytes(b"x")
    for i in range(n_success):
        (d / f"{prefix_success}episode_{i:06d}.mp4").write_bytes(b"x")
    cfg = {"task": "双手分别拿起笔和笔盖，盖上笔盖后把笔放入笔筒",
           "scene": "书桌", "name": name}
    cfg.update(config or {})
    (d / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in cfg.items()), encoding="utf-8")
    return d


# =============================================================================
# 子集解析与 config 校验
# =============================================================================

def test_list_and_resolve_subset(mod, tmp_path):
    assert mod.list_subsets(tmp_path) == []            # 空根
    (tmp_path / "empty").mkdir()                       # 没有 mp4 的目录不算
    a = _make_subset(tmp_path, "task_a", n_fail=1)
    b = _make_subset(tmp_path, "task_b", n_fail=1)
    assert [p.name for p in mod.list_subsets(tmp_path)] == ["task_a", "task_b"]
    assert mod.resolve_subset("task_b", tmp_path) == b
    assert mod.resolve_subset(str(b), tmp_path) == b   # 也认路径
    with pytest.raises(SystemExit):
        mod.resolve_subset("不存在", tmp_path)
    with pytest.raises(SystemExit):
        mod.resolve_subset(None, tmp_path)             # 多个子集必须交互


def test_resolve_subset_single_is_automatic(mod, tmp_path):
    d = _make_subset(tmp_path, "唯一任务", n_fail=1)
    assert mod.resolve_subset(None, tmp_path) == d


def test_load_subset_config_requires_task(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲")
    assert mod.load_subset_config(d)["task"].startswith("双手")

    bad = tmp_path / "坏子集"
    bad.mkdir()
    (bad / "config.yaml").write_text("scene: 书桌\n", encoding="utf-8")
    with pytest.raises(SystemExit):                    # 缺 task
        mod.load_subset_config(bad)

    none_cfg = tmp_path / "无配置"
    none_cfg.mkdir()
    with pytest.raises(SystemExit):                    # 缺 config.yaml
        mod.load_subset_config(none_cfg)


def test_build_instr_template_and_override(mod, monkeypatch):
    monkeypatch.setattr(mod, "load_stim", lambda: _STIM_OVER)
    instr1, instr2 = mod.build_instr(
        {"task": "盖上笔盖", "instr2_text": "自定义指令2"})
    assert instr1 == "请观看视频的第一帧，想象自己盖上笔盖。"
    assert instr2 == "自定义指令2"

    instr1, instr2 = mod.build_instr(
        {"task": "盖上笔盖", "instr1_text": "完整自定义指令"})
    assert instr1 == "完整自定义指令"                   # 整体覆盖
    assert instr2 is None

    monkeypatch.setattr(mod, "load_stim",
                        lambda: {"video_rate": {"instr1_text": "没有占位符"}})
    with pytest.raises(SystemExit):                    # 模板缺 {task}
        mod.build_instr({"task": "x"})


# =============================================================================
# 素材池 / 抽取 / 台账
# =============================================================================

def test_scan_pool_prefix_routing(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=3, n_success=2)
    (d / "unknown_clip.mp4").write_bytes(b"x")
    pool = mod.scan_pool(d)
    assert len(pool["fail"]) == 3
    assert len(pool["success"]) == 2


def test_scan_pool_accepts_sucess_typo(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_success=2, prefix_success="sucess_")
    assert len(mod.scan_pool(d)["success"]) == 2


def test_compose_balanced_and_unique(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=10, n_success=10)
    pool = mod.scan_pool(d)
    rng = random.Random(7)
    for n_trials in (2, 5, 6):
        trials = mod.compose_trials(pool, {}, n_trials, rng)
        stems = [t["video"].stem for t in trials]
        assert len(stems) == len(set(stems)) == n_trials   # 无重复、数量对
        n_fail = sum(1 for t in trials if t["label"] == "fail")
        n_success = n_trials - n_fail
        assert abs(n_fail - n_success) <= 1 and n_fail >= n_success


def test_compose_prefers_least_used(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=4, n_success=4)
    pool = mod.scan_pool(d)
    ledger = {f"success_episode_{i:06d}": {"uses": int(i != 0)}
              for i in range(4)}   # episode_0 未用过(0 次),其余 1 次
    trials = mod.compose_trials(pool, ledger, 4, random.Random(1))
    used_success = [t["video"].stem for t in trials
                    if t["label"] == "success"]
    assert "success_episode_000000" in used_success   # uses=0 最少 → 优先


def test_compose_borrows_when_one_pool_short(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=5, n_success=1)
    pool = mod.scan_pool(d)
    trials = mod.compose_trials(pool, {}, 6, random.Random(3))
    assert len(trials) == 6
    assert sum(1 for t in trials if t["label"] == "success") == 1


def test_compose_empty_pool_raises(mod):
    with pytest.raises(SystemExit):
        mod.compose_trials({"fail": [], "success": []}, {}, 6,
                           random.Random(0))


def test_ledger_roundtrip_per_subset(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=1)
    mod.save_ledger(d, {"success_a": {"uses": 2, "last_session": "s1"},
                        "fail_b": {"uses": 0, "last_session": ""}})
    assert (d / "used.yaml").is_file()
    ledger = mod.load_ledger(d)
    assert ledger["success_a"]["uses"] == 2
    assert ledger["success_a"]["last_session"] == "s1"
    assert ledger["fail_b"]["uses"] == 0
    # 各子集台账互不影响
    other = _make_subset(tmp_path, "任务乙")
    assert mod.load_ledger(other) == {}


def test_load_ledger_corrupt_file_is_empty(mod, tmp_path):
    d = _make_subset(tmp_path, "任务甲", n_fail=1)
    (d / "used.yaml").write_text("[broken: yaml::", encoding="utf-8")
    assert mod.load_ledger(d) == {}


# =============================================================================
# 主流程(dry-run / fail-fast)
# =============================================================================

def test_cli_rejects_too_many_trials(mod, monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["run_session_video.py", "--n-trials", "999"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_aborts_on_undecodable_pool(mod, monkeypatch, tmp_path):
    _make_subset(tmp_path, "坏素材", n_fail=1, n_success=1)  # 伪 mp4 过不了解码
    rc = mod.main(["--video-dir", str(tmp_path), "--dry-run"])
    assert rc == 2


def test_dry_run_prints_and_skips_launch(mod, monkeypatch, tmp_path, capsys):
    d = _make_subset(tmp_path, "任务甲", n_fail=3, n_success=3)
    real_scan = mod.scan_pool
    monkeypatch.setattr(mod, "scan_pool", lambda _: real_scan(d))
    monkeypatch.setattr(mod, "load_ledger", lambda _: {})
    monkeypatch.setattr(mod, "verify_decodable", lambda trials: [])
    monkeypatch.setattr(mod, "load_stim", lambda: _STIM_OVER)
    called = {"queue": False}
    monkeypatch.setattr(mod, "run_queue", lambda *a, **k:
                        called.__setitem__("queue", True) or 0)
    rc = mod.main(["--dry-run", "--video-dir", str(tmp_path)])
    assert rc == 0 and not called["queue"]
    out = capsys.readouterr().out
    assert "任务甲" in out and "dry-run" in out


# =============================================================================
# 逐条录制:stim 产物进录制目录 / 视频拷贝 / 按条台账
# =============================================================================

def test_hooks_stim_cmd_targets_run_dir(mod, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(mod, "load_stim", lambda: _STIM_OVER)
    d = _make_subset(tmp_path, "盖笔盖", n_fail=1, n_success=1)
    cfg = mod.load_subset_config(d)
    instr1, instr2 = mod.build_instr(cfg)
    hooks = mod._hooks(d, cfg, {}, instr1, instr2)
    video = sorted(d.glob("*.mp4"))[0]
    job = {"trial": 1, "total": 2, "video": video, "label": "fail"}
    run_dir = tmp_path / "2026-01-01-00-00-00"
    cmd = hooks.stim_cmd(job, "video_rate", run_dir)
    assert cmd[cmd.index("--result-dir") + 1] == str(run_dir)
    items = json.loads(cmd[cmd.index("--trials-json") + 1])
    assert len(items) == 1                       # 每 trial 一场,单条素材
    assert items[0]["video"] == str(video.resolve())
    assert "{task}" not in items[0]["instr1_text"]  # 占位符已填


def test_hooks_on_result_copy_and_ledger(mod, tmp_path):
    d = _make_subset(tmp_path, "盖笔盖", n_fail=1, n_success=1)
    cfg = mod.load_subset_config(d)
    hooks = mod._hooks(d, cfg, {}, "指令", None)
    video = sorted(d.glob("fail_*.mp4"))[0]
    job = {"trial": 1, "total": 1, "video": video, "label": "fail"}
    run_dir = tmp_path / "2026-01-01-00-00-00"
    run_dir.mkdir()

    hooks.on_result(job, run_dir, "kept", "failed", 2)   # 没录完:不拷不记
    assert not (run_dir / video.name).exists()
    assert mod.load_ledger(d) == {}

    hooks.on_result(job, run_dir, "kept", "success", 0)  # 录完:拷贝 + 记账
    assert (run_dir / video.name).exists()               # 视频拷进录制目录
    ledger = mod.load_ledger(d)
    assert ledger[video.stem]["uses"] == 1
    assert ledger[video.stem]["last_session"] == str(run_dir)

    hooks.on_result(job, run_dir, "kept", "error", 0)    # QC 有错:拷了但不记
    assert mod.load_ledger(d)[video.stem]["uses"] == 1   # 计数不变
