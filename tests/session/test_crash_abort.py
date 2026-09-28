"""recorder 录制中异常退出 → 立即收摊、跳过 QC、直接进选择。

launch() 在录制阶段发现任一子进程退场就提前结束(其余 slot 正常收尾
落盘、stim 终止),rc 记 1 并带 runtime_errors;record_one 据此跳过
run_qc 直接回 run_queue 的 n/r/f/q 选择。这里用真实 spawn 子进程验证
launcher 侧,用桩替换验证编排侧(QC 跳过 / auto_keep 不冒充 success)。
"""

import time
import types

import numpy as np
import pytest

from embodied_brain_collect.recorders.emg import (
    DummyEmgRecorder, EmgRecorderConfig)
from embodied_brain_collect.session.launcher import LaunchResult, launch


# ============================================================================= 
# launcher:崩溃 slot 触发提前收摊,幸存 slot 照常落盘
# =============================================================================

class _CrashAfterCommit(DummyEmgRecorder):
    """commit 后 ~0.5s 在录制循环里崩掉 —— 模拟录制中异常退出。

    ``data_before_stim = False`` 跳过"数据在流动"确认(marker 同款豁免),
    否则它活不到录制阶段。
    """

    data_before_stim = False

    def _record(self):
        self._commit_evt.wait(timeout=15.0)   # 等 launcher 的 commit 广播
        time.sleep(0.5)                       # 给健康 slot 留出录上数据的时间
        raise RuntimeError("simulated mid-recording crash")


def test_launch_aborts_and_saves_survivors_on_recorder_crash(tmp_path):
    from loguru import logger as _loguru

    healthy_dir = tmp_path / "emg"
    healthy = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(healthy_dir),
                                                 hz=500))
    crasher = _CrashAfterCommit(EmgRecorderConfig(session_dir=str(tmp_path / "boom"),
                                                  hz=500))
    try:
        t0 = time.time()
        res = launch({"emg": healthy, "boom": crasher}, duration=0.0,
                     confirm_timeout=15.0)
        elapsed = time.time() - t0
    finally:
        for rec in (healthy, crasher):
            if getattr(rec, "_log_sink_id", None) is not None:
                _loguru.remove(rec._log_sink_id)
                rec._log_sink_id = None

    assert int(res) == 1                          # 提前收摊,rc 记 1
    assert res.open_failures == {}
    assert "boom" in res.runtime_errors           # 崩溃 slot 记运行期错误
    assert elapsed < 60                           # 不会被 duration/stim 兜底拖住
    # 幸存 slot 走了正常收尾:npz 已落盘且是 commit 后的数据
    z = np.load(healthy_dir / "emg" / "emg.npz")
    assert z["emg_timestamps"].size > 0


# =============================================================================
# record_one:runtime_error 跳过 run_qc,干净录制照常跑
# =============================================================================

def _record_one_args(skip_qc=False):
    return types.SimpleNamespace(dummy=True, duration=0.0, recorders=None,
                                 skip_qc=skip_qc)


def _record_one_hooks():
    from embodied_brain_collect.session.run_base import ModeHooks
    return ModeHooks(label="t",
                     preroll=lambda job, run_dir, args: True,
                     meta_kwargs=lambda job: {},
                     stim_cmd=lambda job, stim, run_dir: [],
                     job_id=lambda job: "j1")


def test_record_one_skips_qc_on_runtime_error(tmp_path, monkeypatch):
    from embodied_brain_collect.session import run_base

    qc_calls = []
    monkeypatch.setattr(run_base, "get_dummy_recorders", lambda **k: {})
    monkeypatch.setattr(run_base, "launch", lambda *a, **k: LaunchResult(
        1, runtime_errors={"eeg": "boom"}))
    monkeypatch.setattr(run_base, "run_qc", lambda d: qc_calls.append(d) or 0)

    run_dir, rc = run_base.record_one(tmp_path, {}, stim="simple",
                                      args=_record_one_args(), collect={},
                                      hooks=_record_one_hooks())
    assert int(rc) == 1
    assert qc_calls == []                         # 崩溃录制不跑 check


def test_record_one_runs_qc_when_clean(tmp_path, monkeypatch):
    from embodied_brain_collect.session import run_base

    qc_calls = []
    monkeypatch.setattr(run_base, "get_dummy_recorders", lambda **k: {})
    monkeypatch.setattr(run_base, "launch", lambda *a, **k: LaunchResult(0))
    monkeypatch.setattr(run_base, "run_qc", lambda d: qc_calls.append(d) or 0)

    run_base.record_one(tmp_path, {}, stim="simple", args=_record_one_args(),
                        collect={}, hooks=_record_one_hooks())
    assert len(qc_calls) == 1                     # 正常录制仍跑 check


# =============================================================================
# run_queue:auto_keep 无人值守时,崩溃录制不冒充 success
# =============================================================================

def _run_queue_args(auto_keep):
    return types.SimpleNamespace(auto_keep=auto_keep, pack_episode=False)


def _patch_config(monkeypatch):
    monkeypatch.setattr(
        "embodied_brain_collect.session.config.load_recorders", lambda: {})
    monkeypatch.setattr(
        "embodied_brain_collect.session.config.load_session_run", lambda: {})


def test_auto_keep_marks_crashed_run_failed(tmp_path, monkeypatch):
    from embodied_brain_collect.session import run_base

    _patch_config(monkeypatch)
    seen = []

    def fake_record_one(session_root, job, **kw):
        rd = session_root / "2026-01-01-00-00-01"
        rd.mkdir(parents=True, exist_ok=True)
        return rd, LaunchResult(1, runtime_errors={"eeg": "boom"})

    def on_result(job, run_dir, choice, status, rc):
        seen.append((choice, status))

    hooks = _record_one_hooks()
    hooks.on_result = on_result
    monkeypatch.setattr(run_base, "record_one", fake_record_one)
    rc = run_base.run_queue(tmp_path, [{}], stim="simple",
                            args=_run_queue_args(auto_keep=True),
                            collect={}, hooks=hooks, seed=1)
    assert rc == 0
    assert seen == [("kept", "failed")]           # 没有人确认,不给 success


def test_interactive_keep_respects_operator_choice(tmp_path, monkeypatch):
    """交互模式下 n = 操作员看过崩溃提示后显式保留,尊重其选择。"""
    from embodied_brain_collect.session import run_base

    _patch_config(monkeypatch)
    seen = []
    monkeypatch.setattr(run_base, "record_one", lambda *a, **k: (
        (lambda rd: (rd.mkdir(parents=True, exist_ok=True), rd)[1])(
            tmp_path / "2026-01-01-00-00-01"),
        LaunchResult(1, runtime_errors={"eeg": "boom"})))
    monkeypatch.setattr(run_base, "ask_next", lambda qc_error=False: "n")

    hooks = _record_one_hooks()
    hooks.on_result = lambda job, rd, choice, status, rc: seen.append(status)
    run_base.run_queue(tmp_path, [{}], stim="simple",
                       args=_run_queue_args(auto_keep=False),
                       collect={}, hooks=hooks, seed=1)
    assert seen == ["success"]
