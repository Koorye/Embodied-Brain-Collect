"""两段式启动的 commit 语义:预热数据丢弃、会话时钟锚定 commit。

launcher 的新流程:open 通过后立即开录(预热),全部 slot 确认数据在流动
后才广播 commit —— commit 丢弃预热数据并把会话时钟归零,duration 与落盘
内容都从 commit 起算。这里用真实录制循环(DummyEmg)在进程内验证这套机制。
"""

import threading
import time

import numpy as np

from embodied_brain_collect.recorders.emg import (DummyEmgRecorder,
                                                  EmgRecorderConfig)
from embodied_brain_collect.recorders.marker import UdpMarkerRecorder


def _run_committed(session_dir, pre_s: float, post_s: float):
    """录 pre_s 秒预热 → commit → 再录 post_s 秒 → 停止。返回 (rec, commit 时刻)。"""
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(session_dir),
                                             hz=500))
    t = threading.Thread(target=rec.run, daemon=True)
    t.start()
    time.sleep(pre_s)
    assert rec._count_samples() > 0, "预热期应有数据进缓冲"
    commit_at = time.time()
    rec.request_commit()
    time.sleep(post_s)
    rec.stop_event.set()
    t.join(timeout=15)
    assert not t.is_alive()
    return rec, commit_at


def test_commit_discards_preroll_and_reanchors_clock(tmp_path):
    rec, commit_at = _run_committed(tmp_path, pre_s=1.5, post_s=2.0)
    z = np.load(tmp_path / "emg" / "emg.npz")
    t = z["emg_timestamps"]
    assert t.size > 0
    # 第一帧不早于 commit:预热期的 1.5s 数据已被丢弃
    assert t[0] >= commit_at - 0.05
    # 会话时长锚定 commit(≈2.0s),而不是 open(≈3.5s)
    assert (t[-1] - t[0]) < 2.6


def test_without_commit_duration_anchors_on_record_start(tmp_path):
    """独立 run()(不 commit):行为与旧流程一致,时长从开录起算。"""
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(tmp_path),
                                             duration=1.0, hz=500))
    assert rec.run() == 0
    z = np.load(tmp_path / "emg" / "emg.npz")
    t = z["emg_timestamps"]
    assert (t[-1] - t[0]) < 1.6


def test_discard_recording_skips_save(tmp_path):
    """abort 路径:录了但 discard,teardown 不写 npz。"""
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(tmp_path),
                                             hz=500))
    t = threading.Thread(target=rec.run, daemon=True)
    t.start()
    time.sleep(0.5)
    rec.discard_recording()
    rec.stop_event.set()
    t.join(timeout=15)
    assert not t.is_alive()
    assert not (tmp_path / "emg" / "emg.npz").exists()


def test_marker_is_stim_driven():
    """marker 由 stim 驱动,stim 前天然无数据 —— 确认阶段必须豁免。"""
    assert UdpMarkerRecorder.data_before_stim is False
    assert DummyEmgRecorder.data_before_stim is True
