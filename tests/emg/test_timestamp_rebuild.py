"""Unit tests for the EMG timestamp rebuild, on synthetic sessions.

The recorder stamps every frame of one ``Serial.read()`` with that read's
arrival time, and ``rebuild()`` recovers per-frame times from the shared
8-bit sequence number.  These tests pin the frame-budget guard: it must
tolerate frames that legitimately fall outside the EMG-anchored span (the
session's first transmitted frame being an IMU frame, and IMU frames after
the last EMG one) while still refusing sequences whose wrap lost frames.
"""

import numpy as np
import pytest

from embodied_brain_collect.recorders.emg.timestamp_rebuild import rebuild

FRAME_S = 1 / 2000.0        # combined EMG+IMU wire rate
BATCH_FRAMES = 200          # ~10 reads/s: ~200 frames share one arrival stamp


def _synth_session(n_frames=20000, seed=0, first_frame="emg",
                   last_frame="emg", emg_frac=0.9):
    """Frames at a fixed device rate, delivered in arrival-stamped batches."""
    rng = np.random.default_rng(seed)
    kinds = rng.random(n_frames) < emg_frac          # True = EMG frame
    kinds[0] = first_frame == "emg"
    kinds[-1] = last_frame == "emg"

    k = np.arange(n_frames)
    emg_k = k[kinds]
    imu_k = k[~kinds]

    batch = rng.normal(0, 1e-4, size=n_frames // BATCH_FRAMES + 1)
    t0 = 1_000_000.0                                 # unix-ish epoch
    arrival = t0 + (k // BATCH_FRAMES) * BATCH_FRAMES * FRAME_S \
        + batch[k // BATCH_FRAMES]

    return (arrival[emg_k], (emg_k % 256).astype(np.int64),
            arrival[imu_k], (imu_k % 256).astype(np.int64))


def test_clean_session_rebuilds_strictly_increasing():
    e_ts, e_sn, i_ts, i_sn = _synth_session()
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note
    assert np.all(np.diff(r.emg_timestamps) > 0)
    assert (np.diff(r.emg_timestamps) == 0).mean() == 0.0
    # 期望速率按名义 EMG 占比;随机分配有 ~0.2% 的二项波动,给 1% 容差
    assert r.emg_rate_hz == pytest.approx((1 / FRAME_S) * 0.9, rel=1e-2)


def test_leading_imu_frame_does_not_trip_the_budget():
    """Regression: when the session's very first transmitted frame is an IMU
    frame (k = -1 relative to the first EMG frame), it counts toward the
    received total but sat outside the EMG-anchored span — the old budget
    read that as a one-frame deficit and failed a perfectly clean recording
    (real case: 2026-09-10 session, emg_right rebuilt while emg_left did
    not, on the luck of which frame type opened the session)."""
    e_ts, e_sn, i_ts, i_sn = _synth_session(first_frame="imu")
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note
    assert np.all(np.diff(r.emg_timestamps) > 0)


def test_trailing_imu_frame_does_not_trip_the_budget():
    """Symmetric edge: an IMU frame received after the last EMG frame."""
    e_ts, e_sn, i_ts, i_sn = _synth_session(last_frame="imu")
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note


def test_swallowed_wrap_is_refused():
    """A 256-slot loss makes two consecutive received SNs equal — the unwrap
    cannot see the revolution and the rebuilt series would fold.  The
    fail-safe must keep the arrival timestamps instead."""
    e_ts, e_sn, i_ts, i_sn = _synth_session()
    e_bad = e_sn.copy()
    e_bad[100] = e_bad[99]              # zero step: a whole revolution vanished
    r = rebuild(e_ts, e_bad, i_ts, i_sn)
    assert not r.ok


def test_too_few_batches_is_refused():
    e_ts, e_sn, i_ts, i_sn = _synth_session(n_frames=600)   # ~3 batches
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert not r.ok


def test_short_session_with_five_batches_now_fits():
    """回归:最低锚点 8 → 5。短会话(5 个读取批次)以前被直接拒拟合;
    Theil–Sen 初估让 5 个锚点也能稳定估计斜率。"""
    e_ts, e_sn, i_ts, i_sn = _synth_session(n_frames=1000)   # ~5 batches
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note
    assert np.all(np.diff(r.emg_timestamps) > 0)


def test_arrival_stalls_do_not_refuse_or_tilt():
    """~8% 的读取遇到调度卡顿(整批到达晚 1.2s):普通最小二乘被这些锚点
    倾斜、单轮 3σ 的 σ 又被离群撑大 —— 老实现容易因此拒拟合或产出
    带倾斜的时间线。Theil–Sen 初估 + 迭代剔离群应当照常拟合。"""
    e_ts, e_sn, i_ts, i_sn = _synth_session(n_frames=20000, seed=3)
    stall_vals = np.unique(e_ts)[::12]
    e_ts = e_ts + np.isin(e_ts, stall_vals) * 1.2
    stall_vals_i = np.unique(i_ts)[::12]
    i_ts = i_ts + np.isin(i_ts, stall_vals_i) * 1.2
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note
    assert np.all(np.diff(r.emg_timestamps) > 0)
    assert r.emg_rate_hz == pytest.approx((1 / FRAME_S) * 0.9, rel=1e-2)


def test_second_outlier_round_indexes_kept_anchors():
    """回归:剔离群第 2 轮曾拿剔剩的短掩码去索引全量锚点数组,长度错位
    抛 IndexError,把本可拟合的会话整批拒掉(2026-09-15 批量 rebuild,
    session-night 705 个文件因此 REFUSED)。两级离群让第 1、2 轮各剔一批,
    正是当时的崩溃路径;掩码必须只作用于保留集。"""
    e_ts, e_sn, i_ts, i_sn = _synth_session(n_frames=20000, seed=1)
    big_vals = np.unique(e_ts)[10::25]        # 第 1 轮剔除:20 s 级卡顿
    mid_vals = np.unique(e_ts)[5::50]         # 第 2 轮剔除:1 ms 级卡顿
    e_ts = e_ts + np.isin(e_ts, big_vals) * 20.0
    e_ts = e_ts + np.isin(e_ts, mid_vals) * 1e-3
    r = rebuild(e_ts, e_sn, i_ts, i_sn)
    assert r.ok, r.note
    assert np.all(np.diff(r.emg_timestamps) > 0)
    assert r.emg_rate_hz == pytest.approx((1 / FRAME_S) * 0.9, rel=1e-2)
