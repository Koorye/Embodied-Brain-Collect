"""BrainCo 软件同步: payload 解析 + 包时钟 ↔ marker 映射(无硬件)。"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from embodied_brain_collect.recorders.eeg.base_eeg_recorder import _fit_eeg_to_pc
from embodied_brain_collect.recorders.eeg.brainco_eeg_recorder import (
    BRAINCO_N_CHANNELS, coerce_eeg_payload, marker_times_to_latency,
    pull_eeg_buffer, EEG_BUFFER_TAKE,
)
from embodied_brain_collect.recorders.eeg import (BrainCoEegRecorder,
                                                  BraincoEegRecorderConfig)
from embodied_brain_collect.stim.marker_sender import MarkerSender


def test_factory_registers_brainco_kind():
    from embodied_brain_collect.recorders.factory import REGISTRY, resolve_kind
    assert "brainco_eeg" in REGISTRY
    cls, cfg_cls = resolve_kind("brainco_eeg")
    assert cls is BrainCoEegRecorder
    assert cfg_cls is BraincoEegRecorderConfig
    rec = BrainCoEegRecorder(BraincoEegRecorderConfig(session_dir="",
                                                      sample_rate=500,
                                                      gain=12))
    assert rec.config.sample_rate == 500
    assert rec.config.gain == 12
    assert rec._has_trigger_channel is False


def test_coerce_samples_by_channel_major():
    raw = np.arange(32 * 10, dtype=np.float32).reshape(32, 10)
    block, extra = coerce_eeg_payload(raw)
    assert block.shape == (10, 32)
    assert extra == {}


def test_coerce_33ch_batch_and_flat():
    raw = np.arange(33 * 8, dtype=np.float32).reshape(8, 33)
    block, _ = coerce_eeg_payload(raw)
    assert block.shape == (8, 33)

    flat = np.arange(33 * 5, dtype=np.float32)
    block, _ = coerce_eeg_payload(flat)
    assert block.shape == (5, 33)

    ch_major = np.arange(33 * 6, dtype=np.float32).reshape(33, 6)
    block, _ = coerce_eeg_payload(ch_major)
    assert block.shape == (6, 33)


def test_pull_eeg_buffer_takes_batch_not_bool_true():
    class Fake:
        def __init__(self):
            self.args = None

        def get_eeg_buffer(self, take, clean):
            self.args = (take, clean)
            return [[float(i)] * 33 for i in range(10)]

    sdk = Fake()
    buf = pull_eeg_buffer(sdk)
    assert sdk.args is not None
    assert sdk.args[0] == EEG_BUFFER_TAKE
    assert sdk.args[0] > 1
    assert sdk.args[1] is True
    block, _ = coerce_eeg_payload(buf)
    assert block.shape == (10, 33)


def test_coerce_flat_and_dict_with_seq():
    flat = np.arange(32 * 4, dtype=np.float32)
    block, extra = coerce_eeg_payload({
        "data": flat, "seq": 7, "timestamp": 1.25, "trigger": 0,
    })
    assert block.shape == (4, 32)
    assert extra["seq"] == 7
    assert extra["timestamp"] == 1.25


def test_coerce_list_of_channel_lists():
    chans = [[float(c)] * 5 for c in range(BRAINCO_N_CHANNELS)]
    block, _ = coerce_eeg_payload(chans)
    assert block.shape == (5, BRAINCO_N_CHANNELS)
    assert block[0, 3] == 3.0


def test_packet_clock_fit_and_marker_mapping():
    """包到达时刻带 1ms 抖动时,拟合应找回 slope≈1,并把 marker 映射回样本。"""
    fs = 250.0
    rng = np.random.default_rng(0)
    n_pkt = 80
    samples_per_pkt = 10
    t0 = 1_780_000_000.0
    eeg_t, pc_t = [], []
    for i in range(n_pkt):
        last = (i + 1) * samples_per_pkt - 1
        eeg_t.append(last / fs)
        pc_t.append(t0 + last / fs + rng.normal(0, 0.001))
    fit = _fit_eeg_to_pc(np.asarray(eeg_t), np.asarray(pc_t))
    assert fit["fitted"]
    assert fit["slope_pc_per_eeg"] == pytest.approx(1.0, abs=5e-3)
    assert fit["resid_rms_ms"] < 5.0

    # 刺激时刻落在若干包中间
    want_samples = np.asarray([50, 200, 400, 600], dtype=np.int64)
    t_sent = t0 + want_samples / fs
    lat = marker_times_to_latency(t_sent, fit, fs)
    assert np.max(np.abs(lat - want_samples)) <= 2


def test_packet_clock_fit_is_fast_on_dense_stream():
    """BrainCo 包时钟一场可达 1 万+点:向量化拟合必须在秒级内完成。"""
    import time
    fs = 250.0
    t0 = 1_780_000_000.0
    idx = np.arange(1, 10_001)
    eeg_t = idx / fs
    pc_t = t0 + idx / fs + 1e-4
    t_start = time.perf_counter()
    fit = _fit_eeg_to_pc(eeg_t, pc_t)
    assert time.perf_counter() - t_start < 5.0
    assert fit["fitted"]


def test_marker_sender_copies_to_sync_port():
    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a.bind(("127.0.0.1", 0))
    b.bind(("127.0.0.1", 0))
    a.settimeout(1.0)
    b.settimeout(1.0)
    pa, pb = a.getsockname()[1], b.getsockname()[1]
    sender = MarkerSender(
        enable_serial=False, enable_udp=True,
        udp_host="127.0.0.1", udp_port=pa,
        sync_udp_host="127.0.0.1", sync_udp_port=pb,
    )
    try:
        sender.mark(17, "FIX_ON")
        d1, d2 = a.recv(4096), b.recv(4096)
    finally:
        sender.close()
        a.close()
        b.close()
    assert d1 == d2
    assert b"code=17" in d1
    assert b"t_sent_pc=" in d1


def test_marker_sender_ttl_false_skips_serial():
    """ttl=False 只走 UDP —— 串口未开时本就无副作用,这里锁住参数存在。"""
    sender = MarkerSender(enable_serial=False, enable_udp=False)
    try:
        t = sender.mark(17, "FIX_ON", ttl=False)
        assert t > 0
    finally:
        sender.close()


# =============================================================================
# _fit_eeg_to_pc 向量化等价(任务:确认新拟合与朴素 O(n²) 参考实现一致)
# =============================================================================

def _naive_fit(eeg_t_s, marker_t_pc):
    """v1.3.0 的朴素实现:纯 Python 双循环估斜率,再做内点/polyfit。"""
    from embodied_brain_collect.recorders.eeg.base_eeg_recorder import (
        RESID_MAX_S, SLOPE_COARSE, SLOPE_HI, SLOPE_LO)
    x = np.asarray(eeg_t_s, dtype=np.float64)
    y = np.asarray(marker_t_pc, dtype=np.float64)
    x0, y0 = float(x[0]), float(y[0])
    xr, yr = x - x0, y - y0
    slopes = []
    for i in range(xr.size):
        for j in range(i + 1, xr.size):
            dx = xr[j] - xr[i]
            if dx < 0.5:
                continue
            s = (yr[j] - yr[i]) / dx
            if SLOPE_COARSE[0] <= s <= SLOPE_COARSE[1]:
                slopes.append(s)
    a0 = float(np.median(slopes))
    b0 = float(np.median(yr - a0 * xr))
    inliers = np.abs(yr - (a0 * xr + b0)) <= RESID_MAX_S
    a, b = np.polyfit(xr[inliers], yr[inliers], 1)
    assert SLOPE_LO <= a <= SLOPE_HI
    return a, b


def test_fit_vectorized_matches_naive_reference():
    """向量化版(抽点估斜率)必须与朴素 O(n²) 版在同一内点集上给出一致结果:
    含离群 marker 的规整时钟、抖动时钟、以及 BrainCo 量级的密包时钟。"""
    t0 = 1_780_000_000.0
    cases = []
    # 1) 规整 + 单个延迟离群
    idx = np.arange(1, 201)
    pc = t0 + idx + 0.0
    pc[100] += 3.0                                  # 一个 gross outlier
    cases.append((idx.astype(float), pc))
    # 2) 1ms 抖动
    rng = np.random.default_rng(42)
    cases.append((idx.astype(float), t0 + idx + rng.normal(0, 1e-3, idx.size)))
    # 3) 密包(1 万点,BrainCo 量级)+ 斜率略偏(晶振漂移)
    big = np.arange(1, 10_001, dtype=float)
    cases.append((big, t0 + big * 1.00003))

    for eeg_t, pc_t in cases:
        fit = _fit_eeg_to_pc(eeg_t, pc_t)
        assert fit["fitted"]
        a_ref, b_ref = _naive_fit(eeg_t, pc_t)
        assert fit["slope_pc_per_eeg"] == pytest.approx(a_ref, abs=1e-9)
        assert fit["intercept_s_at_first_marker"] == pytest.approx(
            b_ref, abs=1e-6)
