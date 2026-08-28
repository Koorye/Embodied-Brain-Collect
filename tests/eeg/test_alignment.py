"""EEG<->PC 时钟对齐的纯函数单测(无硬件)。"""

import numpy as np
import pytest

from embodied_brain_collect.recorders.eeg.base_eeg_recorder import (
    _align_code_sequences, _fit_eeg_to_pc)

# 一个典型 run:12 个事件,码唯一。EEG 侧存样本号,marker 侧存 PC 时刻。
CODES = [241, 160, 17, 132, 49, 18, 50, 97, 98, 81, 82, 242]


def make_events(drop_eeg=(), drop_marker=(), delay_ms=0.0):
    """模拟两侧事件流。事件按 1/fs 间隔均匀分布,marker 侧可加恒定延迟。"""
    fs = 1000.0
    eeg_code, eeg_lat = [], []
    for i, c in enumerate(CODES):
        if c in drop_eeg:
            continue
        eeg_code.append(c)
        eeg_lat.append(500000 + i * 2000)
    m_code, m_t = [], []
    t0 = 1_787_000_000.0
    for i, c in enumerate(CODES):
        if c in drop_marker:
            continue
        m_code.append(c)
        m_t.append(t0 + i * 2.0 + delay_ms / 1000.0)
    return (np.asarray(eeg_code), np.asarray(eeg_lat, dtype=np.int64),
            np.asarray(m_code), np.asarray(m_t))


# =============================================================================
# _align_code_sequences
# =============================================================================

def test_clean_pairing_matches_all_events():
    pair = _align_code_sequences(*make_events())
    assert pair is not None
    eeg_t, pc_t = pair
    assert len(eeg_t) == len(CODES)
    # 每侧一个事件: 样本号 500000+2k*i 对应秒,斜率应≈1
    fit = _fit_eeg_to_pc(eeg_t / 1000.0, pc_t)
    assert fit["fitted"] and fit["slope_pc_per_eeg"] == pytest.approx(1.0, abs=1e-4)


def test_unilateral_drops_are_tolerated():
    """一侧丢包(码缺失)不该毁掉整个对齐 —— 幸存码仍能配对。"""
    pair = _align_code_sequences(*make_events(drop_eeg=(17, 50)))
    assert pair is not None and len(pair[0]) == 10
    pair = _align_code_sequences(*make_events(drop_marker=(81, 82, 242)))
    assert pair is not None and len(pair[0]) == 9


def test_order_conflict_is_rejected():
    """码集相同但两侧顺序冲突 —— 拒绝配对而不是硬拟合。"""
    eeg_code, eeg_lat, m_code, m_t = make_events()
    m_code = m_code[::-1]          # marker 侧完全倒序
    assert _align_code_sequences(eeg_code, eeg_lat, m_code, m_t) is None


def test_duplicate_codes_are_rejected():
    """重码让查表配对失去唯一性 — 拒绝而不是猜。"""
    eeg_code, eeg_lat, m_code, m_t = make_events()
    eeg_code = np.asarray([241, 241, 160, 17])     # 重复的 RUN_START
    eeg_lat = np.asarray([0, 100, 200, 300], dtype=np.int64)
    assert _align_code_sequences(eeg_code, eeg_lat, m_code, m_t) is None


def test_too_few_shared_codes_is_rejected():
    eeg_code, eeg_lat, m_code, m_t = make_events(drop_eeg=CODES[1:])
    assert _align_code_sequences(eeg_code, eeg_lat, m_code, m_t) is None


# =============================================================================
# _fit_eeg_to_pc
# =============================================================================

def test_fit_recovers_slope_and_rejects_nothing_clean():
    pair = _align_code_sequences(*make_events())
    fit = _fit_eeg_to_pc(pair[0] / 1000.0, pair[1])
    assert fit["fitted"]
    assert fit["slope_pc_per_eeg"] == pytest.approx(1.0, abs=1e-4)
    assert fit["resid_max_ms"] < 5.0
    assert fit["n_inliers"] == 12 and fit["n_outliers"] == 0


def test_fit_rejects_a_grossly_delayed_marker_as_outlier():
    """单个 marker 延迟 2 秒:内点重拟合后斜率仍正确,坏点被剔除。"""
    eeg_code, eeg_lat, m_code, m_t = make_events()
    m_t = m_t.copy()
    m_t[4] += 2.0               # INSTR_ON 迟到 2 s
    pair = _align_code_sequences(eeg_code, eeg_lat, m_code, m_t)
    fit = _fit_eeg_to_pc(pair[0] / 1000.0, pair[1])
    assert fit["fitted"]
    assert fit["slope_pc_per_eeg"] == pytest.approx(1.0, abs=1e-4)
    assert fit["n_outliers"] >= 1


def test_fit_fails_on_missing_samples():
    """EEG 缺一半样本 → 斜率≈2 → 必须不拟合(旧版会硬给结果)。"""
    eeg_code, eeg_lat, m_code, m_t = make_events()
    eeg_lat = eeg_lat // 2      # 样本号间距减半,模拟丢失
    pair = _align_code_sequences(eeg_code, eeg_lat, m_code, m_t)
    fit = _fit_eeg_to_pc(pair[0] / 1000.0, pair[1])
    assert not fit["fitted"]
    assert "slope" in fit["reason"]


def test_fit_fails_on_fewer_than_two_events():
    fit = _fit_eeg_to_pc(np.array([1.0]), np.array([1.0]))
    assert not fit["fitted"]
