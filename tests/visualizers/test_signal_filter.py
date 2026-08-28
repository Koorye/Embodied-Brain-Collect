"""The display-only filter cascade behind the QC page's 原始/滤波 toggle.

Every test here needs scipy — the module skips wholesale without it, which
is exactly how the page itself degrades (raw-only, no error).
"""

import numpy as np
import pytest

sp = pytest.importorskip("scipy")

from embodied_brain_collect.visualizers.signal_filter import (  # noqa: E402
    PRESETS, FilterPreset, apply_filter, design_sos, preset_for,
    preset_from_dict, scipy_ok)


def _resp(sos, freqs, fs) -> np.ndarray:
    """|H(f)| at exact frequencies (Hz), evaluated via sosfreqz."""
    _, h = sp.signal.sosfreqz(sos, worN=np.asarray(freqs, dtype=float), fs=fs)
    return np.abs(h)


# =============================================================================
# Presets
# =============================================================================

def test_presets_match_research_defaults():
    assert scipy_ok()
    assert set(PRESETS) == {"eeg", "emg"}
    eeg, emg = PRESETS["eeg"], PRESETS["emg"]
    assert (eeg.lo, eeg.hi, eeg.notch_base, eeg.order) == (0.5, 70.0, 50.0, 4)
    assert (emg.lo, emg.hi, emg.notch_base, emg.order) == (20.0, 450.0, 50.0, 4)


def test_preset_from_dict_merges_known_fields_only():
    merged = preset_from_dict({"lo": 30.0, "bogus": 1}, PRESETS["emg"])
    assert merged.lo == 30.0
    assert merged.hi == 450.0 and merged.notch_base == 50.0
    assert preset_from_dict(None, PRESETS["eeg"]) is PRESETS["eeg"]


def test_preset_for_prefix_match_and_overrides():
    assert preset_for("eeg", None) is PRESETS["eeg"]
    assert preset_for("emg_left", None) is PRESETS["emg"]
    assert preset_for("emg_right", {"emg": {"lo": 30.0}}).lo == 30.0
    assert preset_for("emg_right", {"emg": {"lo": 30.0}}).hi == 450.0
    # 未知键与不相关流不干扰
    assert preset_for("eeg", {"eye": {"lo": 1.0}}) is PRESETS["eeg"]
    assert preset_for("hand_pose") is None
    assert preset_for("eye") is None


# =============================================================================
# Frequency response
# =============================================================================

def test_design_sos_eeg_passband_flat_notches_deep():
    sos = design_sos(1000.0, PRESETS["eeg"])
    assert _resp(sos, [2, 5, 10, 20, 30, 40], 1000.0).min() >= 0.9
    notches = np.arange(50.0, 500.0, 50.0)
    assert _resp(sos, notches, 1000.0).max() <= 0.05


def test_design_sos_emg_notches_cover_to_nyquist():
    sos = design_sos(2000.0, PRESETS["emg"])
    # 通带测试点避开 50 Hz 谐波,且远离 450 Hz 截止的过渡带(8 阶在
    # 430 Hz 处 |H|≈0.7 是应有的滚降,不是缺陷)
    assert _resp(sos, [30, 80, 130, 180, 230, 280], 2000.0).min() >= 0.9
    notches = np.arange(50.0, 1000.0, 50.0)
    assert _resp(sos, notches, 2000.0).max() <= 0.05


def test_design_sos_rounds_fs_for_cache():
    # 1999.997 Hz(EMG 推导值)取整到 2000,与精确 2000 命中同一缓存
    assert design_sos(1999.997, PRESETS["emg"]) is design_sos(
        2000.0, PRESETS["emg"])


def test_design_sos_rejects_fs_below_bandpass_edge():
    with pytest.raises(ValueError):
        design_sos(100.0, PRESETS["eeg"])


# =============================================================================
# apply_filter behaviour
# =============================================================================

def test_apply_filter_zero_phase_impulse_symmetric():
    x = np.zeros(2001)
    x[1000] = 1.0
    out = apply_filter(x[:, None], 1000.0, PRESETS["eeg"])[:, 0]
    # 零相位:主瓣落在冲激处,且关于冲激左右对称。13 段 Q=30 级联在双精度
    # 下残余 ~0.1% 的不对称(单段低通为 1e-15),显示层面不可见,容差 1%。
    assert np.argmax(np.abs(out)) == 1000
    k = np.arange(1, 501)
    asym = np.abs(out[1000 - k] - out[1000 + k])
    assert asym.max() < 0.01 * np.abs(out).max()


def test_apply_filter_never_touches_input():
    rng = np.random.default_rng(0)
    inp = rng.normal(size=(4000, 3)).astype(np.float32)
    orig = inp.copy()
    out = apply_filter(inp, 1000.0, PRESETS["eeg"])
    assert np.array_equal(inp, orig)          # 输入原样
    assert out.dtype == np.float64
    assert out.shape == inp.shape
    assert not np.allclose(out[:, 0], inp[:, 0])   # 通道确实被滤了


def test_apply_filter_nan_channel_copied_raw():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4000, 3))
    data[500, 1] = np.nan
    out = apply_filter(data, 1000.0, PRESETS["eeg"])
    # NaN 通道原样拷过(equal_nan:array_equal 会因 NaN!=NaN 误报)
    assert np.allclose(out[:, 1], data[:, 1], rtol=0, atol=0, equal_nan=True)
    assert not np.allclose(out[:, 0], data[:, 0])  # 其余通道被滤


def test_apply_filter_short_array_returns_raw_copy():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(10, 2))
    out = apply_filter(data, 1000.0, PRESETS["eeg"])
    assert np.array_equal(out, data.astype(np.float64))


def test_apply_filter_constant_channel_no_crash():
    out = apply_filter(np.full((4000, 2), 5.0), 1000.0, PRESETS["eeg"])
    assert np.abs(out).max() < 1e-6              # 带通 DC 增益为 0


def test_apply_filter_fs_below_nyquist_returns_raw():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(1000, 2))
    out = apply_filter(data, 100.0, PRESETS["eeg"])   # 100 Hz < 2×70
    assert np.array_equal(out, data.astype(np.float64))


def test_apply_filter_1d_input_returns_copy():
    data = np.arange(100.0)
    out = apply_filter(data, 1000.0, PRESETS["eeg"])
    assert out.shape == data.shape
