"""BrainCo 开录阻抗门禁:回调解析 + 门禁评估 + npz 字段(无硬件)。

与 ``test_pack_impedance_pack.py`` 对齐的是输出 schema:门禁结果与 Curry
同字段落盘(``eeg_impedance_*``),打包器按 ``eeg_channel_names`` 透传。
回调官方签名未公开,``coerce_imp_payload`` 对 (chip, values) tuple /
dict / 属性对象做尽力解析,解析约定在这里锁住。
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from embodied_brain_collect.recorders.eeg import brainco_eeg_recorder as mod
from embodied_brain_collect.recorders.eeg.brainco_eeg_recorder import (
    BRAINCO_CHANNEL_NAMES, BRAINCO_N_CHIPS, BRAINCO_N_CHANNELS,
    BrainCoEegRecorder, BraincoEegRecorderConfig, coerce_imp_payload,
)

_EDGE = ["FT9", "FT10", "TP9", "TP10", "IO"]


class FakeChip:
    """仿 LeadOffChip 枚举成员:name + value(int,1 基)。"""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


def _recorder(max_kohm: float = 100.0, pass_rate: float = 0.87,
              edge: list[str] | None = None) -> BrainCoEegRecorder:
    cfg = BraincoEegRecorderConfig(session_dir="")
    cfg.impedance_max_kohm = max_kohm
    cfg.impedance_pass_rate = pass_rate
    if edge is not None:
        cfg.impedance_edge_channels = edge
    rec = BrainCoEegRecorder(cfg)
    rec._channel_labels = list(BRAINCO_CHANNEL_NAMES)
    return rec


def _window(chip: int, base: float) -> np.ndarray:
    return np.full(8, base, dtype=np.float32)


# =============================================================================
# 回调载荷解析
# =============================================================================

def test_coerce_tuple_enum_chip():
    chip, vals = coerce_imp_payload(
        (FakeChip("Chip2", 2), [10_000.0] * 8))
    assert chip == 1
    assert vals.shape == (8,) and float(vals[0]) == 10_000.0


def test_coerce_flat_tuple_and_dict_and_attrs():
    # (chip, v0..v7) 铺平
    parsed = coerce_imp_payload((3, *([20_000.0] * 8)))
    assert parsed is not None and parsed[0] == 2

    parsed = coerce_imp_payload({"chip": 1, "values": [5_000.0] * 8})
    assert parsed is not None and parsed[0] == 0

    class Obj:
        chip = FakeChip("Chip4", 4)
        impedance = [1.0] * 8

    parsed = coerce_imp_payload(Obj())
    assert parsed is not None and parsed[0] == 3


def test_coerce_garbage_returns_none():
    assert coerce_imp_payload(([1.0] * 8,)) is None          # 只有值没有 chip
    assert coerce_imp_payload((FakeChip("Chip1", 1), [1.0] * 7)) is None
    assert coerce_imp_payload({"values": [1.0] * 8}) is None  # 缺 chip
    assert coerce_imp_payload(None) is None
    assert coerce_imp_payload({"chip": 9, "values": [1.0] * 8}) is None


def test_chip_index_prefers_enum_name_over_value():
    # 判别值不可信时以 name 为准;NONE(0)/越界 → None
    assert coerce_imp_payload((FakeChip("Chip3", 3), [1.0] * 8))[0] == 2
    assert coerce_imp_payload((4, [1.0] * 8))[0] == 3          # 裸 int 1 基
    assert coerce_imp_payload((FakeChip("NONE", 0), [1.0] * 8)) is None
    assert coerce_imp_payload((True, [1.0] * 8)) is None


# =============================================================================
# 窗口收拢 + 门禁评估
# =============================================================================

def test_impedance_mean_averages_per_chip_block():
    rec = _recorder()
    rec._imp_windows = [
        (0, _window(0, 1_000.0)), (0, _window(0, 3_000.0)),   # 均值 2k
        (1, _window(1, 4_000.0)),
        (2, _window(2, 5_000.0)),
        (3, _window(3, 7_000.0)), (3, _window(3, 9_000.0)),   # 均值 8k
    ]
    z, reason = rec._impedance_mean()
    assert reason == ""
    assert z.shape == (BRAINCO_N_CHANNELS,)
    assert z[0] == pytest.approx(2_000.0)
    assert z[8] == pytest.approx(4_000.0)
    assert z[16] == pytest.approx(5_000.0)
    assert z[31] == pytest.approx(8_000.0)
    assert rec._impedance_n == 6


def test_impedance_mean_missing_chip_fails():
    rec = _recorder()
    rec._imp_windows = [(0, _window(0, 1.0)),
                        (1, _window(1, 1.0)),
                        (2, _window(2, 1.0))]
    z, reason = rec._impedance_mean()
    assert z is None
    assert "chip" in reason and "4" in reason


def test_impedance_mean_no_data_fails():
    rec = _recorder()
    z, reason = rec._impedance_mean()
    assert z is None
    assert "未收到阻抗数据" in reason


def test_evaluate_pass_exempts_edge_channels():
    rec = _recorder()
    z = np.full(BRAINCO_N_CHANNELS, 8_000.0, dtype=np.float64)
    for name in _EDGE:   # 边缘位给天量阻抗,不应计入通过率
        z[BRAINCO_CHANNEL_NAMES.index(name)] = 5_000_000.0
    ok, detail = rec._evaluate_impedance(z)
    assert ok and "impedance ok" in detail
    checked = rec._impedance_checked
    assert checked.sum() == BRAINCO_N_CHANNELS - len(_EDGE)
    assert rec._impedance_pass is True
    assert rec._impedance_pass_rate == pytest.approx(1.0)
    for name in _EDGE:
        assert not checked[BRAINCO_CHANNEL_NAMES.index(name)]


def test_evaluate_fail_lists_offending_channels():
    rec = _recorder(pass_rate=0.87)
    z = np.full(BRAINCO_N_CHANNELS, 8_000.0, dtype=np.float64)
    for name in ("C3", "Cz", "Pz", "O1", "O2", "FP1", "FP2",
                 "F3", "F4", "Fz"):                        # 10/27 超标
        z[BRAINCO_CHANNEL_NAMES.index(name)] = 200_000.0
    ok, detail = rec._evaluate_impedance(z)
    assert not ok
    assert "阻抗检查未通过" in detail and "C3(200k)" in detail
    assert rec._impedance_pass is False
    assert rec._impedance_pass_rate == pytest.approx(17.0 / 27.0)


def test_evaluate_over_threshold_but_rate_passes():
    """少量超标不阻断(通过率 ≥ 门槛),但文案里要点名。"""
    rec = _recorder()
    z = np.full(BRAINCO_N_CHANNELS, 8_000.0, dtype=np.float64)
    z[BRAINCO_CHANNEL_NAMES.index("C3")] = 150_000.0
    ok, detail = rec._evaluate_impedance(z)
    assert ok and "超标(未阻断)" in detail and "C3(150k)" in detail
    assert rec._impedance_pass is True


def test_on_sdk_imp_collects_windows():
    rec = _recorder()
    rec._on_sdk_imp(FakeChip("Chip1", 1), [1_000.0] * 8)
    rec._on_sdk_imp({"chip": 2, "values": [2_000.0] * 8})
    rec._on_sdk_imp("garbage")            # 解析失败只告警不进窗口
    assert len(rec._imp_windows) == 2
    assert [c for c, _v in rec._imp_windows] == [0, 1]


# =============================================================================
# 落盘字段(与打包器 schema 对齐)
# =============================================================================

def test_build_output_emits_impedance_fields():
    rec = _recorder()
    z = np.full(BRAINCO_N_CHANNELS, 8_000.0, dtype=np.float64)
    ok, _ = rec._evaluate_impedance(z)
    assert ok
    rec._impedance_n = 3
    out = rec._build_output()
    assert out["eeg_impedance_ohm"].shape == (BRAINCO_N_CHANNELS,)
    assert out["eeg_impedance_checked"].dtype == np.bool_
    assert float(out["eeg_impedance_pass_rate"]) == pytest.approx(1.0)
    assert bool(out["eeg_impedance_check_pass"]) is True
    assert int(out["eeg_impedance_n_snapshots"]) == 3


def test_build_output_pads_trigger_slot_for_33ch_firmware():
    """33 路固件在门禁后才确定通道数:TRIG 槽位补零且 checked=False。"""
    rec = _recorder()
    z = np.full(BRAINCO_N_CHANNELS, 8_000.0, dtype=np.float64)
    rec._evaluate_impedance(z)
    rec._channel_labels = BRAINCO_CHANNEL_NAMES + ["TRIG"]
    out = rec._build_output()
    assert out["eeg_impedance_ohm"].shape == (BRAINCO_N_CHANNELS + 1,)
    checked = out["eeg_impedance_checked"]
    assert not bool(checked[-1])
    assert out["eeg_impedance_ohm"][-1] == 0


def test_build_output_without_gate_has_no_fields():
    rec = _recorder()
    out = rec._build_output()
    assert not [k for k in out if k.startswith("eeg_impedance")]


# =============================================================================
# 门禁流程(假 SDK/客户端:锁住 enable → 收窗 → 评估 → disable 时序)
# =============================================================================

class _FakeSdk:
    registered = None

    class LeadOffFreq:
        Ac31p2hz = "Ac31p2hz"

    class LeadOffCurrent:
        Cur6nA = "Cur6nA"

    class LeadOffChip:
        Chip1 = "Chip1"

    @classmethod
    def set_imp_data_callback(cls, fn):
        cls.registered = fn


class _FakeClient:
    def __init__(self, rec: BrainCoEegRecorder,
                 windows: list[tuple[int, np.ndarray]],
                 enable_raises: Exception | None = None):
        self._rec = rec
        self._windows = windows
        self._enable_raises = enable_raises
        self.enabled = False
        self.disabled = False
        self.enable_kwargs: dict | None = None

    def enable_impedance_detection_mode(self, **kwargs):
        self.enable_kwargs = kwargs
        rec, windows, raises = self._rec, self._windows, self._enable_raises

        async def _go():
            if raises is not None:
                raise raises
            self.enabled = True
            for chip, vals in windows:
                rec._on_sdk_imp(chip + 1, [float(v) for v in vals])
        return _go()

    def disable_impedance_detection_mode(self):
        async def _go():
            self.disabled = True
        return _go()


def test_gate_happy_path_passes_and_disables(monkeypatch):
    monkeypatch.setattr(mod, "_IMPEDANCE_WINDOW_S", 0.2)
    rec = _recorder()
    windows = [(c, _window(c, 8_000.0))
               for c in range(BRAINCO_N_CHIPS) for _ in range(2)]
    client = _FakeClient(rec, windows)
    detail = asyncio.run(rec._impedance_gate(_FakeSdk(), client))
    assert client.enabled and client.disabled
    assert client.enable_kwargs["loop_check"] is True
    assert _FakeSdk.registered is None          # 收尾注销回调
    assert "impedance ok" in detail
    assert rec._impedance_pass is True
    assert rec._impedance_error == ""
    assert rec._impedance_n == len(windows)


def test_gate_reject_sets_error_and_still_disables(monkeypatch):
    monkeypatch.setattr(mod, "_IMPEDANCE_WINDOW_S", 0.2)
    rec = _recorder()
    windows = [(c, _window(c, 500_000.0)) for c in range(BRAINCO_N_CHIPS)]
    client = _FakeClient(rec, windows)
    detail = asyncio.run(rec._impedance_gate(_FakeSdk(), client))
    assert client.disabled
    assert "阻抗检查未通过" in detail
    assert rec._impedance_error == detail
    assert rec._impedance_pass is False


def test_gate_enable_failure_still_disables(monkeypatch):
    monkeypatch.setattr(mod, "_IMPEDANCE_WINDOW_S", 0.2)
    rec = _recorder()
    client = _FakeClient(rec, [], enable_raises=RuntimeError("rejected"))
    detail = asyncio.run(rec._impedance_gate(_FakeSdk(), client))
    assert client.disabled
    assert "阻抗检测失败" in detail and "RuntimeError" in detail
    assert rec._impedance_error == detail


def test_gate_partial_windows_fail_with_chip_hint(monkeypatch):
    monkeypatch.setattr(mod, "_IMPEDANCE_WINDOW_S", 0.2)
    rec = _recorder()
    windows = [(0, _window(0, 8_000.0)), (1, _window(1, 8_000.0))]
    client = _FakeClient(rec, windows)
    detail = asyncio.run(rec._impedance_gate(_FakeSdk(), client))
    assert "阻抗数据不完整" in detail
    assert rec._impedance_error == detail


# =============================================================================
# 配置
# =============================================================================

def test_config_defaults_and_from_dict():
    cfg = BraincoEegRecorderConfig(session_dir="")
    assert cfg.impedance_check is True
    assert cfg.impedance_max_kohm == 100.0
    assert cfg.impedance_pass_rate == pytest.approx(0.87)
    assert cfg.impedance_edge_channels == _EDGE
    cfg2 = BraincoEegRecorderConfig.from_dict(
        {"impedance_check": False, "host": "192.168.1.50",
         "nonexistent_key": 1})
    assert cfg2.impedance_check is False and cfg2.host == "192.168.1.50"


def test_chip_count_matches_channel_layout():
    assert BRAINCO_N_CHIPS == 4
    assert BRAINCO_N_CHIPS * 8 == BRAINCO_N_CHANNELS
