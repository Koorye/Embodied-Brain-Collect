"""Position checker 的设备轴检查(DeviceCount / ValidAlways)单元测试。"""

import numpy as np
import pytest

from embodied_brain_collect.checkers.position import DeviceCount, ValidAlways


def levels(out):
    return [f.level for f in out.findings]


def messages(out):
    return " | ".join(f.message for f in out.findings)


def subjects(out):
    return [f.subject for f in out.findings]


def _pos(n_dev, t=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, (t, n_dev, 3))


def test_count_passes_when_exact(ctx):
    c = ctx(arrays={"positions_m": _pos(3)})
    out = DeviceCount().run(c)
    assert out.findings == []
    assert out.stats["n_devices"] == 3


def test_missing_tracker_is_error(ctx):
    """少连一台 tracker(设备轴只有 2)→ ERROR,可打包门槛据此拦截。"""
    out = DeviceCount().run(ctx(arrays={"positions_m": _pos(2)}))
    assert levels(out) == ["ERROR"]
    assert "仅 2/3" in messages(out)


def test_extra_tracker_is_error(ctx):
    """多出一台同样 ERROR:dev 序号会跨会话错位。"""
    out = DeviceCount().run(ctx(arrays={"positions_m": _pos(4)}))
    assert levels(out) == ["ERROR"]


def test_flat_layout_counts_as_one_device(ctx):
    """(T, 3) 平铺布局按 1 台算;与期望台数不符照常报。"""
    out = DeviceCount(expected=1).run(
        ctx(arrays={"positions_m": _pos(1)[:, 0, :]}))
    assert out.findings == []
    assert out.stats["n_devices"] == 1


def test_expected_is_configurable(ctx):
    out = DeviceCount(expected=2).run(ctx(arrays={"positions_m": _pos(2)}))
    assert out.findings == []


# =============================================================================
# ValidAlways — 所有设备必须始终有效,任一无效样本即 ERROR
# =============================================================================

def test_valid_always_passes_clean(ctx):
    out = ValidAlways().run(ctx(arrays={"positions_m": _pos(3)}))
    assert out.findings == []                    # 无 valid 字段 = 全部有效
    assert out.stats["invalid_samples"] == {f"dev{i}": 0 for i in range(3)}


def test_valid_always_errors_on_any_invalid_sample(ctx):
    """掉一台(哪怕几帧)→ ERROR,subject 按设备。"""
    pos = _pos(3)
    valid = np.ones(pos.shape[:2], dtype=bool)
    valid[10:15, 1] = False
    out = ValidAlways().run(ctx(arrays={
        "positions_m": pos, "valid": valid,
        "serials": np.asarray(["61-A", "61-B", "61-C"])}))
    assert levels(out) == ["ERROR"]
    assert subjects(out) == ["61-B"]
    assert "5 个无效样本" in messages(out)
    assert out.stats["invalid_samples"]["61-B"] == 5


def test_valid_always_reports_each_offender(ctx):
    pos = _pos(3)
    valid = np.ones(pos.shape[:2], dtype=bool)
    valid[3, 0] = False
    valid[7, 2] = False
    out = ValidAlways().run(ctx(arrays={"positions_m": pos, "valid": valid}))
    assert levels(out) == ["ERROR", "ERROR"]
    assert sorted(subjects(out)) == ["dev0", "dev2"]


def test_valid_always_respects_run_window(ctx):
    """窗口外的无效样本(预热前的历史数据)不计入。"""
    pos = _pos(3)
    valid = np.ones(pos.shape[:2], dtype=bool)
    valid[2, 1] = False                          # 窗口前
    out = ValidAlways().run(ctx(arrays={
        "positions_m": pos, "valid": valid,
        "timestamps_s": np.arange(len(pos), dtype=float)},
        window={"t0": 10.0, "t1": 40.0}))
    assert out.findings == []
