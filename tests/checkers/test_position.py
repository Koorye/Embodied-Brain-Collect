"""Position checker 的设备轴检查(DeviceCount)单元测试。"""

import numpy as np
import pytest

from embodied_brain_collect.checkers.position import DeviceCount


def levels(out):
    return [f.level for f in out.findings]


def messages(out):
    return " | ".join(f.message for f in out.findings)


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
