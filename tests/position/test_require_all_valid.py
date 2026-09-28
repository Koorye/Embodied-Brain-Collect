"""require_all_valid 看门狗 —— 录制中任一 tracker 无效 pose 即终止。

规则:所有设备必须始终有效。任一 tracker 出现无效 pose(遮挡/掉线/断
光塔)→ _poll 抛 RuntimeError → base._record 的 finally 照常收尾落盘 →
worker → 子进程非零退出 → launcher 按"异常退出"立即收摊、跳过 QC 直接
进 n/r/f/q 选择。launcher 预热段(commit 前)豁免——那段数据会被整体
丢弃。全部不碰真硬件(_read_frame 打桩)。
"""

import numpy as np
import pytest

from embodied_brain_collect.recorders.position import (
    OpenvrPositionRecorder, PositionRecorderConfig)
from embodied_brain_collect.recorders.position import (
    openvr_position_recorder as mod)

_ROLES = [("left_wrist", "61-A"), ("right_wrist", "61-B"), ("chest", "61-C")]


def _recorder(tmp_path, **cfg):
    rec = OpenvrPositionRecorder(PositionRecorderConfig(
        session_dir=str(tmp_path / "position"), **cfg))
    rec._devices = [{"index": i, "device_class": "tracker",
                     "serial": serial, "model": "T3.0", "role": role}
                    for i, (role, serial) in enumerate(_ROLES)]
    return rec


def _valid(n_dev, bad=None):
    v = np.ones(n_dev, dtype=bool)
    if bad is not None:
        v[bad] = False
    return v


def test_all_valid_no_raise(tmp_path):
    rec = _recorder(tmp_path)
    rec._check_all_valid(_valid(3), ts=100.0)          # 不抛即通过


def test_any_invalid_raises_with_role_and_serial(tmp_path):
    rec = _recorder(tmp_path)
    with pytest.raises(RuntimeError) as ei:
        rec._check_all_valid(_valid(3, bad=2), ts=100.0)
    assert "chest" in str(ei.value) and "61-C" in str(ei.value)
    assert "2/3" in str(ei.value)


def test_unbound_device_reported_as_unbound(tmp_path):
    rec = _recorder(tmp_path)
    rec._devices[1] = {"index": 1, "device_class": "tracker",
                       "serial": "61-B", "model": "T3.0", "role": ""}
    with pytest.raises(RuntimeError) as ei:
        rec._check_all_valid(_valid(3, bad=1), ts=100.0)
    assert "unbound(61-B)" in str(ei.value)


def test_preroll_before_commit_is_exempt(tmp_path):
    """launcher 预热段(commit 前)数据会被丢弃,不参与判定。"""
    rec = _recorder(tmp_path)
    rec._launch_mode = True
    rec._committed = False
    rec._check_all_valid(_valid(3, bad=0), ts=100.0)   # 不抛


def test_post_commit_is_armed(tmp_path):
    rec = _recorder(tmp_path)
    rec._launch_mode = True
    rec._committed = True
    with pytest.raises(RuntimeError):
        rec._check_all_valid(_valid(3, bad=1), ts=100.0)


def test_watchdog_can_be_disabled(tmp_path):
    rec = _recorder(tmp_path, require_all_valid=False)
    rec._launch_mode = True
    rec._committed = True
    rec._check_all_valid(_valid(3, bad=2), ts=100.0)   # 不抛


def test_poll_writes_the_bad_frame_then_raises(tmp_path, monkeypatch):
    """触发帧的数据先落缓冲再抛错 —— 已录的有效段随收尾保存。"""
    rec = _recorder(tmp_path)
    pos = np.zeros((3, 3))
    quat = np.zeros((3, 4))
    euler = np.zeros((3, 3))

    def _read(vr, devs):
        bad = 0 if rec._arr_buf.get("valid") else None   # 第二帧起掉 left_wrist
        return pos, quat, euler, _valid(3, bad=bad)

    monkeypatch.setattr(mod, "_read_frame", _read)
    rec._vr = object()
    rec._poll(100.0)                                  # 首帧全有效
    with pytest.raises(RuntimeError):
        rec._poll(100.1)
    assert len(rec._arr_buf["valid"]) == 2          # 触发帧也在缓冲里,收尾一并落盘
