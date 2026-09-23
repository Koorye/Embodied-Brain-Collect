"""OpenvrPositionRecorder 的 expected_devices 闸门 —— 不碰真硬件。

开录时已连接 tracker 少于期望台数是最常见的漏采(2026-09-20 19-17-56 只录到
2/3 台);闸门必须在 _open 就拒绝,而不是录一份缺轴的数据。
"""

from embodied_brain_collect.recorders.position import (
    OpenvrPositionRecorder, PositionRecorderConfig)
from embodied_brain_collect.recorders.position import openvr_position_recorder as mod


def _devices(n):
    return [{"index": i, "device_class": "tracker",
             "serial": f"serial-{i}", "model": "T3.0"} for i in range(n)]


def _recorder(tmp_path, expected):
    cfg = PositionRecorderConfig(session_dir=str(tmp_path / "position"),
                                 expected_devices=expected)
    return OpenvrPositionRecorder(cfg)


def test_open_fails_when_fewer_than_expected(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_init_openvr", lambda: object())
    monkeypatch.setattr(mod, "_select_devices",
                        lambda vr, classes: _devices(2))
    rec = _recorder(tmp_path, expected=3)
    assert rec._open() is False
    assert "expected 3" in rec._open_error


def test_open_passes_when_count_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_init_openvr", lambda: object())
    monkeypatch.setattr(mod, "_select_devices",
                        lambda vr, classes: _devices(3))
    rec = _recorder(tmp_path, expected=3)
    assert rec._open() is True
    assert len(rec._devices) == 3


def test_open_fails_when_no_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_init_openvr", lambda: object())
    monkeypatch.setattr(mod, "_select_devices", lambda vr, classes: [])
    rec = _recorder(tmp_path, expected=3)
    assert rec._open() is False
