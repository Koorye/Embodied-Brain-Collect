"""EgoHeadbandChecker 的期望相机集合检查(CameraSet)单元测试。

背景:设备端某路相机不发布时,那一流在 npz/目录里直接消失,逐流检查会
全部干净通过(它只检查实际存在的流)。CameraSet 专门拦这种静默缺路。
"""

import numpy as np

from embodied_brain_collect.checkers.ego_headband import (
    CameraSet, EgoHeadbandChecker)


def levels(out):
    return [f.level for f in out.findings]


def messages(out):
    return " | ".join(f.message for f in out.findings)


def _series(*names):
    t = np.arange(100, dtype=float) / 30.0
    return {name: t for name in names}


def _npz(path, *names):
    fields = {f"{n}_timestamps": np.arange(100, dtype=float) / 30.0
              for n in names}
    fields["imu0_ts"] = np.arange(100, dtype=float) / 200.0
    fields["imu0_gyro"] = np.zeros((100, 3), dtype=np.float32)
    fields["imu0_accel"] = np.zeros((100, 3), dtype=np.float32)
    np.savez(path / "ego_headband.npz", **fields)


def test_all_cameras_pass(ctx):
    out = CameraSet().run(
        ctx(series=_series("left", "right", "bleft", "bright")))
    assert out.findings == []
    assert out.stats["n_cameras"] == 4
    assert out.stats["missing"] == []


def test_missing_camera_is_error(ctx):
    out = CameraSet().run(ctx(series=_series("left", "right", "bright")))
    assert levels(out) == ["ERROR"]
    assert "仅 3/4 路相机" in messages(out)
    assert "bleft" in messages(out)
    assert out.stats["missing"] == ["bleft"]


def test_extra_camera_is_warn(ctx):
    out = CameraSet().run(
        ctx(series=_series("left", "right", "bleft", "bright", "front")))
    assert levels(out) == ["WARN"]
    assert "front" in messages(out)
    assert out.stats["extra"] == ["front"]


def test_imu_series_are_not_cameras(ctx):
    series = {**_series("left", "right", "bleft", "bright"),
              "imu0": np.arange(100.0)}
    out = CameraSet().run(ctx(series=series))
    assert out.findings == []
    assert out.stats["n_cameras"] == 4


def test_expected_is_configurable(ctx):
    out = CameraSet(expected=("left", "right")).run(
        ctx(series=_series("left", "right")))
    assert out.findings == []


def test_checker_flags_missing_camera_end_to_end(tmp_path):
    _npz(tmp_path, "left", "right", "bright")
    report = EgoHeadbandChecker().run(tmp_path)
    camera_set = [f for f in report.findings if f.check == "CameraSet"]
    assert [f.level for f in camera_set] == ["ERROR"]
    assert "bleft" in camera_set[0].message


def test_checker_passes_with_all_cameras(tmp_path):
    _npz(tmp_path, "left", "right", "bleft", "bright")
    report = EgoHeadbandChecker().run(tmp_path)
    assert not [f for f in report.findings if f.check == "CameraSet"]


def test_checker_yaml_override(tmp_path):
    """checker_cfg 的 camera_set.expected 覆盖默认的 4 路相机名。"""
    _npz(tmp_path, "left", "right")
    checker = EgoHeadbandChecker(
        {"camera_set": {"expected": ["left", "right"]}})
    report = checker.run(tmp_path)
    assert not [f for f in report.findings if f.check == "CameraSet"]
