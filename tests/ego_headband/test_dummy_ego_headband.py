"""Unit tests for the dummy EGO headband: schema, clocks, and QC pipeline.

The dummy exists so the checker and QC page can run without the headband on
the network; these tests pin the contract that its outputs speak the same
schema the network recorder promises (per-camera ``{name}.mp4`` +
``{name}_timestamps`` in the NPZ, ``imu{j}_ts`` / ``imu{j}_gyro`` /
``imu{j}_accel``) and that the whole offline pipeline dispatches on it.
"""

import numpy as np
import pytest

from embodied_brain_collect.checkers import checker_for
from embodied_brain_collect.checkers.ego_headband import EgoHeadbandChecker
from embodied_brain_collect.recorders.ego_headband import (
    DummyEgoHeadbandRecorder, EgoHeadbandRecorderConfig)
from embodied_brain_collect.visualizers.qc_streams import extractor_for

CAM_FPS = 20.0
IMU_HZ = 100.0


@pytest.fixture
def stream_dir(tmp_path):
    """Run a short dummy session; return the recorder's output directory."""
    cfg = EgoHeadbandRecorderConfig(
        session_dir=str(tmp_path), duration=2.0,
        n_cameras=2, cam_width=32, cam_height=24, cam_fps=CAM_FPS,
        n_imus=1, imu_rate_hz=IMU_HZ)
    assert DummyEgoHeadbandRecorder(cfg).run() == 0
    return tmp_path / "ego_headband"


def test_dummy_schema_matches_network_recorder(stream_dir):
    z = np.load(stream_dir / "ego_headband.npz")
    for key in ("left_timestamps", "right_timestamps",
                "imu0_ts", "imu0_gyro", "imu0_accel"):
        assert key in z.files, key
    # 帧走 {name}.mp4(async HEVC writer),不再进 NPZ
    for name in ("left", "right"):
        assert (stream_dir / f"{name}.mp4").is_file(), name
    assert len(z["imu0_ts"]) == len(z["imu0_gyro"]) == len(z["imu0_accel"])


def test_camera_and_imu_clocks_run_at_configured_rates(stream_dir):
    z = np.load(stream_dir / "ego_headband.npz")
    cam_t = z["left_timestamps"]
    imu_t = z["imu0_ts"]
    assert cam_t.size / (cam_t[-1] - cam_t[0]) == pytest.approx(CAM_FPS, rel=0.15)
    assert imu_t.size / (imu_t[-1] - imu_t[0]) == pytest.approx(IMU_HZ, rel=0.15)


def test_checker_and_qc_dispatch_on_the_schema(stream_dir):
    assert checker_for("ego_headband") is EgoHeadbandChecker

    report = EgoHeadbandChecker().run(stream_dir, None)
    # prepare 按发现的时间戳字段动态注册序列 —— 这是该 checker 的核心契约
    assert set(report.series) == {"left", "right", "imu0"}
    # dummy 的合成时钟继承宿主机调度抖动,心跳/暂停偶发 TimestampGap/Jump
    # (真实网络录制器没有这条路径);结构性问题仍然必须零出现。
    clock_noise = {"TimestampGap", "TimestampJump", "IntervalJitter",
                   "SlidingWindowRate"}
    structural = [f for f in report.findings if f.check not in clock_noise]
    assert not [f for f in structural if f.level != "INFO"], \
        [f.message for f in structural]

    z = np.load(stream_dir / "ego_headband.npz")
    rows, thumbs = extractor_for("ego_headband")(
        z, stream_dir, 0.0, type("Opt", (), {"frames": False})())
    assert rows                                     # cam + imu traces
    assert thumbs == []                             # frames extraction off
