"""两段式启动的写盘管线暖管时序:writer 就绪必须先于 commit。

回归背景(2026-10-15 会话 17-46-37):mp4 写盘器曾在 commit 后第一帧才
懒构造 —— FFmpegWriter 构造(nvenc 探测 + ffmpeg 进程拉起)约 0.8s,期间
8 帧队列溢出丢最老帧;commit→RUN_START 只有 ~0.6s,整个落进冷启动空窗,
四路视频(cam_left_wrist / cam_right_wrist / cam_third / eye scene)各丢
~30 帧,QC 报首端 TimestampGap。现在预热第一帧即暖管,launcher 的确认
阶段等 ``_ready_to_commit`` 翻真才广播 commit —— worker 全热 → commit →
RUN_START,顺序有保证。
"""

import threading
import time

import numpy as np

from embodied_brain_collect.recorders.camera import (
    CameraRecorderConfig, DummyCameraRecorder)
from embodied_brain_collect.session.launcher import _wait_data_flowing

# nvenc 的最小编码尺寸是 145x49 —— 测试帧必须够大,否则编码器初始化
# 失败(InitializeEncoder failed: invalid param),finalize 报错
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def _feed_until_ready(rec, timeout: float = 30.0) -> None:
    """模拟 launcher 预热期:持续投喂帧,直到写盘线程把 writer 构造好。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec.arr_video("frames", time.time(), FRAME)
        if rec._ready_to_commit():
            return
        time.sleep(0.02)
    raise AssertionError("预热期内 writer 未就绪(ffmpeg 拉起失败或过慢)")


def test_writer_warm_before_commit_first_frame_written_immediately(tmp_path):
    """预热首帧暖管:commit 时 writer 已热,commit 后第一帧立刻写掉。"""
    rec = DummyCameraRecorder(CameraRecorderConfig(session_dir=str(tmp_path)))
    rec._launch_mode = True
    _feed_until_ready(rec)

    # 暖管完成:writer 已构造,但预热帧一帧都没写(不进 mp4)
    assert "frames" in rec._video_writers
    assert not rec._written
    assert not rec._ts_buf.get("frames")

    commit_at = time.time()
    rec.request_commit()
    rec._maybe_commit()
    rec.arr_video("frames", time.time(), FRAME)

    # writer 已热:没有冷启动延迟,第一帧应马上写掉
    t0 = time.time()
    while time.time() - t0 < 5.0 and not rec._written.get("frames"):
        time.sleep(0.02)
    assert rec._written.get("frames") == 1
    # 容器第一帧就是 commit 后第一帧:时间戳不早于 commit
    assert rec._ts_buf["frames"][0] >= commit_at - 0.05

    rec._stop_write_worker()
    from embodied_brain_collect.utils.media import ffprobe_count
    assert ffprobe_count(tmp_path / "camera" / "frames.mp4") == 1


def test_discard_deletes_empty_mp4_and_does_not_restart_pipeline(tmp_path):
    """abort 路径:预热拉起的 writer 收掉,0 帧 mp4 删除,后续帧不再写盘。"""
    rec = DummyCameraRecorder(CameraRecorderConfig(session_dir=str(tmp_path)))
    rec._launch_mode = True
    _feed_until_ready(rec)

    rec.discard_recording()
    # 不留空 mp4(否则 has_any_data 误判有数据)。ffmpeg 在首帧写入前可能
    # 根本没创建文件,两种情况都要干净
    assert not list((tmp_path / "camera").glob("*.mp4"))
    rec.arr_video("frames", time.time(), FRAME)
    assert not rec._write_threads  # discard 后投喂不再重启写盘线程
    assert not rec._frames_enqueued


def test_wait_data_flowing_blocks_until_pipeline_ready():
    """确认闸:数据在流动但管线未就绪时,确认不得提前放行。"""

    class _FakeRec:
        def __init__(self):
            self.n = 0
            self.ready = False

        def _recording_signal(self):
            self.n += 1
            return self.n

        def _ready_to_commit(self):
            return self.ready

    rec = _FakeRec()
    abort = threading.Event()
    # 管线一直不就绪 → 超时,报"写盘管线未就绪"而不是"缓冲无新增样本"
    reason = _wait_data_flowing(rec, abort, timeout=0.6, window=0.1)
    assert "写盘管线未就绪" in reason
    # 就绪后立刻放行(旧代码在首拍就返回 "" 的行为即本次回归的病灶)
    rec.ready = True
    assert _wait_data_flowing(rec, abort, timeout=2.0, window=0.1) == ""


def test_base_ready_to_commit_defaults_true(tmp_path):
    """非视频 recorder 没有管线要暖:默认恒就绪,确认行为与旧版一致。"""
    from embodied_brain_collect.recorders.emg import (
        DummyEmgRecorder, EmgRecorderConfig)
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(tmp_path)))
    assert rec._ready_to_commit() is True


def test_eye_ready_to_commit_gate(tmp_path):
    """眼动 scene 写盘器:预热拉起前不放行;no_scene_video 恒就绪。"""
    from embodied_brain_collect.recorders.eye import (
        EyeRecorderConfig, NeonEyeAsyncRecorder)
    rec = NeonEyeAsyncRecorder(EyeRecorderConfig(session_dir=str(tmp_path)))
    assert rec._ready_to_commit() is False    # writer 未拉起
    rec._writer = object()                    # 模拟预热完成
    assert rec._ready_to_commit() is True

    rec2 = NeonEyeAsyncRecorder(EyeRecorderConfig(
        session_dir=str(tmp_path), no_scene_video=True))
    assert rec2._ready_to_commit() is True    # 不录 scene 视频时无需等
