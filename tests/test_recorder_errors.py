"""设备级错误原生传播 —— poll/record 路径不吞错,本来的异常直接往外抛。

原则:recorder 的数据通路里凡是设备/通路错误(对端断开、读数失败、
坏包头、SDK 线程带错退出、写盘管线故障),都不再 log-and-return ——
让异常原样抛出:子进程非零退出,launcher 按"异常退出"立即收摊(其余
模态落盘、跳过 QC 直接进 n/r/f/q 选择)。只有 SDK 用返回值表达失败
(OpenCV cap.read() → False)或错误发生在无法抛出的线程里(写盘线程、
带错退出的 SDK 线程)时,才原地 raise RuntimeError。空闲路径(等帧/
等包超时)是正常流,照常返回。
"""

import socket
import threading

import numpy as np
import pytest

from embodied_brain_collect.recorders.base import BaseRecorder, BaseRecorderConfig
from embodied_brain_collect.recorders.camera import (
    OpencvCameraRecorder, OpencvCameraConfig)
from embodied_brain_collect.recorders.camera import (
    RealsenseCameraRecorder, RealsenseCameraConfig)
from embodied_brain_collect.recorders.eeg import (
    BrainCoEegRecorder, BraincoEegRecorderConfig,
    CurryEegRecorder, EegRecorderConfig, IntanEegRecorder,
    IntanEegRecorderConfig)
from embodied_brain_collect.recorders.eeg.curry_eeg_recorder import (
    _CODE_EEG, _HEADER, _REQ_MAGIC)


# =============================================================================
# OpenCV 相机:cap.read() 失败
# =============================================================================

class _StubCap:
    def __init__(self, result):
        self.result = result

    def read(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_opencv_read_failure_raises(tmp_path):
    rec = OpencvCameraRecorder(OpencvCameraConfig(
        session_dir=str(tmp_path / "cam"), idx=3))
    rec._cap = _StubCap((False, None))
    with pytest.raises(RuntimeError, match="cap.read"):
        rec._poll(1.0)


# =============================================================================
# Realsense:空闲返回 None 不抛;SDK 异常原样穿透
# =============================================================================

class _StubPipeline:
    def __init__(self, result):
        self.result = result

    def try_wait_for_frames(self, timeout_ms):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_realsense_idle_none_is_normal(tmp_path):
    rec = RealsenseCameraRecorder(RealsenseCameraConfig(
        session_dir=str(tmp_path / "cam")))
    rec._pipeline = _StubPipeline(None)
    rec._poll(1.0)                       # 不抛 = 空闲正常


def test_realsense_sdk_error_propagates_unchanged(tmp_path):
    rec = RealsenseCameraRecorder(RealsenseCameraConfig(
        session_dir=str(tmp_path / "cam")))
    boom = RuntimeError("hawk device removed")
    rec._pipeline = _StubPipeline(boom)
    with pytest.raises(RuntimeError) as ei:
        rec._poll(1.0)
    assert ei.value is boom              # 本来的异常,原样往外抛


# =============================================================================
# Curry EEG:对端断开/坏包头/半包超时
# =============================================================================

class _Sock:
    def sendall(self, data):
        pass


def _curry(tmp_path):
    rec = CurryEegRecorder(EegRecorderConfig(
        session_dir=str(tmp_path / "eeg")))
    rec._sock = _Sock()
    return rec


def test_curry_peer_closed_propagates(tmp_path):
    rec = _curry(tmp_path)

    def _recv(*a, **k):
        raise ConnectionError("peer closed")

    rec._recv_exact = _recv
    with pytest.raises(ConnectionError, match="peer closed"):
        rec._poll(1.0)


def test_curry_bad_header_raises(tmp_path):
    rec = _curry(tmp_path)
    rec._recv_exact = lambda *a, **k: bytes(_HEADER.size)
    with pytest.raises(RuntimeError, match="bad packet header"):
        rec._poll(1.0)


def test_curry_partial_body_timeout_propagates(tmp_path):
    """body 读一半超时:流已错位,socket.timeout 原样抛出终止录制。"""
    rec = _curry(tmp_path)
    calls = {"n": 0}

    def _recv(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _HEADER.pack(_REQ_MAGIC, _CODE_EEG, 0, 0, 0, 0)
        raise socket.timeout()

    rec._recv_exact = _recv
    with pytest.raises(socket.timeout):
        rec._poll(1.0)


# =============================================================================
# Intan EEG:波形流断开
# =============================================================================

def test_intan_stream_closed_propagates(tmp_path):
    rec = IntanEegRecorder(IntanEegRecorderConfig(
        session_dir=str(tmp_path / "eeg")))
    rec._data_sock = object()

    def _boom(*a, **k):
        raise OSError("connection reset")

    rec._poll_read = _boom
    with pytest.raises(OSError, match="connection reset"):
        rec._poll(1.0)


# =============================================================================
# BrainCo EEG:SDK 线程带错退出
# =============================================================================

def test_brainco_sdk_thread_death_raises(tmp_path):
    rec = BrainCoEegRecorder(BraincoEegRecorderConfig(
        session_dir=str(tmp_path / "eeg")))
    rec._pkt_q = None
    rec._sync_sock = None
    rec._sdk_error = "BluetoothError: disconnected"
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    rec._sdk_thread = dead
    with pytest.raises(RuntimeError, match="SDK 线程中途退出"):
        rec._poll(1.0)


# =============================================================================
# Ego 头环:设备端断流(_ended / 对端 FIN)原样抛错,不再静默自停
# =============================================================================

def test_ego_stream_end_raises(tmp_path):
    from embodied_brain_collect.recorders.ego_headband import (
        EgoHeadbandRecorderConfig, NetEgoHeadbandRecorder)

    rec = NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig(
        session_dir=str(tmp_path / "ego_headband")))
    rec._drain_thread = None
    rec._ended = True
    rec._ended_reason = "server error: stream failed"
    rec._sock = object()
    with pytest.raises(RuntimeError, match="server error"):
        rec._poll(1.0)


def test_ego_peer_close_raises_directly(tmp_path):
    """对端关闭(FIN)→ _recv 里直接抛,不绕 _end_stream。"""
    from embodied_brain_collect.recorders.ego_headband import (
        EgoHeadbandRecorderConfig, NetEgoHeadbandRecorder)

    class _Sock:
        def settimeout(self, t):
            pass

        def recv(self, n):
            return b""                   # 对端关闭

    rec = NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig(
        session_dir=str(tmp_path / "ego_headband")))
    rec._drain_thread = None
    rec._ended = False
    rec._sock = _Sock()
    with pytest.raises(RuntimeError, match="连接已断开"):
        rec._poll(1.0)


# =============================================================================
# Weili EMG:出过字节后链路静默(serial read 只返回空,没有异常)
# =============================================================================

def test_weili_emg_link_silence_raises(tmp_path):
    """首批字节武装看门狗;之后静默超时 → 臂环断开。"""
    import time as _time
    from embodied_brain_collect.recorders.emg import (
        EmgRecorderConfig, WeiliEmgRecorder)

    class _Ser:
        def __init__(self):
            self.n = 0

        def read(self, size):
            self.n += 1
            return b"\xd2\xd2\xd2" if self.n == 1 else b""   # 首批后静默

    rec = WeiliEmgRecorder(EmgRecorderConfig(
        session_dir=str(tmp_path / "emg_left"), link_timeout=0.1))
    rec._ser = _Ser()
    rec._poll(1.0)                            # 首批字节 → 武装
    rec._last_rx_perf = _time.perf_counter() - 1.0   # 人为拨回 1s
    with pytest.raises(RuntimeError, match="没有任何数据"):
        rec._poll(2.0)


def test_weili_emg_silence_before_first_byte_not_fatal(tmp_path):
    """从未收到字节(慢启动)不判死 —— 交给 launcher 确认阶段。"""
    from embodied_brain_collect.recorders.emg import (
        EmgRecorderConfig, WeiliEmgRecorder)

    class _Ser:
        def read(self, size):
            return b""

    rec = WeiliEmgRecorder(EmgRecorderConfig(
        session_dir=str(tmp_path / "emg_left"), link_timeout=0.1))
    rec._ser = _Ser()
    rec._poll(1.0)                            # 未武装 → 不抛


def test_ego_link_silence_raises(tmp_path):
    """半开 TCP(断电/断网不发 FIN):recv 只会一直超时,没有任何显式
    错误 —— 链路静默超过 link_timeout 就地判死。"""
    import time as _time
    from embodied_brain_collect.recorders.ego_headband import (
        EgoHeadbandRecorderConfig, NetEgoHeadbandRecorder)

    class _DeadSock:
        def settimeout(self, t):
            pass

        def recv(self, n):
            raise socket.timeout()       # 永远只是超时

    rec = NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig(
        session_dir=str(tmp_path / "ego_headband"), link_timeout=5.0))
    rec._drain_thread = None
    rec._sock = _DeadSock()
    rec._ended = False
    rec._last_rx_perf = _time.perf_counter() - 100   # 已静默很久
    with pytest.raises(RuntimeError, match="没有任何数据"):
        rec._poll(1.0)

    rec._last_rx_perf = _time.perf_counter()         # 基线新鲜 → 不判死
    rec._poll(1.0)


# =============================================================================
# Manus 手套:出过数据的手套一旦停发(GetGloveData → None/空)立即抛
# =============================================================================

def test_manus_glove_stops_after_delivering(tmp_path):
    """一只手套断开 → 立即抛错,不等第二只(SDK 对断开的手套返回
    None/空 dict,dict 在场时唯一信号就是内容变空)。"""
    from embodied_brain_collect.recorders.hand_pose import (
        HandPoseRecorderConfig, ManusHandPoseRecorder)

    class _Pub:
        def __init__(self):
            self.ok = True

        def GetGloveData(self, gid):
            if not self.ok:
                return None
            return {
                "ergonomics": [{"type": "ThumbMCPSpread", "value": 0.5}],
                "raw_nodes": [{"position": [0, 0, 0],
                               "rotation": [0, 0, 0, 1]} for _ in range(50)],
            }

    rec = ManusHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=str(tmp_path / "hand_pose")))
    pub = _Pub()
    rec._pub = pub
    rec._glove_ids = [101, 102]
    rec._poll(1.0)                       # 两只都出数据 → 进 seen
    pub.ok = False                       # 手套断了
    with pytest.raises(RuntimeError, match="101"):
        rec._poll(2.0)


def test_manus_startup_silence_not_fatal(tmp_path):
    """SDK 起流后还没出过数据时不判死 —— 是否真在出数由 launcher 的
    确认阶段把关。"""
    from embodied_brain_collect.recorders.hand_pose import (
        HandPoseRecorderConfig, ManusHandPoseRecorder)

    class _Pub:
        def GetGloveData(self, gid):
            return None

    rec = ManusHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=str(tmp_path / "hand_pose")))
    rec._pub = _Pub()
    rec._glove_ids = [101, 102]
    rec._poll(1.0)                       # 从未出过数据 → 不抛


# =============================================================================
# Neon 眼动:录制循环带错收场(依赖 pupil_labs,缺则跳过)
# =============================================================================

def test_neon_record_raises_when_loop_died(tmp_path):
    pytest.importorskip("pupil_labs")
    from embodied_brain_collect.recorders.eye import (
        EyeRecorderConfig, NeonEyeAsyncRecorder)

    rec = NeonEyeAsyncRecorder(EyeRecorderConfig(
        session_dir=str(tmp_path / "eye")))
    rec._record_rc = 1
    rec._loop_done.set()
    with pytest.raises(RuntimeError, match="rc=1"):
        rec._record()


# =============================================================================
# 相机写盘管线:写线程已报错后,下一帧入队即抛
# =============================================================================

def test_camera_enqueue_raises_after_write_failure(tmp_path):
    rec = OpencvCameraRecorder(OpencvCameraConfig(
        session_dir=str(tmp_path / "cam")))
    rec._write_failed = {"frames"}
    with pytest.raises(RuntimeError, match="视频写盘管线已故障"):
        rec._enqueue("frames", 1.0, np.zeros((2, 2, 3), np.uint8))


# =============================================================================
# 端到端语义:poll 抛错 → 已录数据随 finally 落盘 → 异常往外传
# =============================================================================

class _BoomRecorder(BaseRecorder):
    name = "boom"
    output_dir = "boom"

    def __init__(self, config):
        super().__init__(config)
        self._n = 0

    def _open(self):
        return True

    def _close(self):
        pass

    def _poll(self, ts):
        self._acc("x", ts)
        self._n += 1
        if self._n >= 2:
            raise RuntimeError("boom — 设备故障")


def test_exception_stops_recording_but_saves_partial(tmp_path):
    rec = _BoomRecorder(BaseRecorderConfig(
        session_dir=str(tmp_path), hz=500))
    assert rec.run() == 1                # 独立路径:异常被记日志,返回 1
    z = np.load(tmp_path / "boom" / "boom.npz")
    assert z["x"].size >= 1              # 已录样本随 finally 落盘
