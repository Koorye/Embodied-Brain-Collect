"""麦克风预热音频的 commit 门控:wav 必须与视频/元数据一样从 commit 起算。

两段式启动下视频在 commit 时丢弃预热帧;麦克风是流式直写 wav,预热段一旦
写入便无法回退,``_write_microphone`` 必须自己拦截 commit 前的音频块。漏拦
的后果(曾实际发生):wav 比视频长出整个预热期,且 npz 的
``microphone_wav_sample_offset`` 不从 0 起,时间轴无法对齐。这里用
DummyEgoHeadbandRecorder 的真实录制循环(合成麦克风包与真机同 schema)验证。

新时间戳的回归和真机检查也保留在本文件，复用项目 NetEgoHeadbandRecorder：
    python -m pytest tests/ego_headband/test_microphone_commit.py -q
    python tests/ego_headband/test_microphone_commit.py --record --duration 30
    python tests/ego_headband/test_microphone_commit.py --check-session <会话目录>
pytest 只用合成数据；只有显式 --record 才连接头环。成功录制后仅保留
ego_headband.npz：microphone_pcm 为 (N, 320) int16 原始单声道 PCM，
microphone_stamp_ns 为 (N,) int64，每行首采样的 CLOCK_REALTIME 纳秒时间戳
(ALSA 指针估计，物理音视频精度未验证)，microphone_sample_rate 为标量 16000。
复用原录制器写盘并校验，再无损归并；失败时保留原文件供排查。
--check-session 同时支持上述 NPZ 文件/会话目录和旧 WAV+NPZ，检查时不修改文件。
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from embodied_brain_collect.recorders.ego_headband import (
    DummyEgoHeadbandRecorder, EgoHeadbandRecorderConfig, NetEgoHeadbandRecorder)
from embodied_brain_collect.recorders.ego_headband.microphone_writer import MicrophoneWriter

_RATE = 16000
_BLOCK = 320  # MicrophoneWriter 固定块长(采样)


def _config(tmp_path, **kw):
    return EgoHeadbandRecorderConfig(
        session_dir=str(tmp_path), audio_enabled=True,
        n_cameras=2, cam_width=32, cam_height=24, cam_fps=20.0,
        n_imus=1, imu_rate_hz=100.0, hz=200.0, **kw)


def _wav_frames(out_dir):
    with wave.open(str(out_dir / "microphone.wav"), "rb") as w:
        return w.getnframes()


def test_commit_discards_microphone_preroll(tmp_path):
    rec = DummyEgoHeadbandRecorder(_config(tmp_path))
    rec._launch_mode = True            # launcher 子进程在开录前设置(_recorder_main)
    t = threading.Thread(target=rec.run, daemon=True)
    t.start()
    time.sleep(1.5)                      # 预热期:音频块持续产生
    rec.request_commit()
    time.sleep(2.0)
    rec.stop_event.set()
    t.join(timeout=15)
    assert not t.is_alive()

    out_dir = tmp_path / "ego_headband"   # recorder 落盘在 session_dir 的槽位子目录
    z = np.load(out_dir / "ego_headband.npz")
    idx = z["microphone_sample_index"].astype(np.int64)
    off = z["microphone_wav_sample_offset"].astype(np.int64)
    assert idx.size > 20

    # 设备侧计数从 open 起累计 —— 首个落盘块的 index 必然非零,证明预热段
    # 确实产生过(而非压根没数据)。启动/开录有延迟,只设一个宽松下限。
    assert idx[0] > _BLOCK * 10
    # wav 与元数据从 commit 起:offset 从 0 起,wav 恰好铺满设备样本
    # idx[0]..idx[-1](起点非零,但逐块连续无缝)
    assert off[0] == 0
    assert _wav_frames(out_dir) == int(idx[-1]) + _BLOCK - int(idx[0])

    # wav 时长锚定 commit(≈2.0s),而不是 open(≈3.5s)
    secs = _wav_frames(out_dir) / _RATE
    assert secs < 2.6

    # 内容级验证:wav 首采样 = 设备 index[0] 处的合成相位 —— 写进去的就是
    # commit 后那块音频,没有相位跳变(预热段混入会表现为 idx[0] 相位错位)
    with wave.open(str(out_dir / "microphone.wav"), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    expected = (2000 * np.sin(
        2 * np.pi * 440 * np.array([int(idx[0])]) / _RATE)).astype("<i2")[0]
    assert int(pcm[0]) == int(expected)


def test_standalone_run_keeps_full_audio(tmp_path):
    """独立 run()(无 commit,如真机脚本):门控不生效,时长照旧从开录起算。"""
    rec = DummyEgoHeadbandRecorder(_config(tmp_path, duration=1.0))
    assert rec.run() == 0

    out_dir = tmp_path / "ego_headband"
    z = np.load(out_dir / "ego_headband.npz")
    off = z["microphone_wav_sample_offset"].astype(np.int64)
    assert off[0] == 0
    secs = _wav_frames(out_dir) / _RATE
    assert 0.5 < secs <= 1.5


def test_short_preroll_still_aligned(tmp_path):
    """预热极短时同样成立:门控不依赖预热时长。"""
    rec = DummyEgoHeadbandRecorder(_config(tmp_path))
    rec._launch_mode = True
    t = threading.Thread(target=rec.run, daemon=True)
    t.start()
    time.sleep(0.2)
    rec.request_commit()
    time.sleep(1.0)
    rec.stop_event.set()
    t.join(timeout=15)

    out_dir = tmp_path / "ego_headband"
    z = np.load(out_dir / "ego_headband.npz")
    idx = z["microphone_sample_index"].astype(np.int64)
    off = z["microphone_wav_sample_offset"].astype(np.int64)
    assert off[0] == 0
    assert _wav_frames(out_dir) == int(idx[-1]) + _BLOCK - int(idx[0])


def _capture_packet(index):
    """Synthetic new-schema packet: non-float-exact ns, delayed/batched reads."""
    mono = 9_000_000_123 + index * 1_000_000_000 // _RATE
    offset = 1_789_000_000_000_000_000
    stamp = mono + offset
    return {
        "kind": "data", "topic": "/microphone/audio", "type": "audio/PCM",
        "header": {"seq": index // _BLOCK, "stamp_ns": stamp},
        "stream_seq": index // _BLOCK + 1,
        "read_complete_ns": stamp + 120_000_000,
        "timestamp_basis": MicrophoneWriter.CAPTURE_BASIS,
        "timestamp_reference": "first_sample", "timestamp_clock": "CLOCK_REALTIME",
        "capture_monotonic_ns": mono,
        "alsa_pointer_monotonic_ns": mono + 40_000_000,
        "alsa_available_frames": _BLOCK,
        "realtime_minus_monotonic_ns": offset,
        "clock_mapping_uncertainty_ns": 500,
        "audio": {"sample_rate": _RATE, "channels": 1, "format": "S16_LE",
                  "samples": _BLOCK, "sample_index": index},
    }


def _record_synthetic_capture(tmp_path):
    rec = NetEgoHeadbandRecorder(_config(tmp_path))
    rec._audio_available = True
    rec._launch_mode = True
    payload = np.arange(_BLOCK, dtype="<i2").tobytes()
    rec._on_data(_capture_packet(0), payload)
    assert rec._audio_writer is None  # pre-commit audio never touches WAV
    rec.request_commit()
    assert rec._maybe_commit()
    for i in range(1, 5):
        rec._on_data(_capture_packet(i * _BLOCK), payload)
    assert not rec._audio_error
    rec._teardown()  # close WAV, save NPZ, and release the Windows log-file handle
    return tmp_path / "ego_headband"


def _microphone_npz_path(session_dir):
    path = Path(session_dir)
    if path.suffix.lower() == ".npz":
        return path
    if (path / "ego_headband.npz").is_file():
        return path / "ego_headband.npz"
    return path / "ego_headband" / "ego_headband.npz"


def _verify_raw_microphone(path):
    """The compact file supports structure/continuity checks, not ALSA re-derivation."""
    problems = []
    report = {"timestamp_integrity_passed": False, "problems": problems,
              "av_sync_within_10ms_verified": False,
              "scope": "raw PCM and stored timestamp structure/continuity; no ALSA provenance recheck"}
    try:
        with np.load(path, allow_pickle=False) as z:
            pcm, stamp, rate = (z[k] for k in (
                "microphone_pcm", "microphone_stamp_ns", "microphone_sample_rate"))
            if pcm.dtype != np.dtype("<i2") or pcm.ndim != 2 or pcm.shape[1] != _BLOCK:
                raise ValueError("PCM must be int16 with shape (blocks, 320)")
            n = pcm.shape[0]
            if n < 2 or stamp.dtype != np.dtype("int64") or stamp.shape != (n,):
                raise ValueError("need one int64 timestamp per PCM block and at least two blocks")
            if rate.shape != () or rate.dtype.kind not in "iu" or int(rate) != _RATE:
                raise ValueError("sample rate must be the integer scalar 16000")
            if not np.all(stamp > 0):
                problems.append("zero/negative capture timestamp")
            if not np.all(np.diff(stamp) > 0):
                problems.append("capture timestamps repeat or go backwards")
            residual = np.diff(stamp) - _BLOCK * (1_000_000_000 // _RATE)
            max_deviation_ms = float(np.abs(residual).max()) / 1e6
            if max_deviation_ms > 10:
                problems.append("per-block timestamp discontinuity exceeds 10 ms")
            report.update(blocks=n, pcm_samples=int(pcm.size), duration_s=pcm.size / _RATE,
                          first_stamp_ns=int(stamp[0]),
                          max_block_timing_deviation_ms=max_deviation_ms)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        problems.append(str(exc))
    report["timestamp_integrity_passed"] = not problems
    return report


def verify_saved_microphone(session_dir):
    """Read-only check of either raw PCM NPZ or the original WAV+metadata NPZ."""
    path = _microphone_npz_path(session_dir)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "microphone_pcm" in z:
                return _verify_raw_microphone(path)
    except (OSError, ValueError, EOFError):
        pass  # The legacy checker will report the missing/unreadable input.
    return _verify_wav_microphone(path.parent)


def _verify_wav_microphone(out):
    """Check the original WAV and all recorded capture-time diagnostics."""
    problems = []
    def require(ok, message):
        if not ok:
            problems.append(message)
    report = {"timestamp_integrity_passed": False, "problems": problems,
              "av_sync_within_10ms_verified": False,
              "scope": "saved audio timestamp integrity, not physical AV accuracy"}
    try:
        with wave.open(str(out / "microphone.wav"), "rb") as wav:
            frames, rate, ch, width = (wav.getnframes(), wav.getframerate(),
                                       wav.getnchannels(), wav.getsampwidth())
            nbytes = 0
            while True:
                pcm = wav.readframes(_RATE)
                if not pcm:
                    break
                nbytes += len(pcm)
        require((rate, ch, width) == (_RATE, 1, 2), "unexpected WAV format")
        require(nbytes == frames * ch * width, "truncated WAV data")
        with np.load(out / "ego_headband.npz", allow_pickle=False) as z:
            names = ("stamp_ns", "sample_index", "wav_sample_offset", "samples", "stream_seq",
                     "read_complete_ns", "capture_monotonic_ns", "alsa_pointer_monotonic_ns",
                     "alsa_available_frames", "realtime_minus_monotonic_ns",
                     "clock_mapping_uncertainty_ns")
            missing = ["microphone_" + name for name in names if "microphone_" + name not in z]
            if missing:
                raise ValueError("missing capture timestamp fields: " + ", ".join(missing))
            arrays = {name: z["microphone_" + name] for name in names}
            n = arrays["stamp_ns"].size
            if n < 2 or any(a.shape != (n,) or a.dtype.kind not in "iu" for a in arrays.values()):
                raise ValueError("need matching 1D integer arrays for at least two audio blocks")
            # Preserve ns as integers through subtraction; convert only small differences.
            stamp, index, offset, samples = (arrays[k] for k in
                                             ("stamp_ns", "sample_index", "wav_sample_offset", "samples"))
            require(bool(np.all(stamp > 0)), "zero/negative capture timestamp")
            require(bool(np.all(np.diff(stamp) > 0)), "capture timestamps repeat or go backwards")
            require(bool(np.all(samples == _BLOCK)), "unexpected PCM block size")
            require(bool(np.all(np.diff(index) == samples[:-1])), "device sample index gap/overlap")
            require(bool(np.all(np.diff(arrays["stream_seq"]) == 1)), "audio stream sequence gap")
            expected_offsets = np.r_[0, np.cumsum(samples[:-1])]
            require(bool(np.array_equal(offset, expected_offsets)), "WAV offsets must start at zero and be continuous")
            require(int(samples.sum()) == frames, "WAV frame count differs from NPZ sample count")
            require(str(z["microphone_timestamp_basis"]) == MicrophoneWriter.CAPTURE_BASIS,
                    "NPZ still marks timestamps as read-complete/legacy")
            require(str(z["microphone_timestamp_reference"]) == "first_sample", "wrong timestamp reference")
            require(str(z["microphone_timestamp_clock"]) == "CLOCK_REALTIME", "wrong timestamp clock")
            require(not str(z["microphone_error"]), "recorder reported a microphone error")
            expected_mono = arrays["alsa_pointer_monotonic_ns"] - (
                arrays["alsa_available_frames"] + samples) * (1_000_000_000 // _RATE)
            require(bool(np.array_equal(expected_mono, arrays["capture_monotonic_ns"])),
                    "capture time disagrees with ALSA sample position")
            require(bool(np.array_equal(stamp, expected_mono + arrays["realtime_minus_monotonic_ns"])),
                    "capture time disagrees with realtime mapping")
            require(bool(np.all(arrays["read_complete_ns"] >= stamp)), "capture timestamp is after read completion")
            require(bool(np.all((arrays["clock_mapping_uncertainty_ns"] >= 0) &
                                (arrays["clock_mapping_uncertainty_ns"] <= 500_000))),
                    "clock mapping uncertainty exceeds 0.5 ms")
            residual = np.diff(stamp) - samples[:-1] * (1_000_000_000 // _RATE)
            max_deviation_ms = float(np.abs(residual).max()) / 1e6
            require(max_deviation_ms <= 10, "per-block timestamp discontinuity exceeds 10 ms")
            report.update(blocks=int(n), wav_samples=frames, duration_s=frames / rate,
                          first_stamp_ns=int(stamp[0]), first_device_sample_index=int(index[0]),
                          first_wav_sample_offset=int(offset[0]),
                          max_block_timing_deviation_ms=max_deviation_ms)
    except (OSError, ValueError, KeyError, EOFError, wave.Error) as exc:
        problems.append(str(exc))
    report["timestamp_integrity_passed"] = not problems
    return report


def _finalize_raw_microphone(session_dir):
    """Compact only this run's validated output; never discard failed recordings."""
    path = _microphone_npz_path(session_dir)
    out = path.parent
    report = _verify_wav_microphone(out)
    if not report["timestamp_integrity_passed"]:
        raise ValueError("cannot finalize invalid recording: " + "; ".join(report["problems"]))
    with np.load(path, allow_pickle=False) as z:
        stamp = z["microphone_stamp_ns"].copy()
    with wave.open(str(out / "microphone.wav"), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    # The original check established contiguous WAV offsets and 320 samples/block.
    pcm = np.frombuffer(raw, dtype="<i2").reshape(stamp.size, _BLOCK)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=out, prefix=".microphone-", suffix=".npz",
                                         delete=False) as f:
            staged = Path(f.name)
            np.savez(f, microphone_pcm=pcm, microphone_stamp_ns=stamp,
                     microphone_sample_rate=np.asarray(_RATE, dtype=np.int64))
            f.flush()
            os.fsync(f.fileno())
        check = _verify_raw_microphone(staged)
        if not check["timestamp_integrity_passed"]:
            raise ValueError("raw NPZ verification failed: " + "; ".join(check["problems"]))
        with np.load(staged, allow_pickle=False) as z:
            if (z["microphone_pcm"].tobytes() != raw or
                    not np.array_equal(z["microphone_stamp_ns"], stamp)):
                raise ValueError("raw NPZ differs from original PCM or integer timestamps")
        os.replace(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    # Clean up only known files after verified replacement in the new CLI session.
    for name in ("microphone.wav", "ego_headband.log"):
        (out / name).unlink(missing_ok=True)
    return path


def test_new_capture_timestamps_survive_commit_and_npz(tmp_path):
    out = _record_synthetic_capture(tmp_path)
    with np.load(out / "ego_headband.npz", allow_pickle=False) as z:
        stamps = z["microphone_stamp_ns"]
        assert stamps.dtype == np.int64
        expected = [_capture_packet(i * _BLOCK)["header"]["stamp_ns"] for i in range(1, 5)]
        np.testing.assert_array_equal(stamps, expected)  # catches float64 ns rounding
        assert z["microphone_sample_index"][0] == _BLOCK
        assert z["microphone_wav_sample_offset"][0] == 0
        assert str(z["microphone_timestamp_basis"]) == MicrophoneWriter.CAPTURE_BASIS
        assert not bool(z["microphone_av_sync_validated"])
    report = verify_saved_microphone(out)
    assert report["timestamp_integrity_passed"], report
    assert not report["av_sync_within_10ms_verified"]


def test_new_capture_rejects_invalid_timestamps_before_writing(tmp_path):
    payload = b"\0" * (_BLOCK * 2)
    for case in ("zero", "wrong_clock", "read_as_capture"):
        meta = _capture_packet(0)
        if case == "zero":
            meta["header"]["stamp_ns"] = 0
        elif case == "wrong_clock":
            meta["timestamp_clock"] = "CLOCK_MONOTONIC"
        else:
            meta["header"]["stamp_ns"] = meta["read_complete_ns"]
        writer = MicrophoneWriter(tmp_path / (case + ".wav"))
        try:
            with np.testing.assert_raises(ValueError):
                writer.write(meta, payload)
            assert writer.samples_written == 0
            assert not writer.path.exists()
        finally:
            writer.close()


def test_saved_audio_check_rejects_zero_and_mismatched_samples(tmp_path):
    out = _record_synthetic_capture(tmp_path)
    path = out / "ego_headband.npz"
    with np.load(path, allow_pickle=False) as z:
        saved = {key: z[key].copy() for key in z.files}
    saved["microphone_stamp_ns"][0] = 0
    saved["microphone_wav_sample_offset"][0] = _BLOCK
    np.savez(path, **saved)
    report = verify_saved_microphone(out)
    assert not report["timestamp_integrity_passed"]
    assert "zero/negative capture timestamp" in report["problems"]
    assert "WAV offsets must start at zero and be continuous" in report["problems"]


def test_record_cli_saves_only_microphone_from_mixed_stream(tmp_path, monkeypatch):
    """Exercise the CLI's config and real routing/saving, without opening hardware."""
    defaults = EgoHeadbandRecorderConfig()
    topics = {t: "sensor_msgs/CompressedImage" for t in defaults.camera_topics}
    topics.update({t: "sensor_msgs/Imu" for t in defaults.imu_topics})
    topics[defaults.audio_topic] = "audio/PCM"

    def reject_other_sensor(*args):
        raise AssertionError("microphone CLI routed a camera or IMU for recording")

    def run_mixed_stream(rec):
        try:
            rec._synced = True
            rec._on_stream_start({"topics": topics})
            for i in range(4):
                for topic in (*defaults.camera_topics, *defaults.imu_topics):
                    rec._on_data({"kind": "data", "topic": topic,
                                  "header": {"stamp_ns": 1}, "stream_seq": i + 1}, b"")
                pcm = (np.arange(_BLOCK, dtype="<i2") + i * _BLOCK).astype("<i2")
                pcm[0], pcm[-1] = -32768, 32767
                rec._on_data(_capture_packet(i * _BLOCK), pcm.tobytes())
            return 0
        finally:
            rec._teardown()

    monkeypatch.setattr(NetEgoHeadbandRecorder, "run", run_mixed_stream)
    monkeypatch.setattr(NetEgoHeadbandRecorder, "_queue_camera", reject_other_sensor)
    monkeypatch.setattr(NetEgoHeadbandRecorder, "_on_imu", reject_other_sensor)
    assert main(["--record", "--duration", "0.08", "--out", str(tmp_path)]) == 0
    assert not list(tmp_path.rglob("*.mp4"))
    out = next(tmp_path.glob("*-microphone")) / "ego_headband"
    assert {p.name for p in out.iterdir()} == {"ego_headband.npz"}
    with np.load(out / "ego_headband.npz", allow_pickle=False) as z:
        assert set(z.files) == {"microphone_pcm", "microphone_stamp_ns", "microphone_sample_rate"}
        expected = np.arange(4 * _BLOCK, dtype="<i2").reshape(4, _BLOCK)
        expected[:, 0], expected[:, -1] = -32768, 32767
        np.testing.assert_array_equal(z["microphone_pcm"], expected)
        np.testing.assert_array_equal(z["microphone_stamp_ns"],
                                      [_capture_packet(i * _BLOCK)["header"]["stamp_ns"] for i in range(4)])
    assert verify_saved_microphone(out)["timestamp_integrity_passed"]


def test_raw_npz_roundtrip_after_preroll_and_readonly_check(tmp_path):
    out = _record_synthetic_capture(tmp_path)
    with wave.open(str(out / "microphone.wav"), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert main(["--check-session", str(out.parent)]) == 0
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    path = _finalize_raw_microphone(out)
    with np.load(path, allow_pickle=False) as z:
        assert z["microphone_pcm"].tobytes() == raw
        assert z["microphone_stamp_ns"][0] == _capture_packet(_BLOCK)["header"]["stamp_ns"]
    before = path.read_bytes()
    assert main(["--check-session", str(path)]) == 0
    assert {p.name for p in out.iterdir()} == {"ego_headband.npz"}
    assert path.read_bytes() == before


def test_raw_npz_finalize_failure_preserves_recording(tmp_path, monkeypatch):
    out = _record_synthetic_capture(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with np.testing.assert_raises(OSError):
        _finalize_raw_microphone(out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_failed_recording_keeps_wav_and_metadata(tmp_path, monkeypatch):
    def fail_after_audio(rec):
        try:
            rec._synced = True
            rec._on_stream_start({"topics": {rec.config.audio_topic: "audio/PCM"}})
            for i in range(4):
                rec._on_data(_capture_packet(i * _BLOCK), b"\0" * (_BLOCK * 2))
            return 1
        finally:
            rec._teardown()

    monkeypatch.setattr(NetEgoHeadbandRecorder, "run", fail_after_audio)
    assert main(["--record", "--duration", "0.08", "--out", str(tmp_path)]) == 1
    out = next(tmp_path.glob("*-microphone")) / "ego_headband"
    assert _wav_frames(out) == 4 * _BLOCK
    assert (out / "ego_headband.log").is_file()
    with np.load(out / "ego_headband.npz", allow_pickle=False) as z:
        assert "microphone_pcm" not in z
        assert z["microphone_stamp_ns"].size == 4


def test_raw_npz_check_rejects_invalid_data(tmp_path):
    out = _record_synthetic_capture(tmp_path)
    path = _finalize_raw_microphone(out)
    with np.load(path, allow_pickle=False) as z:
        saved = {key: z[key].copy() for key in z.files}
    for case in ("float_pcm", "float_stamp", "block_mismatch", "zero_stamp", "wrong_rate"):
        bad = {key: value.copy() for key, value in saved.items()}
        if case == "float_pcm":
            bad["microphone_pcm"] = bad["microphone_pcm"].astype(np.float32)
        elif case == "float_stamp":
            bad["microphone_stamp_ns"] = bad["microphone_stamp_ns"].astype(np.float64)
        elif case == "block_mismatch":
            bad["microphone_pcm"] = bad["microphone_pcm"][:-1]
        elif case == "zero_stamp":
            bad["microphone_stamp_ns"][0] = 0
        else:
            bad["microphone_sample_rate"] = np.asarray(48000)
        np.savez(path, **bad)
        assert not verify_saved_microphone(path)["timestamp_integrity_passed"], case


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record raw microphone PCM and timestamps into one NPZ")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="record microphone PCM and timestamps; keep only NPZ after validation")
    mode.add_argument("--check-session", type=Path, help="read-only check of an NPZ file or session directory; does not open hardware")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--host", default="192.168.55.6")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--out", type=Path, default=_ROOT / "tests/sessions/ego_headband")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be positive")
    rec, rc = None, 0
    if args.record:
        from datetime import datetime
        session = args.out / (datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + "-microphone")
        config = EgoHeadbandRecorderConfig(session_dir=str(session), duration=args.duration,
                                           host=args.host, port=args.port, audio_enabled=True,
                                           # Empty topic mappings discard camera/IMU packets before saving.
                                           camera_topics=(), camera_names=(), imu_topics=(),
                                           n_cameras=0, n_imus=0,
                                           require_synced=True)
        print("Recording microphone only with project NetEgoHeadbandRecorder:", session, flush=True)
        rec = NetEgoHeadbandRecorder(config)
        rc = rec.run()
    else:
        session = args.check_session
    report = verify_saved_microphone(session)
    if rec is not None:
        report.update(recorder_returncode=rc, clock_synced=rec._synced)
        if rc != 0 or rec._synced is not True or not rec._audio_available:
            report["problems"].append("recording/clock handshake/microphone availability failed")
            report["timestamp_integrity_passed"] = False
        if report.get("duration_s", 0) < args.duration * 0.8:
            report["problems"].append("recorded audio is shorter than 80% of requested duration")
            report["timestamp_integrity_passed"] = False
        if report["timestamp_integrity_passed"]:
            try:
                path = _finalize_raw_microphone(session)
                print("Saved raw audio and timestamps:", path)
            except (OSError, ValueError, EOFError, wave.Error) as exc:
                report["problems"].append(f"raw NPZ finalization failed: {exc}")
                report["timestamp_integrity_passed"] = False
        if not report["timestamp_integrity_passed"]:
            print("Recording did not pass; keeping available files for inspection:", session)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print("PASS: saved microphone timestamps" if report["timestamp_integrity_passed"] else "FAIL: see problems above")
    print("NOT VERIFIED: physical audio/video alignment within 10 ms")
    return 0 if report["timestamp_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
