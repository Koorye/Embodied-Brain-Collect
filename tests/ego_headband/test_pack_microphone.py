"""打包器的麦克风加载:wav 切片、严格采集时间戳、坏数据跳过。

音频 PCM 本体只在 ``microphone.wav``;npz 是逐块索引(``wav_sample_offset`` /
设备侧 ``sample_index`` / 采集时间戳 ``microphone_stamp_ns``)。pack_daily
按索引切片投喂 mf_lerobot 的 AudioFeature,块起始时刻只认采集时间戳 ——
没有有效 stamp 的会话(旧数据)直接跳过音频,不做 read_complete 估计。
这里锁定切片语义、时间戳语义与容错行为。
"""

import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_daily  # noqa: E402

_RATE = 16000
_BLOCK = 320


def _write_session(out_dir: Path, offsets, *, error="", wav_len=0, legacy=False):
    """写一个最小会话。legacy=True 模拟无 stamp 字段的旧数据。"""
    ego = out_dir / "ego_headband"
    ego.mkdir(parents=True, exist_ok=True)
    n = len(offsets)
    off_arr = np.asarray(offsets, dtype=np.int64)
    done = (1_789_000_000_000_000_000 + off_arr * 62500).astype(np.int64)
    z = {
        "microphone_samples": np.full(n, _BLOCK, dtype=np.int64),
        "microphone_read_complete_ns": done,
        "microphone_sample_index": off_arr.copy(),
        "microphone_wav_sample_offset": off_arr.copy(),
        "microphone_sample_rate": np.asarray(_RATE, dtype=np.int64),
        "microphone_channels": np.asarray(1, dtype=np.int64),
        "microphone_sample_width": np.asarray(2, dtype=np.int64),
        "microphone_error": np.asarray(error),
        "left_timestamps": np.zeros(0),
    }
    if not legacy:
        # 采集时刻 = read_complete − 块时长 − 5ms 读延迟(与估计值可区分)
        z["microphone_stamp_ns"] = (done - _BLOCK * 1_000_000_000 // _RATE
                                    - 5_000_000).astype(np.int64)
        z["microphone_has_capture_timestamp"] = np.ones(n, dtype=bool)
    np.savez(ego / "ego_headband.npz", **z)
    if wav_len:
        pcm = (1000 * np.sin(2 * np.pi * 440 *
                             np.arange(wav_len) / _RATE)).astype("<i2")
        with wave.open(str(ego / "microphone.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_RATE)
            w.writeframes(pcm.tobytes())
    return ego


def test_chunks_slice_by_offset(tmp_path):
    """切片以 wav_sample_offset 为准,逐块与 wav 内容一致。"""
    wav_len = 4 * _BLOCK
    offsets = [0, _BLOCK, 2 * _BLOCK]
    _write_session(tmp_path, offsets, wav_len=wav_len)

    mic = pack_daily.load_microphone_index(tmp_path)
    assert mic is not None and mic["rate"] == _RATE
    chunks = pack_daily.load_microphone_chunks(tmp_path, mic)
    assert chunks is not None and len(chunks) == 3
    assert all(c.shape == (_BLOCK,) and c.dtype == np.float32 for c in chunks)
    # 首块相位 = 设备 index 100 处的正弦(不是 wav 采样 0 处)
    expect = 1000 * np.sin(2 * np.pi * 440 * np.arange(_BLOCK) / _RATE)
    assert np.allclose(chunks[0] * 32768.0, expect, atol=1.0)
    # 相邻块内容不同(相位推进)
    assert not np.allclose(chunks[0], chunks[1])


def test_nonzero_first_wav_offset_skips_preroll(tmp_path):
    """wav 头部有无元数据的预热段:切片按 wav_sample_offset,天然跳过。"""
    preroll = 8 * _BLOCK                      # 8s × 16kHz / 320 = 400 块的量
    offsets = [preroll, preroll + _BLOCK]
    _write_session(tmp_path, offsets, wav_len=preroll + 2 * _BLOCK)

    mic = pack_daily.load_microphone_index(tmp_path)
    assert mic is not None
    assert mic["offsets"][0] == preroll
    chunks = pack_daily.load_microphone_chunks(tmp_path, mic)
    assert sum(len(c) for c in chunks) == 2 * _BLOCK


def test_index_gap_warns_but_loads(tmp_path):
    """设备侧 index 有缺口:照常加载(时间戳逐一真实,缺口只是内容缺失)。"""
    offsets = [0, _BLOCK, 3 * _BLOCK]         # 第二块后丢了一块
    _write_session(tmp_path, offsets, wav_len=4 * _BLOCK)
    mic = pack_daily.load_microphone_index(tmp_path)
    assert mic is not None and len(mic["lens"]) == 3
    assert pack_daily.load_microphone_chunks(tmp_path, mic) is not None


def test_start_is_capture_stamp(tmp_path):
    """块起始时刻 = microphone_stamp_ns(硬件采集时刻),逐块精确,不估计。"""
    offsets = [0, _BLOCK, 2 * _BLOCK]
    _write_session(tmp_path, offsets, wav_len=3 * _BLOCK)

    z = np.load(tmp_path / "ego_headband" / "ego_headband.npz",
                allow_pickle=True)
    stamp = z["microphone_stamp_ns"].astype(np.int64)
    done = z["microphone_read_complete_ns"].astype(np.int64)
    est = done / 1e9 - _BLOCK / _RATE         # 旧公式,与 stamp 差 5ms

    mic = pack_daily.load_microphone_index(tmp_path)
    assert mic is not None
    np.testing.assert_allclose(mic["start"], stamp / 1e9, rtol=0, atol=1e-9)
    # 确与读回估计不同(证明确实用了 stamp 而非回推)
    assert np.all(np.abs(mic["start"] - est) > 1e-3)


def test_legacy_session_without_stamp_fields_skips_audio(tmp_path):
    """旧数据没有采集时间戳字段 → 音频整体跳过,不做 read_complete 估计。"""
    _write_session(tmp_path, [0, _BLOCK], wav_len=2 * _BLOCK, legacy=True)
    assert pack_daily.load_microphone_index(tmp_path) is None


def test_any_invalid_stamp_skips_audio(tmp_path):
    """任何一块 stamp 无效(has=False 或 0)→ 整会话跳过,不部分回退。"""
    partial = tmp_path / "partial"
    ego = _write_session(partial, [0, _BLOCK], wav_len=2 * _BLOCK)
    z = dict(np.load(ego / "ego_headband.npz", allow_pickle=True))
    z["microphone_has_capture_timestamp"] = np.array([True, False])
    np.savez(ego / "ego_headband.npz", **z)
    assert pack_daily.load_microphone_index(partial) is None

    zero = tmp_path / "zero"
    ego = _write_session(zero, [0, _BLOCK], wav_len=2 * _BLOCK)
    z = dict(np.load(ego / "ego_headband.npz", allow_pickle=True))
    z["microphone_stamp_ns"] = z["microphone_stamp_ns"].copy()
    z["microphone_stamp_ns"][1] = 0
    np.savez(ego / "ego_headband.npz", **z)
    assert pack_daily.load_microphone_index(zero) is None


def test_error_or_missing_audio_returns_none(tmp_path):
    """录制报错 / 无音频键 / 缺 wav 文件 → 返回 None,打包侧跳过该特征。"""
    _write_session(tmp_path, [0], error="microphone boom", wav_len=_BLOCK)
    assert pack_daily.load_microphone_index(tmp_path) is None

    empty = tmp_path / "empty"
    _write_session(empty, [])
    assert pack_daily.load_microphone_index(empty) is None

    nowav = tmp_path / "nowav"
    _write_session(nowav, [0])
    mic = pack_daily.load_microphone_index(nowav)
    assert mic is not None
    assert pack_daily.load_microphone_chunks(nowav, mic) is None
