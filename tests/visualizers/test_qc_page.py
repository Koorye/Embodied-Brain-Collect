"""The reductions the QC page depends on, and one whole-session build."""

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from embodied_brain_collect.visualizers.qc_page import (Options, _split_findings,
                                                        build_payload,
                                                        render_html)
from embodied_brain_collect.visualizers.qc_payload import (channel_rows,
                                                           encode_i16,
                                                           minmax_downsample,
                                                           series,
                                                           target_points,
                                                           timing_rows)
from embodied_brain_collect.visualizers.qc_streams import (_sampling_rate,
                                                           rows_eeg, rows_emg)

SESSION4 = Path(__file__).resolve().parents[2] / "data" / "session4"


def decode(s, key="y", lo="lo", hi="hi") -> np.ndarray:
    q = np.frombuffer(base64.b64decode(s[key]), dtype=np.int16).astype(float)
    return np.where(q == -32768, np.nan,
                    q * (s[hi] - s[lo]) / 32767 + s[lo])


# =============================================================================
# Decimation
# =============================================================================

def test_downsample_keeps_a_lone_spike():
    """The whole reason for min/max buckets: a one-sample jump is exactly
    what the timing checks flag, and striding would step right over it."""
    y = np.zeros(100_000)
    y[54_321] = 999.0
    t = np.arange(y.size) * 1e-3
    td, yd = minmax_downsample(t, y, 1500)
    assert td.size <= 1500
    assert yd.max() == pytest.approx(999.0)


def test_downsample_keeps_both_extremes_of_a_bucket():
    y = np.zeros(10_000)
    y[4000], y[4001] = -50.0, 70.0        # both inside one bucket
    _, yd = minmax_downsample(np.arange(y.size) * 1e-3, y, 200)
    assert yd.min() == pytest.approx(-50.0)
    assert yd.max() == pytest.approx(70.0)


def test_downsample_marks_an_all_nan_bucket():
    y = np.ones(10_000)
    y[2000:4000] = np.nan
    _, yd = minmax_downsample(np.arange(y.size) * 1e-3, y, 200)
    assert np.isnan(yd).any()             # the stroke must break there


def test_downsample_passes_short_series_through():
    t = np.arange(10.0)
    td, yd = minmax_downsample(t, t * 2, 1500)
    assert np.array_equal(td, t) and np.array_equal(yd, t * 2)


# =============================================================================
# Quantisation
# =============================================================================

def test_int16_roundtrip_within_one_step():
    y = np.linspace(-5.0, 5.0, 4000)
    b64, lo, hi = encode_i16(y)
    q = np.frombuffer(base64.b64decode(b64), dtype=np.int16).astype(float)
    assert np.allclose(q * (hi - lo) / 32767 + lo, y, atol=(hi - lo) / 32767)


def test_int16_nan_sentinel():
    b64, _, _ = encode_i16(np.array([1.0, np.nan, 3.0]))
    q = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
    assert q[1] == -32768 and q[0] != -32768


def test_int16_survives_a_constant_series():
    b64, lo, hi = encode_i16(np.full(50, 7.0))
    assert hi > lo                        # no divide-by-zero
    assert np.frombuffer(base64.b64decode(b64), dtype=np.int16).size == 50


def test_series_rejects_what_cannot_be_drawn():
    t = np.arange(10.0)
    assert series(t, np.full(10, np.nan)) is None
    assert series(np.arange(1.0), np.zeros(1)) is None
    assert series(t, np.zeros(3)) is None          # length mismatch
    assert series(t, t) is not None


# =============================================================================
# Filtered display copies (y_f)
# =============================================================================

def test_series_embeds_filtered_copy():
    rng = np.random.default_rng(0)
    t = np.arange(0, 2.0, 0.001)
    y = rng.normal(scale=100.0, size=t.size)
    yf = y * 0.5
    s = series(t, y, y_f=yf)
    assert {"yf", "flo", "fhi"} <= set(s)
    assert np.allclose(decode(s, "yf", "flo", "fhi"), yf,
                       atol=(s["fhi"] - s["flo"]) / 32767)
    assert s["fhi"] - s["flo"] < s["hi"] - s["lo"]   # 滤波后幅值小得多
    assert np.allclose(decode(s), y, atol=(s["hi"] - s["lo"]) / 32767)


def test_series_without_y_f_has_no_filter_keys():
    s = series(np.arange(10.0), np.arange(10.0))
    assert "yf" not in s and "flo" not in s and "fhi" not in s


def test_series_uniform_ts_embeds_filtered_copy():
    t = np.arange(0, 1.0, 0.0005)                  # EMG 式严格均匀
    y = np.sin(2 * np.pi * 50 * t)
    s = series(t, y, uniform_ts=True, y_f=y * 2)
    assert "tstride" in s and "yf" in s            # stride 分支同样携带
    assert np.allclose(decode(s, "yf", "flo", "fhi"), y * 2,
                       atol=(s["fhi"] - s["flo"]) / 32767)


def test_series_ignores_mismatched_filter_copy():
    s = series(np.arange(10.0), np.arange(10.0), y_f=np.zeros(3))
    assert "yf" not in s                           # 长度不符 -> 忽略,不报错


# =============================================================================
# Channel rows
# =============================================================================

def test_channel_rows_one_row_per_channel_with_names():
    t = np.arange(3000) * 1e-3
    data = np.random.default_rng(0).normal(size=(3000, 132))
    names = [f"ch{n}" for n in range(132)]
    rows = channel_rows(t, data, names=names)
    assert len(rows) == 132
    assert rows[0]["label"] == "ch0"
    assert all(r["src"] == "" for r in rows)      # src 由调用方传入


def test_channel_rows_shrink_with_channel_count():
    """只有行高随通道数变,时间分辨率不降(v1.1.0:全分辨率)。"""
    t = np.arange(0, 24.0, 0.001)                 # 24 s
    big = channel_rows(t, np.random.randn(len(t), 132))
    small = channel_rows(t, np.random.randn(len(t), 8))
    assert big[0]["h"] < small[0]["h"]
    for rows in (big, small):
        tt = np.frombuffer(base64.b64decode(rows[0]["ser"][0]["t"]),
                           dtype=np.float32)
        assert len(tt) == len(t)                   # 每样本都在


def test_channel_rows_rejects_mismatched_shapes():
    assert channel_rows(np.arange(10.0), np.zeros((5, 3))) == []
    assert channel_rows(np.arange(10.0), np.zeros(10)) == []


def test_channel_rows_embed_filtered_per_channel():
    rng = np.random.default_rng(1)
    t = np.arange(3000) * 1e-3
    data = rng.normal(size=(3000, 8))
    rows = channel_rows(t, data, data_f=data * 0.5)
    assert len(rows) == 8
    for c, r in enumerate(rows):
        s = r["ser"][0]
        assert "yf" in s
        yf = decode(s, "yf", "flo", "fhi")
        assert len(yf) == 3000
        assert np.allclose(yf, data[:, c] * 0.5,
                           atol=(s["fhi"] - s["flo"]) / 32767)


def test_channel_rows_ignore_mismatched_filter_matrix():
    rng = np.random.default_rng(2)
    t = np.arange(1000) * 1e-3
    rows = channel_rows(t, rng.normal(size=(1000, 4)),
                        data_f=np.zeros((500, 3)))
    assert rows and all("yf" not in r["ser"][0] for r in rows)   # 忽略,不报错


# =============================================================================
# Per-stream extraction (rows_eeg / rows_emg)
# =============================================================================

def _eeg_z(n=3000, nch=133, t0=1000.0):
    rng = np.random.default_rng(3)
    return {
        "eeg_timestamps_pc": np.arange(n) * 0.001 + t0,
        "eeg_data": rng.normal(size=(n, nch)).astype(np.float32),
        "eeg_channel_names": [f"ch{i}" for i in range(nch - 1)] + ["trigger"],
        "eeg_sample_rate": 1000.0,
    }, t0


def _emg_z(n=6000, nch=8, t0=2000.0):
    rng = np.random.default_rng(4)
    t_emg = np.arange(n) * 0.0005 + t0            # 2 kHz
    t_imu = np.arange(600) * 0.009 + t0           # ~111 Hz
    return {
        "emg_timestamps": t_emg,
        "emg_data": rng.integers(-4000, 4000, size=(n, nch)).astype(np.int32),
        "imu_timestamps": t_imu,
        "imu_gyro": rng.normal(size=(600, 3)).astype(np.float32),
        "imu_accel": rng.normal(size=(600, 3)).astype(np.float32),
    }, t0


def test_rows_eeg_embeds_filtered_copy_same_length():
    pytest.importorskip("scipy")
    z, t0 = _eeg_z()
    rows, _ = rows_eeg(z, Path("."), t0, Options())
    ch_rows = [r for r in rows if r["label"].startswith("ch")]
    assert len(ch_rows) == 132
    for r in ch_rows:
        s = r["ser"][0]
        assert "yf" in s
        assert len(decode(s, "yf", "flo", "fhi")) == 3000
    trig = next(r for r in rows if r["label"] == "Trigger")
    assert "yf" not in trig["ser"][0]             # 触发通道永不滤波
    iv = next(r for r in rows if r["label"].startswith("采样间隔"))
    assert "yf" not in iv["ser"][0]


def test_rows_emg_filters_only_emg_channels():
    pytest.importorskip("scipy")
    z, t0 = _emg_z()
    rows, _ = rows_emg(z, Path("."), t0, Options())
    emg_rows = [r for r in rows if r["label"].startswith("emg ch")]
    assert len(emg_rows) == 8
    for r in emg_rows:
        s = r["ser"][0]
        assert "yf" in s and "tstride" in s       # uniform_ts 分支同样携带
        assert len(decode(s, "yf", "flo", "fhi")) == 6000
    imu_rows = [r for r in rows if r["src"] == "imu"]
    assert imu_rows
    assert all("yf" not in s for r in imu_rows for s in r["ser"])


def test_options_filter_false_omits_all_yf():
    for rows_fn, z_fn in ((rows_eeg, _eeg_z), (rows_emg, _emg_z)):
        z, t0 = z_fn()
        rows, _ = rows_fn(z, Path("."), t0, Options(filter=False))
        assert rows
        assert all("yf" not in s for r in rows for s in r["ser"])


def test_sampling_rate_uses_mean_not_median_for_skewed_emg():
    # 重建后的 EMG 时间戳间隔偏态(实测:中位 0.473 ms、均值 0.5 ms)。
    # 中位间隔会估出 2114 Hz,滤波设计频率整体偏移 5.7%,50 Hz 陷波落空。
    dt = np.tile(np.array([0.000473] * 9 + [0.000743]), 1000)[:9999]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    fs = _sampling_rate({}, "emg", t)
    assert fs == pytest.approx(2000.0, rel=1e-3)
    assert abs(fs - 2000.0) < abs(1.0 / np.median(dt) - 2000.0)


def test_rows_emg_notch_kills_50hz_with_skewed_timestamps():
    pytest.importorskip("scipy")
    # 偏态时间戳 + 50 Hz 工频(应被陷波)+ 130 Hz 参考(应保留)
    n = 6000
    dt = np.tile(np.array([0.000473] * 9 + [0.000743]), n // 10)[:n - 1]
    t_emg = 2000.0 + np.concatenate([[0.0], np.cumsum(dt)])
    tt = t_emg - t_emg[0]
    data = (0.6 * np.sin(2 * np.pi * 50 * tt)
            + 0.3 * np.sin(2 * np.pi * 130 * tt))
    z = {"emg_timestamps": t_emg,
         "emg_data": np.repeat(data[:, None], 8, axis=1).astype(np.float32)}
    rows, _ = rows_emg(z, Path("."), t_emg[0], Options())
    s = next(r for r in rows if r["label"] == "emg ch0")["ser"][0]
    yf = decode(s, "yf", "flo", "fhi")
    Y = np.fft.rfft(yf)
    fr = np.fft.rfftfreq(len(yf), 1 / 2000.0)
    m50 = np.abs(Y[np.argmin(np.abs(fr - 50))])
    m130 = np.abs(Y[np.argmin(np.abs(fr - 130))])
    assert m50 < m130 * 0.1          # 50 Hz 至少衰减 20 dB,130 Hz 保留


def test_target_points_floor():
    """v1.1.0:不降采样,预算 = 样本数本身。"""
    t = np.arange(0, 24.0, 1 / 1000)
    assert target_points(t) == len(t)              # 全分辨率
    assert target_points(np.arange(5.0)) == 5      # 短序列也全保留


# =============================================================================
# Timing rows
# =============================================================================

def test_timing_rows_stay_out_of_the_way_when_clean():
    """A healthy stream should not spend vertical space on a duplicate-stamp
    row that is flat zero."""
    rows = timing_rows(np.arange(0, 10, 0.01))
    assert [r["label"] for r in rows] == ["采样间隔"]


def test_timing_rows_add_the_duplicate_row_when_there_are_duplicates():
    t = np.repeat(np.arange(0, 5, 0.01), 3)
    assert "重复时间戳" in [r["label"] for r in timing_rows(t)]


# =============================================================================
# Findings -> events
# =============================================================================

def test_spans_become_events_on_the_run_clock():
    findings = [
        {"level": "ERROR", "check": "TimestampGap", "message": "缺失",
         "spans": [{"t": 1000.5, "dur": 2.0, "msg": "缺 60 个样本"}]},
        {"level": "WARN", "check": "MadOutlier", "message": "异常值"},
    ]
    events, notes = _split_findings(findings, 1000.0)
    assert len(events) == 1 and len(notes) == 1
    assert events[0]["t"] == pytest.approx(0.5)     # rebased on RUN_START
    assert events[0]["dur"] == pytest.approx(2.0)
    # A finding with no time must NOT become an event at t=0 — it is a
    # property of the whole stream, not something that happened at the start.
    assert notes[0]["check"] == "MadOutlier"


# =============================================================================
# Whole session
# =============================================================================

@pytest.mark.slow
def test_session4_page_builds_and_locates_its_problems():
    from embodied_brain_collect.checkers import qc_session
    if not SESSION4.is_dir():
        pytest.skip("missing fixture")

    report = qc_session(SESSION4).to_dict()
    payload = build_payload(report, SESSION4, Options(frames=False))

    names = {s["name"] for s in payload["streams"]}
    assert {"eeg", "emg_left", "emg_right", "position", "cam_head"} <= names
    assert len(payload["markers"]) == 12
    assert payload["markers"][0]["name"] == "RUN_START"
    assert payload["markers"][0]["t"] == pytest.approx(0.0, abs=1e-6)

    def events(stream, check):
        st = next(s for s in payload["streams"] if s["name"] == stream)
        return [e for e in st["events"] if e["check"] == check]

    # The gaps must land where the checker said they were.
    gaps = events("emg_right", "TimestampGap")
    assert len(gaps) == 7
    assert max(g["dur"] for g in gaps) == pytest.approx(2.97, abs=0.05)

    track = events("position", "TrackingGap")
    assert len(track) == 1 and track[0]["dur"] == pytest.approx(0.45, abs=0.02)

    # Every event sits inside the recorded span.
    for st in payload["streams"]:
        for e in st["events"]:
            assert payload["span"][0] - 1 <= e["t"] <= payload["span"][1] + 1

    html = render_html(payload)
    assert html.count("__QC_PAYLOAD__") == 0        # placeholder consumed
    blob = re.search(r"const DATA = (\{.*?\});\nconst PL", html, re.S)
    assert json.loads(blob.group(1))["level"] == "ERROR"
