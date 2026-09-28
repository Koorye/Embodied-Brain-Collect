"""打包器的阻抗门禁流:单行 parquet、timestamp=0、缺字段会话自然跳过。

Curry 开录阻抗门禁的结果随 ``eeg/eeg.npz`` 落盘(``eeg_impedance_*``),
pack_daily 把它透传成独立特征 ``observation.eeg_impedance``:每 episode
一个 parquet、恰好一行,列 = 各通道均值(Ω,顺序同 ``eeg_channel_names``,
含 Trigger 状态字槽位)+ 门禁汇总(pass_rate / gate_pass / n_snapshots)。
它与时间轴无关 —— 时间戳是 0 哨兵(``CONSTANT_TS``),窗口掩码恒放行、
写帧不平移;BrainCo / 关门禁的会话没有这些字段,不产出该流,也不因
缺它触发"缺模态剔除"。
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_daily as pd  # noqa: E402

_SUMMARY = ["pass_rate", "gate_pass", "n_snapshots"]
_CHANNELS = ["FP1", "FP2", "C3", "C4", "Trigger"]


def _write_session(tmp_dir: Path, *, with_impedance: bool) -> Path:
    """最小会话:窗口前起、跨 RUN_START 的 eeg + RUN_START/END 标记。"""
    t0 = time.time() - 100.0
    eeg = tmp_dir / "eeg"
    eeg.mkdir(parents=True)
    payload = dict(
        eeg_timestamps_pc=t0 + np.arange(100, dtype=np.float64) * 0.05,
        eeg_data=np.random.randn(100, 5).astype(np.float32),
        eeg_sample_rate=np.asarray(1000.0),
        eeg_channel_names=np.asarray(_CHANNELS),
        eeg_n_samples=np.asarray(100),
        eeg_n_channels=np.asarray(5),
        eeg_n_eeg_channels=np.asarray(4),
        eeg_start_amp_sample=np.asarray(0),
    )
    if with_impedance:
        payload.update(
            eeg_impedance_ohm=np.asarray(
                [5123.0, 8011.0, 12345.0, 6800.0, 299.0], dtype=np.float32),
            eeg_impedance_checked=np.asarray(
                [True, True, True, True, False]),
            eeg_impedance_pass_rate=np.asarray(0.8),
            eeg_impedance_check_pass=np.asarray(True),
            eeg_impedance_n_snapshots=np.asarray(3),
        )
    np.savez(eeg / "eeg.npz", **payload)
    marker = tmp_dir / "marker"
    marker.mkdir()
    np.savez(marker / "marker.npz",
             marker_tag=np.asarray(["RUN_START", "RUN_END"]),
             marker_t_sent_pc=np.asarray([t0, t0 + 8.0]))
    return tmp_dir


def _streams_by_key(sd: Path) -> dict:
    return {s[0]: s for s in pd.load_parquet_streams(sd)}


def _probe(sd: Path):
    return pd._probe_session(sd, {}, types.SimpleNamespace(full=False))


def test_impedance_stream_single_row(tmp_path):
    _, ts, vals, names = _streams_by_key(
        _write_session(tmp_path, with_impedance=True))[pd.EEG_IMPEDANCE_KEY]
    assert len(ts) == 1 and ts[0] == pd.CONSTANT_TS
    assert vals.shape == (1, 5 + 3)
    assert names == _CHANNELS + _SUMMARY
    assert float(vals[0, _CHANNELS.index("C3")]) == 12345.0
    assert abs(float(vals[0, 5]) - 0.8) < 1e-6      # pass_rate
    assert float(vals[0, 6]) == 1.0                  # gate_pass
    assert float(vals[0, 7]) == 3.0                  # n_snapshots


def test_no_impedance_fields_no_stream(tmp_path):
    _write_session(tmp_path, with_impedance=False)
    assert pd.EEG_IMPEDANCE_KEY not in _streams_by_key(tmp_path)


def test_stream_index_matches_full_load(tmp_path):
    sd = _write_session(tmp_path, with_impedance=True)
    _, ts, names, width = {
        s[0]: s for s in pd.load_stream_index(sd)}[pd.EEG_IMPEDANCE_KEY]
    assert len(ts) == 1 and ts[0] == pd.CONSTANT_TS
    assert width == 8 and names == _CHANNELS + _SUMMARY
    sd_plain = _write_session(tmp_path / "plain", with_impedance=False)
    assert pd.EEG_IMPEDANCE_KEY not in {
        s[0] for s in pd.load_stream_index(sd_plain)}


def test_constant_stream_always_in_window(tmp_path):
    sd = _write_session(tmp_path, with_impedance=True)
    assert pd.EEG_IMPEDANCE_KEY in _probe(sd)["present"]
    assert "observation.eeg" in _probe(sd)["present"]
    sd_plain = _write_session(tmp_path / "plain", with_impedance=False)
    assert pd.EEG_IMPEDANCE_KEY not in _probe(sd_plain)["present"]


def test_impedance_exempt_from_modality_veto():
    """缺阻抗的会话不被"缺模态剔除"—— 附属流不参与并集否决。"""
    required = {pd.EEG_IMPEDANCE_KEY, "observation.eeg"}
    present_plain = {"observation.eeg"}
    assert sorted((required - present_plain) - {pd.EEG_IMPEDANCE_KEY}) == []
    assert sorted(required - present_plain) == [pd.EEG_IMPEDANCE_KEY]


def test_write_episode_one_row_at_zero(tmp_path):
    """写帧:常量流不过窗口、不平移,rel timestamp 固定 0 落一行。"""
    ml = types.ModuleType("mf_lerobot")
    mlu = types.ModuleType("mf_lerobot.utils")
    mlu.DEFAULT_VIDEO_PATH = ("videos/{episode_chunk:03d}/{video_key}/"
                              "episode_{episode_index:06d}.mp4")
    ml.utils = mlu
    ml.MultiFrequencyLeRobotDataset = object
    sys.modules.setdefault("mf_lerobot", ml)
    sys.modules.setdefault("mf_lerobot.utils", mlu)

    class _Meta:
        total_episodes = 0

        def update_video_info(self):
            pass

    class _Feat:
        def next_episode(self):
            pass

    class StubDS:
        def __init__(self):
            self.meta = _Meta()
            self._features = {k: _Feat() for k in
                              ("task", "observation.eeg", pd.EEG_IMPEDANCE_KEY)}
            self.frames = []

        def add_frame(self, key, v, t):
            self.frames.append((key, np.asarray(v), float(t)))

        def save_episode(self):
            pass

    sd = _write_session(tmp_path, with_impedance=True)
    info = _probe(sd)
    ds = StubDS()
    streams = [s for s in pd.load_parquet_streams(sd)
               if s[0] in info["present"]]
    pd.write_episode(ds, sd, "test", info["master"], streams,
                     info["win"], {})
    imp = [f for f in ds.frames if f[0] == pd.EEG_IMPEDANCE_KEY]
    assert len(imp) == 1 and imp[0][2] == 0.0 and imp[0][1].shape == (8,)
    eeg_rel = [t for k, _v, t in ds.frames if k == "observation.eeg"]
    assert len(eeg_rel) == 100 and all(0.0 <= t <= 8.0 + 1e-6 for t in eeg_rel)


def test_feature_spec_constant_rate(tmp_path):
    """spec:fps 取主轴帧率(避免 0),不参与滑窗。"""
    sd = _write_session(tmp_path, with_impedance=True)
    specs = pd.build_feature_specs([sd], {}, _probe(sd)["present"])
    spec = specs[pd.EEG_IMPEDANCE_KEY]
    assert spec["fps"] == pd.MASTER_FPS and spec["shape"] == (8,)
    assert spec["dtype"] == "float32" and "window" not in spec
    assert specs["observation.eeg"]["window"] == pd.WINDOW


@pytest.mark.parametrize("with_impedance", [True, False])
def test_probe_rejects_windowless_session(tmp_path, with_impedance):
    """无 marker 窗口的会话照旧拒绝(阻抗流不改变既有门槛)。"""
    sd = _write_session(tmp_path, with_impedance=with_impedance)
    (sd / "marker" / "marker.npz").unlink()
    assert _probe(sd) is None
