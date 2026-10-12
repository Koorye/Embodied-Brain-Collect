"""打包器的头环标定边车:meta/calibrations.jsonl 逐 episode 记录标定参数。

录制端把设备端逐字节下发、sha256 校验过的标定 YAML 保存在会话的
``ego_headband/calibrations/``。pack_daily 的 ``write_calibrations_meta``
把它们 YAML 解析成结构化参数随数据集带走;目录缺失记 null。
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_daily as pd  # noqa: E402

# 真实标定目录的三个文件(文件名 → YAML 正文 → 期望的解析结果)
_FILES = {
    "camera-camchain.yaml":
        ("cam0:\n  camera_model: pinhole\n  intrinsics: [1.0, 2.0, 3.0]\n",
         {"cam0": {"camera_model": "pinhole", "intrinsics": [1.0, 2.0, 3.0]}}),
    "camera_imu-camchain-imucam.yaml":
        ("imucam: placeholder\n", {"imucam": "placeholder"}),
    "camera_imu-imu.yaml":
        ("imu: placeholder\n", {"imu": "placeholder"}),
}


def _write_session(tmp_dir: Path, *, with_dir: bool = True, name="s0") -> Path:
    """最小会话:ego_headband/calibrations 下放标定 YAML + manifest(照实
    镜像真实会话目录;加载器不依赖 manifest)。"""
    sd = tmp_dir / name
    if with_dir:
        cal = sd / "ego_headband" / "calibrations"
        cal.mkdir(parents=True)
        for n, (body, _parsed) in _FILES.items():
            (cal / n).write_text(body, encoding="utf-8")
        (cal / "manifest.json").write_text("{}", encoding="utf-8")
    return sd


def test_load_parses_yaml_parameters(tmp_path):
    cal = pd.load_headband_calibration(_write_session(tmp_path))
    assert cal == {Path(n).stem: parsed
                   for n, (_body, parsed) in _FILES.items()}


def test_load_keeps_raw_text_when_yaml_unparsable(tmp_path):
    sd = _write_session(tmp_path)
    bad = sd / "ego_headband" / "calibrations" / "camera-camchain.yaml"
    bad.write_text("cam0: [unclosed\n", encoding="utf-8")
    cal = pd.load_headband_calibration(sd)
    assert cal["camera-camchain"] == "cam0: [unclosed\n"


def test_load_none_without_dir_or_when_empty(tmp_path):
    assert pd.load_headband_calibration(
        _write_session(tmp_path, with_dir=False)) is None
    empty = tmp_path / "empty" / "ego_headband" / "calibrations"
    empty.mkdir(parents=True)
    assert pd.load_headband_calibration(tmp_path / "empty") is None


def test_write_calibrations_meta_per_episode_lines(tmp_path):
    s0 = _write_session(tmp_path, name="a")
    s1 = _write_session(tmp_path, with_dir=False, name="b")   # 无标定 → null
    out = tmp_path / "ds"
    pd.write_calibrations_meta(out, [{"dir": s0}, {"dir": s1}])

    lines = (out / "meta" / "calibrations.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["episode_index"] == 0 and first["session"] == "a"
    assert first["calibrations"]["ego_headband"]["camera-camchain"] \
        == _FILES["camera-camchain.yaml"][1]
    assert second["episode_index"] == 1 and second["session"] == "b"
    assert second["calibrations"] == {"ego_headband": None}
