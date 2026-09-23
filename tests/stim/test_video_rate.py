"""video_rate 范式 —— 码表段/trials 解析/第一帧现读(无硬件、无头)。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


import numpy as np  # noqa: E402
import pytest  # noqa: E402

from embodied_brain_collect.stim import marker_codes as M  # noqa: E402
from embodied_brain_collect.stim.paradigm_video_rate import (  # noqa: E402
    _read_first_frame, _resolve_trials)


# =============================================================================
# video_rate 码段(markers.yaml video_rate_base)
# =============================================================================

def test_vr_segment_enabled_and_capacity():
    assert M.VR_PHASE_BASE == 16
    assert M.MAX_VR_TRIALS == (M.RUN_START - 16 - M.MAX_REPLAY_MARKS) \
        // M.N_VR_PHASES
    assert M.MAX_VR_TRIALS >= 6          # 至少容得下默认一场
    assert M.REPLAY_MARK_BASE == M.VR_PHASE_BASE + M.MAX_VR_TRIALS * M.N_VR_PHASES
    assert M.REPLAY_MARK_BASE + M.MAX_REPLAY_MARKS <= M.RUN_START


def test_vr_codes_unique_across_whole_session():
    """整场所有 trial 的所有阶段码 + 重播打标码两两不同 —— EEG 按码配对的
    前提(markers.yaml 里 RUN_START=241 时上限 12 trial)。"""
    codes = [M.make_vr_code(t, p)
             for t in range(M.MAX_VR_TRIALS) for p in M.VR_PHASES]
    codes += [M.make_replay_mark_code(i) for i in range(M.MAX_REPLAY_MARKS)]
    assert len(codes) == len(set(codes))
    assert all(0 <= c <= 255 for c in codes)
    assert M.RUN_START not in codes and M.RUN_END not in codes


def test_vr_code_roundtrip_and_names():
    code = M.make_vr_code(5, "INSTR2_ON")
    assert M.parse_vr_code(code) == (5, "INSTR2_ON")
    assert M.name_of(code) == "VR_INSTR2_ON_T06"
    assert M.is_known(code)

    mark = M.make_replay_mark_code(47)
    assert M.parse_replay_mark_code(mark) == 47
    assert M.name_of(mark) == "REPLAY_MARK_48"
    assert M.is_known(mark)

    with pytest.raises(ValueError):
        M.make_vr_code(M.MAX_VR_TRIALS, "FIX_ON")
    with pytest.raises(ValueError):
        M.make_vr_code(0, "NO_SUCH_PHASE")
    with pytest.raises(ValueError):
        M.make_replay_mark_code(M.MAX_REPLAY_MARKS)


def test_named_codes_keep_display_priority():
    """命名码 > video_rate 段:重叠区的命名码显示名不变。
    (17 = 命名码 FIX_ON,数值上也是 base16 + 相位1 = T01 的 INSTR_ON。)"""
    assert M.name_of(M.RUN_START) == "RUN_START"
    assert M.name_of(17) == "FIX_ON"


# =============================================================================
# trials 解析(--trials-json > yaml trials > 单素材)
# =============================================================================

def _args(**kw) -> argparse.Namespace:
    defaults = dict(trials_json=None, n_trials=2, image="", video="",
                    instr1_text="i1", instr2_text="i2")
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_resolve_trials_from_cli_json(tmp_path):
    v1 = tmp_path / "fail_a.mp4"
    v2 = tmp_path / "success_b.mp4"
    v1.write_bytes(b"x")
    v2.write_bytes(b"x")
    js = ('[{"video": "%s"}, {"video": "%s", "image": ""}]'
          % (v1, v2))
    trials = _resolve_trials(_args(trials_json=js), {})
    assert [t["video"] for t in trials] == [str(v1), str(v2)]
    assert trials[0]["image"] is None        # 无 image → 第一帧现读
    assert trials[0]["instr1_text"] == "i1"


def test_resolve_trials_json_errors(tmp_path):
    with pytest.raises(SystemExit):
        _resolve_trials(_args(trials_json="{not json"), {})
    with pytest.raises(SystemExit):
        _resolve_trials(_args(trials_json="[]"), {})
    missing = tmp_path / "nope.mp4"
    with pytest.raises(SystemExit):
        _resolve_trials(_args(trials_json='[{"video": "%s"}]' % missing), {})
    # 缺 video 字段
    with pytest.raises(SystemExit):
        _resolve_trials(_args(trials_json='[{"image": "x.png"}]'), {})


def test_resolve_trials_yaml_list_wins_over_single(tmp_path):
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    v1.write_bytes(b"x")
    v2.write_bytes(b"x")
    over = {"trials": [{"video": str(v1)}, {"video": str(v2)}]}
    trials = _resolve_trials(_args(), over)   # 无 --trials-json → 用 yaml 列表
    assert len(trials) == 2
    assert trials[1]["video"] == str(v2)


def test_resolve_trials_single_material_requires_video():
    with pytest.raises(SystemExit):
        _resolve_trials(_args(n_trials=3), {})   # 没给 video


# =============================================================================
# 第一帧现读(不落地)
# =============================================================================

@pytest.fixture()
def tiny_video(tmp_path):
    """cv2 现写一个 3 帧小视频(MJPG/avi,无需编码器)。"""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30,
                        (8, 6))
    assert w.isOpened()
    for i in range(3):
        w.write(np.full((6, 8, 3), i * 40, dtype=np.uint8))
    w.release()
    return path


def test_read_first_frame_returns_rgb(tiny_video):
    frame = _read_first_frame(str(tiny_video))
    assert frame.shape == (6, 8, 3)
    assert frame[0, 0, 0] == 0               # 第一帧全 0(蓝=0 → R 通道 0)
    assert frame[0, 0, 2] == 0


def test_read_first_frame_missing_video(tmp_path):
    with pytest.raises(SystemExit):
        _read_first_frame(str(tmp_path / "nope.avi"))


# =============================================================================
# 解码后端:cv2 优先,ffmpeg 管道回退(AV1 等 cv2 解不动的编码)
# =============================================================================

def test_frame_source_cv2_backend(tiny_video):
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource
    src = _FrameSource(str(tiny_video))
    assert src.backend == "cv2"
    assert src.width == 8 and src.height == 6
    frames = list(src.frames())
    assert len(frames) == 3
    assert frames[0].shape == (6, 8, 3)


def test_frame_source_ffmpeg_fallback_when_cv2_fails(tiny_video,
                                                     monkeypatch):
    """模拟 cv2 读不出帧(AV1 的真实症状:isOpened=True 但 read() 恒 False)
    —— 应自动回退 ffmpeg 管道并解出帧。"""
    import cv2 as cv2_mod
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource

    class BrokenCap:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def get(self, *_):
            return 0.0

        def release(self):
            pass

    monkeypatch.setattr(cv2_mod, "VideoCapture",
                        lambda *a, **k: BrokenCap())
    src = _FrameSource(str(tiny_video))
    assert src.backend == "ffmpeg"
    frame = src.first_frame()
    assert frame is not None and frame.shape == (6, 8, 3)
    # frames() 可从头重放
    assert len(list(src.frames())) == 3


def test_frame_source_undecodable_file_raises(tmp_path):
    """损坏文件:cv2 与 ffmpeg 都读不出帧 → 明确报错而不是静默 0 帧。"""
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"not a video at all" * 10)
    with pytest.raises(SystemExit):
        _FrameSource(str(bad))


def _ffmpeg_can_encode_av1() -> bool:
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    out = subprocess.run([ff, "-hide_banner", "-encoders"],
                         capture_output=True, text=True)
    return "libsvtav1" in out.stdout


def _make_av1(tmp_path, n_frames=3):
    import shutil
    import subprocess
    path = tmp_path / "av1_clip.mp4"
    # SVT-AV1 要求宽高 ≥64
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=64x64:rate=10", "-frames:v", str(n_frames),
         "-c:v", "libsvtav1", "-y", str(path)],
        check=True, capture_output=True, timeout=60)
    return path


def test_frame_source_av1_clip(tmp_path):
    """真实 AV1 素材:无论本机 cv2 带不带 AV1 解码,都要能解出帧。"""
    import shutil
    if not shutil.which("ffmpeg") or not _ffmpeg_can_encode_av1():
        pytest.skip("本机无 ffmpeg/libsvtav1,无法生成 AV1 测试素材")
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource
    path = _make_av1(tmp_path)
    src = _FrameSource(str(path))
    assert src.backend in ("cv2", "ffmpeg")
    assert src.fps == pytest.approx(10.0, abs=0.1)
    assert len(list(src.frames())) == 3


def test_read_first_frame_on_repo_av1_asset():
    """用户实测报障的素材(AV1):存在时必须能读出第一帧(回退路径生效)。"""
    from embodied_brain_collect.stim.paradigm_video_rate import _FrameSource
    videos = sorted((Path(__file__).resolve().parents[2]
                     / "configs" / "videos").rglob("*.mp4"))
    if not videos:
        pytest.skip("configs/videos 里没有实测素材")
    src = _FrameSource(str(videos[0]))
    frame = src.first_frame()
    assert frame is not None and frame.ndim == 3
