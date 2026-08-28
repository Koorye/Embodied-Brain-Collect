"""Context plumbing, the EMG sequence check, and a whole-session run."""

from pathlib import Path

import numpy as np
import pytest

from embodied_brain_collect.checkers import checker_for, qc_session
from embodied_brain_collect.checkers.base import CheckContext
from embodied_brain_collect.checkers.camera import CameraChecker
from embodied_brain_collect.checkers.emg import EmgChecker, SnGap
from embodied_brain_collect.checkers.marker import MarkerChecker, find_run_window
from embodied_brain_collect.checkers.position import PositionChecker


# =============================================================================
# Context
# =============================================================================

def test_artifact_is_produced_once(tmp_path):
    """The whole point of the cache: three video checks, one decode."""
    calls = []
    c = CheckContext(stream="t", directory=tmp_path)
    for _ in range(3):
        got = c.artifact("decode", lambda: (calls.append(1), "OBJ")[1])
    assert len(calls) == 1 and got == "OBJ"


def test_missing_npz_field_returns_none(tmp_path):
    c = CheckContext(stream="t", directory=tmp_path)
    assert c.arr("anything") is None
    assert c.has_npz is False


def test_series_records_what_the_window_excluded(tmp_path):
    c = CheckContext(stream="t", directory=tmp_path,
                     window={"t0": 2.0, "t1": 4.0})
    c.add_series("s", loader=lambda: np.arange(0.0, 6.0, 1.0))
    s = c.series("s")
    assert s.n == 3 and s.n_outside == 3
    assert s.raw.size == 6          # video checks need the unclipped array


# =============================================================================
# Dispatch
# =============================================================================

@pytest.mark.parametrize("name,expected", [
    ("emg_left", EmgChecker), ("emg", EmgChecker),
    ("cam_head", CameraChecker), ("marker", MarkerChecker),
    ("position", PositionChecker), ("nonsense", None),
])
def test_dispatch_by_prefix(name, expected):
    assert checker_for(name) is expected


# =============================================================================
# EMG sequence numbers
# =============================================================================

def _sn_streams(n_emg_between_imu=2, n_frames=300, drop_at=None, drop_len=0):
    """Build emg_sn / imu_sn from one shared counter, as the wire does.

    The stream is trimmed to end on an EMG frame: the check measures the
    span between the first and last EMG frame, so a trailing IMU frame
    would be received-but-not-sent and skew the count by one.
    """
    emg, imu = [], []
    counter = 0
    for i in range(n_frames):
        if drop_at is not None and i == drop_at:
            counter = (counter + drop_len) % 256      # frames never delivered
        is_imu = i % (n_emg_between_imu + 1) == n_emg_between_imu
        (imu if is_imu else emg).append(counter % 256)
        counter = (counter + 1) % 256
    while imu and imu[-1] > emg[-1] % 256 and len(imu) > 1:
        imu.pop()
    return np.array(emg), np.array(imu)


def test_sngap_ignores_interleaved_imu_frames(tmp_path):
    """Regression: counting every step != 1 reported one gap per IMU frame."""
    emg, imu = _sn_streams()
    c = CheckContext(stream="emg", directory=tmp_path)
    c._arrays.update({"emg_sn": emg, "imu_sn": imu})
    out = SnGap().run(c)
    assert out.stats["frames_dropped"] == 0
    assert out.findings == []


def test_sngap_counts_real_losses(tmp_path):
    emg, imu = _sn_streams(drop_at=150, drop_len=20)
    c = CheckContext(stream="emg", directory=tmp_path)
    c._arrays.update({"emg_sn": emg, "imu_sn": imu})
    out = SnGap().run(c)
    assert out.stats["frames_dropped"] == 20
    assert out.findings[0].level == "WARN"


# =============================================================================
# Whole session (real fixtures)
# =============================================================================

@pytest.mark.slow
def test_session4_reproduces_known_findings(session4):
    report = qc_session(session4)
    # ERROR, driven by emg_right: its IMU stream has multi-second holes, and
    # a gap is unrecoverable rather than merely suspicious.
    assert report.level == "ERROR"
    assert report.streams["emg_right"].level == "ERROR"
    assert report.streams["cam_head"].level == "INFO"

    def finding(stream, check):
        return [f for f in report.streams[stream].findings if f.check == check]

    # Run window from the marker pair.
    assert report.window is not None
    assert report.window["t1"] - report.window["t0"] == pytest.approx(20.21, abs=0.01)
    assert report.streams["marker"].stats["markers"]["n_in_window"] == 12

    # EMG: the parse-corruption outliers, and REAL dropped frames rather
    # than one false gap per interleaved IMU frame.
    mad = finding("emg_left", "MadOutlier")[0]
    assert mad.subject == "ch2"          # the worst channel is named
    assert mad.detail["channels"]["ch2"] == pytest.approx(0.1269, abs=1e-3)
    assert report.streams["emg_left"].stats["SnGap"]["frames_dropped"] == 37
    assert report.streams["emg_right"].stats["SnGap"]["frames_dropped"] == 874

    # EEG: the sample count matches what the window and rate imply.
    wsc = report.streams["eeg"].stats["WindowSampleCount"]
    assert abs(wsc["deviation"]) < 0.01

    # Position: one tracker lost line of sight.  10093 consecutive samples
    # are invalid, but 90% of position timestamps are duplicates, so that is
    # 0.45 s of wall clock — not the 5.3 s that samples x median interval
    # used to report.
    gap = finding("position", "TrackingGap")[0]
    assert gap.subject == "61-BH3700181"
    assert gap.observed == pytest.approx(0.448, abs=0.01)

    # Cameras run at a steady 30 fps — no rate findings at all.
    assert report.streams["cam_third"].stats["Freeze"]["frozen_fraction"] \
        == pytest.approx(0.25, abs=0.01)
    assert not finding("cam_third", "SlidingWindowRate")


@pytest.mark.slow
def test_session4_report_is_json_serializable(session4):
    import json
    payload = json.dumps(qc_session(session4).to_dict(), ensure_ascii=False)
    assert '"level"' in payload and len(payload) > 1000
