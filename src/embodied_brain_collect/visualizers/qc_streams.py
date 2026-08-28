"""Per-modality extraction: what to draw for each kind of recorder.

One function per modality, chosen by directory prefix exactly as the checkers
are.  Each returns the rows to plot plus, for the camera-like streams, the
thumbnails to show — and each decides what is worth drawing rather than
dumping every channel, because 132 EEG traces or 50 skeleton joints on one
screen tell a reader nothing.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .qc_payload import (FRAME_FPS, FRAME_W, JPEG_Q, channel_rows, row,
                         series, timing_rows, xyz_rows)

TRACE_FPS = 2.0          # luminance / frame-difference sampling


# =============================================================================
# Video
# =============================================================================

@dataclass
class VideoPass:
    """One decode of one mp4, giving both the thumbnails and the traces."""

    thumbs: list = field(default_factory=list)     # {t, b64}
    t_samp: np.ndarray = None
    lums: np.ndarray = None
    diffs: np.ndarray = None
    n_frames: int = 0
    fps: float = 30.0


def scan_video(mp4: Path, ts_rel, *, want_times=(), frames: bool = True,
               fps: float = FRAME_FPS, thumb_w: int = FRAME_W,
               jpeg_q: int = JPEG_Q, gaze=None) -> VideoPass:
    """Decode once; sample luminance and frame difference; keep thumbnails.

    Thumbnails land on a fixed cadence AND at ``want_times`` — the moments the
    QC flagged.  A freeze or a black stretch is only diagnosable if you can
    see the frame it happened on, and the fixed cadence alone will miss it.

    ``gaze`` is ``(t_rel, xy)`` for the eye scene: the gaze point is drawn
    onto the thumbnail, which is the only place the two can be compared.
    """
    import cv2

    out = VideoPass()
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return out
    out.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    stride = max(1, int(round(out.fps / TRACE_FPS)))
    thumb_every = max(1, int(round(TRACE_FPS / max(fps, 1e-6))))

    ts = np.asarray(ts_rel, dtype=np.float64) if ts_rel is not None else None
    want = np.asarray(sorted(want_times), dtype=np.float64)
    g_t, g_xy = (gaze if gaze is not None else (None, None))

    lums, diffs, t_samp = [], [], []
    prev = None
    n = 0
    taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % stride:
            continue
        t = float(ts[n - 1]) if ts is not None and n - 1 < ts.size \
            else (n - 1) / out.fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lums.append(float(gray.mean()))
        diffs.append(float(np.abs(gray.astype(np.float32) - prev).mean())
                     if prev is not None else 0.0)
        t_samp.append(t)
        prev = gray

        if frames:
            near = want.size and bool(np.min(np.abs(want - t)) < 1.0)
            if taken % thumb_every == 0 or near:
                out.thumbs.append(_thumb(cv2, frame, t, thumb_w, jpeg_q,
                                         g_t, g_xy, flagged=near))
            taken += 1
    cap.release()

    out.n_frames = n
    out.t_samp = np.asarray(t_samp)
    out.lums = np.asarray(lums)
    out.diffs = np.asarray(diffs)
    return out


def _thumb(cv2, frame, t, thumb_w, jpeg_q, g_t, g_xy, *, flagged=False) -> dict:
    h = max(1, int(frame.shape[0] * thumb_w / frame.shape[1]))
    small = cv2.resize(frame, (thumb_w, h), interpolation=cv2.INTER_AREA)
    if g_t is not None and g_xy is not None and g_t.size:
        sel = np.abs(g_t - t) < 0.25
        if sel.any():
            k = thumb_w / frame.shape[1]
            pts = (g_xy[sel] * k).astype(np.int32)
            if pts.shape[0] > 1:
                cv2.polylines(small, [pts.reshape(-1, 1, 2)], False,
                              (90, 90, 90), 1)
            x, y = int(pts[-1][0]), int(pts[-1][1])
            cv2.circle(small, (x, y), 4, (30, 30, 30), -1)
            cv2.circle(small, (x, y), 3, (255, 210, 0), -1)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return {"t": round(t, 4), "flag": flagged,
            "b64": base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""}


# =============================================================================
# Per-modality rows
# =============================================================================

def _rel(z, key, t0):
    a = z.get(key)
    return None if a is None else np.asarray(a, dtype=np.float64).ravel() - t0


def _sampling_rate(z, family: str, t) -> float | None:
    """Sampling rate for *family*: an NPZ rate field when present (EEG),
    else the mean rate (samples / span) of the timestamps.

    The mean, not the median interval: rebuilt EMG timestamps are skewed
    (median 0.473 ms, mean 0.5 ms at a true 2000 Hz — per-packet steps of
    ~0.47 ms plus inter-packet jumps of ~0.95 ms).  The median estimates
    2113 Hz, and a filter designed at 2113 Hz applied to 2000 Hz data
    shifts every frequency by 5.7% — the 50 Hz mains notch lands on
    47.3 Hz and the mains sails straight through.
    """
    key = {"eeg": "eeg_sample_rate", "emg": "emg_sample_rate"}.get(family)
    if key:
        v = z.get(key)
        if v is not None and float(v) > 0:
            return float(v)
    if t is not None and t.size > 2:
        span = float(t[-1] - t[0])
        if span > 0:
            return (t.size - 1) / span
    return None


def _filtered_copy(z, data, family: str, t, opt) -> np.ndarray | None:
    """Filtered float64 copy of an (N, C) channel matrix, or None when
    filtering is off, scipy is missing, or the data cannot be filtered.

    Display only: the copy lives inside qc.html; the NPZ is never touched.
    The import is lazy so the page still builds on a scipy-less machine.
    """
    if not getattr(opt, "filter", True):
        return None
    from .signal_filter import apply_filter, preset_for, scipy_ok
    preset = preset_for(family, getattr(opt, "filter_presets", None))
    if preset is None or not scipy_ok():
        return None
    fs = _sampling_rate(z, family, t)
    if fs is None:
        return None
    return apply_filter(data, fs, preset)


def rows_camera(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t = _rel(z, "frames_timestamps", t0)
    rows = timing_rows(t, "frames") if t is not None else []
    mp4 = next(iter(sorted(d.glob("*.mp4"))), None)
    thumbs: list = []
    if mp4 is not None:
        v = scan_video(mp4, t, want_times=opt.want_times(d.name),
                       frames=opt.frames, fps=opt.fps,
                       thumb_w=opt.thumb_w, jpeg_q=opt.jpeg_q)
        thumbs = v.thumbs
        if v.t_samp is not None and v.t_samp.size > 2:
            r = row("画面亮度", [series(v.t_samp, v.lums, label="亮度", slot=1)],
                    h=52, src="frames")
            if r:
                rows.append(r)
            r = row("帧间差异", [series(v.t_samp, v.diffs, label="帧差", slot=2)],
                    h=52, src="frames")
            if r:
                rows.append(r)
    return rows, thumbs


def rows_emg(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t_emg = _rel(z, "emg_timestamps", t0)
    t_imu = _rel(z, "imu_timestamps", t0)

    rows = timing_rows(t_emg, "emg") if t_emg is not None else []
    rows += timing_rows(t_imu, "imu") if t_imu is not None else []

    data = z.get("emg_data")
    if t_emg is not None and data is not None and len(data) == t_emg.size:
        data = np.asarray(data, dtype=np.float64)
        data_f = _filtered_copy(z, data, "emg", t_emg, opt)
        # One thin row per channel rather than eight overlaid lines: a
        # single series per row needs no colour to carry identity, and a
        # dead or corrupt channel is obvious against its neighbours.
        # uniform_ts asks series() to ship the x axis as a stride where the
        # stamps are truly uniform; it self-checks and falls back to the
        # full array otherwise (e.g. EMG frames interleaved with IMU frames
        # step by one or two shared clock ticks).
        for c in range(data.shape[1]):
            r = row(f"emg ch{c}",
                    [series(t_emg, data[:, c], label=f"ch{c}", slot=1,
                            uniform_ts=True,
                            y_f=None if data_f is None else data_f[:, c])],
                    h=26,
                    src="emg")
            if r:
                rows.append(r)
    if t_imu is not None:
        for key, label, unit in (("imu_gyro", "IMU 陀螺", "rad/s"),
                                 ("imu_accel", "IMU 加速度", "m/s²")):
            a = z.get(key)
            if a is not None and len(a) == t_imu.size:
                rows += xyz_rows(label, t_imu, a, unit=unit, src="imu")
    return rows, []


def rows_eeg(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t = _rel(z, "eeg_timestamps_pc", t0)
    rows = timing_rows(t, "eeg") if t is not None else []
    data = z.get("eeg_data")
    if t is None or data is None or len(data) != t.size:
        return rows, []
    data = np.asarray(data, dtype=np.float64)
    names = z.get("eeg_channel_names")
    names = [str(x) for x in names] if names is not None else None

    # The trigger channel is an event code, not a signal — plotting it in the
    # same heatmap would swamp the scale.
    trig = None
    if names and names[-1].lower().startswith("trig"):
        trig, data, names = data[:, -1], data[:, :-1], names[:-1]

    # Filtered display copy for the signal channels only; the trigger row
    # below stays raw (filtering an event code would smear its edges).
    data_f = _filtered_copy(z, data, "eeg", t, opt)
    rows += channel_rows(t, data, names=names, unit="µV", src="eeg",
                         data_f=data_f)
    if trig is not None:
        r = row("Trigger", [series(t, trig, label="trigger", slot=4)], h=44,
                src="eeg")
        if r:
            rows.append(r)
    return rows, []


def rows_eye(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t_gaze = _rel(z, "gaze_timestamps", t0)
    t_imu = _rel(z, "imu_timestamps", t0)
    t_scene = _rel(z, "scene_timestamps", t0)
    rows = timing_rows(t_gaze, "gaze") if t_gaze is not None else []
    rows += timing_rows(t_imu, "imu") if t_imu is not None else []

    xy = z.get("gaze_xy")
    if t_gaze is not None and xy is not None and len(xy) == t_gaze.size:
        xy = np.asarray(xy, dtype=np.float64)
        r = row("注视点", [series(t_gaze, xy[:, 0], label="x", slot=1, unit="px"),
                        series(t_gaze, xy[:, 1], label="y", slot=2, unit="px")],
                h=72, unit="px", src="gaze")
        if r:
            rows.append(r)
    if t_imu is not None:
        for key, label, unit in (("imu_gyro", "IMU 陀螺", "rad/s"),
                                 ("imu_accel", "IMU 加速度", "g")):
            a = z.get(key)
            if a is not None and len(a) == t_imu.size:
                rows += xyz_rows(label, t_imu, a, unit=unit, src="imu")

    thumbs: list = []
    mp4 = next(iter(sorted(d.glob("*.mp4"))), None)
    if mp4 is not None and t_scene is not None:
        gaze = (t_gaze, np.asarray(xy, dtype=np.float64)) \
            if (t_gaze is not None and xy is not None) else None
        v = scan_video(mp4, t_scene, want_times=opt.want_times(d.name),
                       frames=opt.frames, fps=opt.fps, thumb_w=opt.thumb_w,
                       jpeg_q=opt.jpeg_q, gaze=gaze)
        thumbs = v.thumbs
        if v.t_samp is not None and v.t_samp.size > 2:
            r = row("画面亮度", [series(v.t_samp, v.lums, label="亮度", slot=1)],
                    h=48, src="scene")
            if r:
                rows.append(r)
    return rows, thumbs


def rows_hand_pose(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t = _rel(z, "ergo_timestamps", t0)
    if t is None:
        t = _rel(z, "skeleton_timestamps", t0)
    rows = timing_rows(t, "samples") if t is not None else []
    if t is None:
        return rows, []

    ergo = z.get("ergo_data")
    if ergo is not None and len(ergo) == t.size:
        rows += channel_rows(t, np.asarray(ergo, dtype=np.float64),
                             prefix="ergo", unit="°", src="samples")

    skel = z.get("skeleton_positions")
    if skel is not None and len(skel) == t.size:
        skel = np.asarray(skel, dtype=np.float64)
        # 50 joints cannot all be drawn; the centroid says where the hand is
        # and the mean node speed says whether it is moving at all.
        with np.errstate(invalid="ignore"):
            centroid = np.nanmean(skel, axis=1)
            speed = np.nanmean(
                np.linalg.norm(np.diff(skel, axis=0), axis=-1), axis=1)
        rows += xyz_rows("骨架质心", t, centroid, unit="m", src="samples")
        r = row("平均节点速度",
                [series(t[1:], speed, label="speed", slot=3, unit="m/样本")],
                h=52, src="samples")
        if r:
            rows.append(r)
    return rows, []


def rows_position(z, d: Path, t0: float, opt) -> tuple[list, list]:
    t = _rel(z, "timestamps_s", t0)
    rows = timing_rows(t, "combined") if t is not None else []
    pos = z.get("positions_m")
    if t is None or pos is None:
        return rows, []
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[:, None, :]
    valid = z.get("valid")
    if valid is None:
        valid = ~np.isnan(pos).any(axis=-1)
    else:
        valid = np.asarray(valid)
        if valid.ndim == 1:
            valid = valid[:, None]
    ser = z.get("serials")
    names = ([str(s) for s in np.asarray(ser).ravel()] if ser is not None
             else [f"dev{i}" for i in range(pos.shape[1])])

    for i, name in enumerate(names):
        if i >= pos.shape[1]:
            break
        # Blank the invalid samples so a tracking dropout is a break in the
        # line rather than a straight segment across it.
        xyz = np.where(valid[:, i, None], pos[:, i, :], np.nan)
        rows += xyz_rows(name, t, xyz, unit="m", h=72, src=name)
    return rows, []


def rows_marker(z, d: Path, t0: float, opt) -> tuple[list, list]:
    return [], []          # markers are drawn on every card, not in one


EXTRACTORS: tuple[tuple[str, object], ...] = (
    ("emg", rows_emg),
    ("eeg", rows_eeg),
    ("eye", rows_eye),
    ("hand_pose", rows_hand_pose),
    ("position", rows_position),
    ("marker", rows_marker),
    ("cam", rows_camera),
)


def extractor_for(name: str):
    for prefix, fn in EXTRACTORS:
        if name.startswith(prefix):
            return fn
    return None
