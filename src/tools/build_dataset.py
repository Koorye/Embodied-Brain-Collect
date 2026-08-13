"""Build a self-contained, LeRobot-style multimodal dataset under record/dataset/.

Organization (task -> many episodes), modeled on LeRobot v2.1 conventions but
with per-episode HDF5 for the multi-rate neuro/kinematic signals (EEG 1000Hz,
EMG 2000Hz, glove 200Hz, vive 60Hz, eye 100Hz) which a single-rate Parquet
table cannot represent. Videos are cut per episode into MP4 (stream-copy).

  record/dataset/
  ├── meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl,
  │         modalities.json, quality_report.json}
  ├── data/chunk-000/episode_000000.h5
  └── videos/chunk-000/observation.<cam>/episode_000000.mp4

An *episode* = one trial (baseline -> instruction -> imagery(MI) ->
execution(ME)); phase boundaries are stored as metadata. EEG is embedded
(sliced to the episode window from the raw Curry .cdt). Master clock = PC unix
seconds (t_pc); each stream also carries t_rel = t_pc - episode_onset.

Usage:
  python -m record.tools.build_dataset --limit-sessions 2        # smoke test
  python -m record.tools.build_dataset                           # full build
  python -m record.tools.build_dataset --no-video                # skip video cut
  python -m record.tools.build_dataset --only 2026-06-18
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import h5py  # noqa: E402

from sync import marker_codes as M  # noqa: E402
from tools.curry_io import (  # noqa: E402
    decode_triggers, find_curry_acquisition_for_session,
    load_curry_eeg_meta, open_curry_eeg_memmap)
from session.aligner import (  # noqa: E402
    _match_marker_sequence, _refine_marker_match_by_timing, _fit_eeg_to_pc)
from tools.analyze_sync import analyze_session, _session_dt  # noqa: E402

ACQ_ROOT = Path(
    os.environ.get(
        "RECORD_ACQUISITION_ROOT",
        r"C:\Users\31454\Desktop\Acquisition",
    )
)
SESSIONS_DIR = ROOT / "sessions"
DATASET_ROOT = ROOT / "dataset"
VERSION = "bci-multimodal-0.1"
CHUNK_SIZE = 1000
H5_OPTS = dict(compression="gzip", compression_opts=4)
EEG_AUX_LABELS = ["VEOG", "HEOG", "EKG", "EMG", "Trigger"]
EPISODE_MARGIN_S = 0.5  # pad before FIX_ON / after EXEC_END

PHASES = [
    ("baseline", "FIX_ON", "FIX_OFF"),
    ("instruction", "INSTR_ON", "INSTR_OFF"),
    ("imagery", "IMG_START", "IMG_END"),
    ("execution", "EXEC_START", "EXEC_END"),
]
CAMERAS = {
    "observation.eye": ("eye/eye.mp4", "eye_scene"),
    "observation.tactile_cam": ("tactile/tactile_cam.mp4", "tactile_cam"),
    "observation.wrist_cam0": ("wrist_cam/cam0.mp4", "wrist_cam0"),
    "observation.wrist_cam1": ("wrist_cam/cam1.mp4", "wrist_cam1"),
}


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _load_npz(p: Path):
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _chunk_dir(idx: int) -> str:
    return f"chunk-{idx // CHUNK_SIZE:03d}"


def _ds(grp, name, arr, **kw):
    arr = np.asarray(arr)
    if arr.dtype.kind == "U":
        arr = arr.astype("S")
    if arr.size >= 256:
        grp.create_dataset(name, data=arr, **H5_OPTS, **kw)
    else:
        grp.create_dataset(name, data=arr, **kw)


def _stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.shape[0]),
            "mean": round(float(np.nanmean(a)), 6),
            "std": round(float(np.nanstd(a)), 6),
            "min": round(float(np.nanmin(a)), 6),
            "max": round(float(np.nanmax(a)), 6)}


# --------------------------------------------------------------------------
# session-level resources (loaded once, reused across that session's episodes)
# --------------------------------------------------------------------------

class SessionRes:
    def __init__(self, sd: Path):
        self.sd = sd
        self.name = sd.name
        self.markers = _load_npz(sd / "markers.npz")
        self.eye = _load_npz(sd / "eye" / "eye.npz")
        self.emg = _load_npz(sd / "emg" / "emg.npz")
        self.tac = _load_npz(sd / "tactile" / "tactile.npz")
        self.wc = _load_npz(sd / "wrist_cam" / "wrist_cam.npz")
        self.vive = _load_npz(sd / "vive" / "vive.npz")
        self.eye_off = (float(self.eye["pc_to_phone_offset_ms"]) / 1000.0
                        if self.eye is not None
                        and "pc_to_phone_offset_ms" in self.eye else 0.0)
        # eye imu re-anchor shift (different epoch base than gaze)
        self.imu_shift = 0.0
        if self.eye is not None and "gaze_timestamps" in self.eye \
                and len(self.eye["gaze_timestamps"]) \
                and "imu_timestamps" in self.eye \
                and len(self.eye["imu_timestamps"]):
            self.imu_shift = (float(self.eye["gaze_timestamps"][0])
                              - float(self.eye["imu_timestamps"][0]))
        self.emap = self._eeg_mapping()
        self._eeg_mm = None

    def _eeg_mapping(self) -> dict:
        dt = _session_dt(self.name)
        if dt is None:
            return {"present": False}
        dpo = find_curry_acquisition_for_session(dt, ACQ_ROOT)
        if dpo is None:
            return {"present": False}
        try:
            meta = load_curry_eeg_meta(dpo)
            _, runs = decode_triggers(dpo, min_duration_s=0.005)
            ecode = np.array([r["code"] for r in runs], dtype=np.int64)
            et = np.array([r["t_start_s"] for r in runs], dtype=np.float64)
            codes = self.markers["code"].astype(np.int32)
            mpc = self.markers["t_pc_recv"].astype(np.float64)
            match = _match_marker_sequence(ecode, codes)
            if match is None:
                return {"present": True, "fitted": False, "dpo": str(dpo)}
            ei, mi = match
            ep_s = self.markers["t_eprime_ms"].astype(np.float64) / 1000.0
            ei, mi = _refine_marker_match_by_timing(et, ep_s, ei, mi)
            fit = _fit_eeg_to_pc(et[ei], mpc[mi], ep_s[mi])
            if not fit.get("fitted"):
                return {"present": True, "fitted": False, "dpo": str(dpo)}
            return {"present": True, "fitted": True, "dpo": str(dpo),
                    "meta": meta,
                    "slope": fit["slope_pc_per_eeg"],
                    "intercept": fit["intercept_s_at_first_marker"],
                    "eeg_t0": fit["eeg_t0_s"], "pc_t0": fit["pc_t0_s"],
                    "resid_rms_ms": fit.get("resid_rms_ms"),
                    "marker_eeg_idx": ei, "marker_idx": mi,
                    "marker_eeg_t_s": et[ei]}
        except Exception as exc:
            return {"present": True, "fitted": False, "err": str(exc)}

    def eeg_memmap(self):
        if self._eeg_mm is None:
            _, data = open_curry_eeg_memmap(Path(self.emap["dpo"]))
            self._eeg_mm = data
        return self._eeg_mm

    def t_pc_to_eeg_sample(self, t_pc: float) -> int:
        e = self.emap
        t_eeg = e["eeg_t0"] + (t_pc - e["pc_t0"] - e["intercept"]) / e["slope"]
        return int(round(t_eeg * e["meta"]["sample_freq_hz"]))


# --------------------------------------------------------------------------
# episode enumeration (pass 1, lightweight: markers only)
# --------------------------------------------------------------------------

def enumerate_episodes(sd: Path) -> list[dict]:
    m = _load_npz(sd / "markers.npz")
    if m is None or "code" not in m or len(m["code"]) < 2:
        return []
    tags = [str(t) for t in m["tag"]]
    trials = m["trial"].astype(int)
    codes = m["code"].astype(int)
    mpc = m["t_pc_recv"].astype(np.float64)
    dt = _session_dt(sd.name)
    eps = []
    for tr in sorted(set(int(t) for t in trials if int(t) >= 1)):
        idx = [i for i in range(len(tags)) if int(trials[i]) == tr]
        if not idx:
            continue
        pos = {tags[i]: i for i in idx}
        t_first = float(mpc[idx[0]])
        t_last = float(mpc[idx[-1]])
        onset = float(mpc[pos["FIX_ON"]]) if "FIX_ON" in pos else t_first
        offset = float(mpc[pos["EXEC_END"]]) if "EXEC_END" in pos else t_last
        if offset <= onset:
            offset = t_last
        task_id = (int(codes[pos["TASK_ID"]]) - M.TASK_ID_BASE
                   if "TASK_ID" in pos else None)
        phases = {}
        for ph, on, off in PHASES:
            if on in pos and off in pos:
                phases[ph] = [round(float(mpc[pos[on]]) - onset, 4),
                              round(float(mpc[pos[off]]) - onset, 4)]
        eps.append({
            "session": sd.name, "session_dt": dt, "trial": tr,
            "task_id": task_id,
            "win_on": onset - EPISODE_MARGIN_S,
            "win_off": offset + EPISODE_MARGIN_S,
            "onset": onset, "offset": offset,
            "phases": phases,
            "has_exec": "EXEC_START" in pos and "EXEC_END" in pos,
        })
    return eps


# --------------------------------------------------------------------------
# per-episode HDF5 (pass 2)
# --------------------------------------------------------------------------

def _slice(ts: np.ndarray, t0: float, t1: float):
    i0 = int(np.searchsorted(ts, t0, "left"))
    i1 = int(np.searchsorted(ts, t1, "right"))
    return i0, i1


def write_episode_h5(res: SessionRes, ep: dict, out_h5: Path,
                     embed_eeg: bool) -> dict:
    t0, t1 = ep["win_on"], ep["win_off"]
    onset = ep["onset"]
    cov = {}            # modality coverage flags
    stats = {}
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5:
        a = h5.attrs
        a["version"] = VERSION
        a["session_id"] = res.name
        a["trial"] = ep["trial"]
        a["task_id"] = ep["task_id"] if ep["task_id"] is not None else -1
        a["task_index"] = ep["task_index"]
        a["task_slug"] = ep["slug"]
        a["task_desc"] = ep["desc"]
        a["task_zh"] = ep["zh"]
        a["episode_index"] = ep["episode_index"]
        a["master_clock"] = "pc_unix_s"
        a["onset_pc"] = onset
        a["window_pc"] = [t0, t1]
        a["duration_s"] = round(t1 - t0, 4)
        a["created_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        for ph, (s, e) in ep["phases"].items():
            a[f"phase_{ph}"] = [s, e]

        # markers within window
        m = res.markers
        mpc = m["t_pc_recv"].astype(np.float64)
        mi0, mi1 = _slice(mpc, t0, t1)
        g = h5.create_group("markers")
        _ds(g, "code", m["code"][mi0:mi1].astype(np.int32))
        _ds(g, "tag", m["tag"][mi0:mi1])
        _ds(g, "trial", m["trial"][mi0:mi1].astype(np.int32))
        _ds(g, "t_pc", mpc[mi0:mi1])
        _ds(g, "t_rel", mpc[mi0:mi1] - onset)
        _ds(g, "t_eprime_ms", m["t_eprime_ms"][mi0:mi1].astype(np.int64))

        # EEG (sliced + embedded)
        e = res.emap
        if e.get("fitted"):
            fs = float(e["meta"]["sample_freq_hz"])
            nch = int(e["meta"]["num_channels"])
            ntot = int(e["meta"]["num_samples"])
            s0 = max(0, res.t_pc_to_eeg_sample(t0))
            s1 = min(ntot, res.t_pc_to_eeg_sample(t1))
            if s1 > s0:
                geeg = h5.create_group("eeg")
                geeg.attrs["fs_hz"] = fs
                geeg.attrs["unit"] = "uV"
                geeg.attrs["n_channels"] = nch
                geeg.attrs["n_eeg_channels"] = nch - 5
                names = [f"EEG_{i:03d}" for i in range(nch - 5)] + EEG_AUX_LABELS
                geeg.attrs["resid_rms_ms"] = e.get("resid_rms_ms") or -1
                geeg.attrs["sample_range"] = [s0, s1]
                samp = np.arange(s0, s1, dtype=np.float64)
                t_eeg = samp / fs
                t_pc_eeg = e["pc_t0"] + e["slope"] * (t_eeg - e["eeg_t0"]) + e["intercept"]
                _ds(geeg, "t_pc", t_pc_eeg)
                _ds(geeg, "t_rel", t_pc_eeg - onset)
                _ds(geeg, "channel_names", np.array(names, dtype="S"))
                if embed_eeg:
                    data = np.asarray(res.eeg_memmap()[s0:s1, :], dtype="<f4")
                    geeg.create_dataset("data", data=data,
                                        chunks=(min(data.shape[0], 50000), nch),
                                        **H5_OPTS)
                    stats["eeg"] = _stats(data[:, :nch - 5])
                cov["eeg"] = int(s1 - s0)

        # eye gaze / imu / scene
        eye = res.eye
        if eye is not None:
            if "gaze_timestamps" in eye and len(eye["gaze_timestamps"]):
                ts = eye["gaze_timestamps"].astype(np.float64) + res.eye_off
                i0, i1 = _slice(ts, t0, t1)
                if i1 > i0:
                    g = h5.create_group("eye_gaze")
                    g.attrs["fs_hz_nominal"] = 100.0
                    _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                    _ds(g, "xy", eye["gaze_xy"][i0:i1])
                    if "gaze_worn" in eye:
                        _ds(g, "worn", eye["gaze_worn"][i0:i1])
                    cov["eye_gaze"] = i1 - i0
                    stats["eye_gaze_xy"] = _stats(eye["gaze_xy"][i0:i1])
            if "imu_timestamps" in eye and len(eye["imu_timestamps"]):
                ts = eye["imu_timestamps"].astype(np.float64) + res.imu_shift + res.eye_off
                i0, i1 = _slice(ts, t0, t1)
                if i1 > i0:
                    g = h5.create_group("eye_imu")
                    _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                    for k in ("imu_gyro", "imu_accel", "imu_quat"):
                        if k in eye:
                            _ds(g, k.replace("imu_", ""), eye[k][i0:i1])
                    cov["eye_imu"] = i1 - i0

        # emg + emg imu
        emg = res.emg
        if emg is not None and "emg_timestamps" in emg and len(emg["emg_timestamps"]):
            ts = emg["emg_timestamps"].astype(np.float64)
            i0, i1 = _slice(ts, t0, t1)
            if i1 > i0:
                g = h5.create_group("emg")
                g.attrs["fs_hz_nominal"] = 2000.0
                g.attrs["unit"] = "uV"
                _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                _ds(g, "data", emg["emg_data"][i0:i1])
                if "emg_sn" in emg:
                    _ds(g, "sn", emg["emg_sn"][i0:i1])
                cov["emg"] = i1 - i0
                stats["emg"] = _stats(emg["emg_data"][i0:i1])
            if "imu_timestamps" in emg and len(emg["imu_timestamps"]):
                ts = emg["imu_timestamps"].astype(np.float64)
                i0, i1 = _slice(ts, t0, t1)
                if i1 > i0:
                    g = h5.create_group("emg_imu")
                    _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                    for k in ("imu_gyro", "imu_accel"):
                        if k in emg:
                            _ds(g, k.replace("imu_", ""), emg[k][i0:i1])
                    cov["emg_imu"] = i1 - i0

        # tactile glove
        tac = res.tac
        if tac is not None and "glove_timestamps" in tac and len(tac["glove_timestamps"]):
            ts = tac["glove_timestamps"].astype(np.float64)
            i0, i1 = _slice(ts, t0, t1)
            if i1 > i0:
                g = h5.create_group("tactile_glove")
                g.attrs["fs_hz_nominal"] = 200.0
                _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                _ds(g, "data", tac["glove_data"][i0:i1])
                if "glove_channel_names" in tac:
                    _ds(g, "channel_names", tac["glove_channel_names"])
                cov["tactile_glove"] = i1 - i0
                stats["tactile_glove"] = _stats(tac["glove_data"][i0:i1])

        # vive
        vive = res.vive
        if vive is not None and "timestamps_s" in vive and len(vive["timestamps_s"]):
            ts = vive["timestamps_s"].astype(np.float64)
            i0, i1 = _slice(ts, t0, t1)
            if i1 > i0:
                g = h5.create_group("vive")
                g.attrs["fs_hz_nominal"] = 60.0
                _ds(g, "t_pc", ts[i0:i1]); _ds(g, "t_rel", ts[i0:i1] - onset)
                for k in ("positions_m", "quaternions_wxyz", "euler_rpy_deg", "valid"):
                    if k in vive:
                        _ds(g, k, vive[k][i0:i1])
                if "serials" in vive:
                    _ds(g, "serials", vive["serials"])
                cov["vive"] = i1 - i0
                if "positions_m" in vive:
                    stats["vive_pos"] = _stats(vive["positions_m"][i0:i1])

        # camera frame timestamps (video files cut separately)
        gcam = h5.create_group("video_frames")
        for cam_key, (rel, grp) in CAMERAS.items():
            ts = _cam_timestamps(res, grp)
            if ts is None:
                continue
            i0, i1 = _slice(ts, t0, t1)
            if i1 > i0:
                cg = gcam.create_group(cam_key)
                cg.attrs["mp4"] = f"videos/{_chunk_dir(ep['episode_index'])}/{cam_key}/episode_{ep['episode_index']:06d}.mp4"
                cg.attrs["fps_nominal"] = 30.0
                _ds(cg, "t_pc", ts[i0:i1]); _ds(cg, "t_rel", ts[i0:i1] - onset)
                cov[cam_key] = i1 - i0

    return {"coverage": cov, "stats": stats}


def _cam_timestamps(res: SessionRes, grp: str):
    if grp == "eye_scene":
        if res.eye is not None and "scene_timestamps" in res.eye and len(res.eye["scene_timestamps"]):
            return res.eye["scene_timestamps"].astype(np.float64) + res.eye_off
    elif grp == "tactile_cam":
        if res.tac is not None and "cam_timestamps" in res.tac and len(res.tac["cam_timestamps"]):
            return res.tac["cam_timestamps"].astype(np.float64)
    elif grp == "wrist_cam0":
        if res.wc is not None and "cam0_timestamps" in res.wc and len(res.wc["cam0_timestamps"]):
            return res.wc["cam0_timestamps"].astype(np.float64)
    elif grp == "wrist_cam1":
        if res.wc is not None and "cam1_timestamps" in res.wc and len(res.wc["cam1_timestamps"]):
            return res.wc["cam1_timestamps"].astype(np.float64)
    return None


def cut_video(ffmpeg: str, src: Path, t_start_in_video: float, dur: float,
              out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    ss = max(0.0, t_start_in_video)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{ss:.3f}",
           "-i", str(src), "-t", f"{dur:.3f}", "-c", "copy",
           "-avoid_negative_ts", "make_zero", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and out.is_file() and out.stat().st_size > 1024:
        return True
    # fallback: re-encode
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{ss:.3f}",
           "-i", str(src), "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and out.is_file() and out.stat().st_size > 1024


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit-sessions", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-eeg", action="store_true", help="don't embed EEG matrix")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    # task english names
    ten = json.loads((ROOT / "config" / "task_names_en.json").read_text(encoding="utf-8"))["tasks"]
    cfg = json.loads((ROOT / "config" / "collection.json").read_text(encoding="utf-8"))
    zh = {int(t["task_id"]): t["task_name"] for t in cfg["tasks"]}

    sessions = sorted([p for p in SESSIONS_DIR.iterdir()
                       if p.is_dir() and "subj" in p.name
                       and (p / "markers.npz").is_file()])
    if args.only:
        sessions = [p for p in sessions if args.only in p.name]
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    if not sessions:
        print("[dataset] no sessions matched"); return 1

    # ---- pass 1: enumerate + globally order episodes ----
    print(f"[dataset] pass1: enumerating episodes from {len(sessions)} sessions")
    all_eps = []
    for sd in sessions:
        all_eps.extend(enumerate_episodes(sd))
    # order by task_id, then session time, then trial
    all_eps.sort(key=lambda e: (e["task_id"] if e["task_id"] is not None else 9999,
                                e["session_dt"] or datetime.min, e["trial"]))
    # assign task_index (dense over present task_ids) + episode_index
    present_tasks = sorted({e["task_id"] for e in all_eps if e["task_id"] is not None})
    task_index = {tid: i for i, tid in enumerate(present_tasks)}
    for ei, e in enumerate(all_eps):
        tid = e["task_id"]
        e["episode_index"] = ei
        e["task_index"] = task_index.get(tid, -1)
        info = ten.get(str(tid), {}) if tid is not None else {}
        e["slug"] = info.get("slug", f"task_{tid:03d}" if tid is not None else "unknown")
        e["desc"] = info.get("desc", zh.get(tid, "n/a"))
        e["zh"] = zh.get(tid, "n/a")
    print(f"[dataset] {len(all_eps)} episodes over {len(present_tasks)} tasks")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "meta").mkdir(exist_ok=True)
    ffmpeg = None if args.no_video else _ffmpeg()

    # ---- pass 2: build per session (load resources once) ----
    from collections import defaultdict
    by_session = defaultdict(list)
    for e in all_eps:
        by_session[e["session"]].append(e)

    ep_records = []
    ep_stats_records = []
    sync_by_session = {}
    t_start = time.time()
    done = 0
    sess_map = {s.name: s for s in sessions}
    for si, (sname, eps) in enumerate(by_session.items(), 1):
        sd = sess_map[sname]
        print(f"[sess {si}/{len(by_session)}] {sname}  ({len(eps)} eps)", flush=True)
        try:
            res = SessionRes(sd)
        except Exception as exc:
            print(f"  !! load failed: {exc}")
            continue
        try:
            sync_by_session[sname] = analyze_session(sd, with_eeg=True)
        except Exception:
            sync_by_session[sname] = {}
        for ep in eps:
            ci = _chunk_dir(ep["episode_index"])
            out_h5 = DATASET_ROOT / "data" / ci / f"episode_{ep['episode_index']:06d}.h5"
            if out_h5.is_file() and not args.overwrite:
                done += 1
                continue
            try:
                meta = write_episode_h5(res, ep, out_h5, embed_eeg=not args.no_eeg)
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  !! episode {ep['episode_index']} failed: {exc}")
                continue
            # videos
            vids = {}
            if ffmpeg is not None:
                for cam_key, (rel, grp) in CAMERAS.items():
                    src = sd / rel
                    ts = _cam_timestamps(res, grp)
                    if not src.is_file() or ts is None:
                        continue
                    i0, i1 = _slice(ts, ep["win_on"], ep["win_off"])
                    if i1 <= i0:
                        continue
                    out_mp4 = (DATASET_ROOT / "videos" / ci / cam_key
                               / f"episode_{ep['episode_index']:06d}.mp4")
                    t_start_in = float(ts[i0]) - float(ts[0])
                    dur = float(ts[i1 - 1]) - float(ts[i0]) + 0.05
                    if cut_video(ffmpeg, src, t_start_in, dur, out_mp4):
                        vids[cam_key] = str(out_mp4.relative_to(DATASET_ROOT)).replace("\\", "/")
            sy = sync_by_session.get(sname, {})
            rec = {
                "episode_index": ep["episode_index"],
                "task_index": ep["task_index"],
                "task_id": ep["task_id"],
                "tasks": [ep["desc"]],
                "task_slug": ep["slug"],
                "task_zh": ep["zh"],
                "source_session": sname,
                "trial": ep["trial"],
                "duration_s": round(ep["win_off"] - ep["win_on"], 3),
                "phases": ep["phases"],
                "has_execution": ep["has_exec"],
                "modalities": sorted(meta["coverage"].keys()),
                "coverage": meta["coverage"],
                "videos": vids,
                "quality_tier": sy.get("tier", "n/a"),
                "eeg_resid_rms_ms": (sy.get("eeg") or {}).get("resid_rms_ms"),
                "eeg_resid_vs_eprime_rms_ms": (sy.get("eeg") or {}).get("resid_vs_eprime_rms_ms"),
            }
            ep_records.append(rec)
            ep_stats_records.append({"episode_index": ep["episode_index"],
                                     "stats": meta["stats"]})
            done += 1
        # free big memmap between sessions
        res._eeg_mm = None

    # ---- meta files ----
    _write_meta(ten, zh, present_tasks, task_index, all_eps, ep_records,
                ep_stats_records, sync_by_session, args)
    print(f"\n[dataset] done in {time.time()-t_start:.1f}s  episodes={done}  -> {DATASET_ROOT}")
    return 0


def _write_meta(ten, zh, present_tasks, task_index, all_eps, ep_records,
                ep_stats_records, sync_by_session, args):
    meta = DATASET_ROOT / "meta"
    # tasks.jsonl
    ep_per_task = {}
    for e in all_eps:
        ep_per_task[e["task_index"]] = ep_per_task.get(e["task_index"], 0) + 1
    with open(meta / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for tid in present_tasks:
            ti = task_index[tid]
            info = ten.get(str(tid), {})
            fh.write(json.dumps({
                "task_index": ti, "task_id": tid,
                "task": info.get("desc", zh.get(tid, "n/a")),
                "task_zh": zh.get(tid, "n/a"),
                "slug": info.get("slug", f"task_{tid:03d}"),
                "n_episodes": ep_per_task.get(ti, 0),
            }, ensure_ascii=False) + "\n")

    # episodes.jsonl + episodes_stats.jsonl (sorted by episode_index)
    ep_records.sort(key=lambda r: r["episode_index"])
    with open(meta / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for r in ep_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ep_stats_records.sort(key=lambda r: r["episode_index"])
    with open(meta / "episodes_stats.jsonl", "w", encoding="utf-8") as fh:
        for r in ep_stats_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # modalities.json
    modalities = {
        "master_clock": "pc_unix_s (each stream also has t_rel = t_pc - episode onset)",
        "streams": {
            "eeg": {"fs_hz": 1000, "unit": "uV", "channels": "256 EEG + VEOG/HEOG/EKG/EMG + Trigger (261)", "storage": "embedded sliced f32"},
            "emg": {"fs_hz": 2000, "unit": "uV", "channels": 8},
            "emg_imu": {"fs_hz": "~110", "fields": "gyro(3), accel(3)"},
            "eye_gaze": {"fs_hz": 100, "fields": "xy(2), worn", "clock": "neon_unix+offset"},
            "eye_imu": {"fs_hz": "~100", "fields": "gyro(3), accel(3), quat(4)", "clock": "reanchored_to_gaze"},
            "tactile_glove": {"fs_hz": 200, "channels": 135},
            "vive": {"fs_hz": 60, "fields": "positions_m(3,3), quaternions_wxyz(3,4), euler_rpy_deg(3,3), valid(3)"},
            "video": {"fps": 30, "cameras": list(CAMERAS.keys()), "format": "mp4 per episode"},
        },
        "phases": {p[0]: [p[1], p[2]] for p in PHASES},
    }
    (meta / "modalities.json").write_text(json.dumps(modalities, ensure_ascii=False, indent=2), encoding="utf-8")

    # info.json (LeRobot-style)
    info = {
        "codebase_version": VERSION,
        "robot_type": "human_subj01_multimodal_bci",
        "format": "lerobot-style (task/episode) + per-episode HDF5 for multi-rate signals",
        "total_episodes": len(all_eps),
        "total_tasks": len(present_tasks),
        "chunks_size": CHUNK_SIZE,
        "fps": 30,
        "data_path": "data/chunk-{ep_chunk:03d}/episode_{episode_index:06d}.h5",
        "video_path": "videos/chunk-{ep_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "episode_definition": "one trial: baseline -> instruction -> imagery(MI) -> execution(ME)",
        "master_clock": "pc_unix_s",
        "eeg_alignment": "hardware-trigger linear fit to PC clock (sub-ms vs eprime)",
        "video_embedded": not args.no_video,
        "eeg_embedded": not args.no_eeg,
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # quality_report.json (dataset-wide analysis)
    _write_quality_report(meta, all_eps, ep_records, present_tasks, task_index,
                          ten, zh, sync_by_session)


def _write_quality_report(meta, all_eps, ep_records, present_tasks, task_index,
                          ten, zh, sync_by_session):
    n_ep = len(ep_records)
    expected_mods = ["eeg", "emg", "eye_gaze", "tactile_glove", "vive",
                     "observation.eye", "observation.tactile_cam",
                     "observation.wrist_cam0", "observation.wrist_cam1"]
    missing_counts = {m: 0 for m in expected_mods}
    no_video = no_eeg = no_exec = 0
    tiers = {}
    for r in ep_records:
        mods = set(r["modalities"])
        for m in expected_mods:
            if m not in mods:
                missing_counts[m] += 1
        if not r["videos"]:
            no_video += 1
        if "eeg" not in mods:
            no_eeg += 1
        if not r["has_execution"]:
            no_exec += 1
        tiers[r["quality_tier"]] = tiers.get(r["quality_tier"], 0) + 1

    # sync residuals across sessions
    eeg_pc, eeg_ep, ep_pc = [], [], []
    for sy in sync_by_session.values():
        eg = sy.get("eeg") or {}
        if eg.get("resid_rms_ms") is not None:
            eeg_pc.append(eg["resid_rms_ms"])
        if eg.get("resid_vs_eprime_rms_ms") is not None:
            eeg_ep.append(eg["resid_vs_eprime_rms_ms"])
        if sy.get("eprime_pc_resid_ms") is not None:
            ep_pc.append(sy["eprime_pc_resid_ms"])

    def summ(x):
        if not x:
            return None
        a = np.array(x, dtype=float)
        return {"n": len(x), "median": round(float(np.median(a)), 3),
                "p90": round(float(np.percentile(a, 90)), 3),
                "max": round(float(a.max()), 3)}

    # per-task episode counts
    per_task = []
    for tid in present_tasks:
        ti = task_index[tid]
        cnt = sum(1 for r in ep_records if r["task_index"] == ti)
        per_task.append({"task_index": ti, "task_id": tid,
                         "slug": ten.get(str(tid), {}).get("slug", ""),
                         "n_episodes": cnt})

    report = {
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_episodes": n_ep,
        "n_tasks": len(present_tasks),
        "quality_tiers": tiers,
        "episodes_without_video": no_video,
        "episodes_without_eeg": no_eeg,
        "episodes_without_execution": no_exec,
        "missing_modality_counts": missing_counts,
        "sync_eeg_vs_pc_rms_ms": summ(eeg_pc),
        "sync_eeg_vs_eprime_rms_ms": summ(eeg_ep),
        "sync_eprime_vs_pc_rms_ms": summ(ep_pc),
        "notes": [
            "eeg_vs_pc residual reflects PC network-receive jitter (~30ms); "
            "eeg_vs_eprime (hardware-trigger truth) is sub-ms and is the real EEG alignment quality.",
            "episodes_without_execution are imagery-only or aborted trials.",
        ],
        "per_task_episode_counts": per_task,
    }
    (meta / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
