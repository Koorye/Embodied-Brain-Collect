"""Build the processed multimodal dataset under record/processed/.

For each valid session (has markers.npz) this writes:
  sub-subj01/<session_id>/
    aligned.h5      all modalities at native rate + unified master clock t_pc
    events.tsv      BIDS-style epoch index (trial x phase), no data copy
    sync_report.json  alignment fit + quality tier
    manifest.json   modality list + raw-file pointers

Master clock = PC unix seconds (t_pc).  EEG is mapped onto t_pc via the
hardware-trigger <-> marker linear fit (see record.session.aligner).  EEG raw
matrix and videos are *referenced* (not copied) by default to save space.

Dataset-level files (written once):
  dataset_info.json, task_library.json, marker_dictionary.json,
  quality_summary.csv

Usage:
  python -m record.tools.build_processed --limit 3          # smoke test
  python -m record.tools.build_processed                    # all valid sessions
  python -m record.tools.build_processed --only 2026-06-18
  python -m record.tools.build_processed --embed-eeg --copy-video   # self-contained
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py  # noqa: E402

from sync import marker_codes as M  # noqa: E402
from tools.curry_io import (  # noqa: E402
    decode_triggers, find_curry_acquisition_for_session, load_curry_eeg_meta)
from session.aligner import (  # noqa: E402
    _match_marker_sequence, _fit_eeg_to_pc)
from tools.analyze_sync import analyze_session, _session_dt  # noqa: E402

ACQUISITION_ROOT = Path(r"C:\Users\31454\Desktop\Acquisition")
SESSIONS_DIR = ROOT / "sessions"
PROCESSED_ROOT = ROOT / "processed"
PIPELINE_VERSION = "1.0"
H5_OPTS = dict(compression="gzip", compression_opts=4)

EEG_AUX_LABELS = ["VEOG", "HEOG", "EKG", "EMG", "Trigger"]


def _load_npz(p: Path):
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------
# EEG mapping (hardware trigger -> master clock)
# --------------------------------------------------------------------------

def _eeg_mapping(session_name: str, codes: np.ndarray, mpc: np.ndarray,
                 t_ep_ms: np.ndarray) -> dict:
    dt = _session_dt(session_name)
    if dt is None:
        return {"present": False, "reason": "bad name"}
    dpo = find_curry_acquisition_for_session(dt, ACQUISITION_ROOT)
    if dpo is None:
        return {"present": False, "reason": "no Curry match"}
    try:
        meta = load_curry_eeg_meta(dpo)
        _, runs = decode_triggers(dpo, min_duration_s=0.005)
        ecode = np.array([r["code"] for r in runs], dtype=np.int64)
        et = np.array([r["t_start_s"] for r in runs], dtype=np.float64)
        match = _match_marker_sequence(ecode, codes.astype(np.int32))
        if match is None:
            return {"present": True, "fitted": False, "reason": "no match",
                    "dpo": str(dpo), "meta": meta}
        ei, mi = match
        fit = _fit_eeg_to_pc(et[ei], mpc[mi])
        return {
            "present": True, "fitted": bool(fit.get("fitted")),
            "dpo": str(dpo), "meta": meta,
            "slope_pc_per_eeg": fit["slope_pc_per_eeg"],
            "intercept_s": fit["intercept_s_at_first_marker"],
            "eeg_t0_s": fit["eeg_t0_s"], "pc_t0_s": fit["pc_t0_s"],
            "resid_rms_ms": fit.get("resid_rms_ms"),
            "marker_eeg_idx": ei, "marker_idx": mi,
            "marker_eeg_t_s": et[ei],
        }
    except Exception as exc:
        return {"present": True, "fitted": False, "reason": f"error: {exc}"}


def _eeg_sample_t_pc(emap: dict, n_samples: int, fs: float) -> np.ndarray:
    t_eeg = np.arange(n_samples, dtype=np.float64) / fs
    return (emap["pc_t0_s"]
            + emap["slope_pc_per_eeg"] * (t_eeg - emap["eeg_t0_s"])
            + emap["intercept_s"])


# --------------------------------------------------------------------------
# aligned.h5
# --------------------------------------------------------------------------

def _ds(grp, name, arr, **kw):
    arr = np.asarray(arr)
    if arr.dtype.kind == "U":
        arr = arr.astype("S")
    # only chunk/compress arrays big enough to benefit
    if arr.size >= 256:
        grp.create_dataset(name, data=arr, **H5_OPTS, **kw)
    else:
        grp.create_dataset(name, data=arr, **kw)


def build_h5(sd: Path, out_h5: Path, sj: dict, emap: dict,
             embed_eeg: bool) -> dict:
    name = sd.name
    markers = _load_npz(sd / "markers.npz")
    codes = markers["code"].astype(np.int32)
    mpc = markers["t_pc_recv"].astype(np.float64)
    summary = {"groups": []}

    eye = _load_npz(sd / "eye" / "eye.npz")
    eye_off_s = (float(eye["pc_to_phone_offset_ms"]) / 1000.0
                 if eye and "pc_to_phone_offset_ms" in eye else 0.0)

    with h5py.File(out_h5, "w") as h5:
        a = h5.attrs
        a["session_id"] = name
        a["subject"] = sj.get("subject", "subj01")
        a["task_id"] = sj.get("task_id", -1)
        a["run"] = sj.get("run", -1)
        a["paradigm"] = str(sj.get("paradigm", "1"))
        a["created_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        a["pipeline_version"] = PIPELINE_VERSION
        a["master_clock"] = "pc_unix_s"
        a["pc_to_phone_offset_ms"] = eye_off_s * 1000.0

        # markers
        g = h5.create_group("markers")
        _ds(g, "code", codes)
        _ds(g, "tag", markers["tag"])
        _ds(g, "trial", markers["trial"].astype(np.int32))
        _ds(g, "t_pc", mpc)
        _ds(g, "t_eprime_ms", markers["t_eprime_ms"].astype(np.int64))
        t_eeg_per_marker = np.full(len(codes), np.nan)
        if emap.get("fitted"):
            t_eeg_per_marker[emap["marker_idx"]] = emap["marker_eeg_t_s"]
        _ds(g, "t_eeg_s", t_eeg_per_marker)
        summary["groups"].append("markers")

        # ---- EEG ----
        a["eeg_present"] = bool(emap.get("present"))
        a["eeg_fitted"] = bool(emap.get("fitted"))
        if emap.get("fitted"):
            meta = emap["meta"]
            fs = float(meta["sample_freq_hz"])
            n = int(meta["num_samples"])
            nch = int(meta["num_channels"])
            geeg = h5.create_group("eeg")
            geeg.attrs["fs_hz"] = fs
            geeg.attrs["unit"] = "uV"
            geeg.attrs["n_channels"] = nch
            geeg.attrs["n_eeg_channels"] = nch - 5
            geeg.attrs["aux_labels"] = np.array(EEG_AUX_LABELS, dtype="S")
            geeg.attrs["cdt_path"] = emap["dpo"].replace(".cdt.dpo", ".cdt")
            geeg.attrs["dpo_path"] = emap["dpo"]
            geeg.attrs["slope_pc_per_eeg"] = emap["slope_pc_per_eeg"]
            geeg.attrs["intercept_s"] = emap["intercept_s"]
            geeg.attrs["eeg_t0_s"] = emap["eeg_t0_s"]
            geeg.attrs["pc_t0_s"] = emap["pc_t0_s"]
            geeg.attrs["resid_rms_ms"] = emap.get("resid_rms_ms") or -1
            geeg.attrs["embedded"] = bool(embed_eeg)
            t_pc_eeg = _eeg_sample_t_pc(emap, n, fs)
            _ds(geeg, "t_pc", t_pc_eeg)
            a["eeg_slope_pc_per_eeg"] = emap["slope_pc_per_eeg"]
            a["eeg_resid_rms_ms"] = emap.get("resid_rms_ms") or -1
            if embed_eeg:
                data = np.memmap(meta["data_path"], dtype="<f4", mode="r",
                                 shape=(n, nch))
                geeg.create_dataset("data", shape=(n, nch), dtype="<f4",
                                    chunks=(min(n, 100000), nch), **H5_OPTS)
                step = 100000
                for i in range(0, n, step):
                    geeg["data"][i:i + step] = data[i:i + step]
            summary["groups"].append("eeg")

        # ---- eye ----
        if eye is not None:
            gaze_t0 = None
            if "gaze_timestamps" in eye and len(eye["gaze_timestamps"]):
                gz = eye["gaze_timestamps"].astype(np.float64)
                gaze_t0 = float(gz[0])
                g = h5.create_group("eye_gaze")
                g.attrs["fs_hz_nominal"] = 100.0
                g.attrs["clock_src"] = "neon_unix+offset"
                _ds(g, "t_pc", gz + eye_off_s)
                _ds(g, "xy", eye["gaze_xy"])
                if "gaze_worn" in eye:
                    _ds(g, "worn", eye["gaze_worn"])
                summary["groups"].append("eye_gaze")
            if "imu_timestamps" in eye and len(eye["imu_timestamps"]):
                # Neon IMU uses a different epoch base than gaze; re-anchor to the
                # gaze/unix clock by aligning starts (spans match within ~1s).
                im = eye["imu_timestamps"].astype(np.float64)
                imu_shift = (gaze_t0 - float(im[0])) if gaze_t0 is not None else 0.0
                g = h5.create_group("eye_imu")
                g.attrs["clock_src"] = "neon_imu_reanchored_to_gaze"
                g.attrs["imu_shift_s"] = imu_shift
                _ds(g, "t_pc", im + imu_shift + eye_off_s)
                for k in ("imu_gyro", "imu_accel", "imu_quat"):
                    if k in eye:
                        _ds(g, k.replace("imu_", ""), eye[k])
                summary["groups"].append("eye_imu")
            if "scene_timestamps" in eye and len(eye["scene_timestamps"]):
                g = h5.create_group("eye_scene")
                g.attrs["video"] = "eye/eye.mp4"
                g.attrs["fps"] = 30.0
                _ds(g, "t_pc", eye["scene_timestamps"].astype(np.float64) + eye_off_s)
                summary["groups"].append("eye_scene")

        # ---- emg ----
        emg = _load_npz(sd / "emg" / "emg.npz")
        if emg is not None and "emg_timestamps" in emg and len(emg["emg_timestamps"]):
            g = h5.create_group("emg")
            g.attrs["fs_hz_nominal"] = 2000.0
            g.attrs["unit"] = "uV"
            _ds(g, "t_pc", emg["emg_timestamps"].astype(np.float64))
            _ds(g, "data", emg["emg_data"])
            if "emg_sn" in emg:
                _ds(g, "sn", emg["emg_sn"])
            summary["groups"].append("emg")
            if "imu_timestamps" in emg and len(emg["imu_timestamps"]):
                g = h5.create_group("emg_imu")
                _ds(g, "t_pc", emg["imu_timestamps"].astype(np.float64))
                for k in ("imu_gyro", "imu_accel"):
                    if k in emg:
                        _ds(g, k.replace("imu_", ""), emg[k])
                summary["groups"].append("emg_imu")

        # ---- tactile ----
        tac = _load_npz(sd / "tactile" / "tactile.npz")
        if tac is not None:
            if "glove_timestamps" in tac and len(tac["glove_timestamps"]):
                g = h5.create_group("tactile_glove")
                g.attrs["fs_hz_nominal"] = 200.0
                _ds(g, "t_pc", tac["glove_timestamps"].astype(np.float64))
                _ds(g, "data", tac["glove_data"])
                if "glove_channel_names" in tac:
                    _ds(g, "channel_names", tac["glove_channel_names"])
                summary["groups"].append("tactile_glove")
            if "cam_timestamps" in tac and len(tac["cam_timestamps"]):
                g = h5.create_group("tactile_cam")
                g.attrs["video"] = "tactile/tactile_cam.mp4"
                g.attrs["fps"] = 30.0
                _ds(g, "t_pc", tac["cam_timestamps"].astype(np.float64))
                summary["groups"].append("tactile_cam")

        # ---- wrist_cam ----
        wc = _load_npz(sd / "wrist_cam" / "wrist_cam.npz")
        if wc is not None:
            g = h5.create_group("wrist_cam")
            g.attrs["video0"] = "wrist_cam/cam0.mp4"
            g.attrs["video1"] = "wrist_cam/cam1.mp4"
            g.attrs["fps"] = 30.0
            for k in ("cam0_timestamps", "cam1_timestamps"):
                if k in wc:
                    _ds(g, "t_pc_" + k.split("_")[0], wc[k].astype(np.float64))
            summary["groups"].append("wrist_cam")

        # ---- vive ----
        vive = _load_npz(sd / "vive" / "vive.npz")
        if vive is not None and "timestamps_s" in vive and len(vive["timestamps_s"]):
            g = h5.create_group("vive")
            g.attrs["fs_hz_nominal"] = 60.0
            _ds(g, "t_pc", vive["timestamps_s"].astype(np.float64))
            for k in ("positions_m", "quaternions_wxyz", "euler_rpy_deg", "valid"):
                if k in vive:
                    _ds(g, k, vive[k])
            if "serials" in vive:
                _ds(g, "serials", vive["serials"])
            summary["groups"].append("vive")

    return summary


# --------------------------------------------------------------------------
# events.tsv  (trial x phase epochs)
# --------------------------------------------------------------------------

PHASES = [
    ("baseline", "FIX_ON", "FIX_OFF"),
    ("instruction", "INSTR_ON", "INSTR_OFF"),
    ("imagery", "IMG_START", "IMG_END"),
    ("execution", "EXEC_START", "EXEC_END"),
]
MOD_TS_GROUPS = ["eeg", "eye_gaze", "eye_imu", "emg", "tactile_glove", "vive"]


def build_events(sd: Path, out_tsv: Path, task_lib: dict, out_h5: Path) -> int:
    markers = _load_npz(sd / "markers.npz")
    tags = [str(t) for t in markers["tag"]]
    trials = markers["trial"].astype(int)
    codes = markers["code"].astype(int)
    mpc = markers["t_pc_recv"].astype(np.float64)
    run_t0 = float(mpc[0])

    # gather per-modality t_pc arrays from the h5 for index lookup
    mod_t: dict[str, np.ndarray] = {}
    with h5py.File(out_h5, "r") as h5:
        for grp in MOD_TS_GROUPS:
            if grp in h5 and "t_pc" in h5[grp]:
                mod_t[grp] = h5[grp]["t_pc"][:]

    # index events by trial
    rows = []
    uniq_trials = sorted(set(int(t) for t in trials if int(t) >= 1))
    for tr in uniq_trials:
        idx = [i for i in range(len(tags)) if int(trials[i]) == tr]
        tagpos = {tags[i]: i for i in idx}
        task_id = None
        if "TASK_ID" in tagpos:
            task_id = int(codes[tagpos["TASK_ID"]]) - M.TASK_ID_BASE
        for phase, on_tag, off_tag in PHASES:
            if on_tag in tagpos and off_tag in tagpos:
                t_on = float(mpc[tagpos[on_tag]])
                t_off = float(mpc[tagpos[off_tag]])
                row = {
                    "onset": round(t_on - run_t0, 4),
                    "duration": round(t_off - t_on, 4),
                    "trial": tr,
                    "phase": phase,
                    "task_id": task_id if task_id is not None else "n/a",
                    "task_name": task_lib.get(task_id, "n/a"),
                    "onset_pc": round(t_on, 4),
                }
                for grp, ts in mod_t.items():
                    # guard: only emit indices when the phase window actually
                    # overlaps this stream's time range (else mis-clocked/empty)
                    if len(ts) == 0 or t_off < ts[0] or t_on > ts[-1]:
                        row[f"{grp}_idx0"] = "n/a"
                        row[f"{grp}_idx1"] = "n/a"
                    else:
                        row[f"{grp}_idx0"] = int(np.searchsorted(ts, t_on, "left"))
                        row[f"{grp}_idx1"] = int(np.searchsorted(ts, t_off, "right"))
                rows.append(row)

    if not rows:
        return 0
    cols = (["onset", "duration", "trial", "phase", "task_id", "task_name",
             "onset_pc"]
            + [f"{g}_idx{k}" for g in mod_t for k in (0, 1)])
    with open(out_tsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "n/a") for c in cols})
    return len(rows)


# --------------------------------------------------------------------------
# dataset-level metadata
# --------------------------------------------------------------------------

def write_dataset_meta(out_root: Path, sessions: list[Path]) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    # task library from master config
    cfg = json.loads((ROOT / "config" / "collection.json").read_text(encoding="utf-8"))
    task_lib = {int(t["task_id"]): t["task_name"] for t in cfg.get("tasks", [])}
    (out_root / "task_library.json").write_text(
        json.dumps(task_lib, ensure_ascii=False, indent=2), encoding="utf-8")

    # marker dictionary
    mdict = {
        "fixed_codes": {M.name_of(c): c for c in sorted(M.NAMED)},
        "task_id": {"base": M.TASK_ID_BASE, "formula": "code = 128 + task_id"},
        "scene_id": {"base": M.SCENE_ID_BASE, "formula": "code = 160 + scene"},
        "phases": {p[0]: [p[1], p[2]] for p in PHASES},
    }
    (out_root / "marker_dictionary.json").write_text(
        json.dumps(mdict, ensure_ascii=False, indent=2), encoding="utf-8")

    info = {
        "subject": cfg.get("subject", "subj01"),
        "paradigm": cfg.get("paradigm", "1"),
        "pipeline_version": PIPELINE_VERSION,
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "master_clock": "pc_unix_s",
        "eeg_alignment": "hardware-trigger linear fit to PC clock (~0.5ms vs eprime)",
        "n_sessions_processed": len(sessions),
        "n_tasks": len(task_lib),
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return task_lib


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--embed-eeg", action="store_true")
    ap.add_argument("--copy-video", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if aligned.h5 exists")
    args = ap.parse_args(argv)

    sess = sorted([p for p in SESSIONS_DIR.iterdir()
                   if p.is_dir() and "subj" in p.name
                   and (p / "markers.npz").is_file()])
    if args.only:
        sess = [p for p in sess if args.only in p.name]
    if args.limit:
        sess = sess[:args.limit]
    if not sess:
        print("[build] no valid sessions matched")
        return 1

    task_lib = write_dataset_meta(PROCESSED_ROOT, sess)
    import shutil
    quality_rows = []
    ok = skip = fail = 0
    t_start = time.time()
    for i, sd in enumerate(sess, 1):
        name = sd.name
        out_dir = PROCESSED_ROOT / "sub-subj01" / name
        out_h5 = out_dir / "aligned.h5"
        print(f"[{i}/{len(sess)}] {name}", flush=True)
        if out_h5.is_file() and not args.overwrite:
            skip += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            sj = {}
            if (sd / "session.json").is_file():
                sj = json.loads((sd / "session.json").read_text(encoding="utf-8"))
            markers = _load_npz(sd / "markers.npz")
            codes = markers["code"].astype(np.int32)
            mpc = markers["t_pc_recv"].astype(np.float64)
            t_ep = markers["t_eprime_ms"].astype(np.int64)

            emap = _eeg_mapping(name, codes, mpc, t_ep)
            h5_summary = build_h5(sd, out_h5, sj, emap, args.embed_eeg)
            n_epochs = build_events(sd, out_dir / "events.tsv", task_lib, out_h5)

            report = analyze_session(sd, with_eeg=True)
            (out_dir / "sync_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            quality_rows.append(report)

            manifest = {
                "session_id": name,
                "source_dir": str(sd),
                "groups": h5_summary["groups"],
                "n_epochs": n_epochs,
                "eeg_referenced": (not args.embed_eeg) and emap.get("fitted", False),
                "eeg_cdt": emap.get("dpo", "").replace(".cdt.dpo", ".cdt"),
                "videos": {},
            }
            for rel in ["eye/eye.mp4", "tactile/tactile_cam.mp4",
                        "wrist_cam/cam0.mp4", "wrist_cam/cam1.mp4"]:
                src = sd / rel
                if src.is_file():
                    if args.copy_video:
                        vdir = out_dir / "video"
                        vdir.mkdir(exist_ok=True)
                        dst = vdir / src.name
                        if not dst.is_file():
                            shutil.copy2(src, dst)
                        manifest["videos"][rel] = f"video/{src.name}"
                    else:
                        manifest["videos"][rel] = str(src)
            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[build] FAILED {name}: {exc}")
            fail += 1

    # quality summary csv: aggregate from ALL existing sync_report.json so the
    # table stays complete across partial/resumed runs.
    from tools.analyze_sync import CSV_COLS, _csv_row
    all_reports = sorted((PROCESSED_ROOT / "sub-subj01").glob("*/sync_report.json"))
    if all_reports:
        with open(PROCESSED_ROOT / "quality_summary.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS)
            w.writeheader()
            for rp in all_reports:
                try:
                    w.writerow(_csv_row(json.loads(rp.read_text(encoding="utf-8"))))
                except Exception:
                    pass

    print(f"\n[build] done in {time.time()-t_start:.1f}s  "
          f"ok={ok} skip={skip} fail={fail}  -> {PROCESSED_ROOT}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
