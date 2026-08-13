"""Batch synchronization + data-quality analysis across all sessions.

For every ``record/sessions/<SESSION>`` this computes, from the raw npz only
(fast) and optionally from the paired Curry EEG (--with-eeg, slow):

  * paradigm structure : n_trials, n_imagery, n_execution, task_ids/names
  * marker transport   : E-Prime->PC residual (jitter of t_pc_recv)
  * per-modality cover  : lead/tail vs marker window, max gap, early-dropout
  * EEG hardware sync   : trigger match-rate + residual after fit (the gold
                          metric; this is what the new master-clock relies on)
  * quality tier        : PASS / WARN / FAIL  + human-readable reasons

Outputs (under record/sessions/_reports/):
  sync_analysis_<stamp>.json   full per-session detail
  sync_analysis_<stamp>.csv    one row per session (open in Excel)

Usage:
  python -m record.tools.analyze_sync                 # fast, all sessions
  python -m record.tools.analyze_sync --with-eeg      # + EEG (slow)
  python -m record.tools.analyze_sync --with-eeg --only 2026-06-18
  python -m record.tools.analyze_sync --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ACQUISITION_ROOT = Path(
    os.environ.get(
        "RECORD_ACQUISITION_ROOT",
        r"C:\Users\31454\Desktop\Acquisition",
    )
)
SESSIONS_DIR = ROOT / "sessions"

from sync import marker_codes as M  # noqa: E402

# Continuous streams: (npz rel-path, timestamp key, is_phone_clock, nominal_hz)
MODALITY_TS = [
    ("eye",        "eye/eye.npz",             "gaze_timestamps", True,  200.0),
    ("eye_scene",  "eye/eye.npz",             "scene_timestamps", True, 30.0),
    ("tactile",    "tactile/tactile.npz",     "glove_timestamps", False, 200.0),
    ("tactile_cam","tactile/tactile.npz",     "cam_timestamps",  False, 30.0),
    ("wrist_cam",  "wrist_cam/wrist_cam.npz", "cam0_timestamps", False, 30.0),
    ("emg",        "emg/emg.npz",             "emg_timestamps",  False, 2000.0),
    ("vive",       "vive/vive.npz",           "timestamps_s",    False, 60.0),
]
# Which modalities count as "core" (their absence -> FAIL if they were enabled)
CORE = {"eye", "tactile", "emg", "vive"}


def _session_dt(name: str) -> datetime | None:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_", name)
    if not m:
        return None
    return datetime(*(int(x) for x in m.groups()))


def _fit_eprime_resid_ms(t_ep_ms: np.ndarray, t_pc: np.ndarray) -> float | None:
    valid = t_ep_ms > 0
    if valid.sum() < 2:
        return None
    x = t_ep_ms[valid].astype(np.float64) / 1000.0
    y = t_pc[valid].astype(np.float64)
    a, b = np.polyfit(x - x[0], y - y[0], 1)
    resid = (y - y[0]) - (a * (x - x[0]) + b)
    return float(np.abs(resid).max() * 1000.0)


def _coverage(ts: np.ndarray, t0: float, t1: float, nominal_hz: float) -> dict:
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    if n < 2:
        return {"n": n, "empty": True}
    dd = np.diff(ts)
    med = float(np.median(dd))
    nominal_dt = 1.0 / nominal_hz if nominal_hz else med
    # a "gap" = interval > 5x the larger of (median, nominal)
    thr = 5.0 * max(med, nominal_dt)
    return {
        "n": n,
        "fps": float((n - 1) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else 0.0,
        "lead_s": float(t0 - ts[0]),      # >0 = started before first marker (good)
        "tail_s": float(ts[-1] - t1),     # >0 = ended after last marker (good)
        "max_gap_ms": float(dd.max() * 1000.0),
        "n_gaps": int((dd > thr).sum()),
        "ends_early": bool(ts[-1] < t1 - 1.0),
        "starts_late": bool(ts[0] > t0 + 1.0),
        "covers_window": bool(ts[0] <= t0 + 1.0 and ts[-1] >= t1 - 1.0),
    }


def analyze_session(sd: Path, with_eeg: bool) -> dict:
    name = sd.name
    out: dict = {"session": name, "tier": "PASS", "reasons": []}
    mp = sd / "markers.npz"
    if not mp.is_file():
        out["tier"] = "FAIL"; out["reasons"].append("no markers.npz")
        return out
    try:
        m = np.load(mp, allow_pickle=True)
    except Exception as exc:
        out["tier"] = "FAIL"; out["reasons"].append(f"markers load error: {exc}")
        return out

    tags = [str(t) for t in m["tag"]]
    codes = m["code"].astype(np.int64)
    mpc = m["t_pc_recv"].astype(np.float64)
    t_ep = m["t_eprime_ms"].astype(np.int64)
    t0, t1 = float(mpc[0]), float(mpc[-1])

    # paradigm structure
    n_fix = tags.count("FIX_ON")
    n_img = tags.count("IMG_START")
    n_exec = tags.count("EXEC_START")
    task_ids = sorted({int(c) - M.TASK_ID_BASE
                       for t, c in zip(tags, codes) if t == "TASK_ID"})
    out.update({
        "n_markers": int(len(codes)),
        "n_trials": n_fix,
        "n_imagery": n_img,
        "n_execution": n_exec,
        "task_ids": task_ids,
        "n_distinct_tasks": len(task_ids),
        "window_dur_s": round(t1 - t0, 2),
        "eprime_pc_resid_ms": (lambda r: round(r, 2) if r is not None else None)(
            _fit_eprime_resid_ms(t_ep, mpc)),
    })

    # enabled modalities from session.json (fallback: detect from disk)
    enabled = None
    sj = sd / "session.json"
    if sj.is_file():
        try:
            enabled = set(json.loads(sj.read_text(encoding="utf-8"))
                          .get("enabled_modalities", []) or [])
        except Exception:
            enabled = None

    # per-modality coverage
    eye_off_s = 0.0
    eye_npz = sd / "eye" / "eye.npz"
    if eye_npz.is_file():
        try:
            ed = np.load(eye_npz, allow_pickle=True)
            if "pc_to_phone_offset_ms" in ed.files:
                eye_off_s = float(ed["pc_to_phone_offset_ms"]) / 1000.0
        except Exception:
            pass

    mods: dict = {}
    npz_cache: dict[str, dict] = {}
    for mod, rel, key, is_phone, hz in MODALITY_TS:
        p = sd / rel
        if not p.is_file():
            mods[mod] = {"present": False}
            base = rel.split("/")[0]
            if enabled and base in enabled and base in CORE:
                out["tier"] = "FAIL"
                out["reasons"].append(f"{base} enabled but npz missing")
            continue
        try:
            if rel not in npz_cache:
                npz_cache[rel] = np.load(p, allow_pickle=True)
            d = npz_cache[rel]
            if key not in d.files:
                mods[mod] = {"present": True, "no_key": key}
                continue
            ts = d[key].astype(np.float64)
            if is_phone:
                ts = ts + eye_off_s
            cov = _coverage(ts, t0, t1, hz)
            cov["present"] = True
            mods[mod] = cov
            base = rel.split("/")[0]
            if cov.get("ends_early") and base in CORE:
                _bump(out, "WARN", f"{mod} ends early (tail={cov['tail_s']:.0f}s)")
            if cov.get("max_gap_ms", 0) > 300 and base in CORE:
                _bump(out, "WARN", f"{mod} gap {cov['max_gap_ms']:.0f}ms")
        except Exception as exc:
            mods[mod] = {"present": True, "error": str(exc)}
    out["modalities"] = mods

    # paradigm sanity
    if n_fix == 0:
        _bump(out, "FAIL", "no trials (FIX_ON=0)")
    if n_exec == 0:
        _bump(out, "WARN", "no EXEC_START")

    # EEG hardware sync (slow, optional)
    if with_eeg:
        out["eeg"] = _eeg_sync(name, codes, mpc, t_ep)
        e = out["eeg"]
        if not e.get("present"):
            _bump(out, "WARN", "no paired EEG")
        elif not e.get("fitted"):
            _bump(out, "WARN", f"EEG fit failed: {e.get('reason', '')}")
        elif e.get("resid_rms_ms", 1e9) > 50:
            _bump(out, "WARN", f"EEG resid_rms {e['resid_rms_ms']:.0f}ms")
    return out


def _bump(out: dict, tier: str, reason: str) -> None:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    if order[tier] > order[out["tier"]]:
        out["tier"] = tier
    out["reasons"].append(reason)


def _resid_ms(x_s: np.ndarray, y_s: np.ndarray) -> dict:
    """Linear fit y~x, return rms/max residual in ms (jitter between two clocks)."""
    if len(x_s) < 2:
        return {"rms_ms": None, "max_ms": None}
    a, b = np.polyfit(x_s - x_s[0], y_s - y_s[0], 1)
    resid = (y_s - y_s[0]) - (a * (x_s - x_s[0]) + b)
    return {"rms_ms": round(float(np.sqrt(np.mean(resid ** 2)) * 1000), 3),
            "max_ms": round(float(np.abs(resid).max() * 1000), 3)}


def _eeg_sync(name: str, marker_codes: np.ndarray, mpc: np.ndarray,
              t_ep_ms: np.ndarray) -> dict:
    from tools.curry_io import (
        decode_triggers, find_curry_acquisition_for_session)
    from session.aligner import (
        _match_marker_sequence,
        _refine_marker_match_by_timing,
        _fit_eeg_to_pc,
    )

    dt = _session_dt(name)
    if dt is None:
        return {"present": False, "reason": "bad session name"}
    dpo = find_curry_acquisition_for_session(dt, ACQUISITION_ROOT)
    if dpo is None:
        return {"present": False, "reason": "no Curry match"}
    try:
        meta, runs = decode_triggers(dpo, min_duration_s=0.005)
        ecode = np.array([r["code"] for r in runs], dtype=np.int64)
        et = np.array([r["t_start_s"] for r in runs], dtype=np.float64)
        match = _match_marker_sequence(ecode, marker_codes.astype(np.int32))
        res = {
            "present": True, "dpo": dpo.name,
            "n_eeg_triggers": int(len(ecode)),
            "n_markers": int(len(marker_codes)),
        }
        if match is None:
            res.update({"fitted": False, "reason": "no code match"})
            return res
        ei, mi = match
        ep_all_s = t_ep_ms.astype(np.float64) / 1000.0
        ei, mi = _refine_marker_match_by_timing(et, ep_all_s, ei, mi)
        fit = _fit_eeg_to_pc(et[ei], mpc[mi], ep_all_s[mi])
        # The headline comparison:
        #   vs t_pc_recv  -> dominated by network/OS jitter (~30 ms)
        #   vs t_eprime_ms-> hardware-trigger vs stimulus clock (should be ~ms)
        ep_s = ep_all_s[mi]
        ep_valid = t_ep_ms[mi] > 0
        fit_inliers = np.asarray(
            fit.get("inlier_mask", np.ones(len(ei), dtype=bool)),
            dtype=bool,
        )
        ep_fit = ep_valid & fit_inliers
        vs_eprime = (_resid_ms(et[ei][ep_fit], ep_s[ep_fit])
                     if ep_fit.sum() >= 2 else {"rms_ms": None, "max_ms": None})
        res.update({
            "fitted": bool(fit.get("fitted")),
            "n_matched": int(len(ei)),
            "n_fit_inliers": int(fit_inliers.sum()),
            "n_fit_outliers": int((~fit_inliers).sum()),
            "match_rate": round(len(ei) / max(1, len(marker_codes)), 3),
            "slope_ppm": round((fit.get("slope_pc_per_eeg", 1.0) - 1.0) * 1e6, 1),
            "resid_max_ms": round(fit.get("resid_max_ms", -1), 3),
            "resid_rms_ms": round(fit.get("resid_rms_ms", -1), 3),
            "resid_vs_eprime_rms_ms": vs_eprime["rms_ms"],
            "resid_vs_eprime_max_ms": vs_eprime["max_ms"],
        })
        return res
    except Exception as exc:
        return {"present": True, "dpo": dpo.name, "fitted": False,
                "reason": f"error: {exc}"}


CSV_COLS = [
    "session", "tier", "n_trials", "n_imagery", "n_execution",
    "n_distinct_tasks", "window_dur_s", "eprime_pc_resid_ms",
    "eeg_match_rate", "eeg_resid_rms_ms", "eeg_vs_eprime_rms_ms", "eeg_slope_ppm",
    "eye_n", "tactile_n", "emg_n", "vive_n",
    "tactile_ends_early", "emg_max_gap_ms", "reasons",
]


def _csv_row(r: dict) -> dict:
    mods = r.get("modalities", {})
    eeg = r.get("eeg", {})

    def mn(k):
        return mods.get(k, {}).get("n", "")
    return {
        "session": r["session"], "tier": r["tier"],
        "n_trials": r.get("n_trials", ""), "n_imagery": r.get("n_imagery", ""),
        "n_execution": r.get("n_execution", ""),
        "n_distinct_tasks": r.get("n_distinct_tasks", ""),
        "window_dur_s": r.get("window_dur_s", ""),
        "eprime_pc_resid_ms": r.get("eprime_pc_resid_ms", ""),
        "eeg_match_rate": eeg.get("match_rate", ""),
        "eeg_resid_rms_ms": eeg.get("resid_rms_ms", ""),
        "eeg_vs_eprime_rms_ms": eeg.get("resid_vs_eprime_rms_ms", ""),
        "eeg_slope_ppm": eeg.get("slope_ppm", ""),
        "eye_n": mn("eye"), "tactile_n": mn("tactile"),
        "emg_n": mn("emg"), "vive_n": mn("vive"),
        "tactile_ends_early": mods.get("tactile", {}).get("ends_early", ""),
        "emg_max_gap_ms": mods.get("emg", {}).get("max_gap_ms", ""),
        "reasons": "; ".join(r.get("reasons", [])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-eeg", action="store_true",
                    help="also decode + fit paired Curry EEG (slow)")
    ap.add_argument("--only", default=None, help="substring filter on session name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=SESSIONS_DIR / "_reports")
    args = ap.parse_args(argv)

    sess = sorted([p for p in SESSIONS_DIR.iterdir()
                   if p.is_dir() and "subj" in p.name])
    if args.only:
        sess = [p for p in sess if args.only in p.name]
    if args.limit:
        sess = sess[:args.limit]
    if not sess:
        print("[analyze_sync] no sessions matched")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    t_start = time.time()
    for i, sd in enumerate(sess, 1):
        print(f"[{i}/{len(sess)}] {sd.name}", flush=True)
        try:
            rows.append(analyze_session(sd, args.with_eeg))
        except Exception as exc:
            rows.append({"session": sd.name, "tier": "FAIL",
                         "reasons": [f"analyzer crash: {exc}"]})

    json_path = args.out_dir / f"sync_analysis_{stamp}.json"
    csv_path = args.out_dir / f"sync_analysis_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(_csv_row(r))

    tiers = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in rows:
        tiers[r.get("tier", "FAIL")] = tiers.get(r.get("tier", "FAIL"), 0) + 1
    n_multi = sum(1 for r in rows if (r.get("n_trials") or 0) > 1)
    total_exec = sum(r.get("n_execution") or 0 for r in rows)
    print(f"\n[analyze_sync] {len(rows)} sessions in {time.time()-t_start:.1f}s")
    print(f"  tiers: PASS={tiers['PASS']}  WARN={tiers['WARN']}  FAIL={tiers['FAIL']}")
    print(f"  multi-trial sessions: {n_multi}   total executions: {total_exec}")
    print(f"  JSON -> {json_path}")
    print(f"  CSV  -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
