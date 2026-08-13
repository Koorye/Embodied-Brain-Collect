"""Resample an aligned session onto 30/60/120/900 Hz PC-time grids.

Requires:
  <session>/aligned/aligned.npz
  <session>/aligned/align_report.json
  --eeg-path pointing at the paired Curry .cdt.dpo (for 900 Hz EEG)

Outputs:
  <session>/aligned/resampled/
    grid_30hz.npz   scene/cam timestamps + nearest-sample indices
    grid_60hz.npz   vive poses
    grid_120hz.npz  gaze, IMU, glove, EMG, tactile glove
    grid_900hz.npz  EEG (Curry, PC-clock mapped)
    resample_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.curry_io import load_curry_eeg_meta


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def _pc_span(aligned: dict[str, np.ndarray]) -> tuple[float, float]:
    """Use marker PC clock span only (avoid bad/outlier modality clocks)."""
    mt = aligned["marker_t_pc"].astype(np.float64)
    margin_s = 30.0
    return float(mt[0]) - margin_s, float(mt[-1]) + margin_s


def _make_grid(t0: float, t1: float, hz: float) -> np.ndarray:
    start = np.ceil(t0 * hz) / hz
    n = int(np.floor((t1 - start) * hz)) + 1
    if n < 1:
        return np.array([start], dtype=np.float64)
    return start + np.arange(n, dtype=np.float64) / hz


def _interp_rows(ts: np.ndarray, values: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    if len(ts) < 2:
        tail = arr.shape[1:]
        out = np.full((len(grid_t),) + tail, np.nan, dtype=np.float64)
        if arr.size:
            out[:] = arr[0]
        return out
    if arr.ndim == 1:
        return np.interp(grid_t, ts, arr.astype(np.float64))
    # Flatten trailing dims so (T,D,3) Vive poses interpolate cleanly.
    n0 = arr.shape[0]
    flat = arr.reshape(n0, -1).astype(np.float64)
    out = np.empty((len(grid_t), flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(grid_t, ts, flat[:, c])
    return out.reshape((len(grid_t),) + arr.shape[1:])


def _nearest_indices(ts: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """For video: index of last sample at or before each grid time."""
    idx = np.searchsorted(ts, grid_t, side="right") - 1
    return np.clip(idx, 0, len(ts) - 1).astype(np.int64)


def _eeg_pc_timestamps(fit: dict, n_samples: int, fs: float) -> np.ndarray:
    x0 = float(fit["eeg_t0_s"])
    y0 = float(fit["pc_t0_s"])
    a = float(fit["slope_pc_per_eeg"])
    b = float(fit["intercept_s_at_first_marker"])
    t_eeg = np.arange(n_samples, dtype=np.float64) / fs
    return y0 + a * (t_eeg - x0) + b


def _resample_eeg_900(
    eeg_path: Path,
    fit: dict,
    t0: float,
    t1: float,
    hz: float = 900.0,
) -> dict[str, np.ndarray]:
    """Resample EEG to 900 Hz on PC clock; processes channels one-by-one to limit RAM."""
    meta = load_curry_eeg_meta(eeg_path)
    data = np.memmap(
        meta["data_path"], dtype="<f4", mode="r",
        shape=(meta["num_samples"], meta["num_channels"]),
    )
    fs = float(meta["sample_freq_hz"])
    n_eeg = meta["eeg_channels"]
    t_pc = _eeg_pc_timestamps(fit, meta["num_samples"], fs)

    i0 = int(np.searchsorted(t_pc, t0, side="left"))
    i1 = int(np.searchsorted(t_pc, t1, side="right"))
    i0 = max(0, i0)
    i1 = max(i0 + 2, min(i1, meta["num_samples"]))

    t_seg = t_pc[i0:i1]
    grid_t = _make_grid(t0, t1, hz)
    print(f"[resample] EEG {i1 - i0} samples x {n_eeg} ch -> {len(grid_t)} @ {hz} Hz",
          flush=True)
    out = np.empty((len(grid_t), n_eeg), dtype=np.float32)
    for c in range(n_eeg):
        out[:, c] = np.interp(
            grid_t, t_seg, data[i0:i1, c].astype(np.float64)
        ).astype(np.float32)
        if (c + 1) % 64 == 0:
            print(f"[resample]   EEG channel {c + 1}/{n_eeg}", flush=True)
    return {
        "timestamps_pc": grid_t,
        "data_uV": out,
        "native_fs_hz": np.float64(fs),
        "target_hz": np.float64(hz),
        "eeg_channels": np.int32(n_eeg),
    }


def resample_session(session_dir: Path, eeg_path: Path | None, *, skip_eeg: bool = False) -> dict:
    aligned_dir = session_dir / "aligned"
    aligned_npz = aligned_dir / "aligned.npz"
    report_path = aligned_dir / "align_report.json"
    if not aligned_npz.is_file():
        raise FileNotFoundError(f"missing {aligned_npz}")
    if not report_path.is_file():
        raise FileNotFoundError(f"missing {report_path}")

    aligned = _load_npz(aligned_npz)
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    t0, t1 = _pc_span(aligned)
    out_dir = aligned_dir / "resampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"session_dir": str(session_dir), "t0_pc": t0, "t1_pc": t1, "grids": {}}

    # --- 30 Hz: video-related timestamps (indices for offline frame lookup) ---
    g30 = _make_grid(t0, t1, 30)
    pack30: dict[str, np.ndarray] = {"timestamps_pc": g30, "target_hz": np.int32(30)}
    for key, out_key in (
        ("eye_scene_timestamps_pc", "eye_scene_index"),
        ("tactile_cam_timestamps", "tactile_cam_index"),
        ("wrist_cam0_timestamps", "wrist_cam0_index"),
        ("wrist_cam1_timestamps", "wrist_cam1_index"),
    ):
        if key in aligned and len(aligned[key]) > 1:
            pack30[out_key] = _nearest_indices(aligned[key].astype(np.float64), g30)
    np.savez_compressed(out_dir / "grid_30hz.npz", **pack30)
    summary["grids"]["30"] = {"n": int(len(g30)), "keys": sorted(pack30.keys())}
    print(f"[resample] wrote grid_30hz ({len(g30)} samples)", flush=True)

    # --- 60 Hz: Vive ---
    g60 = _make_grid(t0, t1, 60)
    pack60: dict[str, np.ndarray] = {"timestamps_pc": g60, "target_hz": np.int32(60)}
    if "vive_timestamps_s" in aligned and len(aligned["vive_timestamps_s"]) > 1:
        ts = aligned["vive_timestamps_s"].astype(np.float64)
        for k in ("positions_m", "quaternions_wxyz", "euler_rpy_deg", "valid"):
            fk = f"vive_{k}"
            if fk in aligned:
                pack60[k] = _interp_rows(ts, aligned[fk], g60)
    np.savez_compressed(out_dir / "grid_60hz.npz", **pack60)
    summary["grids"]["60"] = {"n": int(len(g60)), "keys": sorted(pack60.keys())}
    print(f"[resample] wrote grid_60hz ({len(g60)} samples)", flush=True)

    # --- 120 Hz: gaze / IMU / glove / EMG ---
    g120 = _make_grid(t0, t1, 120)
    pack120: dict[str, np.ndarray] = {"timestamps_pc": g120, "target_hz": np.int32(120)}
    streams_120 = [
        ("eye_gaze_timestamps_pc", "eye_gaze_xy", "gaze_xy"),
        ("eye_imu_timestamps_pc", "eye_imu_gyro", "imu_gyro"),
        ("eye_imu_timestamps_pc", "eye_imu_accel", "imu_accel"),
        ("tactile_glove_timestamps", "tactile_glove_data", "tactile_glove"),
        ("emg_emg_timestamps", "emg_emg_data", "emg"),
        ("emg_imu_timestamps", "emg_imu_gyro", "emg_imu_gyro"),
        ("emg_imu_timestamps", "emg_imu_accel", "emg_imu_accel"),
    ]
    for ts_key, val_key, out_prefix in streams_120:
        if ts_key in aligned and val_key in aligned and len(aligned[ts_key]) > 1:
            pack120[out_prefix] = _interp_rows(
                aligned[ts_key].astype(np.float64),
                aligned[val_key],
                g120,
            )
    np.savez_compressed(out_dir / "grid_120hz.npz", **pack120)
    summary["grids"]["120"] = {"n": int(len(g120)), "keys": sorted(pack120.keys())}
    print(f"[resample] wrote grid_120hz ({len(g120)} samples)", flush=True)

    # --- 900 Hz: EEG ---
    g900 = _make_grid(t0, t1, 900)
    pack900: dict[str, np.ndarray] = {"timestamps_pc": g900, "target_hz": np.int32(900)}
    eeg_fit = report.get("eeg", {}).get("fit_to_pc", {})
    if not skip_eeg and eeg_path is not None and eeg_fit.get("fitted"):
        pack900.update(_resample_eeg_900(eeg_path, eeg_fit, t0, t1, hz=900.0))
    elif not skip_eeg:
        print("[resample] WARN: EEG skipped (no path or fit_to_pc failed)", flush=True)
    np.savez_compressed(out_dir / "grid_900hz.npz", **pack900)
    print(f"[resample] wrote grid_900hz ({len(g900)} samples)", flush=True)
    summary["grids"]["900"] = {
        "n": int(len(g900)),
        "keys": sorted(pack900.keys()),
        "eeg_fitted": bool(eeg_fit.get("fitted")),
    }

    # Markers on every grid (for epoch slicing)
    mc = aligned["marker_code"].astype(np.int32)
    mt = aligned["marker_t_pc"].astype(np.float64)
    for hz, name in ((30, "grid_30hz"), (60, "grid_60hz"), (120, "grid_120hz"), (900, "grid_900hz")):
        path = out_dir / f"{name}.npz"
        d = dict(np.load(path, allow_pickle=False))
        d["marker_code"] = mc
        d["marker_t_pc"] = mt
        np.savez_compressed(path, **d)

    with open(out_dir / "resample_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--eeg-path", type=Path, default=None,
                    help="Curry .cdt.dpo for 900 Hz EEG resampling")
    ap.add_argument("--skip-eeg", action="store_true",
                    help="Only write 30/60/120 Hz grids (fast)")
    args = ap.parse_args(argv)
    try:
        summary = resample_session(
            args.session_dir.resolve(), args.eeg_path, skip_eeg=args.skip_eeg
        )
    except Exception as exc:
        print(f"[resample] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[resample] OK  span={summary['t1_pc'] - summary['t0_pc']:.1f}s")
    for hz, info in summary["grids"].items():
        print(f"  {hz:>4} Hz  n={info['n']:>7}  keys={info['keys']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
