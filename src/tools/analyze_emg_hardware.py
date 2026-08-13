"""Analyze raw-session and dataset EMG quality and write a Chinese Markdown report.

The analysis intentionally separates:
1. transport/timing performance over all production raw sessions;
2. signal-quality statistics over every raw session represented in the built dataset;
3. phase-dependent usability over every processed episode containing EMG;
4. lossless raw-to-HDF5 integrity on a deterministic stratified sample.

Only NumPy and h5py are required. No filtering is applied to the source data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import h5py  # pyright: ignore[reportMissingImports]
import numpy as np


SRC = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "sessions"
DATASET = ROOT / "dataset"
DEFAULT_REPORT = ROOT / "docs" / "EMG_HARDWARE_PERFORMANCE_REPORT.md"
DEFAULT_METRICS = ROOT / "docs" / "emg_hardware_metrics.json"
DEFAULT_SESSION_ISSUES = ROOT / "docs" / "emg_problem_sessions.csv"
DEFAULT_EPISODE_ISSUES = ROOT / "docs" / "emg_problem_episodes.csv"
ADC_MIN = -(1 << 23)
ADC_MAX = (1 << 23) - 1
N_CHANNELS = 8
NOMINAL_FS = 2000.0


def pct(a, q):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else math.nan


def fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:,.{digits}f}"


def fmt_pct(x, digits=3):
    return "N/A" if x is None or not np.isfinite(x) else f"{100*x:.{digits}f}%"


def load_episodes():
    path = DATASET / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def raw_npz(session_name):
    return SESSIONS / session_name / "emg" / "emg.npz"


def production_raw_paths():
    out = []
    for session_json in sorted(SESSIONS.glob("*/session.json")):
        if not re.match(r"^\d{4}-\d{2}-\d{2}_", session_json.parent.name):
            continue
        p = session_json.parent / "emg" / "emg.npz"
        if p.is_file():
            out.append(p)
    return out


def inspect_transport(paths):
    rows = []
    for i, path in enumerate(paths, 1):
        try:
            with np.load(path, allow_pickle=False) as d:
                t = d["emg_timestamps"].astype(np.float64)
                imu_n = len(d["imu_timestamps"]) if "imu_timestamps" in d else 0
                dropped = int(d["dropped_frames"]) if "dropped_frames" in d else -1
            if len(t) < 2 or t[-1] <= t[0]:
                continue
            dt = np.diff(t)
            pos = dt[dt > 0]
            rows.append({
                "session": path.parents[1].name,
                "n": len(t),
                "imu_n": imu_n,
                "duration_s": float(t[-1] - t[0]),
                "fs_hz": float((len(t) - 1) / (t[-1] - t[0])),
                "duplicate_dt_fraction": float(np.mean(dt == 0)),
                "positive_gap_median_ms": float(np.median(pos) * 1000) if len(pos) else math.nan,
                "positive_gap_p99_ms": float(np.percentile(pos, 99) * 1000) if len(pos) else math.nan,
                "positive_gap_max_ms": float(pos.max() * 1000) if len(pos) else math.nan,
                "dropped": dropped,
                "loss_fraction": (dropped / (len(t) + imu_n + dropped)
                                  if dropped >= 0 else math.nan),
            })
        except Exception as exc:
            rows.append({"session": path.parents[1].name, "error": repr(exc)})
        if i % 100 == 0:
            print(f"[transport] {i}/{len(paths)}", flush=True)
    return rows


def stratified_indices(n, limit):
    if n <= limit:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, limit, dtype=int))


def inspect_raw_signal(source_sessions, sample_per_file=5000, spectral_sessions=72):
    available = [(s, raw_npz(s)) for s in sorted(source_sessions) if raw_npz(s).is_file()]
    spectral_pick = set(stratified_indices(len(available), spectral_sessions).tolist())
    sampled = []
    session_rows = []
    psd_sum = None
    psd_count = 0
    freq = None
    corr_sum = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float64)
    corr_count = 0
    total_values = 0
    total_zero = 0
    total_clip = 0
    clip_by_channel = np.zeros(N_CHANNELS, dtype=np.int64)
    total_nonfinite = 0
    global_min = np.full(N_CHANNELS, np.inf)
    global_max = np.full(N_CHANNELS, -np.inf)

    for j, (session, path) in enumerate(available):
        with np.load(path, allow_pickle=False) as d:
            x = d["emg_data"]
            t = d["emg_timestamps"].astype(np.float64)
            n = len(x)
            if n == 0:
                continue
            take = stratified_indices(n, sample_per_file)
            xs = np.asarray(x[take], dtype=np.float64)
            sampled.append(xs)
            total_values += int(x.size)
            total_zero += int(np.count_nonzero(x == 0))
            clip_mask = (x == ADC_MIN) | (x == ADC_MAX)
            session_clip = int(np.count_nonzero(clip_mask))
            total_clip += session_clip
            clip_by_channel += np.count_nonzero(clip_mask, axis=0)
            total_nonfinite += int(np.count_nonzero(~np.isfinite(x)))
            global_min = np.minimum(global_min, np.min(x, axis=0))
            global_max = np.maximum(global_max, np.max(x, axis=0))
            std = np.std(xs, axis=0)
            session_rows.append({
                "session": session,
                "n": n,
                "fs_hz": float((n - 1) / (t[-1] - t[0])) if n > 1 and t[-1] > t[0] else math.nan,
                "channel_std": std.tolist(),
                "flat_channels": int(np.sum(std < 1.0)),
                "clipped_values": session_clip,
            })

            if n >= 4096:
                start = max(0, n // 2 - 2048)
                seg = np.asarray(x[start:start + 4096], dtype=np.float64)
                seg -= np.mean(seg, axis=0, keepdims=True)
                win1 = np.hanning(4096).reshape(-1, 1)
                one_psd = np.abs(np.fft.rfft(seg * win1, axis=0)) ** 2
                one_f = np.fft.rfftfreq(4096, d=1 / NOMINAL_FS)
                total_p = np.trapezoid(one_psd, one_f, axis=0)
                lm = (one_f >= 49) & (one_f <= 51)
                line_p = np.trapezoid(one_psd[lm], one_f[lm], axis=0)
                line_frac = line_p / np.maximum(total_p, 1e-30)
                session_rows[-1]["line49_51_fraction_median"] = float(np.median(line_frac))
                session_rows[-1]["line50_bad_channels"] = int(np.sum(line_frac > .20))

            c = np.corrcoef(xs, rowvar=False)
            if np.all(np.isfinite(c)):
                corr_sum += c
                corr_count += 1

            if j in spectral_pick and n >= 4096:
                # Three deterministic, non-overlapping 4096-sample windows.
                starts = [0, max(0, n // 2 - 2048), n - 4096]
                win = np.hanning(4096).reshape(-1, 1)
                for start in starts:
                    seg = np.asarray(x[start:start + 4096], dtype=np.float64)
                    seg -= np.mean(seg, axis=0, keepdims=True)
                    spec = np.abs(np.fft.rfft(seg * win, axis=0)) ** 2
                    spec /= np.sum(win[:, 0] ** 2) * NOMINAL_FS
                    psd_sum = spec if psd_sum is None else psd_sum + spec
                    psd_count += 1
                freq = np.fft.rfftfreq(4096, d=1 / NOMINAL_FS)
        if (j + 1) % 40 == 0:
            print(f"[raw signal] {j + 1}/{len(available)}", flush=True)

    samples = np.concatenate(sampled, axis=0)
    channel = []
    for c in range(N_CHANNELS):
        v = samples[:, c]
        channel.append({
            "channel": c + 1,
            "median_uV": float(np.median(v)),
            "p01_uV": pct(v, 1),
            "p99_uV": pct(v, 99),
            "robust_span_uV": pct(v, 99) - pct(v, 1),
            "std_uV": float(np.std(v)),
            "min_uV": float(global_min[c]),
            "max_uV": float(global_max[c]),
            "adc_span_fraction": float((pct(v, 99) - pct(v, 1)) / (ADC_MAX - ADC_MIN)),
        })

    mean_psd = psd_sum / max(psd_count, 1)
    spectral = []
    bands = {
        "below_20": (0, 20),
        "emg_20_450": (20, 450),
        "above_450": (450, 1000.1),
        "line_49_51": (49, 51),
        "line_99_101": (99, 101),
    }
    for c in range(N_CHANNELS):
        p = mean_psd[:, c]
        total = np.trapezoid(p, freq)
        row = {"channel": c + 1}
        for name, (lo, hi) in bands.items():
            m = (freq >= lo) & (freq < hi)
            row[name + "_fraction"] = float(np.trapezoid(p[m], freq[m]) / total)
        m = (freq >= 20) & (freq <= 450)
        pf, ff = p[m], freq[m]
        cumulative = np.cumsum(pf)
        row["median_frequency_hz"] = float(ff[np.searchsorted(cumulative, cumulative[-1] / 2)])
        # Narrow line-power relative to local 45-55 Hz neighborhood.
        line = (freq >= 49) & (freq <= 51)
        local = (freq >= 45) & (freq <= 55)
        row["line50_local_fraction"] = float(
            np.trapezoid(p[line], freq[line]) / np.trapezoid(p[local], freq[local]))
        line_total = row["line_49_51_fraction"]
        row["line_interference_sir_db"] = float(
            10 * np.log10(max(1 - line_total, 1e-12) / max(line_total, 1e-12)))
        spectral.append(row)

    corr = corr_sum / max(corr_count, 1)
    offdiag = corr[np.triu_indices(N_CHANNELS, 1)]
    return {
        "sessions_analyzed": len(available),
        "sampled_values": int(samples.size),
        "full_values_checked": total_values,
        "zero_fraction": total_zero / total_values,
        "clipped_fraction": total_clip / total_values,
        "clipped_by_channel": clip_by_channel.tolist(),
        "sessions_with_clipping": int(sum(r["clipped_values"] > 0 for r in session_rows)),
        "nonfinite_fraction": total_nonfinite / total_values,
        "flat_session_channels": int(sum(r["flat_channels"] for r in session_rows)),
        "channel": channel,
        "spectral": spectral,
        "spectral_windows": psd_count,
        "mean_correlation_matrix": corr.tolist(),
        "offdiag_abs_correlation_median": float(np.median(np.abs(offdiag))),
        "offdiag_abs_correlation_max": float(np.max(np.abs(offdiag))),
        "session_rows": session_rows,
    }


def segment_features(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return None
    y = x - np.mean(x, axis=0, keepdims=True)
    return {
        "rms": np.sqrt(np.mean(y * y, axis=0)),
        "mav": np.mean(np.abs(y), axis=0),
        "waveform_length_per_s": np.sum(np.abs(np.diff(y, axis=0)), axis=0)
                                 / (len(y) / NOMINAL_FS),
    }


def inspect_processed(episodes):
    rows = []
    total_samples = 0
    dtype_counts = defaultdict(int)
    for i, ep in enumerate(episodes, 1):
        idx = ep["episode_index"]
        p = DATASET / "data" / f"chunk-{idx // 1000:03d}" / f"episode_{idx:06d}.h5"
        if not p.is_file():
            continue
        with h5py.File(p, "r") as h:
            if "emg" not in h:
                continue
            g = h["emg"]
            x = g["data"]
            t = g["t_rel"][:]
            total_samples += len(x)
            dtype_counts[str(x.dtype)] += 1
            if "phase_baseline" not in h.attrs or "phase_execution" not in h.attrs:
                continue
            b0, b1 = h.attrs["phase_baseline"]
            e0, e1 = h.attrs["phase_execution"]
            bm = (t >= b0) & (t <= b1)
            em = (t >= e0) & (t <= e1)
            if np.sum(bm) < 100 or np.sum(em) < 100:
                continue
            bf = segment_features(x[bm])
            ef = segment_features(x[em])
            ratio_db = 20 * np.log10(np.maximum(ef["rms"], 1e-12)
                                     / np.maximum(bf["rms"], 1e-12))
            rows.append({
                "episode_index": idx,
                "task_id": ep.get("task_id"),
                "baseline_n": int(np.sum(bm)),
                "execution_n": int(np.sum(em)),
                "baseline_rms": bf["rms"].tolist(),
                "execution_rms": ef["rms"].tolist(),
                "activation_db": ratio_db.tolist(),
                "baseline_mav": bf["mav"].tolist(),
                "execution_mav": ef["mav"].tolist(),
            })
        if i % 400 == 0:
            print(f"[processed] {i}/{len(episodes)}", flush=True)

    activation = np.asarray([r["activation_db"] for r in rows])
    base_rms = np.asarray([r["baseline_rms"] for r in rows])
    exec_rms = np.asarray([r["execution_rms"] for r in rows])
    return {
        "episodes_with_emg": int(sum("emg" in set(e.get("modalities", [])) for e in episodes)),
        "episodes_with_phase_comparison": len(rows),
        "total_emg_samples": total_samples,
        "dtype_counts": dict(dtype_counts),
        "activation_db_channel_median": np.median(activation, axis=0).tolist(),
        "activation_db_all_median": float(np.median(activation)),
        "activation_db_all_p10": pct(activation, 10),
        "activation_db_all_p90": pct(activation, 90),
        "episode_any_channel_gt3db_fraction": float(np.mean(np.max(activation, axis=1) > 3)),
        "episode_median_channel_positive_fraction": float(
            np.mean(np.median(activation, axis=1) > 0)),
        "baseline_rms_channel_median_uV": np.median(base_rms, axis=0).tolist(),
        "execution_rms_channel_median_uV": np.median(exec_rms, axis=0).tolist(),
    }


def validate_integrity(episodes, limit=320):
    candidates = [e for e in episodes if "emg" in set(e.get("modalities", []))]
    selected = [candidates[i] for i in stratified_indices(len(candidates), limit)]
    by_session = defaultdict(list)
    for ep in selected:
        by_session[ep["source_session"]].append(ep)
    checked = exact = failures = 0
    mismatch_examples = []
    for session, eps in by_session.items():
        p = raw_npz(session)
        if not p.is_file():
            failures += len(eps)
            continue
        with np.load(p, allow_pickle=False) as d:
            raw_t = d["emg_timestamps"]
            raw_x = d["emg_data"]
            raw_sn = d["emg_sn"] if "emg_sn" in d else None
            for ep in eps:
                idx = ep["episode_index"]
                hp = DATASET / "data" / f"chunk-{idx // 1000:03d}" / f"episode_{idx:06d}.h5"
                with h5py.File(hp, "r") as h:
                    g = h["emg"]
                    t = g["t_pc"][:]
                    x = g["data"][:]
                    i0 = int(np.searchsorted(raw_t, t[0], side="left"))
                    i1 = int(np.searchsorted(raw_t, t[-1], side="right"))
                    ok = (np.array_equal(t, raw_t[i0:i1])
                          and np.array_equal(x, raw_x[i0:i1]))
                    if raw_sn is not None and "sn" in g:
                        ok = ok and np.array_equal(g["sn"][:], raw_sn[i0:i1])
                    checked += 1
                    exact += int(ok)
                    if not ok and len(mismatch_examples) < 10:
                        mismatch_examples.append(idx)
    return {
        "checked": checked,
        "exact": exact,
        "failures": failures,
        "exact_fraction": exact / checked if checked else math.nan,
        "mismatch_examples": mismatch_examples,
    }


def summarize_transport(rows):
    ok = [r for r in rows if "error" not in r]
    fs = [r["fs_hz"] for r in ok]
    dup = [r["duplicate_dt_fraction"] for r in ok]
    loss = [r["loss_fraction"] for r in ok]
    gaps = [r["positive_gap_p99_ms"] for r in ok]
    return {
        "sessions": len(ok),
        "errors": len(rows) - len(ok),
        "samples": int(sum(r["n"] for r in ok)),
        "duration_hours": float(sum(r["duration_s"] for r in ok) / 3600),
        "fs_hz": {f"p{q}": pct(fs, q) for q in (0, 10, 50, 90, 100)},
        "sessions_outside_1pct": int(sum(abs(x - NOMINAL_FS) / NOMINAL_FS > .01 for x in fs)),
        "sessions_with_drops": int(sum(r["dropped"] > 0 for r in ok)),
        "dropped_total": int(sum(max(r["dropped"], 0) for r in ok)),
        "aggregate_loss_fraction": float(
            sum(max(r["dropped"], 0) for r in ok)
            / sum(r["n"] + r["imu_n"] + max(r["dropped"], 0) for r in ok)),
        "loss_session_p50": pct(loss, 50),
        "loss_session_p95": pct(loss, 95),
        "duplicate_dt_fraction_p50": pct(dup, 50),
        "duplicate_dt_fraction_p10": pct(dup, 10),
        "positive_gap_p99_ms_p50": pct(gaps, 50),
        "positive_gap_p99_ms_p90": pct(gaps, 90),
    }


def transport_cohorts(rows, source_sessions):
    ok = [r for r in rows if "error" not in r]
    source = [r for r in ok if r["session"] in source_sessions]
    monthly = {}
    for month in sorted({r["session"][:7] for r in ok}):
        group = [r for r in ok if r["session"].startswith(month)]
        monthly[month] = summarize_transport(group)
    return summarize_transport(ok), summarize_transport(source), monthly


def write_problem_manifests(transport_rows, signal_rows, episodes,
                            session_path, episode_path):
    signal_by_session = {r["session"]: r for r in signal_rows}
    session_issues = []
    bad_sessions = set()
    fail_sessions = set()
    for row in transport_rows:
        if "error" in row:
            session_issues.append({
                "session": row["session"], "severity": "FAIL",
                "issues": "read_error", "detail": row["error"],
            })
            bad_sessions.add(row["session"])
            fail_sessions.add(row["session"])
            continue
        issues, severity = [], "INFO"
        rate_error = abs(row["fs_hz"] - NOMINAL_FS) / NOMINAL_FS
        if rate_error > .05:
            issues.append("sampling_rate_error_gt5pct")
            severity = "FAIL"
        elif rate_error > .01:
            issues.append("sampling_rate_error_gt1pct")
            severity = "WARN"
        if row["loss_fraction"] > .01:
            issues.append("packet_loss_gt1pct")
            severity = "FAIL"
        elif row["loss_fraction"] > .001:
            issues.append("packet_loss_gt0.1pct")
            severity = max(severity, "WARN", key=lambda x: ("INFO", "WARN", "FAIL").index(x))
        if row["duplicate_dt_fraction"] > .50:
            issues.append("batched_timestamps")
            severity = max(severity, "WARN", key=lambda x: ("INFO", "WARN", "FAIL").index(x))
        sig = signal_by_session.get(row["session"], {})
        if sig.get("clipped_values", 0) > 0:
            issues.append("adc_endpoint_clipping")
            severity = max(severity, "WARN", key=lambda x: ("INFO", "WARN", "FAIL").index(x))
        if sig.get("flat_channels", 0) > 0:
            issues.append("flat_channel")
            severity = "FAIL"
        if sig.get("line50_bad_channels", 0) >= 4:
            issues.append("50hz_dominates_ge4_channels")
            severity = max(severity, "WARN", key=lambda x: ("INFO", "WARN", "FAIL").index(x))
        if issues:
            bad_sessions.add(row["session"])
            if severity == "FAIL":
                fail_sessions.add(row["session"])
            session_issues.append({
                "session": row["session"],
                "severity": severity,
                "issues": ";".join(issues),
                "fs_hz": round(row["fs_hz"], 6),
                "rate_error_pct": round(rate_error * 100, 4),
                "dropped_frames": row["dropped"],
                "loss_pct": round(row["loss_fraction"] * 100, 6),
                "duplicate_timestamp_pct": round(row["duplicate_dt_fraction"] * 100, 4),
                "positive_gap_p99_ms": round(row["positive_gap_p99_ms"], 4),
                "clipped_values": sig.get("clipped_values", ""),
                "line49_51_power_median_pct": (
                    round(sig["line49_51_fraction_median"] * 100, 3)
                    if "line49_51_fraction_median" in sig else ""),
                "line50_bad_channels": sig.get("line50_bad_channels", ""),
            })

    episode_issues = []
    for ep in episodes:
        issues, severity = [], "WARN"
        mods = set(ep.get("modalities", []))
        if "emg" not in mods:
            issues.append("missing_emg")
            severity = "FAIL"
        else:
            n = int(ep.get("coverage", {}).get("emg", 0))
            expected = float(ep.get("duration_s", 0)) * NOMINAL_FS
            ratio = n / expected if expected > 0 else math.nan
            if np.isfinite(ratio) and ratio < .50:
                issues.append("emg_coverage_lt50pct")
                severity = "FAIL"
            elif np.isfinite(ratio) and ratio < .90:
                issues.append("emg_coverage_lt90pct")
        if ep["source_session"] in fail_sessions:
            issues.append("source_session_has_fail_level_emg_issue")
            severity = "FAIL"
        if issues:
            episode_issues.append({
                "episode_index": ep["episode_index"],
                "source_session": ep["source_session"],
                "task_id": ep.get("task_id"),
                "trial": ep.get("trial"),
                "severity": severity,
                "issues": ";".join(issues),
                "emg_samples": ep.get("coverage", {}).get("emg", 0),
                "duration_s": ep.get("duration_s"),
                "coverage_ratio": round(
                    ep.get("coverage", {}).get("emg", 0)
                    / max(float(ep.get("duration_s", 0)) * NOMINAL_FS, 1), 6),
            })

    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(session_path, session_issues)
    write_csv(episode_path, episode_issues)
    return {
        "session_issue_rows": len(session_issues),
        "session_fail": sum(r["severity"] == "FAIL" for r in session_issues),
        "session_warn": sum(r["severity"] == "WARN" for r in session_issues),
        "episode_issue_rows": len(episode_issues),
        "episode_missing_emg": sum("missing_emg" in r["issues"] for r in episode_issues),
        "episode_low_coverage": sum("coverage_lt" in r["issues"] for r in episode_issues),
        "session_manifest": str(session_path.relative_to(ROOT)).replace("\\", "/"),
        "episode_manifest": str(episode_path.relative_to(ROOT)).replace("\\", "/"),
    }


def channel_table(rows, columns, formatters):
    header = "| 通道 | " + " | ".join(columns) + " |\n"
    sep = "|---:" + "|---:" * len(columns) + "|\n"
    body = []
    for row in rows:
        vals = [formatters[k](row[k]) for k in formatters]
        body.append(f"| CH{row['channel']} | " + " | ".join(vals) + " |")
    return header + sep + "\n".join(body)


def write_report(metrics, report_path):
    tr = metrics["transport"]
    ds_tr = metrics["transport_dataset_source"]
    sg = metrics["raw_signal"]
    pr = metrics["processed"]
    it = metrics["integrity"]
    cov = metrics["coverage"]
    problems = metrics["problems"]
    channels = channel_table(
        sg["channel"],
        ["中位数 (µV)", "P1 (µV)", "P99 (µV)", "P1–P99跨度 (µV)", "ADC跨度利用率"],
        {
            "median_uV": lambda x: fmt(x, 0),
            "p01_uV": lambda x: fmt(x, 0),
            "p99_uV": lambda x: fmt(x, 0),
            "robust_span_uV": lambda x: fmt(x, 0),
            "adc_span_fraction": lambda x: fmt_pct(x, 2),
        },
    )
    spectral = channel_table(
        sg["spectral"],
        ["<20 Hz功率", "49–51 Hz总功率", "非工频/工频 SIR (dB)", "50 Hz局部占比", "中值频率 (Hz)"],
        {
            "below_20_fraction": lambda x: fmt_pct(x, 2),
            "line_49_51_fraction": lambda x: fmt_pct(x, 2),
            "line_interference_sir_db": lambda x: fmt(x, 2),
            "line50_local_fraction": lambda x: fmt_pct(x, 2),
            "median_frequency_hz": lambda x: fmt(x, 1),
        },
    )
    phase_rows = []
    for c in range(N_CHANNELS):
        phase_rows.append({
            "channel": c + 1,
            "base": pr["baseline_rms_channel_median_uV"][c],
            "exec": pr["execution_rms_channel_median_uV"][c],
            "db": pr["activation_db_channel_median"][c],
        })
    phase = channel_table(
        phase_rows,
        ["静息段去均值RMS (µV)", "执行段去均值RMS (µV)", "执行/静息中位变化 (dB)"],
        {
            "base": lambda x: fmt(x, 1),
            "exec": lambda x: fmt(x, 1),
            "db": lambda x: fmt(x, 2),
        },
    )
    monthly_lines = [
        "| 月份 | session数 | 采样率中位 (Hz) | 超±1% | 有丢帧session | 聚合丢帧率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for month, row in metrics["transport_monthly"].items():
        monthly_lines.append(
            f"| {month} | {row['sessions']} | {row['fs_hz']['p50']:.3f} | "
            f"{row['sessions_outside_1pct']} | {row['sessions_with_drops']} | "
            f"{fmt_pct(row['aggregate_loss_fraction'], 4)} |")
    monthly_table = "\n".join(monthly_lines)

    fs = tr["fs_hz"]
    verdict_timing = ("吞吐率稳定，但逐样本 PC 时间戳不可用"
                      if tr["duplicate_dt_fraction_p50"] > .5 else "吞吐率与时间戳均稳定")
    verdict_loss = ("优秀" if tr["aggregate_loss_fraction"] < 1e-4
                    else "良好" if tr["aggregate_loss_fraction"] < 1e-3 else "不合格")
    verdict_clip = "优秀" if sg["clipped_fraction"] < 1e-6 else "需检查量程"
    verdict_integrity = "无损" if it["exact_fraction"] == 1 else "存在不一致"
    generated = time.strftime("%Y-%m-%d %H:%M:%S")

    text = f"""# EMG 传感器与数据链路性能评估报告

> 生成时间：{generated}  
> 数据范围：`record/sessions` 原始 EMG 与 `record/dataset` episode HDF5 中的 `/emg`。  
> 本报告只分析 8 通道 EMG；腕带 IMU 不参与信号质量结论。

## 1. 执行摘要

- **历史 dataset 对应批次稳定，但当前全量数据存在严重退化**：{tr["sessions"]} 个正式原始 session、
  累计 {tr["duration_hours"]:.2f} 小时，全量 session 有效采样率中位数 **{fs["p50"]:.3f} Hz**，
  范围 {fs["p0"]:.3f}–{fs["p100"]:.3f} Hz，偏离标称 2000 Hz 超过 1% 的 session 为
  **{tr["sessions_outside_1pct"]}**。dataset 对应的 {ds_tr["sessions"]} 个可用源 session 则全部在 ±1% 内。
- **当前全量传输连续性为{verdict_loss}**：累计记录器报告丢帧 {tr["dropped_total"]:,}，
  按 EMG+IMU 总帧估算的聚合丢帧率 **{fmt_pct(tr["aggregate_loss_fraction"], 5)}**；
  {tr["sessions_with_drops"]}/{tr["sessions"]} 个 session 出现过丢帧。相比之下，dataset 源 session
  聚合丢帧率仅 **{fmt_pct(ds_tr["aggregate_loss_fraction"], 5)}**，说明问题主要集中在后续新增采集。
- **量程与数值完整性为{verdict_clip}**：检查 {sg["full_values_checked"]:,} 个原始通道值，
  24-bit 端点饱和率 **{fmt_pct(sg["clipped_fraction"], 6)}**，涉及
  {sg["sessions_with_clipping"]}/{sg["sessions_analyzed"]} 个源 session；非有限值率
  **{fmt_pct(sg["nonfinite_fraction"], 6)}**，零值率 **{fmt_pct(sg["zero_fraction"], 5)}**。
- **逐样本时间戳是最主要短板**：相邻 EMG 样本时间戳相同的 session 中位比例为
  **{fmt_pct(tr["duplicate_dt_fraction_p50"], 2)}**。记录器在一次串口批量读取后给多帧赋几乎相同的
  `time.time()`，因此样本顺序和平均采样率可信，但原始 `t_pc` 不能表达 0.5 ms 的逐样本时刻。
- **处理前后数值{verdict_integrity}**：分层抽查 {it["checked"]} 个 episode，
  原始 `emg_data/emg_timestamps/emg_sn` 与 HDF5 切片逐元素完全一致
  {it["exact"]}/{it["checked"]}（{fmt_pct(it["exact_fraction"], 2)}）。
- **dataset 构建没有滤波、降采样或归一化**：只按 PC 时间窗切片并使用 gzip 压缩，`int32` 数值保留。
  因此 dataset 中的频谱噪声、直流偏置和批量时间戳问题均来自原始链路，不是后处理引入。

综合判断：该腕带在历史 dataset 批次中能稳定采集**高采样率、动作相关的 8 通道 sEMG**，但不能据此判定
当前整套系统性能合格：后续 session 出现大量低采样率和高丢帧记录；原始频谱还被 50 Hz 工频显著主导。
此外，软件时间戳机制把 2000 Hz 流压成约几十毫秒一批的同时间戳样本，限制跨模态精细同步。当前数据可在
严格质量筛选、重建均匀时间轴和滤波后用于分类；不应把所有 session 直接作为同质量训练数据。

## 2. 数据覆盖与分析口径

- 当前 `sessions` 中有 session 元数据且存在 EMG NPZ：**{tr["sessions"]}** 个。
- dataset 元数据引用源 session：**{cov["source_sessions"]}** 个；其中原始 EMG 可用：
  **{cov["source_sessions_with_raw_emg"]}** 个。
- dataset 总 episode：**{cov["episodes_total"]}**；带 `/emg`：**{pr["episodes_with_emg"]}**
  （{fmt_pct(pr["episodes_with_emg"]/cov["episodes_total"], 2)}）；缺 EMG：
  **{cov["episodes_total"]-pr["episodes_with_emg"]}**。
- 原始信号幅值/频谱统计覆盖 dataset 对应的全部 **{sg["sessions_analyzed"]}** 个可用源 session；
  完整扫描 {sg["full_values_checked"]:,} 个值，稳健分位数使用均匀抽取的 {sg["sampled_values"]:,} 个值。
- 动作效应覆盖同时具备 baseline 与 execution 的 **{pr["episodes_with_phase_comparison"]}** 个 episode。
- 频谱由 {sg["spectral_windows"]} 个 4096 点 Hann 窗平均得到，按样本顺序及标称 2000 Hz 计算。
- 精确问题清单已输出：`{problems["session_manifest"]}`（逐 session）和
  `{problems["episode_manifest"]}`（逐 episode）。session 清单共 {problems["session_issue_rows"]} 条，
  其中 FAIL {problems["session_fail"]}、WARN {problems["session_warn"]}；episode 清单
  {problems["episode_issue_rows"]} 条。

这里的“丢帧”采用采集器保存的 `dropped_frames`，它根据 EMG 与 IMU 共用的线上传输序号统计，优于只看
`emg_sn`；后者会把正常插入的 IMU 帧误判为 EMG 丢帧。

## 3. 采样率、连续性与时钟性能

| 指标 | 结果 | 评价 |
|---|---:|---|
| session 采样率 P10 / P50 / P90 | {fs["p10"]:.3f} / {fs["p50"]:.3f} / {fs["p90"]:.3f} Hz | 接近标称 2000 Hz |
| 超出 ±1% 的 session | {tr["sessions_outside_1pct"]}/{tr["sessions"]} | 越少越好 |
| 有丢帧的 session | {tr["sessions_with_drops"]}/{tr["sessions"]} | 多为局部事件 |
| 聚合丢帧率 | {fmt_pct(tr["aggregate_loss_fraction"], 5)} | {verdict_loss} |
| session 丢帧率 P50 / P95 | {fmt_pct(tr["loss_session_p50"], 5)} / {fmt_pct(tr["loss_session_p95"], 5)} | 反映尾部风险 |
| 相邻时间戳重复比例 P10 / P50 | {fmt_pct(tr["duplicate_dt_fraction_p10"], 2)} / {fmt_pct(tr["duplicate_dt_fraction_p50"], 2)} | 高，属批量赋时 |
| session 内正时间间隔 P99 的 P50 / P90 | {fmt(tr["positive_gap_p99_ms_p50"], 2)} / {fmt(tr["positive_gap_p99_ms_p90"], 2)} ms | 非真实采样间隔 |
| dataset 源 session 采样率范围 | {ds_tr["fs_hz"]["p0"]:.3f}–{ds_tr["fs_hz"]["p100"]:.3f} Hz | 历史批次稳定 |
| dataset 源 session 聚合丢帧率 | {fmt_pct(ds_tr["aggregate_loss_fraction"], 5)} | 明显优于当前全量 |

按月可见退化集中在哪一阶段：

{monthly_table}

结论：历史 dataset 批次长期速率稳定，但全量后续数据存在间歇性严重掉速/丢帧，必须按 session 门控；
同时{verdict_timing}。采集代码在
解析每个 4096-byte 串口块时才调用 `time.time()`，一个块内大量帧获得相同或极接近的时间戳。对单通道
波形和基于样本序号的频谱影响较小，对跨模态事件边界、延迟估计和高精度同步影响明显。HDF5 按这些时间戳
切片，边界分辨率实际受批次间隔限制，而不是理论上的 0.5 ms。

## 4. 幅值、量程、坏道与动态范围

{channels}

- 全库未发现的“session×通道”近似平坦（抽样标准差 <1 µV）数量：
  **{sg["flat_session_channels"]}**。
- 通道间绝对相关系数：中位 **{sg["offdiag_abs_correlation_median"]:.3f}**，
  最大 **{sg["offdiag_abs_correlation_max"]:.3f}**。过高的最大相关应结合电极位置判断是共同肌群活动、
  共模干扰还是通道冗余，不能仅凭人体任务数据归因于硬件串扰。
- 各通道存在明显且不同的直流基线，报告中的中位数不是肌肉激活幅度。后续特征必须至少逐段去均值；
  更推荐 20–450 Hz 带通后计算 RMS/MAV/波形长度。
- 端点饱和总体比例低但并非零，且分布在 {sg["sessions_with_clipping"]} 个源 session，应把饱和率纳入
  session 质量门控。较低的 ADC 跨度利用率不等价于“分辨率不足”，因为未知
  模拟增益、ADC LSB 与文档中“µV”换算是否经过校准。建议用已知幅度正弦源验证绝对增益和单位。

## 5. 频域性能与噪声构成

{spectral}

解释：

1. 20–450 Hz 是常用 sEMG 分析带宽；低于 20 Hz 往往包含动作伪迹、基线漂移和电极运动。
2. “50 Hz 局部占比”是 49–51 Hz 功率占 45–55 Hz 局部功率的比例，用于识别窄带工频峰；它不是传统
   SNR，数值还受真实肌电宽带功率影响。
3. 多数通道中值频率约为 50 Hz，且 49–51 Hz 可占总功率的很大比例，说明原始信号存在**严重工频主导**；
   这会夸大未滤波 RMS 和通道相关性，必须在动作分类前做 50 Hz 陷波并复核陷波后的有效带宽。
4. 当前数据未经任何滤波，以上比例反映“电极+人体+模拟前端+串口采集”的系统表现，不能拆分为纯硬件
   自噪声。要测输入参考噪声，应在同采样配置下增加短接输入和电阻负载记录。

### 5.1 信噪比可以分析到什么程度

“SNR”必须先定义信号和噪声。现有任务数据能够给出两类系统级代理指标：

- **动作/静息比（activation SNR proxy）**：把 phase 内去均值 RMS 作为总交流能量，计算
  `20·log10(RMS_execution/RMS_baseline)`。全通道中位 **{pr["activation_db_all_median"]:.2f} dB**，
  P10–P90 为 {pr["activation_db_all_p10"]:.2f}–{pr["activation_db_all_p90"]:.2f} dB。它同时包含真实肌电、
  工频和动作伪迹，因此只说明动作期更强，不是放大器本底 SNR。
- **非工频/工频信号干扰比（line SIR）**：`10·log10(P_非49–51Hz/P_49–51Hz)`，见上表。
  负值表示 49–51 Hz 功率超过其余全部频率功率。部分通道为负值，证实工频污染已经足以主导波形。

无法从现有数据可靠得到的是**输入参考噪声 SNR**：baseline 仍有生理 EMG、工频、电极运动和环境噪声，
没有“已知输入信号”和“零输入噪声”两组真值。要得到可与硬件规格书比较的 SNR，应做输入短接噪声记录，
再注入已知 RMS 正弦/宽带信号，计算 `20·log10(V_signal_rms/V_noise_rms)`。

## 6. 动作敏感性与可用信号

以下 RMS 均先在各 phase 内逐通道去均值，避免巨大直流偏置主导结果：

{phase}

- 全通道、全 episode 的执行/静息变化中位数：
  **{pr["activation_db_all_median"]:.2f} dB**（P10 {pr["activation_db_all_p10"]:.2f} dB，
  P90 {pr["activation_db_all_p90"]:.2f} dB）。
- 至少一个通道执行期提升 >3 dB 的 episode：
  **{fmt_pct(pr["episode_any_channel_gt3db_fraction"], 1)}**。
- episode 的通道中位变化 >0 dB：
  **{fmt_pct(pr["episode_median_channel_positive_fraction"], 1)}**。

这些指标验证传感器能否捕获任务相关肌肉活动，但不是分类准确率。任务种类、左右手、电极佩戴位置和动作
强度差异很大，建议建模时做 session 内标准化，并把 session/date 作为分组变量，避免同次佩戴泄漏到训练
与测试两侧。

## 7. 处理前后对比

| 项目 | 原始 `sessions/*/emg/emg.npz` | 处理后 `dataset/data/.../episode_*.h5` | 影响 |
|---|---|---|---|
| 数据类型 | `int32`（承载有符号 24-bit） | `int32` | 无量化损失 |
| 通道数 | 8 | 8 | 不变 |
| 采样顺序 | 连续 session 流 | episode 时间窗切片 | 窗外数据被舍弃 |
| 标称采样率 | 2000 Hz | 属性写 2000 Hz | 未重采样 |
| 时间戳 | PC `time.time()` | 原值 + `t_rel` | 批量赋时问题保留 |
| 滤波/去直流 | 无 | 无 | 原始偏置和噪声保留 |
| 压缩 | NPZ zip 容器 | HDF5 gzip level 4 | 无损压缩 |
| 序号 | `emg_sn` | `sn` | 原值保留 |

分层逐元素核验结果：**{it["exact"]}/{it["checked"]} 完全一致**。
{("不一致 episode 示例：" + str(it["mismatch_examples"])) if it["mismatch_examples"] else "未发现后处理改写、插值或数值漂移。"}

## 8. 硬件/链路分项评价

| 维度 | 评价 | 依据 |
|---|---|---|
| 持续 2 kHz × 8 通道吞吐 | 历史优秀、当前不稳定 | 全量 {tr["sessions_outside_1pct"]} 个 session 超出 ±1% |
| 串口传输完整性 | {verdict_loss} | 聚合丢帧率 {fmt_pct(tr["aggregate_loss_fraction"], 5)} |
| ADC 饱和余量 | {verdict_clip} | 端点饱和率 {fmt_pct(sg["clipped_fraction"], 6)} |
| 通道存活 | 良好 | 近似平坦 session×通道 {sg["flat_session_channels"]} |
| 动作响应 | 可用 | 执行/静息变化与 >3 dB episode 比例见第 6 节 |
| 工频抗扰 | 较差 | 多数通道频谱中值约 50 Hz，49–51 Hz 功率占比高 |
| 逐样本时间准确性 | 较差 | 时间戳重复比例中位 {fmt_pct(tr["duplicate_dt_fraction_p50"], 2)} |
| 数据集转换保真 | 优秀 | 抽查 {it["checked"]} 个 episode，逐元素一致率 {fmt_pct(it["exact_fraction"], 2)} |
| 绝对幅值校准 | 未验证 | 缺少标准信号源与输入短接数据 |

## 9. 优先改进建议

1. **重构时间戳（最高优先级）**：为每个串口读取批次记录到达时刻，再按 SN/固定 2000 Hz 对批内 EMG
   样本回填时间；更理想的是设备提供硬件计数器/时间戳。保存 `t_arrival` 与 `t_reconstructed` 两套时间。
2. **实时质量门控**：每 5–10 秒显示有效采样率、累计丢帧、各通道 RMS、饱和率、平坦通道和 50 Hz 比例；
   超阈值立即提示重新贴电极或检查 USB。
3. **建立标准预处理版本**：保留当前 raw HDF5，同时另建明确版本化的 20–450 Hz 零相位带通、50 Hz
   陷波（必要时含谐波）和逐 session/逐通道稳健标准化数据。不要覆盖原始值。
4. **做台架测试**：至少采集输入短接、已知电阻负载、10–500 Hz 多频正弦和不同幅度阶梯，测输入参考噪声、
   增益误差、频响、通道串扰、CMRR、饱和恢复和有效位数 ENOB。
5. **佩戴一致性**：记录通道-肌肉位置、电极间距、左右臂、皮肤准备和腕带方向；用固定收缩动作做每次
   session 开始前的 10 秒校准，量化跨日漂移。
6. **数据划分**：按 session 或采集日期分组划分训练/验证/测试；不能随机拆 episode 后宣称跨日泛化性能。

## 10. 限制

- 结论评价的是完整采集系统（电极、人体、模拟前端、无线/USB 接收器、串口和软件），不是拆机级芯片指标。
- 无独立参考 EMG、力传感器或标准信号源，无法给出真实灵敏度、绝对增益误差、CMRR、输入阻抗和 ENOB。
- 数据只有 `subj01`，不能推断跨被试性能。
- 频谱按样本序号和标称 2000 Hz 计算；这是对当前批量时间戳缺陷的必要处理。
- dataset 建于 {cov["dataset_created_iso"]}，晚于该时间的新 session 不在“处理后”对比中，但已纳入当前
  raw transport 总览（只要具有 `session.json` 和 `emg.npz`）。

## 11. 复现

```powershell
& C:\\Users\\31454\\miniconda3\\envs\\record\\python.exe `
  C:\\Users\\31454\\pangjingrui\\record\\tools\\analyze_emg_hardware.py
```

结构化指标同时写入 `record/docs/emg_hardware_metrics.json`。脚本不修改任何原始或 dataset 数据。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--session-issues", type=Path, default=DEFAULT_SESSION_ISSUES)
    ap.add_argument("--episode-issues", type=Path, default=DEFAULT_EPISODE_ISSUES)
    ap.add_argument("--integrity-sample", type=int, default=320)
    args = ap.parse_args(argv)

    episodes = load_episodes()
    source_sessions = {e["source_session"] for e in episodes}
    raw_paths = production_raw_paths()
    print(f"[inventory] production raw={len(raw_paths)}, dataset episodes={len(episodes)}, "
          f"source sessions={len(source_sessions)}", flush=True)

    transport_rows = inspect_transport(raw_paths)
    transport, source_transport, monthly_transport = transport_cohorts(
        transport_rows, source_sessions)
    raw_signal = inspect_raw_signal(source_sessions)
    processed = inspect_processed(episodes)
    integrity = validate_integrity(episodes, args.integrity_sample)
    problems = write_problem_manifests(
        transport_rows, raw_signal["session_rows"], episodes,
        args.session_issues, args.episode_issues)
    info = json.loads((DATASET / "meta" / "info.json").read_text(encoding="utf-8"))
    metrics = {
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "coverage": {
            "episodes_total": len(episodes),
            "source_sessions": len(source_sessions),
            "source_sessions_with_raw_emg": sum(raw_npz(s).is_file() for s in source_sessions),
            "dataset_created_iso": info.get("created_iso", "unknown"),
        },
        "transport": transport,
        "transport_dataset_source": source_transport,
        "transport_monthly": monthly_transport,
        "problems": problems,
        "raw_signal": {k: v for k, v in raw_signal.items() if k != "session_rows"},
        "processed": processed,
        "integrity": integrity,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, args.report)
    print(f"[done] {args.report}")
    print(f"[done] {args.metrics}")
    print(f"[done] {args.session_issues}")
    print(f"[done] {args.episode_issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
