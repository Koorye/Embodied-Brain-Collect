"""Curry7 I/O: legacy .dap/.dat and current .cdt/.cdt.dpo acquisitions."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _parse_meta_number(text: str, key: str, cast):
    patterns = [
        rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)",
        rf"(?im)^\s*{re.escape(key)}\s*:\s*([^\r\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip().split()[0]
            return cast(value)
    raise ValueError(f"Could not find {key!r} in Curry metadata")


def _decode_trigger_runs(
    raw: np.ndarray,
    sample_freq_hz: float,
    *,
    baseline: int,
    min_duration_s: float,
    meta_base: dict,
) -> tuple[dict, list[dict]]:
    unique = np.unique(raw)
    change_idx = np.flatnonzero(np.diff(raw) != 0) + 1
    bounds = np.r_[0, change_idx, len(raw)]

    runs = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        raw_value = int(raw[start])
        code = raw_value - baseline
        duration_s = (stop - start) / sample_freq_hz
        if code == 0 or duration_s < min_duration_s:
            continue
        runs.append({
            "sample_start": int(start),
            "sample_stop_exclusive": int(stop),
            "t_start_s": float(start / sample_freq_hz),
            "t_stop_s": float(stop / sample_freq_hz),
            "duration_s": float(duration_s),
            "raw_value": raw_value,
            "code": int(code),
        })

    meta = {
        **meta_base,
        "baseline": baseline,
        "raw_min": int(np.nanmin(raw)),
        "raw_max": int(np.nanmax(raw)),
        "raw_unique_count": int(len(unique)),
        "raw_unique_values": ",".join(str(int(v)) for v in unique[:64]),
        "change_count": int(len(change_idx)),
    }
    return meta, runs


def decode_triggers_dap_dat(
    dap_path: Path,
    *,
    baseline: int = 65280,
    channel: int | None = None,
    min_duration_s: float = 0.0,
) -> tuple[dict, list[dict]]:
    dap_path = dap_path.resolve()
    dat_path = dap_path.with_suffix(".dat")
    if not dat_path.is_file():
        raise FileNotFoundError(f"Missing DAT next to DAP: {dat_path}")

    text = dap_path.read_text(encoding="utf-8", errors="ignore")
    num_samples = _parse_meta_number(text, "NumSamples", int)
    num_channels = _parse_meta_number(text, "NumChannels", int)
    sample_freq_hz = _parse_meta_number(text, "SampleFreqHz", float)

    channel_1based = channel or num_channels
    expected_bytes = num_samples * num_channels * np.dtype("<f4").itemsize
    if dat_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"DAT size mismatch: expected {expected_bytes} bytes, "
            f"got {dat_path.stat().st_size}"
        )

    data = np.memmap(dat_path, dtype="<f4", mode="r", shape=(num_samples, num_channels))
    raw = np.asarray(data[:, channel_1based - 1]).astype(np.int64)
    meta_base = {
        "format": "dap_dat",
        "meta_path": str(dap_path),
        "data_path": str(dat_path),
        "num_samples": num_samples,
        "num_channels": num_channels,
        "sample_freq_hz": sample_freq_hz,
        "duration_s": float(num_samples / sample_freq_hz),
        "trigger_channel_1based": channel_1based,
    }
    return _decode_trigger_runs(
        raw, sample_freq_hz, baseline=baseline, min_duration_s=min_duration_s, meta_base=meta_base
    )


def decode_triggers_cdt(
    dpo_path: Path,
    *,
    baseline: int = 65280,
    channel: int | None = None,
    min_duration_s: float = 0.0,
) -> tuple[dict, list[dict]]:
    dpo_path = dpo_path.resolve()
    if dpo_path.name.endswith(".cdt.dpo"):
        cdt_path = dpo_path.with_name(dpo_path.name.replace(".cdt.dpo", ".cdt"))
    else:
        cdt_path = dpo_path.with_suffix(".cdt")
    if not cdt_path.is_file():
        raise FileNotFoundError(f"Missing CDT next to DPO: {cdt_path}")

    text = dpo_path.read_text(encoding="utf-8", errors="ignore")
    num_samples = _parse_meta_number(text, "NumSamples", int)
    num_channels = _parse_meta_number(text, "NumChannels", int)
    sample_freq_hz = _parse_meta_number(text, "SampleFreqHz", float)

    channel_1based = channel or num_channels
    expected_bytes = num_samples * num_channels * np.dtype("<f4").itemsize
    if cdt_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"CDT size mismatch: expected {expected_bytes} bytes, "
            f"got {cdt_path.stat().st_size}"
        )

    data = np.memmap(cdt_path, dtype="<f4", mode="r", shape=(num_samples, num_channels))
    raw = np.asarray(data[:, channel_1based - 1]).astype(np.int64)
    meta_base = {
        "format": "cdt_dpo",
        "meta_path": str(dpo_path),
        "data_path": str(cdt_path),
        "num_samples": num_samples,
        "num_channels": num_channels,
        "sample_freq_hz": sample_freq_hz,
        "duration_s": float(num_samples / sample_freq_hz),
        "trigger_channel_1based": channel_1based,
    }
    return _decode_trigger_runs(
        raw, sample_freq_hz, baseline=baseline, min_duration_s=min_duration_s, meta_base=meta_base
    )


def resolve_curry_paths(eeg_path: Path) -> tuple[Path, Path]:
    """Return (metadata_path, data_path) for a Curry acquisition."""
    eeg_path = eeg_path.resolve()
    if eeg_path.name.endswith(".cdt.dpo"):
        return eeg_path, eeg_path.with_name(eeg_path.name.replace(".cdt.dpo", ".cdt"))
    if eeg_path.suffix == ".cdt" and eeg_path.with_suffix(".cdt.dpo").is_file():
        return eeg_path.with_suffix(".cdt.dpo"), eeg_path
    if eeg_path.suffix == ".dap":
        return eeg_path, eeg_path.with_suffix(".dat")
    raise ValueError(f"Unrecognized Curry path: {eeg_path}")


def decode_triggers(
    eeg_path: Path,
    *,
    baseline: int = 65280,
    channel: int | None = None,
    min_duration_s: float = 0.0,
) -> tuple[dict, list[dict]]:
    """Decode ParallelBox triggers from Curry .dap/.dat or .cdt/.cdt.dpo."""
    eeg_path = eeg_path.resolve()
    if eeg_path.suffix == ".dap" or (
        eeg_path.suffix == ".dat" and eeg_path.with_suffix(".dap").is_file()
    ):
        dap = eeg_path if eeg_path.suffix == ".dap" else eeg_path.with_suffix(".dap")
        return decode_triggers_dap_dat(
            dap, baseline=baseline, channel=channel, min_duration_s=min_duration_s
        )
    if eeg_path.name.endswith(".cdt.dpo") or (
        eeg_path.suffix == ".cdt" and eeg_path.with_suffix(".cdt.dpo").is_file()
    ):
        dpo = eeg_path if eeg_path.name.endswith(".cdt.dpo") else eeg_path.with_suffix(".cdt.dpo")
        return decode_triggers_cdt(
            dpo, baseline=baseline, channel=channel, min_duration_s=min_duration_s
        )
    raise ValueError(f"Unrecognized Curry acquisition path: {eeg_path}")


def load_curry_eeg_meta(eeg_path: Path) -> dict:
    """Metadata only (no bulk data load)."""
    meta_path, data_path = resolve_curry_paths(eeg_path)
    text = meta_path.read_text(encoding="utf-8", errors="ignore")
    num_samples = _parse_meta_number(text, "NumSamples", int)
    num_channels = _parse_meta_number(text, "NumChannels", int)
    sample_freq_hz = _parse_meta_number(text, "SampleFreqHz", float)
    return {
        "meta_path": str(meta_path),
        "data_path": str(data_path),
        "num_samples": num_samples,
        "num_channels": num_channels,
        "sample_freq_hz": sample_freq_hz,
        "duration_s": float(num_samples / sample_freq_hz),
        "trigger_channel_1based": num_channels,
        "eeg_channels": num_channels - 1,
    }


def open_curry_eeg_memmap(eeg_path: Path):
    """Return (meta dict, memmap shape (samples, channels))."""
    meta = load_curry_eeg_meta(eeg_path)
    data = np.memmap(
        meta["data_path"], dtype="<f4", mode="r",
        shape=(meta["num_samples"], meta["num_channels"]),
    )
    return meta, data


def curry_acquisition_start(dpo_path: Path) -> "datetime.datetime | None":
    """Parse Curry DATA_PARAMETERS start time from a .cdt.dpo file."""
    from datetime import datetime

    text = dpo_path.read_text(encoding="utf-8", errors="ignore")

    def g(key: str) -> int:
        return _parse_meta_number(text, key, int)

    try:
        return datetime(
            g("StartYear"), g("StartMonth"), g("StartDay"),
            g("StartHour"), g("StartMin"), g("StartSec"),
        )
    except Exception:
        return None


def find_curry_acquisition_for_session(
    session_dt,
    acquisition_root: Path,
) -> Path | None:
    """Pick the .cdt.dpo whose Curry start time is closest to session folder time."""
    day_dir = acquisition_root / session_dt.strftime("%Y_%m_%d")
    if not day_dir.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for dpo in day_dir.glob("Acq *.cdt.dpo"):
        if dpo.stat().st_size < 100:
            continue
        cdt = dpo.with_name(dpo.name.replace(".cdt.dpo", ".cdt"))
        if not cdt.is_file() or cdt.stat().st_size < 1_000_000:
            continue
        start = curry_acquisition_start(dpo)
        if start is None:
            continue
        delta = abs((session_dt - start).total_seconds())
        if delta <= 90 * 60:
            candidates.append((delta, dpo))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
