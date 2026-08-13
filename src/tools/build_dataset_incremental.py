"""Build a safe standalone or incremental dataset from newly collected sessions.

This is intentionally separate from ``build_dataset.py``.  The original full
builder globally renumbers episodes and rewrites metadata, so it is not safe
for appends.

Workflow:
  1. Prepare an immutable plan (read-only discovery + a small JSON file):
       python -m record.tools.build_dataset_incremental --fresh \
         --exclude-dataset record/dataset --dataset-root record/dataset01 \
         --prepare PLAN.json
  2. Review PLAN.json, then apply it:
       python -m record.tools.build_dataset_incremental --apply-plan PLAN.json

The apply step preserves all existing episode indices, writes each new artifact
through a temporary file, stores resumable per-episode sidecars, backs up
metadata, and only publishes merged metadata after every planned episode has
finished.  Structurally valid data is retained; strict training selection is
regenerated afterwards by analyze_dataset.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools import build_dataset as base  # noqa: E402
from tools import curry_io  # noqa: E402
from tools import analyze_dataset as analyze_module  # noqa: E402
from tools import report_problems as problems_module  # noqa: E402

DEFAULT_DATASET = ROOT / "dataset"
SESSIONS_DIR = ROOT / "sessions"
PLAN_VERSION = 1

# Recent collection days can contain hundreds of short sessions inside a small
# number of long Curry acquisitions.  Both SessionRes and analyze_sync decode
# the same trigger channel, so cache that immutable result across sessions.
_decode_triggers_uncached = curry_io.decode_triggers


@lru_cache(maxsize=32)
def _decode_triggers_cached(dpo: Path, min_duration_s: float = 0.005):
    return _decode_triggers_uncached(dpo, min_duration_s=min_duration_s)


curry_io.decode_triggers = _decode_triggers_cached
base.decode_triggers = _decode_triggers_cached


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _jsonl_text(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _session_screen(sd: Path) -> dict:
    source_files = {
        "eye": sd / "eye" / "eye.npz",
        "emg": sd / "emg" / "emg.npz",
        "wrist_cam": sd / "wrist_cam" / "wrist_cam.npz",
        "vive": sd / "vive" / "vive.npz",
    }
    tactile = sd / "tactile" / "tactile.npz"
    manus = sd / "manus" / "manus.npz"
    result = {
        "session": sd.name,
        "sanity_overall_ok": None,
        "aborted": False,
        "degraded_modalities": [],
        "missing_sources": [
            name
            for name, path in source_files.items()
            if not path.is_file() or path.stat().st_size < 1024
        ],
        "hand_source": (
            "tactile"
            if tactile.is_file()
            else "manus"
            if manus.is_file()
            else None
        ),
    }
    if result["hand_source"] is None:
        result["missing_sources"].append("hand_glove")
    session_dt = base._session_dt(sd.name)
    eeg_dpo = (
        base.find_curry_acquisition_for_session(session_dt, base.ACQ_ROOT)
        if session_dt is not None
        else None
    )
    result["eeg_dpo"] = str(eeg_dpo) if eeg_dpo is not None else None
    sj = sd / "session.json"
    if sj.is_file():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            result["aborted"] = bool(data.get("aborted", False))
            result["degraded_modalities"] = data.get("degraded_modalities") or []
        except Exception as exc:
            result["session_json_error"] = str(exc)
    sr = sd / "sanity_report.json"
    if sr.is_file():
        try:
            data = json.loads(sr.read_text(encoding="utf-8"))
            result["sanity_overall_ok"] = data.get("overall_ok")
            result["failed_checks"] = [
                c.get("name")
                for c in data.get("checks", [])
                if c.get("status") not in ("OK", "DISABLED")
            ]
        except Exception as exc:
            result["sanity_report_error"] = str(exc)
    return result


def _load_session_list(path: Path) -> list[Path]:
    sessions = []
    seen = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PACKAGE_PARENT / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise RuntimeError(
                f"{path}:{line_number}: session directory not found: {candidate}"
            )
        if not (candidate / "markers.npz").is_file():
            raise RuntimeError(
                f"{path}:{line_number}: markers.npz missing: {candidate}"
            )
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        sessions.append(candidate)
    if not sessions:
        raise RuntimeError(f"session list is empty: {path}")
    return sorted(sessions)


def prepare_plan(
    dataset_root: Path,
    plan_path: Path,
    only: str | None,
    limit_sessions: int | None,
    fresh: bool,
    exclude_dataset: Path | None,
    task_reference_dataset: Path | None,
    session_list: Path | None,
) -> dict:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    old_eps = _load_jsonl(episodes_path)
    old_tasks = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    if not old_eps and fresh:
        old_tasks = []
    elif not old_eps or not old_tasks:
        raise RuntimeError(f"existing dataset metadata is incomplete: {dataset_root}")

    excluded_sources = {row["source_session"] for row in old_eps}
    if exclude_dataset is not None:
        excluded_sources.update(
            row["source_session"]
            for row in _load_jsonl(exclude_dataset / "meta" / "episodes.jsonl")
        )
    if session_list is not None:
        sessions = [
            p for p in _load_session_list(session_list)
            if p.name not in excluded_sources
        ]
    else:
        sessions = sorted(
            p for p in SESSIONS_DIR.iterdir()
            if p.is_dir()
            and "subj" in p.name
            and (p / "markers.npz").is_file()
            and p.name not in excluded_sources
            and (only is None or only in p.name)
        )
    if limit_sessions is not None:
        sessions = sessions[:limit_sessions]

    task_index = {int(t["task_id"]): int(t["task_index"]) for t in old_tasks}
    next_task_index = max(task_index.values(), default=-1) + 1
    names_en = json.loads(
        (ROOT / "config" / "task_names_en.json").read_text(encoding="utf-8")
    )["tasks"]
    cfg = json.loads(
        (ROOT / "config" / "collection.json").read_text(encoding="utf-8")
    )
    names_zh = {int(t["task_id"]): t["task_name"] for t in cfg["tasks"]}
    canonical_by_zh: dict[str, dict] = {}
    task_reference = task_reference_dataset or exclude_dataset
    if task_reference is not None:
        for task in _load_jsonl(task_reference / "meta" / "tasks.jsonl"):
            canonical_by_zh[task["task_zh"]] = {
                "slug": task["slug"],
                "desc": task["task"],
            }

    enumerated: list[dict] = []
    screens: list[dict] = []
    excluded: list[dict] = []
    for sd in sessions:
        screen = _session_screen(sd)
        eps = base.enumerate_episodes(sd)
        screen["n_episodes"] = len(eps)
        screen["n_without_execution"] = sum(not e["has_exec"] for e in eps)
        screens.append(screen)
        if not eps:
            excluded.append({"session": sd.name, "reason": "no episodes"})
            continue
        enumerated.extend(eps)

    enumerated.sort(
        key=lambda e: (
            e["task_id"] if e["task_id"] is not None else 9999,
            e["session_dt"] or datetime.min,
            e["trial"],
        )
    )
    next_episode = max((int(e["episode_index"]) for e in old_eps), default=-1) + 1
    planned_eps = []
    for offset, ep in enumerate(enumerated):
        task_id = ep["task_id"]
        if task_id is not None and task_id not in task_index:
            task_index[task_id] = next_task_index
            next_task_index += 1
        zh_name = names_zh.get(task_id, "n/a")
        info = (
            canonical_by_zh.get(zh_name)
            or (names_en.get(str(task_id), {}) if task_id is not None else {})
        )
        planned_eps.append({
            "episode_index": next_episode + offset,
            "source_session": ep["session"],
            "trial": ep["trial"],
            "task_id": task_id,
            "task_index": task_index.get(task_id, -1),
            "task_slug": info.get(
                "slug", f"task_{task_id:03d}" if task_id is not None else "unknown"
            ),
            "task_desc": info.get("desc", zh_name),
            "task_zh": zh_name,
            "win_on": ep["win_on"],
            "win_off": ep["win_off"],
            "onset": ep["onset"],
            "offset": ep["offset"],
            "phases": ep["phases"],
            "has_execution": ep["has_exec"],
        })

    plan = {
        "plan_version": PLAN_VERSION,
        "mode": "fresh" if fresh else "append",
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_root": str(dataset_root.resolve()),
        "episodes_meta_sha256": _sha256(episodes_path) if episodes_path.is_file() else None,
        "base_episode_count": len(old_eps),
        "base_max_episode_index": next_episode - 1,
        "candidate_session_count": len(sessions),
        "planned_session_count": len({e["source_session"] for e in planned_eps}),
        "planned_episode_count": len(planned_eps),
        "session_selection": (
            {"mode": "list", "path": str(session_list.resolve())}
            if session_list is not None
            else {"mode": "scan", "only": only}
        ),
        "eeg_match_count": sum(bool(s["eeg_dpo"]) for s in screens),
        "eeg_missing_count": sum(not s["eeg_dpo"] for s in screens),
        "excluded": excluded,
        "screens": screens,
        "episodes": planned_eps,
    }
    _atomic_json(plan_path, plan)
    return plan


def _episode_from_plan(row: dict) -> dict:
    return {
        "session": row["source_session"],
        "trial": row["trial"],
        "task_id": row["task_id"],
        "episode_index": row["episode_index"],
        "task_index": row["task_index"],
        "slug": row["task_slug"],
        "desc": row["task_desc"],
        "zh": row["task_zh"],
        "win_on": row["win_on"],
        "win_off": row["win_off"],
        "onset": row["onset"],
        "offset": row["offset"],
        "phases": row["phases"],
        "has_exec": row["has_execution"],
    }


def _load_npz(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _append_new_modalities(
    res: base.SessionRes,
    ep: dict,
    out_h5: Path,
    meta: dict,
) -> None:
    """Append modalities introduced after the original dataset build."""
    manus = _load_npz(res.sd / "manus" / "manus.npz")
    oak = _load_npz(res.sd / "oak_camera" / "oak_camera.npz")
    with base.h5py.File(out_h5, "a") as h5:
        if manus is not None and "ergo_timestamps" in manus:
            timestamps = manus["ergo_timestamps"].astype(np.float64)
            i0, i1 = base._slice(timestamps, ep["win_on"], ep["win_off"])
            if i1 > i0:
                group = h5.create_group("manus_glove")
                group.attrs["fs_hz_nominal"] = 200.0
                group.attrs["schema"] = "Manus Ergo 40-channel"
                base._ds(group, "t_pc", timestamps[i0:i1])
                base._ds(group, "t_rel", timestamps[i0:i1] - ep["onset"])
                base._ds(group, "data", manus["ergo_data"][i0:i1])
                for source, target in (
                    ("ergo_glove_id", "glove_id"),
                    ("ergo_is_user_id", "is_user_id"),
                    ("ergo_publish_time", "publish_time"),
                    ("channel_names", "channel_names"),
                ):
                    if source in manus:
                        values = manus[source]
                        if source != "channel_names":
                            values = values[i0:i1]
                        base._ds(group, target, values)
                meta["coverage"]["manus_glove"] = i1 - i0
                meta["stats"]["manus_glove"] = base._stats(
                    manus["ergo_data"][i0:i1]
                )

        # OAK replaced the former tactile camera in the collection rig.  Keep
        # the stable public camera key while recording the physical source.
        if oak is not None and "cam_timestamps" in oak:
            timestamps = oak["cam_timestamps"].astype(np.float64)
            i0, i1 = base._slice(timestamps, ep["win_on"], ep["win_off"])
            frames = h5.require_group("video_frames")
            key = "observation.tactile_cam"
            if i1 > i0 and key not in frames:
                group = frames.create_group(key)
                group.attrs["mp4"] = (
                    f"videos/{base._chunk_dir(ep['episode_index'])}/{key}/"
                    f"episode_{ep['episode_index']:06d}.mp4"
                )
                group.attrs["fps_nominal"] = 30.0
                group.attrs["physical_source"] = "oak_camera"
                base._ds(group, "t_pc", timestamps[i0:i1])
                base._ds(group, "t_rel", timestamps[i0:i1] - ep["onset"])
                meta["coverage"][key] = i1 - i0


def _camera_source(res: base.SessionRes, camera: str):
    relative_src, group = base.CAMERAS[camera]
    timestamps = base._cam_timestamps(res, group)
    if camera == "observation.tactile_cam" and timestamps is None:
        oak = _load_npz(res.sd / "oak_camera" / "oak_camera.npz")
        if oak is not None and "cam_timestamps" in oak:
            return (
                res.sd / "oak_camera" / "oak_camera.mp4",
                oak["cam_timestamps"].astype(np.float64),
            )
    return res.sd / relative_src, timestamps


def _build_one(
    res: base.SessionRes,
    row: dict,
    output_root: Path,
    sync: dict,
    state_dir: Path,
) -> tuple[dict, dict]:
    idx = int(row["episode_index"])
    state_path = state_dir / f"episode_{idx:06d}.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        h5 = output_root / state["h5"]
        if h5.is_file():
            return state["episode_record"], state["stats_record"]

    ep = _episode_from_plan(row)
    chunk = base._chunk_dir(idx)
    out_h5 = output_root / "data" / chunk / f"episode_{idx:06d}.h5"
    tmp_h5 = out_h5.with_name(out_h5.name + ".tmp")
    tmp_h5.parent.mkdir(parents=True, exist_ok=True)
    if tmp_h5.exists():
        tmp_h5.unlink()
    meta = base.write_episode_h5(res, ep, tmp_h5, embed_eeg=True)
    _append_new_modalities(res, ep, tmp_h5, meta)
    os.replace(tmp_h5, out_h5)

    videos = {}
    ffmpeg = base._ffmpeg()
    for camera in base.CAMERAS:
        src, timestamps = _camera_source(res, camera)
        if not src.is_file() or timestamps is None:
            continue
        i0, i1 = base._slice(timestamps, ep["win_on"], ep["win_off"])
        if i1 <= i0:
            continue
        out_video = (
            output_root / "videos" / chunk / camera / f"episode_{idx:06d}.mp4"
        )
        tmp_video = out_video.with_name(out_video.stem + ".tmp.mp4")
        if tmp_video.exists():
            tmp_video.unlink()
        start = float(timestamps[i0]) - float(timestamps[0])
        duration = float(timestamps[i1 - 1]) - float(timestamps[i0]) + 0.05
        if base.cut_video(ffmpeg, src, start, duration, tmp_video):
            out_video.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_video, out_video)
            videos[camera] = str(out_video.relative_to(output_root)).replace("\\", "/")
        elif tmp_video.exists():
            tmp_video.unlink()

    episode_record = {
        "episode_index": idx,
        "task_index": row["task_index"],
        "task_id": row["task_id"],
        "tasks": [row["task_desc"]],
        "task_slug": row["task_slug"],
        "task_zh": row["task_zh"],
        "source_session": row["source_session"],
        "trial": row["trial"],
        "duration_s": round(row["win_off"] - row["win_on"], 3),
        "phases": row["phases"],
        "has_execution": row["has_execution"],
        "modalities": sorted(meta["coverage"]),
        "coverage": meta["coverage"],
        "videos": videos,
        "quality_tier": sync.get("tier", "n/a"),
        "eeg_resid_rms_ms": (sync.get("eeg") or {}).get("resid_rms_ms"),
        "eeg_resid_vs_eprime_rms_ms": (
            sync.get("eeg") or {}
        ).get("resid_vs_eprime_rms_ms"),
    }
    stats_record = {"episode_index": idx, "stats": meta["stats"]}
    state = {
        "h5": str(out_h5.relative_to(output_root)).replace("\\", "/"),
        "episode_record": episode_record,
        "stats_record": stats_record,
    }
    _atomic_json(state_path, state)
    return episode_record, stats_record


def _quality_report(episodes: list[dict], tasks: list[dict]) -> dict:
    expected = [
        "eeg", "emg", "eye_gaze", "vive",
        "observation.eye", "observation.tactile_cam",
        "observation.wrist_cam0", "observation.wrist_cam1",
    ]
    missing = {m: 0 for m in expected}
    missing["hand_glove"] = 0
    tiers = Counter()
    for ep in episodes:
        mods = set(ep["modalities"])
        tiers[ep.get("quality_tier", "n/a")] += 1
        for modality in expected:
            missing[modality] += modality not in mods
        missing["hand_glove"] += not (
            {"tactile_glove", "manus_glove"} & mods
        )

    def summary(field: str) -> dict | None:
        values = [e.get(field) for e in episodes if e.get(field) is not None]
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "n": len(values),
            "median": round(float(np.median(arr)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "max": round(float(arr.max()), 3),
        }

    return {
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_episodes": len(episodes),
        "n_tasks": len(tasks),
        "quality_tiers": dict(tiers),
        "episodes_without_video": sum(not e.get("videos") for e in episodes),
        "episodes_without_eeg": sum("eeg" not in e["modalities"] for e in episodes),
        "episodes_without_execution": sum(
            not e.get("has_execution") for e in episodes
        ),
        "missing_modality_counts": missing,
        "sync_eeg_vs_pc_rms_ms": summary("eeg_resid_rms_ms"),
        "sync_eeg_vs_eprime_rms_ms": summary(
            "eeg_resid_vs_eprime_rms_ms"
        ),
        "per_task_episode_counts": [
            {
                "task_index": task["task_index"],
                "task_id": task["task_id"],
                "slug": task["slug"],
                "n_episodes": task["n_episodes"],
            }
            for task in tasks
        ],
    }


def _finalize_metadata(
    dataset_root: Path,
    new_episodes: list[dict],
    new_stats: list[dict],
    plan: dict,
) -> None:
    meta = dataset_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    old_episodes = _load_jsonl(meta / "episodes.jsonl")
    old_stats = _load_jsonl(meta / "episodes_stats.jsonl")
    existing_keys = {
        (e["source_session"], int(e["trial"])) for e in old_episodes
    }
    new_episodes = [
        e for e in new_episodes
        if (e["source_session"], int(e["trial"])) not in existing_keys
    ]
    new_indices = {int(e["episode_index"]) for e in new_episodes}
    new_stats = [s for s in new_stats if int(s["episode_index"]) in new_indices]
    episodes = sorted(old_episodes + new_episodes, key=lambda e: e["episode_index"])
    stats = sorted(old_stats + new_stats, key=lambda e: e["episode_index"])

    old_tasks = _load_jsonl(meta / "tasks.jsonl")
    task_by_id = {int(t["task_id"]): dict(t) for t in old_tasks}
    for row in plan["episodes"]:
        task_id = row["task_id"]
        if task_id is None or task_id in task_by_id:
            continue
        task_by_id[task_id] = {
            "task_index": row["task_index"],
            "task_id": task_id,
            "task": row["task_desc"],
            "task_zh": row["task_zh"],
            "slug": row["task_slug"],
            "n_episodes": 0,
        }
    counts = Counter(e["task_index"] for e in episodes)
    tasks = sorted(task_by_id.values(), key=lambda t: t["task_index"])
    for task in tasks:
        task["n_episodes"] = counts[task["task_index"]]

    info_path = meta / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = {
            "robot_type": "human_subj01_multimodal_bci",
            "format": (
                "lerobot-style (task/episode) + per-episode HDF5 "
                "for multi-rate signals"
            ),
            "chunks_size": base.CHUNK_SIZE,
            "fps": 30,
            "data_path": (
                "data/chunk-{ep_chunk:03d}/episode_{episode_index:06d}.h5"
            ),
            "video_path": (
                "videos/chunk-{ep_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "episode_definition": (
                "one trial: baseline -> instruction -> imagery(MI) -> "
                "execution(ME)"
            ),
            "master_clock": "pc_unix_s",
            "eeg_alignment": (
                "hardware-trigger linear fit to PC clock "
                "(validated against E-Prime)"
            ),
            "video_embedded": True,
            "eeg_embedded": True,
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    info["total_episodes"] = len(episodes)
    info["total_tasks"] = len(tasks)
    info["codebase_version"] = "bci-multimodal-0.2"
    info["task_id_namespace"] = (
        "collection-local; IDs 54-89 were reassigned in the recent campaign"
    )
    info["recommended_merge_key"] = "task_slug"
    info["updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    modalities_path = meta / "modalities.json"
    if modalities_path.is_file():
        modalities = json.loads(modalities_path.read_text(encoding="utf-8"))
    else:
        modalities = {
            "master_clock": (
                "pc_unix_s (each stream also has "
                "t_rel = t_pc - episode onset)"
            ),
            "streams": {
                "emg": {"fs_hz": 2000, "unit": "uV", "channels": 8},
                "emg_imu": {
                    "fs_hz": "~110",
                    "fields": "gyro(3), accel(3)",
                },
                "eye_gaze": {
                    "fs_hz": 100,
                    "fields": "xy(2), worn",
                },
                "eye_imu": {
                    "fs_hz": "~100",
                    "fields": "gyro(3), accel(3), quat(4)",
                },
                "tactile_glove": {"fs_hz": 200, "channels": 135},
                "vive": {
                    "fs_hz": 60,
                    "fields": (
                        "positions_m(3,3), quaternions_wxyz(3,4), "
                        "euler_rpy_deg(3,3), valid(3)"
                    ),
                },
                "video": {
                    "fps": 30,
                    "cameras": list(base.CAMERAS),
                    "format": "mp4 per episode",
                },
            },
            "phases": {p[0]: [p[1], p[2]] for p in base.PHASES},
        }
    streams = modalities.setdefault("streams", {})
    streams["eeg"] = {
        "fs_hz": 1000,
        "unit": "uV",
        "channel_variants": {
            "legacy": "256 EEG + VEOG/HEOG/EKG/EMG/Trigger (261)",
            "new": "128 EEG + VEOG/HEOG/EKG/EMG/Trigger (133)",
        },
        "storage": "embedded sliced f32",
    }
    streams["manus_glove"] = {
        "fs_hz": "~200",
        "channels": 40,
        "schema": "Manus Ergo",
        "alternative_to": "tactile_glove",
    }
    video = streams.setdefault("video", {})
    video["tactile_cam_sources"] = [
        "legacy tactile/tactile_cam.mp4",
        "new oak_camera/oak_camera.mp4",
    ]

    payloads = {
        "episodes.jsonl": _jsonl_text(episodes),
        "episodes_stats.jsonl": _jsonl_text(stats),
        "tasks.jsonl": _jsonl_text(tasks),
        "info.json": json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        "quality_report.json": json.dumps(
            _quality_report(episodes, tasks), ensure_ascii=False, indent=2
        ) + "\n",
        "modalities.json": json.dumps(
            modalities, ensure_ascii=False, indent=2
        ) + "\n",
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = dataset_root / "_incremental" / "backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for name in payloads:
        src = meta / name
        if src.is_file():
            shutil.copy2(src, backup / name)
    for name, text in payloads.items():
        _atomic_text(meta / name, text)

    history = dataset_root / "_incremental" / "history.jsonl"
    event = {
        "applied_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan_created_iso": plan["created_iso"],
        "added_episodes": len(new_episodes),
        "new_total_episodes": len(episodes),
        "backup": str(backup.relative_to(dataset_root)).replace("\\", "/"),
    }
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    readme_path = dataset_root / "README.md"
    template = DEFAULT_DATASET / "README.md"
    if not readme_path.is_file() and template.is_file():
        shutil.copy2(template, readme_path)


def apply_plan(
    plan_path: Path,
    output_root: Path | None,
    no_finalize: bool,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_version") != PLAN_VERSION:
        raise RuntimeError("unsupported plan version")
    dataset_root = Path(plan["dataset_root"])
    output_root = output_root or dataset_root
    planned_hash = plan.get("episodes_meta_sha256")
    current_meta = dataset_root / "meta" / "episodes.jsonl"
    if planned_hash is None:
        if current_meta.is_file():
            raise RuntimeError(
                "fresh target now has metadata; prepare a fresh plan"
            )
    elif not current_meta.is_file() or _sha256(current_meta) != planned_hash:
        raise RuntimeError(
            "episodes.jsonl changed after planning; prepare a fresh plan"
        )

    state_dir = (
        output_root / "_incremental" / "state"
        / hashlib.sha256(plan_path.read_bytes()).hexdigest()[:16]
    )
    rows_by_session: dict[str, list[dict]] = {}
    for row in plan["episodes"]:
        rows_by_session.setdefault(row["source_session"], []).append(row)

    records: list[dict] = []
    stats: list[dict] = []
    from tools.analyze_sync import analyze_session

    ordered_sessions = sorted(rows_by_session)
    for number, session_name in enumerate(ordered_sessions, 1):
        print(
            f"[increment {number}/{len(ordered_sessions)}] {session_name}",
            flush=True,
        )
        sd = SESSIONS_DIR / session_name
        res = base.SessionRes(sd)
        try:
            sync = analyze_session(sd, with_eeg=True)
        except Exception as exc:
            print(f"  warning: sync analysis failed: {exc}")
            sync = {}
        for row in rows_by_session[session_name]:
            record, stat = _build_one(res, row, output_root, sync, state_dir)
            records.append(record)
            stats.append(stat)
        res._eeg_mm = None

    if no_finalize:
        print(f"[increment] smoke artifacts complete: {output_root}")
        return
    if output_root.resolve() != dataset_root.resolve():
        raise RuntimeError("metadata can only be finalized into the planned dataset")
    if len(records) != int(plan["planned_episode_count"]):
        raise RuntimeError("not all planned episodes completed; metadata not changed")

    _finalize_metadata(dataset_root, records, stats, plan)
    analyze_module.DS = dataset_root
    analyze_module.META = dataset_root / "meta"
    problems_module.DS = dataset_root
    problems_module.META = dataset_root / "meta"
    analyze_module.main()
    problems_module.main()
    flags = json.loads(
        (dataset_root / "meta" / "flags.json").read_text(encoding="utf-8")
    )
    readme_path = dataset_root / "README.md"
    if readme_path.is_file():
        import re

        readme = readme_path.read_text(encoding="utf-8")
        readme = readme.replace(
            "(bci-multimodal-0.1)", "(bci-multimodal-0.2)"
        )
        readme = re.sub(
            r"^- 规模：.*$",
            (
                f"- 规模：**{flags['n_episodes']} episodes / "
                f"{len(_load_jsonl(dataset_root / 'meta' / 'tasks.jsonl'))} tasks**；"
                f"训练级干净子集 **{flags['clean_subset_count']}**。"
            ),
            readme,
            flags=re.MULTILINE,
        )
        quality = json.loads(
            (dataset_root / "meta" / "quality_report.json").read_text(
                encoding="utf-8"
            )
        )
        sync = quality.get("sync_eeg_vs_eprime_rms_ms") or {}
        problems = flags.get("problem_counts", {})
        quality_section = (
            "## 数据质量与同步（详见 `meta/ANALYSIS.md`）\n\n"
            f"- 总 episodes：**{flags['n_episodes']}**；训练级 clean subset："
            f"**{flags['clean_subset_count']}**。\n"
            f"- 无 EEG：{problems.get('no_eeg', 0)}；无视频："
            f"{problems.get('no_video', 0)}；无 execution："
            f"{problems.get('no_execution', 0)}。\n"
            f"- EEG vs E-Prime 残差中位数："
            f"{sync.get('median', 'n/a')} ms。\n"
            "- 详细问题列表：`meta/PROBLEMS.md` 和 "
            "`meta/problem_episodes.csv`。\n\n"
        )
        readme = re.sub(
            r"## 数据质量与同步.*?(?=## 复现)",
            quality_section,
            readme,
            flags=re.DOTALL,
        )
        readme = readme.replace(
            "| `/eeg` | `data(N,261) f32, t_pc, t_rel, channel_names` | 1000 | "
            "256 EEG + VEOG/HEOG/EKG/EMG + Trigger；嵌入切片，µV |",
            "| `/eeg` | `data(N,C) f32, t_pc, t_rel, channel_names` | 1000 | "
            "兼容 256+5（旧）与 128+5（新）通道；嵌入切片，µV |",
        )
        manus_row = (
            "| `/manus_glove` | `data(N,40), t_pc, t_rel, channel_names` | ~200 | "
            "新采集 Manus Ergo；与旧 `/tactile_glove` 并列的手套模态 |"
        )
        if manus_row not in readme:
            tactile_row = (
                "| `/tactile_glove` | `data(N,135), channel_names, t_pc, t_rel` "
                "| 200 | |"
            )
            readme = readme.replace(tactile_row, tactile_row + "\n" + manus_row)
        schema_note = (
            "\n## v0.2 新采集批次说明\n"
            "- 手套使用 `/manus_glove`（40 通道 Manus Ergo）；旧批次可使用 "
            "`/tactile_glove`（135 通道）。\n"
            "- `observation.tactile_cam` 在新批次由 OAK 相机提供。\n"
            "- EEG 同时兼容 128+5 与 256+5 通道配置。\n"
            "- 新批次重新分配过原始 `task_id`；跨数据集合并必须以 "
            "`task_slug` 为任务主键。\n"
        )
        if "## v0.2 新采集批次说明" not in readme:
            readme += schema_note
        _atomic_text(readme_path, readme)
    print(
        f"[increment] finalized {len(records)} episodes into {dataset_root}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path, metavar="PLAN")
    mode.add_argument("--apply-plan", type=Path, metavar="PLAN")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--only", default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="build a standalone dataset with indices starting at zero",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        default=None,
        help="exclude source sessions already listed in another dataset",
    )
    parser.add_argument(
        "--task-reference-dataset",
        type=Path,
        default=None,
        help="reuse canonical task slug/English names by matching task_zh",
    )
    parser.add_argument(
        "--session-list",
        type=Path,
        default=None,
        help="UTF-8 file containing one explicit session directory per line",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="alternate output for smoke tests",
    )
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args(argv)

    if args.prepare:
        plan = prepare_plan(
            args.dataset_root,
            args.prepare,
            args.only,
            args.limit_sessions,
            args.fresh,
            args.exclude_dataset,
            args.task_reference_dataset,
            args.session_list,
        )
        print(
            "[plan] "
            f"candidates={plan['candidate_session_count']} "
            f"sessions={plan['planned_session_count']} "
            f"episodes={plan['planned_episode_count']} "
            f"excluded={len(plan['excluded'])} "
            f"eeg={plan['eeg_match_count']} "
            f"eeg_missing={plan['eeg_missing_count']}"
        )
        print(f"[plan] wrote {args.prepare}")
        return 0
    apply_plan(args.apply_plan, args.output_root, args.no_finalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
