"""Comprehensive per-episode problem inventory for record/dataset/.

Reads meta/episodes.jsonl and emits, listing the *specific* episodes for
every issue:
  meta/problem_episodes.csv   one row per problematic episode (all flags)
  meta/PROBLEMS.md            detailed human-readable inventory (tables)

Checks (per episode):
  - missing modality (each of EEG/EMG/eye_gaze/eye_imu/emg_imu/glove/vive)
  - missing camera / no video at all
  - no EEG trigger match (no eeg group) / EEG misaligned (vs eprime > 2ms)
  - pathological-clock session membership (huge residuals)
  - low coverage (present stream has < 50% of expected samples = dropout)
  - missing phase (baseline/instruction/imagery/execution absent)
  - invalid phase window (end <= start)
  - no execution phase / no TASK_ID (task_index == -1)
  - duration too short (<5s) / too long (>60s)

Usage:  python -m record.tools.report_problems
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset"
META = DS / "meta"

SIGNAL_MODS = ["eeg", "emg", "emg_imu", "eye_gaze", "eye_imu", "vive"]
HAND_GLOVE_ALTERNATIVES = {"tactile_glove", "manus_glove"}
CAMERAS = ["observation.eye", "observation.tactile_cam",
           "observation.wrist_cam0", "observation.wrist_cam1"]
FS = {"eeg": 1000, "emg": 2000, "emg_imu": 110, "eye_gaze": 100, "eye_imu": 100,
      "tactile_glove": 200, "manus_glove": 200, "vive": 60,
      "observation.eye": 30, "observation.tactile_cam": 30,
      "observation.wrist_cam0": 30, "observation.wrist_cam1": 30}
PHASES = ["baseline", "instruction", "imagery", "execution"]
VS_EPRIME_OK_MS = 2.0
DUR_MIN_S, DUR_MAX_S = 5.0, 60.0
LOW_COV = 0.5


def _load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    eps = _load_jsonl(META / "episodes.jsonl")
    tasks = {t["task_index"]: t for t in _load_jsonl(META / "tasks.jsonl")}

    # pathological-clock sessions (same rule as analyze_dataset)
    sess_resid = {}
    for e in eps:
        s = e["source_session"]
        if s not in sess_resid:
            sess_resid[s] = (e.get("eeg_resid_vs_eprime_rms_ms"),
                             e.get("eeg_resid_rms_ms"))
    bad_sessions = {}
    for s, (vep, vpc) in sess_resid.items():
        if (vep is not None and vep > 50) or (vpc is not None and vpc > 1000):
            bad_sessions[s] = (vep, vpc)

    rows = []
    cat = {k: [] for k in [
        "missing_eeg", "missing_emg", "missing_emg_imu", "missing_eye_gaze",
        "missing_eye_imu", "missing_hand_glove", "missing_vive",
        "no_video", "missing_camera", "eeg_misaligned",
        "pathological_clock", "low_coverage", "missing_phase",
        "invalid_phase", "no_execution", "no_task_id",
        "short_duration", "long_duration"]}

    for e in eps:
        idx = e["episode_index"]
        mods = set(e["modalities"])
        cov = e.get("coverage", {})
        dur = e["duration_s"]
        vep = e.get("eeg_resid_vs_eprime_rms_ms")
        vpc = e.get("eeg_resid_rms_ms")
        phases = e.get("phases", {})
        probs = []
        miss_mod = []
        low_cov = []
        miss_ph = []

        for m in SIGNAL_MODS:
            if m not in mods:
                miss_mod.append(m)
                if f"missing_{m}" in cat:
                    cat[f"missing_{m}"].append(idx)
        if not (HAND_GLOVE_ALTERNATIVES & mods):
            miss_mod.append("hand_glove")
            cat["missing_hand_glove"].append(idx)
        miss_cam = [c for c in CAMERAS if c not in mods]
        if not e.get("videos"):
            probs.append("no_video"); cat["no_video"].append(idx)
        elif miss_cam:
            probs.append("missing_camera:" + ",".join(c.split(".")[1] for c in miss_cam))
            cat["missing_camera"].append(idx)
        if miss_mod:
            probs.append("missing:" + ",".join(miss_mod))
        if "eeg" not in mods:
            probs.append("no_eeg")
        elif vep is not None and vep > VS_EPRIME_OK_MS:
            probs.append(f"eeg_misaligned({vep:.2f}ms)")
            cat["eeg_misaligned"].append(idx)
        if e["source_session"] in bad_sessions:
            probs.append("pathological_clock_session")
            cat["pathological_clock"].append(idx)

        # low coverage (dropout) for present streams
        for m in mods:
            if m in FS and m in cov and dur > 0:
                expected = dur * FS[m]
                if expected > 0 and cov[m] / expected < LOW_COV:
                    low_cov.append(f"{m}:{cov[m]}/{int(expected)}")
        if low_cov:
            probs.append("low_coverage:" + ";".join(low_cov))
            cat["low_coverage"].append(idx)

        # phase completeness / validity
        for ph in PHASES:
            if ph not in phases:
                miss_ph.append(ph)
        if miss_ph:
            probs.append("missing_phase:" + ",".join(miss_ph))
            cat["missing_phase"].append(idx)
        for ph, (s0, s1) in phases.items():
            if s1 <= s0:
                probs.append(f"invalid_phase:{ph}")
                cat["invalid_phase"].append(idx)
                break
        if not e.get("has_execution"):
            cat["no_execution"].append(idx)
            if "missing_phase:execution" not in " ".join(probs):
                probs.append("no_execution")
        if e["task_index"] == -1:
            probs.append("no_task_id"); cat["no_task_id"].append(idx)
        if dur < DUR_MIN_S:
            probs.append(f"short_duration({dur:.1f}s)"); cat["short_duration"].append(idx)
        if dur > DUR_MAX_S:
            probs.append(f"long_duration({dur:.1f}s)"); cat["long_duration"].append(idx)

        if probs:
            rows.append({
                "episode_index": idx,
                "source_session": e["source_session"],
                "task_index": e["task_index"],
                "task_id": e["task_id"],
                "task_slug": e.get("task_slug", ""),
                "trial": e["trial"],
                "duration_s": dur,
                "quality_tier": e.get("quality_tier"),
                "eeg_vs_eprime_ms": vep,
                "eeg_vs_pc_ms": vpc,
                "problems": " | ".join(probs),
            })

    # ---- CSV ----
    cols = ["episode_index", "source_session", "task_index", "task_id",
            "task_slug", "trial", "duration_s", "quality_tier",
            "eeg_vs_eprime_ms", "eeg_vs_pc_ms", "problems"]
    with open(META / "problem_episodes.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["episode_index"]):
            w.writerow(r)

    # ---- MD ----
    n = len(eps)
    nprob = len(rows)
    L = []
    L.append(f"# 问题数据全面排查清单 (record/{DS.name}/)\n")
    L.append(f"- 总 episodes: **{n}**；存在问题的: **{nprob}** ({100*nprob/n:.1f}%)；"
             f"完全无问题: **{n-nprob}** ({100*(n-nprob)/n:.1f}%)。")
    L.append(f"- 每条问题 episode 的完整明细见同目录 `problem_episodes.csv`。\n")

    L.append("## 0. 各类问题计数")
    L.append("| 问题类型 | 条数 |")
    L.append("|---|---|")
    label = {
        "missing_eeg": "缺 EEG", "missing_emg": "缺 EMG",
        "missing_emg_imu": "缺 EMG-IMU", "missing_eye_gaze": "缺 眼动注视",
        "missing_eye_imu": "缺 眼动IMU", "missing_hand_glove": "缺手套数据（触觉/Manus均无）",
        "missing_vive": "缺 Vive", "no_video": "完全无视频",
        "missing_camera": "缺至少一路相机", "eeg_misaligned": f"EEG错配(vs_eprime>{VS_EPRIME_OK_MS}ms)",
        "pathological_clock": "坏钟 session 的 episode", "low_coverage": "模态掉线/低覆盖(<50%)",
        "missing_phase": "缺相位", "invalid_phase": "无效相位窗",
        "no_execution": "无执行段", "no_task_id": "无 TASK_ID 标记",
        "short_duration": "时长过短(<5s)", "long_duration": "时长过长(>60s)"}
    for k in cat:
        L.append(f"| {label.get(k,k)} | {len(cat[k])} |")
    L.append("")

    def table(idxs, title, max_rows=None):
        L.append(f"## {title}（{len(idxs)} 条）")
        if not idxs:
            L.append("- 无。\n"); return
        rowmap = {r["episode_index"]: r for r in rows}
        L.append("| episode | session | task_slug | trial | dur(s) | vs_eprime(ms) | vs_pc(ms) | 问题 |")
        L.append("|---|---|---|---|---|---|---|---|")
        show = idxs if max_rows is None else idxs[:max_rows]
        for i in show:
            r = rowmap.get(i)
            if not r:
                continue
            vep = "" if r["eeg_vs_eprime_ms"] is None else f"{r['eeg_vs_eprime_ms']:.3f}"
            vpc = "" if r["eeg_vs_pc_ms"] is None else f"{r['eeg_vs_pc_ms']:.1f}"
            L.append(f"| {i} | {r['source_session']} | {r['task_slug']} | {r['trial']} | "
                     f"{r['duration_s']:.1f} | {vep} | {vpc} | {r['problems']} |")
        if max_rows is not None and len(idxs) > max_rows:
            L.append(f"| ... | 其余 {len(idxs)-max_rows} 条见 CSV | | | | | | |")
        L.append("")

    # critical first (full tables)
    L.append("---\n## 严重问题（建议排除，逐条列出）")
    table(sorted(set(cat["pathological_clock"])), "1. 坏钟 session 的 episode")
    # bad session summary
    L.append("### 坏钟 session 汇总")
    L.append("| session | vs_eprime(ms) | vs_pc(ms) | episode_index 列表 |")
    L.append("|---|---|---|---|")
    for s, (vep, vpc) in sorted(bad_sessions.items(), key=lambda x: -(x[1][0] or 0)):
        idxs = [r["episode_index"] for r in rows if r["source_session"] == s]
        veps = "" if vep is None else f"{vep:.3f}"
        vpcs = "" if vpc is None else f"{vpc:.1f}"
        L.append(f"| {s} | {veps} | {vpcs} | {idxs} |")
    L.append("")
    table(sorted(set(cat["eeg_misaligned"])), "2. EEG 错配 (vs_eprime>2ms)")
    table(sorted(set(cat["invalid_phase"])), "3. 无效相位窗")
    table(sorted(set(cat["short_duration"])), "4. 时长过短")
    table(sorted(set(cat["long_duration"])), "5. 时长过长")
    table(sorted(set(cat["no_task_id"])), "6. 无 TASK_ID 标记")
    table(sorted(set(cat["low_coverage"])), "7. 模态掉线/低覆盖")

    L.append("---\n## 缺失类问题（逐条列出）")
    table(sorted(
        [r["episode_index"] for r in rows if "no_eeg" in r["problems"]]),
        "8. 无 EEG")
    table(sorted(set(cat["no_video"])), "9. 完全无视频")
    table(sorted(set(cat["missing_camera"])), "10. 缺至少一路相机", max_rows=60)
    table(sorted(set(cat["missing_hand_glove"])), "11. 缺手套数据", max_rows=60)
    table(sorted(set(cat["missing_eye_gaze"])), "12. 缺眼动注视", max_rows=60)
    table(sorted(set(cat["missing_vive"])), "13. 缺 Vive", max_rows=60)
    table(sorted(set(cat["missing_emg"])), "14. 缺 EMG", max_rows=60)
    table(sorted(set(cat["missing_phase"])), "15. 缺相位（含无执行）", max_rows=60)

    (META / "PROBLEMS.md").write_text("\n".join(L), encoding="utf-8")

    print(f"[problems] episodes={n} with_problems={nprob} clean={n-nprob}")
    for k in cat:
        if cat[k]:
            print(f"  {label.get(k,k):28s} {len(set(cat[k]))}")
    print(f"[problems] wrote {META/'PROBLEMS.md'} + {META/'problem_episodes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
