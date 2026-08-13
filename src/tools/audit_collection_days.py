#!/usr/bin/env python3
"""Audit multi-day collection: sessions + Curry EEG on Desktop.

Usage:
  python tools/audit_collection_days.py --days 2026-05-28 2026-05-29
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "sessions"
EEG_ROOT = Path(r"C:\Users\31454\Desktop\Acquisition")

ESSENTIAL = ("eye", "tactile", "emg", "wrist_cam")
OPTIONAL = ("vive",)
IGNORE = ("manus",)

SESSION_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<subject>[^_]+)_t(?P<task>\d+)_run(?P<run>\d+)_p(?P<paradigm>\d+)$"
)
EEG_RE = re.compile(r"Acq (?P<date>\d{4}_\d{2}_\d{2})_(?P<time>\d{4})\.cdt$")


@dataclass
class EegFile:
    path: Path
    dt: datetime
    size_mb: float


@dataclass
class SessionAudit:
    name: str
    dt: datetime
    subject: str
    task_id: int
    run: int
    paradigm: str
    enabled: list[str]
    end_reason: str | None
    aborted: bool
    degraded: list[str]
    modality_status: dict[str, str] = field(default_factory=dict)
    markers_mb: float = 0.0
    sanity_overall_ok: bool | None = None
    eeg_match: EegFile | None = None
    tier: str = "UNKNOWN"
    notes: list[str] = field(default_factory=list)


def _parse_session_name(name: str) -> dict | None:
    if "smoketest" in name or "smoke_" in name:
        return None
    m = SESSION_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    dt = datetime.strptime(f"{g['date']} {g['time'].replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
    return {**g, "dt": dt}


def _load_session(path: Path) -> dict:
    sj = path / "session.json"
    if not sj.exists():
        return {}
    try:
        return json.loads(sj.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _modality_status_from_sanity(session: Path, enabled: set[str]) -> dict[str, str]:
    sr = session / "sanity_report.json"
    if sr.exists():
        try:
            rep = json.loads(sr.read_text(encoding="utf-8"))
            return {c["name"]: c["status"] for c in rep.get("checks", [])}
        except Exception:
            pass
    # fallback: file presence
    out: dict[str, str] = {}
    for m in ESSENTIAL + OPTIONAL + IGNORE:
        if m not in enabled and enabled:
            out[m] = "DISABLED"
            continue
        sub = session / m
        npz = sub / f"{m}.npz"
        if npz.exists() and npz.stat().st_size >= 1024:
            out[m] = "OK"
        elif (sub.exists() and any(sub.iterdir())):
            out[m] = "PARTIAL"
        else:
            out[m] = "MISSING" if m in enabled else "DISABLED"
    return out


def _scan_eeg(days: list[str]) -> list[EegFile]:
    files: list[EegFile] = []
    for day in days:
        folder = EEG_ROOT / day.replace("-", "_")
        if not folder.is_dir():
            continue
        for p in folder.glob("Acq *.cdt"):
            if p.name.endswith(".cdt.ceo") or p.name.endswith(".cdt.dpo"):
                continue
            m = EEG_RE.search(p.name)
            if not m:
                continue
            d = m.group("date").replace("_", "-")
            t = m.group("time")
            dt = datetime.strptime(f"{d} {t[:2]}:{t[2:4]}", "%Y-%m-%d %H:%M")
            files.append(EegFile(path=p, dt=dt, size_mb=p.stat().st_size / (1024 * 1024)))
    return sorted(files, key=lambda x: x.dt)


def _match_eeg_greedy(
    sessions: list[SessionAudit], eeg_files: list[EegFile]
) -> set[Path]:
    """Pair sessions with Curry .dpo files by closest clock time.

    Curry file names use minute resolution (HHMM) and may reflect either
    start or stop time depending on operator habit, so we allow a wide
    window (±90 min) and assign each .dpo to at most one session.
    """
    available = [e for e in eeg_files if e.size_mb >= 0.5]
    used: set[Path] = set()
    for a in sorted(sessions, key=lambda x: x.dt):
        best: EegFile | None = None
        best_abs: float = 9999.0
        for e in available:
            if e.path in used:
                continue
            delta = (a.dt - e.dt).total_seconds() / 60.0
            if abs(delta) > 90:
                continue
            if abs(delta) < best_abs:
                best_abs = abs(delta)
                best = e
        if best is not None:
            used.add(best.path)
            a.eeg_match = best
    return used


def _classify(a: SessionAudit) -> str:
    if a.aborted:
        return "ABORTED"
    essential_ok = all(a.modality_status.get(m) == "OK" for m in ESSENTIAL if m in a.enabled)
    if not essential_ok:
        missing = [m for m in ESSENTIAL if m in a.enabled and a.modality_status.get(m) != "OK"]
        if missing:
            a.notes.append(f"essential missing: {missing}")
        return "FAILED"

    markers_ok = a.markers_mb >= 0.001 or a.sanity_overall_ok is True
    if not markers_ok:
        a.notes.append("markers.npz tiny/missing")
        return "PARTIAL"

    vive_ok = a.modality_status.get("vive") == "OK" if "vive" in a.enabled else True
    eeg_ok = a.eeg_match is not None and a.eeg_match.size_mb >= 50.0

    if a.sanity_overall_ok and eeg_ok and vive_ok:
        return "FULL"
    if a.sanity_overall_ok and eeg_ok:
        if not vive_ok:
            a.notes.append("vive missing/degraded")
        return "USABLE"
    if a.sanity_overall_ok:
        a.notes.append("no matched EEG .cdt")
        return "USABLE_MM"  # multimodal OK, EEG pairing uncertain
    if essential_ok and markers_ok and eeg_ok:
        return "USABLE"
    if essential_ok and markers_ok:
        a.notes.append("sanity not run or failed")
        return "PARTIAL"
    return "PARTIAL"


def audit(days: list[str]) -> list[SessionAudit]:
    eeg_files = _scan_eeg(days)
    used_eeg: set[Path] = set()
    audits: list[SessionAudit] = []

    for sess in sorted(SESSIONS.iterdir()):
        if not sess.is_dir():
            continue
        meta = _parse_session_name(sess.name)
        if meta is None:
            continue
        if meta["date"] not in days:
            continue
        if meta["subject"] != "subj01":
            continue

        data = _load_session(sess)
        enabled = [r["name"] for r in data.get("recorders", []) if isinstance(r, dict)]
        enabled_set = set(enabled)

        sr_path = sess / "sanity_report.json"
        sanity_ok = None
        if sr_path.exists():
            try:
                sanity_ok = json.loads(sr_path.read_text(encoding="utf-8")).get("overall_ok")
            except Exception:
                pass

        markers = sess / "markers.npz"
        markers_mb = markers.stat().st_size / (1024 * 1024) if markers.exists() else 0.0

        a = SessionAudit(
            name=sess.name,
            dt=meta["dt"],
            subject=meta["subject"],
            task_id=int(meta["task"]),
            run=int(meta["run"]),
            paradigm=meta["paradigm"],
            enabled=enabled,
            end_reason=data.get("end_reason"),
            aborted=bool(data.get("aborted", False)),
            degraded=list(data.get("degraded_modalities") or []),
            modality_status=_modality_status_from_sanity(sess, enabled_set),
            markers_mb=markers_mb,
            sanity_overall_ok=sanity_ok,
        )

        audits.append(a)

    used_eeg = _match_eeg_greedy(audits, eeg_files)
    for a in audits:
        a.notes.clear()
        a.tier = _classify(a)

    return audits, eeg_files, used_eeg


def _print_report(audits: list[SessionAudit], eeg_files: list[EegFile], used_eeg: set[Path], days: list[str]) -> None:
    from collections import Counter

    tiers = Counter(a.tier for a in audits)
    by_task: dict[int, list[SessionAudit]] = {}
    for a in audits:
        by_task.setdefault(a.task_id, []).append(a)

    print("=" * 72)
    print(f"采集审计  days={', '.join(days)}  subject=subj01")
    print("=" * 72)
    print(f"\nSession 总数: {len(audits)}")
    print("分级统计:")
    for tier in ("FULL", "USABLE", "USABLE_MM", "PARTIAL", "FAILED", "ABORTED"):
        if tiers.get(tier):
            print(f"  {tier:8s}: {tiers[tier]}")

    print(f"\nEEG (Curry .cdt.dpo) 文件数: {len(eeg_files)}")
    unmatched = [e for e in eeg_files if e.path not in used_eeg]
    print(f"  已匹配到 session: {len(used_eeg)}")
    print(f"  未匹配（可能单独录/测试）: {len(unmatched)}")
    if unmatched:
        for e in unmatched:
            print(f"    - {e.path.name}  ({e.size_mb:.1f} MB)  {e.dt}")

    print("\n--- 按 task 汇总（每个 task 取最佳 run）---")
    tasks = {
        0: "把石榴放进粉色的碗中",
        1: "把石榴从粉色的碗中取出",
        2: "将葡萄放进蓝色的碗中",
        3: "把葡萄从蓝色的碗中取出",
        4: "葡萄取出并放入粉色碗",
        5: "魔方右侧一列调蓝",
        6: "右手拿起魔方并放下",
        7: "右手拧开黑色瓶盖",
        8: "右手拧紧黑色瓶盖",
    }
    for tid in sorted(by_task.keys()):
        tier_rank = {"FULL": 0, "USABLE": 1, "USABLE_MM": 2, "PARTIAL": 3, "FAILED": 4, "ABORTED": 5}
        runs = sorted(by_task[tid], key=lambda x: (tier_rank.get(x.tier, 9), -x.run))
        best = runs[0]
        full_count = sum(1 for r in runs if r.tier == "FULL")
        usable_count = sum(1 for r in runs if r.tier in ("USABLE", "USABLE_MM"))
        print(f"  t{tid} {tasks.get(tid, '?')[:20]:20s}  runs={len(runs)}  FULL={full_count} USABLE*={usable_count}  best={best.name[-24:]} tier={best.tier}")

    print("\n--- 全部 session 明细 ---")
    hdr = f"{'time':16} {'task':3} {'run':4} {'tier':8} {'sanity':6} {'EEG':5} {'vive':5}  modalities / notes"
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(audits, key=lambda x: x.dt):
        ess = "".join(
            (a.modality_status.get(m, "?")[0] if m in a.enabled else "-")
            for m in ("e", "t", "g", "w")  # eye tactile emg wrist
        )
        vive = a.modality_status.get("vive", "-")[:3] if "vive" in a.enabled else "off"
        eeg = "yes" if a.eeg_match else "no"
        sanity = "OK" if a.sanity_overall_ok else ("-" if a.sanity_overall_ok is None else "NO")
        note = "; ".join(a.notes) if a.notes else a.end_reason or ""
        print(
            f"{a.dt.strftime('%m-%d %H:%M'):16} t{a.task_id:<2} r{a.run:<3} "
            f"{a.tier:8} {sanity:6} {eeg:5} {vive:5}  [{ess}] {note}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=["2026-05-28", "2026-05-29"])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    audits, eeg_files, used_eeg = audit(args.days)
    _print_report(audits, eeg_files, used_eeg, args.days)

    if args.json:
        payload = [
            {
                "session": a.name,
                "dt": a.dt.isoformat(),
                "task_id": a.task_id,
                "run": a.run,
                "tier": a.tier,
                "enabled": a.enabled,
                "modality_status": a.modality_status,
                "eeg": str(a.eeg_match.path) if a.eeg_match else None,
                "notes": a.notes,
            }
            for a in audits
        ]
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
