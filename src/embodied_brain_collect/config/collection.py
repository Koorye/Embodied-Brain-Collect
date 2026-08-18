"""Collection config and per-task run numbering.

Files under ``record/config/``:

  collection.json       Task library + subject / stim knobs + recorder configs.
  run_counters.json     Auto-maintained: next run index per (subject, task_id).
  active_session.json   Written when launcher starts; stim reads the same run.

Each recording session picks **one** task (``--task-id``). Run numbers for the
same task auto-increment across sessions until you reset counters manually.

Recorder configurations are stored in ``collection.json`` under the
``recorders`` key, keyed by recorder name (e.g. ``"eye"``, ``"emg"``).
Use ``add_recorder()`` / ``remove_recorder()`` / ``enable()`` / ``disable()``
to manage which devices are active.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..recorders.base import *


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION_PATH = CONFIG_DIR / "collection.json"
EXAMPLE_COLLECTION_PATH = CONFIG_DIR / "collection.example.json"
RUN_COUNTERS_PATH = CONFIG_DIR / "run_counters.json"
ACTIVE_SESSION_PATH = CONFIG_DIR / "active_session.json"

# Mapping from recorder name to its config class (for JSON deserialization).
RECORDER_CONFIG_CLASSES: dict[str, type[BaseRecorderConfig]] = {
    # "emg": DummyEmgRecorderConfig,
}


def _serialize_recorders(recorders: dict[str, BaseRecorderConfig]) -> dict[str, Any]:
    """Serialize all recorder configs to a JSON-safe dict."""
    return {name: cfg.to_dict() for name, cfg in recorders.items()}


def _deserialize_recorders(raw: dict[str, Any]) -> dict[str, BaseRecorderConfig]:
    """Deserialize recorder configs from JSON, using the class registry."""
    result: dict[str, BaseRecorderConfig] = {}
    for name, data in raw.items():
        cls = RECORDER_CONFIG_CLASSES.get(name)
        if cls is None:
            print(f"[config] WARN: unknown recorder '{name}', skipping.")
            continue
        result[name] = cls.from_dict(data)
    return result


@dataclass
class SyncHubConfig:
    """sync_hub network settings."""
    bind: str = "127.0.0.1"
    udp_port: int = 9999
    zmq_port: int = 9998
    neon_ip: str = "172.16.19.213"


@dataclass
class StimConfig:
    parallelbox: str = "COM14"
    baud: int = 115200
    hold_s: float = 0.020
    marker_host: str = "127.0.0.1"
    marker_port: int = 9999
    fullscreen: bool = True
    fix_pre_s: float = 2.0
    instr_s: float = 10.0
    font_size: int = 64
    fix_radius: int = 14


@dataclass
class TaskConfig:
    task_id: int
    task_name: str
    scene: int = 0


@dataclass
class CollectionConfig:
    """Master config for a collection session.

    Manages tasks, stim settings, sync_hub, and a flexible dict of recorder
    configs that can be added/removed/enabled/disabled independently.
    """

    subject: str = "subj01"
    paradigm: str = "1"
    notes: str = ""
    sync_hub: SyncHubConfig = field(default_factory=SyncHubConfig)
    stim: StimConfig = field(default_factory=StimConfig)
    tasks: list[TaskConfig] = field(default_factory=list)
    recorders: dict[str, BaseRecorderConfig] = field(default_factory=dict)

    # -- Recorder management (fluent API) ----------------------------------

    def add_recorder(self, name: str, config: BaseRecorderConfig | None = None) -> "CollectionConfig":
        """Add or replace a recorder.  If *config* is None, a default config is
        created from ``RECORDER_CONFIG_CLASSES``."""
        if config is None:
            cls = RECORDER_CONFIG_CLASSES.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown recorder '{name}'. Known: {sorted(RECORDER_CONFIG_CLASSES)}"
                )
            config = cls(enabled=True)
        self.recorders[name] = config
        return self

    def remove_recorder(self, name: str) -> "CollectionConfig":
        """Remove a recorder entirely."""
        self.recorders.pop(name, None)
        return self

    def enable(self, name: str) -> "CollectionConfig":
        """Enable a recorder for collection."""
        if name in self.recorders:
            self.recorders[name].enabled = True
        return self

    def disable(self, name: str) -> "CollectionConfig":
        """Disable a recorder (keep config but skip during collection)."""
        if name in self.recorders:
            self.recorders[name].enabled = False
        return self

    def get_enabled(self) -> dict[str, BaseRecorderConfig]:
        """Return only enabled recorder configs."""
        return {k: v for k, v in self.recorders.items() if v.enabled}

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recorders"] = _serialize_recorders(self.recorders)
        return d

    def to_json_file(self, path: Path) -> None:
        """Write the full config to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        path.write_text(
            path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CollectionConfig":
        # Sync hub
        sync_raw = dict(d.pop("sync_hub", {}) or {})
        sync_hub = SyncHubConfig(
            **{k: v for k, v in sync_raw.items()
               if k in SyncHubConfig.__dataclass_fields__}
        )
        # Stim
        stim_raw = dict(d.pop("stim", {}) or {})
        stim = StimConfig(
            **{k: v for k, v in stim_raw.items() if k in StimConfig.__dataclass_fields__}
        )
        # Tasks
        tasks_raw = d.pop("tasks", None)
        if tasks_raw is None:
            tasks_raw = d.pop("trials", []) or []
        tasks: list[TaskConfig] = []
        for row in tasks_raw:
            if "task_name" not in row and "instruction" in row:
                row = {**row, "task_name": row["instruction"]}
            if "task_id" not in row and "task" in row:
                row = {**row, "task_id": row["task"]}
            tasks.append(TaskConfig(
                task_id=int(row["task_id"]),
                task_name=str(row["task_name"]).strip(),
                scene=int(row.get("scene", 0)),
            ))
        # Recorders
        recorders_raw = d.pop("recorders", {}) or {}
        recorders = _deserialize_recorders(recorders_raw)
        # Remaining fields
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(sync_hub=sync_hub, stim=stim, tasks=tasks,
                   recorders=recorders, **fields)

    def validate(self) -> None:
        if not self.subject.strip():
            raise ValueError("subject must not be empty")
        if not self.tasks:
            raise ValueError("tasks list must not be empty")
        seen: set[int] = set()
        for i, t in enumerate(self.tasks):
            if not t.task_name:
                raise ValueError(f"tasks[{i}].task_name must not be empty")
            if t.task_id in seen:
                raise ValueError(f"duplicate task_id {t.task_id}")
            seen.add(t.task_id)

    def get_task(self, task_id: int) -> TaskConfig:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        ids = ", ".join(str(t.task_id) for t in self.tasks)
        raise KeyError(f"task_id {task_id} not in config (available: {ids})")

    def format_task_list(self) -> str:
        lines = [f"subject={self.subject}  tasks={len(self.tasks)}", ""]
        for t in self.tasks:
            lines.append(f"  --task-id {t.task_id:2d}  scene={t.scene}  {t.task_name}")
        return "\n".join(lines)

    def format_recorder_list(self) -> str:
        lines = [f"recorders ({len(self.recorders)} total, "
                 f"{len(self.get_enabled())} enabled):"]
        for name, cfg in sorted(self.recorders.items()):
            status = "🟢 ON" if cfg.enabled else "⚫ OFF"
            lines.append(f"  {status}  {name:12s}  {type(cfg).__name__}")
        return "\n".join(lines)


@dataclass
class ActiveSession:
    subject: str
    task_id: int
    task_name: str
    run: int
    scene: int
    paradigm: str
    neon_ip: str
    notes: str
    started_at: str
    session_dir: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_collection_config(path: Path | str | None = None) -> CollectionConfig:
    p = Path(path) if path else DEFAULT_COLLECTION_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"collection config not found: {p}\n"
            f"Copy {EXAMPLE_COLLECTION_PATH.name} to {p.name} and edit."
        )
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cfg = CollectionConfig.from_dict(raw)
    cfg.validate()
    return cfg


def _load_run_counters() -> dict[str, dict[str, int]]:
    if not RUN_COUNTERS_PATH.is_file():
        return {}
    with RUN_COUNTERS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): {str(tid): int(v) for tid, v in sub.items()}
            for k, sub in data.items()}


def _save_run_counters(data: dict[str, dict[str, int]]) -> None:
    RUN_COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_COUNTERS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def peek_next_run(subject: str, task_id: int) -> int:
    """Next run number that would be allocated (without consuming)."""
    data = _load_run_counters()
    return data.get(subject, {}).get(str(task_id), 1)


def allocate_run(subject: str, task_id: int) -> int:
    """Reserve the next run index for (subject, task_id) and persist."""
    data = _load_run_counters()
    subj = data.setdefault(subject, {})
    run = int(subj.get(str(task_id), 1))
    subj[str(task_id)] = run + 1
    _save_run_counters(data)
    return run


def write_active_session(session: ActiveSession) -> Path:
    ACTIVE_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE_SESSION_PATH.open("w", encoding="utf-8") as fh:
        json.dump(session.to_json(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return ACTIVE_SESSION_PATH


def read_active_session() -> ActiveSession | None:
    if not ACTIVE_SESSION_PATH.is_file():
        return None
    with ACTIVE_SESSION_PATH.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    return ActiveSession(**d)


def prepare_session(
    task_id: int,
    *,
    config_path: Path | str | None = None,
    consume_run: bool = True,
) -> tuple[CollectionConfig, ActiveSession]:
    """Load config, pick one task, allocate run, write active_session.json."""
    cfg = load_collection_config(config_path)
    task = cfg.get_task(task_id)
    run = allocate_run(cfg.subject, task_id) if consume_run else peek_next_run(cfg.subject, task_id)
    active = ActiveSession(
        subject=cfg.subject,
        task_id=task.task_id,
        task_name=task.task_name,
        run=run,
        scene=task.scene,
        paradigm=cfg.paradigm,
        neon_ip=cfg.sync_hub.neon_ip,
        notes=cfg.notes,
        started_at=time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    write_active_session(active)
    return cfg, active


def resolve_active_for_stim(
    task_id: int | None,
    *,
    config_path: Path | str | None = None,
    allow_allocate: bool = False,
) -> tuple[CollectionConfig, ActiveSession]:
    """Stim: use active_session if task_id matches; else require explicit task_id."""
    cfg = load_collection_config(config_path)
    active = read_active_session()

    if task_id is None:
        if active is None:
            raise SystemExit(
                "no --task-id and no active_session.json.\n"
                "Start launcher first, or pass --task-id explicitly.\n\n"
                + cfg.format_task_list()
            )
        if active.subject != cfg.subject:
            print(f"[stim] warning: active subject {active.subject!r} != config {cfg.subject!r}")
        return cfg, active

    task = cfg.get_task(task_id)
    if active is not None and active.task_id == task_id and active.subject == cfg.subject:
        return cfg, active

    if allow_allocate:
        run = allocate_run(cfg.subject, task_id)
    else:
        run = peek_next_run(cfg.subject, task_id)
        print(
            f"[stim] no matching active_session; using run={run} (peek only). "
            "Start launcher with --task-id first for a reserved run number."
        )

    active = ActiveSession(
        subject=cfg.subject,
        task_id=task.task_id,
        task_name=task.task_name,
        run=run,
        scene=task.scene,
        paradigm=cfg.paradigm,
        neon_ip=cfg.sync_hub.neon_ip,
        notes=cfg.notes,
        started_at=time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    write_active_session(active)
    return cfg, active


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="List tasks or show next run numbers.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--show-runs", action="store_true", help="Show next run per task")
    args = ap.parse_args()
    cfg = load_collection_config(args.config)
    print(cfg.format_task_list())
    if args.show_runs:
        print("\nNext run (per subject + task_id):")
        for t in cfg.tasks:
            n = peek_next_run(cfg.subject, t.task_id)
            print(f"  task {t.task_id}: run {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())