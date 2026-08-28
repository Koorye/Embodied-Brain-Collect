"""Load the deployment configuration from the repo-root ``configs/`` dir.

The config files live OUTSIDE the package (``<repo>/configs/``), all in
YAML, so an operator edits them without touching the code.  Locating them:
walk up from this file to the repo root; the ``EMBODIED_BRAIN_COLLECT_CONFIGS``
env var overrides the directory for tests and unusual deployments.

All loaders return plain dicts/lists and raise ``FileNotFoundError`` when a
file is missing — pre-flight scripts should catch that and say which file.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml


def configs_dir() -> Path:
    """The directory holding the *.yaml config files."""
    env = os.environ.get("EMBODIED_BRAIN_COLLECT_CONFIGS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "configs"


def _read_yaml(name: str) -> dict:
    # 不缓存:文件都很小,且缓存键只有文件名会在测试/多目录场景串读
    path = configs_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} 不存在 — 请补全 configs/ 目录")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def load_recorders() -> dict:
    """``{slot: {kind, ...params}}`` from recorders.yaml."""
    return _read_yaml("recorders.yaml").get("recorders", {})


@lru_cache(maxsize=None)
def load_tasks() -> list[dict]:
    """``[{task_id, task_name}, ...]`` from tasks.yaml."""
    return _read_yaml("tasks.yaml").get("tasks", [])


@lru_cache(maxsize=None)
def load_stim() -> dict:
    """Stim/marker transport settings (ParallelBox COM, UDP, timing)."""
    return _read_yaml("stim.yaml")


@lru_cache(maxsize=None)
def load_checker() -> dict:
    """``{check_class_lower: {param: value}}`` threshold overrides."""
    return _read_yaml("checker.yaml")


@lru_cache(maxsize=None)
def load_meta() -> dict:
    """Version / framework metadata, copied into each session dir."""
    return _read_yaml("meta.yaml")


def task_by_id(task_id: int) -> dict | None:
    for t in load_tasks():
        if t.get("task_id") == task_id:
            return t
    return None


def task_name(task_id: int) -> str | None:
    t = task_by_id(task_id)
    return t.get("task_name") if t else None


# tasks.yaml 只读:任务库与默认顺序(task_id 升序)。运行期的随机顺序在
# run_session 内存中采样,不再改写文件 —— 文件被外部改写反而是 bug 来源。
