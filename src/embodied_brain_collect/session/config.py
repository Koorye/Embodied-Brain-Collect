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


def load_stim() -> dict:
    """Stim/marker transport settings (ParallelBox COM, UDP, timing).

    刻意不缓存(其他 loader 缓存没问题):stim 程序一次进程只读一次,而
    测试/多目录部署靠 ``EMBODIED_BRAIN_COLLECT_CONFIGS`` 切目录 —— 缓存
    键没有目录,先读仓库配置再切隔离目录会串读(曾经让 headless 测试拿到
    仓库的 serial: true 去开不存在的串口)。
    """
    return _read_yaml("stim.yaml")


@lru_cache(maxsize=None)
def load_checker() -> dict:
    """``{check_class_lower: {param: value}}`` threshold overrides."""
    return _read_yaml("checker.yaml")


def load_displays() -> dict:
    """每屏显示设置(configs/displays.yaml);文件缺失 = 全部默认全屏。"""
    path = configs_dir() / "displays.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def display_settings(n: int) -> dict:
    """屏幕 n 的显示设置:{mode: fullscreen|windowed, width, height}。

    所有程序(图纸/镜像/stim/辅助台)显示到该屏时一律采用这里的设置。
    未列出的屏幕默认全屏、分辨率跟随该屏;windowed 缺 width/height 同样
    跟随该屏;mode 写错的按 fullscreen 处理。
    """
    displays = load_displays().get("displays") or {}
    # yaml 会把数字键解析成 int,同时匹配 int/str 两种键型
    item = (displays.get(n) or displays.get(str(n)) or {})
    if not isinstance(item, dict):
        item = {}
    mode = str(item.get("mode", "fullscreen")).strip().lower()
    if mode not in ("fullscreen", "windowed"):
        mode = "fullscreen"
    return {"mode": mode,
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0)}


def load_session_run() -> dict:
    """run_session 的编排配置(mode/stim);session.yaml 缺失返回 {}。

    该文件是可选的运行期开关,不强制部署 —— CLI 显式传参优先于这里。
    """
    path = configs_dir() / "session.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


#: session.yaml 里属于运行期开关的键,不算采集信息
SESSION_RUN_KEYS = ("mode", "stim", "pack_episode")

#: session meta.yaml 的框架保留键 —— 采集信息不得占用;打包时也从信息
#: 集中剔除(其余键视为操作员维护的采集信息)
FRAMEWORK_KEYS = frozenset({
    "version", "framework", "collect_version", "session_dir", "started_at",
    "task_id", "task_name", "environment", "recorders", "objects", "scene",
})


def load_collect() -> dict:
    """session.yaml 顶层除运行期开关与框架保留键外的所有键 = 采集信息。

    编号(collector_id)等由操作员维护,键不设 schema —— 写什么数据集就
    带什么。开录时随 session 固化为 meta.yaml 的顶层字段(launcher 抄写),
    打包时由 pack_daily 汇总进 meta/collect_info.jsonl(同样顶层平铺,
    不嵌套)。run_session 的 CLI(--collector-id/--set)可逐键覆盖。
    缺失返回 {}。
    """
    return {k: v for k, v in load_session_run().items()
            if k not in SESSION_RUN_KEYS and k not in FRAMEWORK_KEYS
            and v is not None}


@lru_cache(maxsize=None)
def load_meta() -> dict:
    """Version / framework metadata, copied into each session dir."""
    return _read_yaml("meta.yaml")


@lru_cache(maxsize=None)
def load_markers() -> dict:
    """marker 码表(markers.yaml 的 codes/hand_cue_base);缺失返回 {}。

    代码侧有内置默认表兜底 —— 本文件缺失不影响启动。
    """
    return _read_yaml("markers.yaml")


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
