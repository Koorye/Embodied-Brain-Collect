"""Stim 工厂 —— launcher 按名字构建刺激程序子进程命令。

与 recorder 的 ``recorder_presets.FACTORY_BY_KIND`` 同构:每个 kind 一个
模块名,``build_stim_cmd`` 返回 launcher 可以直接 ``subprocess.Popen`` 的
argv。除了 kind 与任务 id(paradigm1 的任务轮转)之外,**任何参数都不在这里
传** —— 全屏/窗口、时间压缩、串口开关、各阶段时长等一律由 stim 程序自己
从 ``configs/stim.yaml`` 读取,保证参数只有一处来源。
"""

from __future__ import annotations

import sys

#: kind -> 模块名
STIM_KINDS: dict[str, str] = {
    "paradigm1": "embodied_brain_collect.stim.paradigm1_pickplace",
    "sync_test": "embodied_brain_collect.stim.sync_test",
}


def build_stim_cmd(kind: str, *, task_id: int | None = None) -> list[str]:
    """一个刺激程序的子进程 argv。"""
    if kind not in STIM_KINDS:
        raise ValueError(f"未知 stim kind: {kind!r} (可用: {sorted(STIM_KINDS)})")

    argv = [sys.executable, "-m", STIM_KINDS[kind]]
    if kind == "paradigm1":
        argv.append("--once")
        if task_id is not None:
            argv += ["--task-id", str(task_id)]
    return argv
