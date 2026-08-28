"""tasks.yaml 只读加载与查询(随机顺序由 run_session 在内存中采样)。"""

import pytest

from embodied_brain_collect.config import load as L


@pytest.fixture
def isolated_configs(tmp_path, monkeypatch):
    """隔离的 configs 目录,避免读到真实 tasks.yaml。"""
    monkeypatch.setenv("EMBODIED_BRAIN_COLLECT_CONFIGS", str(tmp_path))
    L.load_tasks.cache_clear()
    (tmp_path / "tasks.yaml").write_text(
        "# header\n"
        "tasks:\n"
        "- task_id: 1\n  task_name: 喝水\n"
        "- task_id: 2\n  task_name: 写字\n"
        "- task_id: 3\n  task_name: 开门\n",
        encoding="utf-8")
    return tmp_path


def test_load_tasks_reads_yaml_list(isolated_configs):
    tasks = L.load_tasks()
    assert [t["task_id"] for t in tasks] == [1, 2, 3]
    assert tasks[0]["task_name"] == "喝水"


def test_load_tasks_never_rewrites_the_file(isolated_configs):
    """任务库文件只读:加载前后内容一致(随机顺序在内存里,不落盘)。"""
    before = (isolated_configs / "tasks.yaml").read_text(encoding="utf-8")
    for _ in range(3):
        L.load_tasks.cache_clear()
        L.load_tasks()
    assert (isolated_configs / "tasks.yaml").read_text(encoding="utf-8") == before


def test_task_lookup_helpers(isolated_configs):
    assert L.task_by_id(2)["task_name"] == "写字"
    assert L.task_name(2) == "写字"
    assert L.task_by_id(99) is None
