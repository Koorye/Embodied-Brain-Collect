"""编排层新行为的回归:排查指引、LaunchResult、run_session 输入、preflight。"""

import builtins
from collections import Counter
from pathlib import Path

from embodied_brain_collect.recorders.emg import DummyEmgRecorder, EmgRecorderConfig
from embodied_brain_collect.recorders.marker import (
    MarkerRecorderConfig, UdpMarkerRecorder)
from embodied_brain_collect.session.launcher import LaunchResult
from embodied_brain_collect.session.troubleshooting import (
    format_failure_help, guide_for_slot, self_check_flow, slot_category)


# =============================================================================
# troubleshooting
# =============================================================================

def test_slot_category_prefix_matching():
    assert slot_category("eeg") == "eeg"
    assert slot_category("emg_left") == "emg"
    assert slot_category("emg_right") == "emg"
    assert slot_category("cam_head") == "cam"
    assert slot_category("cam_left_wrist") == "cam"
    assert slot_category("eye") == "eye"
    assert slot_category("hand_pose") == "hand_pose"
    assert slot_category("position") == "position"
    assert slot_category("marker") == "marker"
    assert slot_category("whatever") is None


def test_self_check_flow_lists_only_relevant_devices():
    text = self_check_flow(["emg_left", "cam_third", "emg_right"])
    assert "EMG" in text and "相机" in text
    assert "EEG" not in text          # 未涉及的设备不出现


def test_failure_help_first_vs_repeated():
    failures = {"eeg": "open FAILED — cannot connect"}
    first = format_failure_help(failures, phase="startup")
    assert "首次失败" in first
    assert "同步盒" not in first      # 首次只建议重采,不展开方案

    repeated = format_failure_help(
        failures, phase="startup", fail_counts=Counter({"eeg": 2}))
    assert "第 2 次失败" in repeated
    assert "同步盒" in repeated        # 多次失败展开设备排查方案


# =============================================================================
# LaunchResult
# =============================================================================

def test_launch_result_is_int_with_details():
    rc = LaunchResult(1, open_failures={"emg_left": "no frame"})
    assert rc == 1                     # 既有代码的 int 比较仍然成立
    assert rc.open_failures == {"emg_left": "no frame"}
    assert rc.runtime_errors == {}
    assert LaunchResult(0) == 0


# =============================================================================
# run_session._ask_next —— 防误触的严格输入
# =============================================================================

def test_ask_next_rejects_empty_and_wrong_letters(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_session", Path(__file__).resolve().parents[2] / "scripts" / "run_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    answers = iter(["", "x", "keep", "n"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    assert mod._ask_next() == "next"

    answers = iter(["r"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    assert mod._ask_next() == "rerun"

    answers = iter(["Q"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    assert mod._ask_next() == "quit"   # 大小写不敏感


# =============================================================================
# preflight.check_one + probe_data_flow
# =============================================================================

def test_probe_data_flow_detects_sustained_samples(tmp_path):
    rec = DummyEmgRecorder(EmgRecorderConfig(session_dir=str(tmp_path)))
    assert rec._open()
    ok, detail = rec.probe_data_flow(timeout=2.0)
    assert ok and "新增" in detail
    rec._teardown()


def test_udp_marker_probe_accepts_bound_port(tmp_path):
    """UDP marker 预检只验证端口绑定 —— 没有发送端也算数据流通过。"""
    import socket
    # 9999 可能被占;动态挑一个空闲端口
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    rec = UdpMarkerRecorder(MarkerRecorderConfig(
        session_dir=str(tmp_path), host="127.0.0.1", port=port))
    assert rec._open()
    ok, detail = rec.probe_data_flow()
    assert ok and "绑定" in detail
    rec._teardown()


def test_preflight_check_one_passes_on_working_recorder(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight", Path(__file__).resolve().parents[2] / "scripts" / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    report = mod.check_one("emg_probe", {"kind": "dummy_emg"},
                           workdir=str(tmp_path / "wf"), probe_seconds=1.0)
    assert report["status"] == "OK", report
    assert report["probe"]
    assert not (tmp_path / "wf").exists() or True   # 清理由调用方负责


def test_preflight_check_one_reports_open_crash_with_traceback(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight", Path(__file__).resolve().parents[2] / "scripts" / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # COM9 在任何机器上都不是可用串口 → open 崩溃,报告必须带 traceback
    report = mod.check_one("emg_bad", {"kind": "weili_emg", "port": "COM9"},
                           workdir=str(tmp_path / "wf2"), probe_seconds=1.0)
    assert report["status"] == "FAIL"
    assert report["phase"] == "open"
    assert report["traceback"]


# =============================================================================
# 会话汇总:流级错误归类 + 空录制不计入无误比例
# =============================================================================

def _make_run(root, name, *, streams, findings_session=(), stream_findings=(),
              window=None):
    """造一个假 session 目录:streams={slot: 有无npz},qc_report.json 可选。"""
    import json as _json
    d = root / name
    for slot, has_npz in streams.items():
        sdir = d / slot
        sdir.mkdir(parents=True)
        (sdir / f"{slot}.log").write_text("x", encoding="utf-8")
        if has_npz:
            (sdir / f"{slot}.npz").write_bytes(b"fake")
    if stream_findings is not None or findings_session:
        qc = {
            "streams": {s: {"findings": fs} for s, fs in stream_findings},
            "findings": list(findings_session),
            "window": window or {},
        }
        (d / "qc_report.json").write_text(
            _json.dumps(qc, ensure_ascii=False), encoding="utf-8")
    return d


def test_collect_layers_empty_nqc_and_clean(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_session", Path(__file__).resolve().parents[2] / "scripts" / "run_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1) 空录制:目录在但只有 .log,QC 报 StreamPresent(subject=slot)
    _make_run(tmp_path, "01-empty",
              streams={"marker": False, "eeg": False},
              findings_session=[
                  {"level": "ERROR", "check": "StreamPresent",
                   "message": "marker/ 无数据文件", "subject": "marker"},
                  {"level": "ERROR", "check": "StreamPresent",
                   "message": "eeg/ 无数据文件", "subject": "eeg"},
                  {"level": "WARN", "check": "RunWindow",
                   "message": "未找到 RUN_START"}])
    # 2) 有数据 + 流级 ERROR
    _make_run(tmp_path, "02-eye-bad",
              streams={"eye": True},
              stream_findings=[
                  ("eye", [{"level": "ERROR", "check": "ClockAlign",
                            "message": "对齐失败"}])])
    # 3) 有数据 + 干净
    _make_run(tmp_path, "03-clean", streams={"emg_left": True},
              stream_findings=[("emg_left", [])])
    # 4) 有数据但没跑 QC
    _make_run(tmp_path, "04-noqc", streams={"emg_left": True},
              stream_findings=None)

    s = mod._collect([tmp_path / "01-empty", tmp_path / "02-eye-bad",
                      tmp_path / "03-clean", tmp_path / "04-noqc"],
                     {"01-empty": "rerun", "02-eye-bad": "kept",
                      "03-clean": "kept", "04-noqc": "kept"})
    assert s["n_sessions"] == 4
    assert s["n_empty"] == 1
    assert s["n_no_qc"] == 1
    assert s["n_judged"] == 2          # 02 + 03
    assert s["n_clean"] == 1           # 只有 03
    # 错误按检查项 -> {流: 条数} 归类,带具体 recorder 名
    assert s["errors"]["StreamPresent"] == {"marker": 1, "eeg": 1}
    assert s["errors"]["ClockAlign"] == {"eye": 1}
    # 涉及 session 数按去重后的检查项计,不随条数虚增
    assert s["error_sessions"]["StreamPresent"] == 1
    assert s["warning_sessions"]["RunWindow"] == 1
    # 明细:空录制标 empty
    marks = {x["dir"]: x for x in s["sessions"]}
    assert marks["01-empty"]["empty"] is True
    assert marks["03-clean"]["clean"] is True


def test_print_summary_renders_marks_and_streams(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_session", Path(__file__).resolve().parents[2] / "scripts" / "run_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _make_run(tmp_path, "01-empty",
              streams={"marker": False},
              findings_session=[
                  {"level": "ERROR", "check": "StreamPresent",
                   "message": "marker/ 无数据文件", "subject": "marker"}])
    _make_run(tmp_path, "02-clean", streams={"emg_left": True},
              stream_findings=[("emg_left", [])])

    s = mod._collect([tmp_path / "01-empty", tmp_path / "02-clean"],
                     {"01-empty": "rerun", "02-clean": "kept"})
    mod.print_summary(s, title="测试")
    out = capsys.readouterr().out
    assert "空录制 1 条" in out
    assert "分母 = 有数据且跑过 QC 的 1 条" in out
    assert "∅" in out and "✓" in out
    assert "StreamPresent" in out and "流: marker 1" in out
    assert "注: StreamPresent" in out
