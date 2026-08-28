"""错误排查指引 —— 按 slot 归类设备,给出人工排查方案。

两类错误共用这套指引:

* **启动错误**:recorder 的 open 首帧闸门失败/超时/崩溃,本次没有任何数据
  (launcher 返回码 1,preflight 报 FAIL)。
* **保存/录制错误**:录制中途进程退出、崩溃、落盘失败,或自动 QC 报 ERROR
  (数据可能不完整,需人工判断保留还是重采)。

自查流程:错误**第一次**发生 → 直接重采(输入 r);**多次**发生 → 看
error 行的关键词定位设备,按对应条目排查硬件。每个 recorder 的完整
traceback 在 ``<session>/<slot>/<slot>.log``。
"""

from __future__ import annotations

from collections import Counter

# slot 前缀 → 设备类别(与自查流程的编号对应)。
_SLOT_CATEGORY = (
    ("cam", "cam"),
    ("eeg", "eeg"),
    ("emg", "emg"),
    ("eye", "eye"),
    ("hand", "hand_pose"),
    ("position", "position"),
    ("marker", "marker"),
)

#: 类别 → 排查方案(顺序即自查流程的展示顺序)。
SLOT_GUIDES: dict[str, str] = {
    "eeg":        "拔插同步盒(EEG 同步盒 USB 重插,确认 Curry 端 NetStream 在发)",
    "emg":        "拔插 USB 适配器,按紧接线(臂环接口容易松)",
    "eye":        "拔插网线;退出并重启手机上的 Neon app(Companion)",
    "hand_pose":  "拔插接收器(dongle),确保手套全蓝常亮(Manus Core 里两只都在线)",
    "position":   "确保 app 连接(SteamVR 正在运行,tracker 显示在线)",
    "cam":        "拔插对应 USB 口(逐个口试,换口后重新预检)",
    "marker":     "检查 stim 与并行盒/UDP 配置(host/port 是否与 stim.yaml 一致)",
}

_CATEGORY_LABEL = {
    "eeg": "EEG", "emg": "EMG", "eye": "眼动", "hand_pose": "手部姿态",
    "position": "位置", "cam": "相机", "marker": "marker",
}


def slot_category(slot: str) -> str | None:
    """slot 名 → 设备类别;识别不了返回 None(用通用指引)。"""
    s = slot.lower()
    for prefix, cat in _SLOT_CATEGORY:
        if s == prefix or s.startswith(prefix + "_") or s.startswith(prefix):
            return cat
    return None


def guide_for_slot(slot: str) -> str:
    cat = slot_category(slot)
    if cat is None or cat not in SLOT_GUIDES:
        return "查看该 slot 的 .log 里 error 行的 traceback,定位具体原因"
    return f"{slot}: {SLOT_GUIDES[cat]}"


def self_check_flow(slots: list[str] | None = None) -> str:
    """错误自查流程全文;给 ``slots`` 时只列涉及的设备,否则列全部。"""
    cats: list[str] = []
    for slot in slots or []:
        cat = slot_category(slot)
        if cat and cat not in cats:
            cats.append(cat)
    if not cats:
        cats = list(SLOT_GUIDES)
    lines = ["错误自查流程:如果错误第一次发生,直接重采;"
             "如果错误多次发生,根据 error 行定位关键词:"]
    for i, cat in enumerate(cats, 1):
        lines.append(f"  {i}. {_CATEGORY_LABEL.get(cat, cat)} — {SLOT_GUIDES[cat]}")
    return "\n".join(lines)


def format_failure_help(
    failures: dict[str, str],
    *,
    phase: str,
    fail_counts: Counter | None = None,
    log_root: str = "",
) -> str:
    """一段可直接打印的排查提示。

    ``failures``: slot → 错误原因(启动 open 失败,或录制/保存阶段出错)。
    ``phase``: "startup"(设备打开失败)或 "data"(录制/保存阶段)。
    ``fail_counts``: 该会话内每个 slot 的累计失败次数 —— 首次失败只提示
    重采,多次失败才展开对应设备的排查方案(与自查流程一致)。
    ``log_root``: session 目录,用来提示 traceback 日志的位置。
    """
    fail_counts = fail_counts or Counter()
    if phase == "startup":
        head = "启动错误 — 设备打开失败,本次没有录到任何数据"
    else:
        head = "录制/保存阶段错误 — 数据可能不完整"
    lines = ["", "─" * 68, f"⚠ {head}"]
    repeated: list[str] = []
    for slot, reason in failures.items():
        n = fail_counts.get(slot, 0)
        tag = f"(第 {n} 次失败)" if n > 1 else "(首次失败)"
        lines.append(f"  ✗ {slot} {tag}: {reason}")
        if n > 1:
            lines.append(f"      排查: {guide_for_slot(slot)}")
            repeated.append(slot)
        else:
            lines.append("      首次失败可直接重采(输入 r);再失败按下面的流程排查")
    if repeated:
        lines.append("")
        lines.append(self_check_flow(repeated))
    if log_root:
        lines.append(f"  详细 traceback: {log_root}/<slot>/<slot>.log")
    lines.append("─" * 68)
    return "\n".join(lines)
