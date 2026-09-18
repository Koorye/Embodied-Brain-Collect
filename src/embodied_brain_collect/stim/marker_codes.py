"""Marker 码表 —— 单一事实源,数值一律来自 configs/markers.yaml。

Used by:
  - stim/*                        (发码)
  - checkers/marker.py            (窗口配对 + 码校验 + 显示名)
  - recorders/eeg/dummy_eeg_recorder.py  (dummy 事件流)
  - intan 的 digital_map 默认映射  (字型 → 边界码)

**代码只认键名**(RUN_START/RUN_END/...),数值一律来自 markers.yaml:
进程导入本模块时读取一次(`EMBODIED_BRAIN_COLLECT_CONFIGS` 可换目录),
键名写错/缺键/数值越界/重复在导入时立即报错 —— 绝不带病发码。
**没有内置默认表**:markers.yaml 缺失或缺 codes 段,导入即失败。

P1_RUN_START/P1_RUN_END 已取消:所有 stim 统一发 RUN_START/RUN_END;两线
接法的台架直接在 markers.yaml 里把这对值改成单比特码(见 markers.yaml
末尾的示例),代码不再区分。
"""

from __future__ import annotations

from typing import Final

# ---- 合法码名清单(仅键名;数值一律来自 markers.yaml) ----------------------
_KEYS: Final[tuple[str, ...]] = (
    "RUN_START", "RUN_END", "BLOCK_START", "BLOCK_END",
    "FIX_ON", "FIX_OFF", "CUE_AUDIO", "INSTR_ON", "INSTR_OFF",
    "GO_AUDIO", "EXEC_START", "EXEC_END", "IMG_START", "IMG_END",
    "VIDEO_START", "VIDEO_END", "ERROR", "PANIC", "IDLE",
)

HAND_ACTIONS: Final[tuple[str, ...]] = ("LIFT_LEFT", "PUT_LEFT",
                                        "LIFT_RIGHT", "PUT_RIGHT")


def _load_table() -> tuple[dict[str, int], int | None]:
    """读 markers.yaml(codes + hand_cue_base),校验后返回。

    码表没有内置默认,以下情况一律在导入时立即失败:markers.yaml 缺失/
    缺 codes 段/缺任一码名/未知码名/数值出 0..255/码值重复/缺
    hand_cue_base(重复码会让 EEG 按码配对与校验失效 —— 宁可起不来
    也不带病发码)。
    """
    from embodied_brain_collect.session.config import load_markers
    try:
        cfg = load_markers() or {}
    except FileNotFoundError as exc:
        raise ValueError(
            f"markers.yaml 缺失 — 码表没有内置默认({exc});从仓库恢复该文件"
            "或按台架补全") from None

    overrides = cfg.get("codes")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(
            "configs/markers.yaml 缺少 codes 段 — 码表没有内置默认,"
            "从仓库恢复该文件或按台架补全")
    names = {str(k).upper() for k in overrides}
    missing = [k for k in _KEYS if k not in names]
    if missing:
        raise ValueError(f"markers.yaml codes 缺少码名: {', '.join(missing)}"
                         f"(必须齐全: {', '.join(_KEYS)})")

    codes: dict[str, int] = {}
    for key, value in overrides.items():
        key = str(key).upper()
        if key not in _KEYS:
            raise ValueError(f"markers.yaml 未知码名: {key!r} (可用: "
                             f"{', '.join(_KEYS)})")
        value = int(value)
        if not 0 <= value <= 255:
            raise ValueError(f"markers.yaml {key}={value} 出界 — 码是 8 位 "
                             "(0..255)")
        codes[key] = value

    sent = [v for k, v in codes.items() if k != "IDLE"]
    dupes = {v for v in sent if sent.count(v) > 1}
    if dupes:
        named = {f"{k}({v})" for k, v in codes.items() if v in dupes}
        raise ValueError(f"markers.yaml 码值重复: {', '.join(sorted(named))}"
                         " — 同码在一次采集里必须唯一,否则配对失效")

    if "hand_cue_base" not in cfg:
        raise ValueError("markers.yaml 缺少 hand_cue_base — 码表没有内置"
                         "默认;sync_test 用手势段就写基址(0xC0),不用写 null")
    raw = cfg["hand_cue_base"]
    hand_cue_base = None if raw is None else int(raw)
    if hand_cue_base is not None and not 0 <= hand_cue_base <= 227:
        raise ValueError(f"markers.yaml hand_cue_base={hand_cue_base} "
                         "出界 — 手势段要留 32 个码(0..227)")
    return codes, hand_cue_base


_codes, HAND_CUE_BASE = _load_table()

# 键名 → 模块属性:全项目的 `from marker_codes import RUN_START` 与
# `marker_codes.RUN_START` 都拿到表里的值
for _name, _value in _codes.items():
    globals()[_name] = _value

NAMED: Final[dict[int, str]] = {v: k for k, v in _codes.items()}


def make_hand_cue(cycle: int, action: int) -> int:
    """第 ``cycle`` 轮(0-based)第 ``action`` 个动作的手势码。"""
    if HAND_CUE_BASE is None:
        raise ValueError("markers.yaml 已禁用手势码段(hand_cue_base: null)")
    if not 0 <= action < len(HAND_ACTIONS):
        raise ValueError(f"action {action} out of range [0, 3]")
    return HAND_CUE_BASE + cycle * 4 + action


def name_of(code: int) -> str:
    """Resolve numeric code -> human tag.  Ranged codes get a synthetic name.

    Task and scene identity markers were removed in v1.1.0 — a session records
    exactly one task, so the task id lives in the session's meta.yaml and the
    marker stream carries only event timing.
    """
    if code in NAMED:
        return NAMED[code]
    if HAND_CUE_BASE is not None and HAND_CUE_BASE <= code <= HAND_CUE_BASE + 31:
        cycle = (code - HAND_CUE_BASE) // 4 + 1
        action = HAND_ACTIONS[(code - HAND_CUE_BASE) % 4]
        return f"{action}_{cycle}"
    return f"UNKNOWN_0x{code:02X}"


def is_known(code: int) -> bool:
    if code in NAMED:
        return True
    return (HAND_CUE_BASE is not None
            and HAND_CUE_BASE <= code <= HAND_CUE_BASE + 31)


# 一个完整 trial 的事件序列(dummy recorder 用它模拟真实刺激流)。
# RUN_END 不在此列:_close 时由 recorder 补发,保证一个 session 恰有一对
# RUN_START/RUN_END 且所有码唯一。
DUMMY_TRIAL_CODES: tuple[int, ...] = (
    RUN_START, FIX_ON, INSTR_ON, FIX_OFF, INSTR_OFF,
    IMG_START, IMG_END, EXEC_START, EXEC_END,
)

if __name__ == "__main__":
    print("码表(configs/markers.yaml):")
    for _name, _value in _codes.items():
        print(f"  {_name:<12} = {_value:3d}  (0x{_value:02X})")
    print(f"  HAND_CUE_BASE = {HAND_CUE_BASE} "
          f"(0x{HAND_CUE_BASE:02X}-0x{HAND_CUE_BASE + 31:02X})"
          if HAND_CUE_BASE is not None else "  手势码段已禁用")
