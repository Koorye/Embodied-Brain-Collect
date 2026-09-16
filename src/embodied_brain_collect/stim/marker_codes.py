"""Single source of truth for E-Prime marker codes.

Used by:
  - record.sync.sync_hub          (validation + decoding)
  - record.sync.eprime_simulator  (simulating E-Prime UDP packets without E-Prime)
  - record.tools.inspect_markers  (pretty-printing recorded markers.npz)
  - record/eprime/inline_template.ebs2 must keep the same hex values
    (E-Basic does not import Python, copy by hand if you change here)

LPT side is 8-bit, so codes are 0-255.
0x00 and 0xFF are reserved sentinels:
  0x00 = port idle (cleared)  -- never sent as a real event
  0xFF = panic / abort        -- session-level error sentinel
"""

from __future__ import annotations

from typing import Final


# ---- run / block boundary ---------------------------------------------------
RUN_START:    Final[int] = 0xF1   # 241
RUN_END:      Final[int] = 0xF2   # 242
BLOCK_START:  Final[int] = 0xE1   # 225
BLOCK_END:    Final[int] = 0xE2   # 226

# ---- fixation / cue ---------------------------------------------------------
FIX_ON:       Final[int] = 0x11   #  17
FIX_OFF:      Final[int] = 0x12   #  18
CUE_AUDIO:    Final[int] = 0x21   #  33
INSTR_ON:     Final[int] = 0x31   #  49
INSTR_OFF:    Final[int] = 0x32   #  50
GO_AUDIO:     Final[int] = 0x41   #  65

# ---- execution / imagery / video --------------------------------------------
EXEC_START:   Final[int] = 0x51   #  81
EXEC_END:     Final[int] = 0x52   #  82
IMG_START:    Final[int] = 0x61   #  97
IMG_END:      Final[int] = 0x62   #  98
VIDEO_START:  Final[int] = 0x71   # 113
VIDEO_END:    Final[int] = 0x72   # 114

# ---- hand cues (同步测试用,0xC0-0xDF 段,每轮 4 个动作) -----------------------
# 码必须唯一:EEG 对齐按码配对,同一码在同一 session 出现两次会被拒绝。
# 布局: cycle(0-based) * 4 + action(0=抬左 1=放左 2=抬右 3=放右)
HAND_CUE_BASE: Final[int] = 0xC0   # 192 .. 223 (8 轮 x 4 动作)
HAND_ACTIONS: Final[tuple[str, ...]] = ("LIFT_LEFT", "PUT_LEFT",
                                        "LIFT_RIGHT", "PUT_RIGHT")


def make_hand_cue(cycle: int, action: int) -> int:
    """第 ``cycle`` 轮(0-based)第 ``action`` 个动作的手势码。"""
    if not 0 <= action < len(HAND_ACTIONS):
        raise ValueError(f"action {action} out of range [0, 3]")
    return HAND_CUE_BASE + cycle * 4 + action

# ---- error sentinels --------------------------------------------------------
ERROR:        Final[int] = 0xFE   # 254
PANIC:        Final[int] = 0xFF   # 255
IDLE:         Final[int] = 0x00   #   0

# ---- reverse map (for sync_hub validation + inspect_markers pretty-print) ---
NAMED: Final[dict[int, str]] = {
    RUN_START:   "RUN_START",
    RUN_END:     "RUN_END",
    BLOCK_START: "BLOCK_START",
    BLOCK_END:   "BLOCK_END",
    FIX_ON:      "FIX_ON",
    FIX_OFF:     "FIX_OFF",
    CUE_AUDIO:   "CUE_AUDIO",
    INSTR_ON:    "INSTR_ON",
    INSTR_OFF:   "INSTR_OFF",
    GO_AUDIO:    "GO_AUDIO",
    EXEC_START:  "EXEC_START",
    EXEC_END:    "EXEC_END",
    IMG_START:   "IMG_START",
    IMG_END:     "IMG_END",
    VIDEO_START: "VIDEO_START",
    VIDEO_END:   "VIDEO_END",
    ERROR:       "ERROR",
    PANIC:       "PANIC",
    IDLE:        "IDLE",
}


def name_of(code: int) -> str:
    """Resolve numeric code -> human tag.  Ranged codes get a synthetic name.

    Task and scene identity markers were removed in v1.1.0 — a session records
    exactly one task, so the task id lives in the session's meta.yaml and the
    marker stream carries only event timing.  Freeing the 0x80..0xBF range also
    lifts the old 32-slot cap that limited the task library size.
    """
    if code in NAMED:
        return NAMED[code]
    if HAND_CUE_BASE <= code <= HAND_CUE_BASE + 31:
        cycle = (code - HAND_CUE_BASE) // 4 + 1
        action = HAND_ACTIONS[(code - HAND_CUE_BASE) % 4]
        return f"{action}_{cycle}"
    return f"UNKNOWN_0x{code:02X}"


def is_known(code: int) -> bool:
    return code in NAMED or HAND_CUE_BASE <= code <= HAND_CUE_BASE + 31


# 一个完整 trial 的事件序列(dummy recorder 用它模拟真实刺激流)。
# RUN_END 不在此列:_close 时由 recorder 补发,保证一个 session 恰有一对
# RUN_START/RUN_END 且所有码唯一。
DUMMY_TRIAL_CODES: tuple[int, ...] = (
    RUN_START, FIX_ON, INSTR_ON, FIX_OFF, INSTR_OFF,
    IMG_START, IMG_END, EXEC_START, EXEC_END,
)

if __name__ == "__main__":
    for c in sorted(NAMED):
        print(f"  0x{c:02X} ({c:3d})  {NAMED[c]}")
    print(f"  0x{HAND_CUE_BASE:02X}-0x{HAND_CUE_BASE + 31:02X}  HAND_CUE (8轮x4动作)")
