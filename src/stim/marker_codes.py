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

# ---- ranged identity codes (compose at runtime) -----------------------------
TASK_ID_BASE:  Final[int] = 0x80   # 128 .. 159
TASK_ID_LAST:  Final[int] = 0x9F
SCENE_ID_BASE: Final[int] = 0xA0   # 160 .. 191
SCENE_ID_LAST: Final[int] = 0xBF

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
    """Resolve numeric code -> human tag.  Ranged codes get a synthetic name."""
    if code in NAMED:
        return NAMED[code]
    if TASK_ID_BASE <= code <= TASK_ID_LAST:
        return f"TASK_ID_{code - TASK_ID_BASE:02d}"
    if SCENE_ID_BASE <= code <= SCENE_ID_LAST:
        return f"SCENE_ID_{code - SCENE_ID_BASE:02d}"
    return f"UNKNOWN_0x{code:02X}"


def is_known(code: int) -> bool:
    return (code in NAMED
            or TASK_ID_BASE  <= code <= TASK_ID_LAST
            or SCENE_ID_BASE <= code <= SCENE_ID_LAST)


def make_task_id(n: int) -> int:
    if not 0 <= n <= TASK_ID_LAST - TASK_ID_BASE:
        # raise ValueError(f"task index {n} out of range [0, 31]")
        pass
    return TASK_ID_BASE + n


def make_scene_id(n: int) -> int:
    """Compose a SCENE_ID code for scene index n in [0, 31]."""
    if not 0 <= n <= SCENE_ID_LAST - SCENE_ID_BASE:
        raise ValueError(f"scene index {n} out of range [0, 31]")
    return SCENE_ID_BASE + n


if __name__ == "__main__":
    for c in sorted(NAMED):
        print(f"  0x{c:02X} ({c:3d})  {NAMED[c]}")
    print(f"  0x{TASK_ID_BASE:02X}-0x{TASK_ID_LAST:02X}  TASK_ID_00..31")
    print(f"  0x{SCENE_ID_BASE:02X}-0x{SCENE_ID_LAST:02X}  SCENE_ID_00..31")
