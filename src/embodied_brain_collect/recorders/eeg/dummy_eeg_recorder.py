"""Dummy EEG — 8ch sine + noise at 250 Hz, with synthetic events.

Used by the launcher's ``--dummy`` mode.  The synthetic events replay one
trial's semantic sequence (RUN_START, FIX_ON, INSTR_ON, ..., EXEC_END at
1 Hz, RUN_END at close) in lockstep with the dummy marker recorder — the
real pipeline's TTL and UDP markers come from one source, and the dummy
pair simulates that so the close-time alignment actually succeeds in dummy
runs.
"""

import time

import numpy as np

from .base_eeg_recorder import BaseEegRecorder
from .eeg_recorder_config import EegRecorderConfig

_DUMMY_FS = 250.0
_DUMMY_CHANNELS = 8
_DUMMY_BLOCK_S = 0.04            # 10 samples per block
_DUMMY_EVENT_S = 1.0             # 与 dummy marker 同节奏

from ...stim.marker_codes import (DUMMY_TRIAL_CODES, P1_RUN_END,   # noqa: E402
                                  RUN_START)


class DummyEegRecorder(BaseEegRecorder):
    config: EegRecorderConfig

    def __init__(self, config: EegRecorderConfig):
        super().__init__(config)
        self._amp_index = 0
        self._t_start: float | None = None
        self._last_event: float | None = None
        self._event_i = 0

    def _open(self) -> bool:
        self._sample_rate = _DUMMY_FS
        self._channel_labels = [f"Ch{i + 1}" for i in range(_DUMMY_CHANNELS - 1)] + ["Trigger"]
        self._schedule: list[tuple[float, int]] | None = None
        self._schedule_i = 0
        if self.config.dummy_events == "sync_test":
            self._schedule = self._build_sync_test_schedule()
            self._log("[eeg:dummy] 事件节奏 = sync_test(与刺激程序同源)")
        self._log(f"[eeg:dummy] synthetic {_DUMMY_CHANNELS}-ch EEG @ "
                  f"{_DUMMY_FS:g} Hz")
        return True

    def _build_sync_test_schedule(self) -> list[tuple[float, int]]:
        """复刻 sync_test 的事件时刻表(读 configs/stim.yaml 的 sync_test 段)。

        dummy eeg 模拟 TTL 通路 —— 真实刺激的 marker 经串口打给放大器,
        所以 dummy eeg 的事件必须与所选 stim 的节奏完全一致,否则对齐
        拟合的配对点不共线,斜率守卫会正确拒绝。
        """
        from embodied_brain_collect.config.load import load_stim
        from embodied_brain_collect.stim.marker_codes import (
            EXEC_END, EXEC_START, IMG_END, IMG_START, make_hand_cue)
        try:
            stim = load_stim()
        except FileNotFoundError:
            stim = {}
        sec = {k: v for k, v in stim.items()
               if isinstance(v, dict)}.get("sync_test", {})
        imag = float(sec.get("imag_s", 10.0))
        cyc = float(sec.get("cycle_s", 2.5))
        cycles = int(sec.get("cycles", 3))

        events: list[tuple[float, int]] = [
            (0.0, RUN_START), (0.0, IMG_START),
            (imag, IMG_END), (imag, EXEC_START),
        ]
        for k in range(cycles * 4):
            events.append((imag + k * cyc, make_hand_cue(k // 4, k % 4)))
        events.append((imag + cycles * 4 * cyc, EXEC_END))
        # 只按时刻排序:同刻事件保持构造顺序(码值不参与排序)
        return sorted(events, key=lambda e: e[0])

    def _reset_stream_state(self) -> None:
        # 统一开录后从样本 0 / 事件表头重新开始
        self._amp_index = 0
        self._t_start = None
        self._last_event = None
        self._event_i = 0
        self._schedule_i = 0

    def _close(self) -> None:
        # 收尾事件:与 stim 的 P1_RUN_END 同码,保证两端序列配对完整
        if self._amp_index > 0:
            self._on_event(P1_RUN_END, self._amp_index - 1)
        self._log(f"[eeg:dummy] stopped (samples={self._total_samples}, "
                  f"events={len(self._buf.get('eeg_event_code', []))})")
        super()._close()

    def _poll(self, ts: float) -> None:
        if self._t_start is None:
            self._t_start = ts
            self._last_event = ts
        # 样本号与真实时间连续积分(round 到最近样本),绝不能逐块 int()
        # 截断累积 —— 6 秒 150 块会累计出 ~1% 的时钟偏差,足以让斜率守卫
        # 误杀 dummy 对齐。
        target = int(round((ts - self._t_start) * _DUMMY_FS))
        if target > self._amp_index:
            n = target - self._amp_index
            t = (np.arange(n, dtype=np.float32)
                 + self._amp_index) / _DUMMY_FS
            phase = np.arange(_DUMMY_CHANNELS - 1) * 0.5
            data = (np.sin(2 * np.pi * 10.0 * t[:, None]
                           + phase[None, :]) * 40.0
                    + np.random.randn(n, _DUMMY_CHANNELS - 1) * 2.0)
            block = np.zeros((n, _DUMMY_CHANNELS), dtype=np.float32)
            block[:, :-1] = data.astype(np.float32)
            self._on_block(self._amp_index, block)
            self._amp_index = target
        if self._schedule is not None:
            # sync_test 模式:按时刻表发事件
            elapsed = ts - self._t_start
            while (self._schedule_i < len(self._schedule)
                   and elapsed >= self._schedule[self._schedule_i][0]):
                self._on_event(self._schedule[self._schedule_i][1],
                               self._amp_index)
                self._schedule_i += 1
        elif (ts - self._last_event >= _DUMMY_EVENT_S
                and self._event_i < len(DUMMY_TRIAL_CODES)):
            self._on_event(DUMMY_TRIAL_CODES[self._event_i], self._amp_index)
            self._event_i += 1
            self._last_event = ts
        time.sleep(0.01)
