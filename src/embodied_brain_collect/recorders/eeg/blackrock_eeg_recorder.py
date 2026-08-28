"""BlackrockEegRecorder — Cerebus/NeuroPort via the CereLink SDK (pycbsdk).

``pip install pycbsdk`` wraps Blackrock's cbSDK (the wheel bundles cbsdk.dll).
The session auto-discovers the device the same way Central does: NSP on its
default 192.168.137.x subnet or a local Central instance — the 10.x API does
not expose explicit addressing.

Continuous data: channels the operator assigned to a *sample group* in
Central arrive as int16 group packets.  We subscribe with the batch callback,
which hands over ``(n_samples, n_channels) int16`` plus per-sample device
timestamps in one copy.  Sample indices are derived from the device clock
(period inferred from the timestamps themselves, wrap-corrected for legacy
32-bit firmware), so a dropped datagram shows up as a block gap the QC step
can see — same semantics as Curry's absolute sample numbers.

Marker events: wire the ParallelBox TTL output to the NSP's digital input
port.  Digital-in change packets carry the word + device timestamp; every
non-zero word is forwarded as an EEG event whose code is the word value,
which the base recorder pairs with the marker bus for the amp->PC clock fit.
Raw event words are also saved for post-hoc reprocessing.

The recorder never changes device configuration on its own: if no frontend
channels are sampling, open fails with instructions — assign the group in
Central, or set ``auto_enable_group`` to let the recorder do it once.
"""

from __future__ import annotations

import time

import numpy as np

from .base_eeg_recorder import BaseEegRecorder
from .eeg_recorder_config import BlackrockEegRecorderConfig

try:
    from pycbsdk import ChannelType, DeviceType, SampleRate, Session
    _HAVE_PYCBSDK = True
    _IMPORT_ERROR = ""
except Exception as _exc:  # ImportError or a broken cffi/dll load
    _HAVE_PYCBSDK = False
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

#: config sample_group/auto_enable_group 的速率值 -> pycbsdk SampleRate
_RATE_HZ_TO_GROUP = {500: 1, 1000: 2, 2000: 3, 10000: 4, 30000: 5}

# int16 原始计数 -> uV 的兜底换算(cbSDK 前端组的经典步长);设备上报的
# 通道换算结构可用时优先用它,并把最终步长写进 npz 备查。
_FALLBACK_STEP_UV = 0.25


def wrap_delta(ts_new: int, ts_old: int, wrap_mod: int) -> int:
    """Forward distance between two device timestamps.

    ``wrap_mod`` = 2**32 for legacy 32-bit clocks (wrap at ~39.7 h @ 30 kHz),
    0 for 64-bit nanosecond clocks (never wraps in practice).  A negative
    delta on a wrapping clock is a wrap, not time going backwards.
    """
    d = ts_new - ts_old
    if d < 0 and wrap_mod:
        d += wrap_mod
    return d


def infer_period(timestamps: np.ndarray) -> float:
    """Device-clock units per sample from one batch's timestamps.

    Units differ across firmware (ns vs ticks), so the period is measured,
    not assumed: the median positive diff of consecutive timestamps.  0 when
    the batch carries fewer than two distinct timestamps (nothing to measure
    yet — caller retries with the next batch).
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size < 2:
        return 0.0
    d = np.diff(ts)
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


class GroupIndexTracker:
    """Map device timestamps onto absolute stream sample indices.

    The first received sample defines index 0 and the reference timestamp;
    afterwards the gap between the previous batch's last timestamp and the
    new batch's first timestamp, divided by the (measured) clock period,
    tells how many samples the link dropped in between.  Drops therefore
    surface as block-start discontinuities — exactly what QC's
    ``BlockContinuity`` check reads.
    """

    def __init__(self, wrap_mod: int):
        self.wrap_mod = wrap_mod
        self.period = 0.0        # device-clock units per sample; 0 = unmeasured
        self.last_ts: int | None = None
        self.abs_next = 0        # absolute index the next contiguous sample gets
        self.total_dropped = 0

    def feed(self, timestamps: np.ndarray) -> tuple[int, int]:
        """One batch -> ``(start_index, dropped_before_batch)``."""
        ts = np.asarray(timestamps, dtype=np.int64)
        if ts.size == 0:
            return self.abs_next, 0
        if self.period <= 0.0:
            self.period = infer_period(ts)
        t0 = int(ts[0])
        if self.last_ts is None:
            start, gap = self.abs_next, 0
        else:
            d = wrap_delta(t0, self.last_ts, self.wrap_mod)
            gap = (int(round(d / self.period)) - 1
                   if self.period > 0 else 0)
            gap = max(0, gap)      # 时钟抖动导致的负间隙按连续处理
            start = self.abs_next + gap
            self.total_dropped += gap
        self.abs_next = start + int(ts.size)
        self.last_ts = int(ts[-1])
        return start, gap

    def index_of(self, ts_event: int) -> int:
        """Nearest stream index for a single event timestamp (now-ish)."""
        if self.period <= 0.0 or self.last_ts is None:
            return self.abs_next
        d = wrap_delta(ts_event, self.last_ts, self.wrap_mod)
        return self.abs_next + max(0, int(round(d / self.period)))


def event_word(data) -> int:
    """16-bit little-endian word from a digital-in event packet payload."""
    try:
        return int(data[0]) | (int(data[1]) << 8)
    except (IndexError, TypeError):
        return 0


class BlackrockEegRecorder(BaseEegRecorder):
    """Records EEG/ECoG from a Blackrock Cerebus/NeuroPort via pycbsdk."""

    config: BlackrockEegRecorderConfig
    _has_trigger_channel = False   # 数据里没有 Trigger 列,事件走数字输入包

    def __init__(self, config: BlackrockEegRecorderConfig):
        super().__init__(config)
        # 硬件会话只在 _open() 里建 —— Windows spawn 下 recorder 会被 pickle
        self._session = None
        self._tracker: GroupIndexTracker | None = None
        self._scale_step = np.asarray([_FALLBACK_STEP_UV], dtype=np.float32)
        self._scale_offset = np.zeros(1, dtype=np.float32)
        self._last_dig_word = -1
        self._sdk_errors = 0
        self._last_dropped = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fail(self, reason: str) -> bool:
        self._open_error = reason
        self._log(f"[eeg:blackrock] open failed — {reason}")
        self._teardown_session()
        return False

    def _teardown_session(self) -> None:
        if self._session is None:
            return
        try:
            self._session.close()
        except Exception as exc:  # noqa: BLE001 - 收尾必须继续
            self._log(f"[eeg:blackrock] session close error: {exc}", level="WARNING")
        self._session = None

    def _pick_group(self, session) -> tuple[int, list[int]]:
        """(group_id, channel ids) — configured group, else the one holding
        the most frontend channels."""
        fe_max = session.num_fe_chans()
        group_ch = {g: session.get_group_channels(g) for g in range(1, 7)}
        want = int(self.config.sample_group)
        if want:
            ch = group_ch.get(want, [])
            if ch:
                return want, ch
            raise ConnectionError(
                f"sample_group={want} 里没有通道(Central 里可用组: "
                + ", ".join(f"{g}:{len(c)}ch" for g, c in group_ch.items() if c)
                + ")")
        best = max(group_ch,
                   key=lambda g: sum(1 for c in group_ch[g] if 1 <= c <= fe_max),
                   default=0)
        ch = group_ch.get(best, [])
        if not any(1 <= c <= fe_max for c in ch):
            raise ConnectionError(
                "没有任何采样组包含前端通道 — 在 Central 里把通道设到某个"
                "采样组,或把 auto_enable_group 设为目标速率(如 1000)")
        return best, ch

    def _build_scaling(self, session, chids: list[int]) -> None:
        """int16 -> uV 的逐通道线性换算(cbSDK 的 cbSCALING 结构)。

        结构拿不到或退化时退回 0.25 uV/LSB;最终步长存进 npz,事后可再标定。
        """
        steps, offsets, seen = [], [], False
        for chid in chids:
            try:
                sc = session.get_channel_scaling(chid)
            except Exception:  # noqa: BLE001 - 换算失败不阻断采集
                sc = None
            if sc and sc.get("digmax", 0) > sc.get("digmin", 0):
                span_d = sc["digmax"] - sc["digmin"]
                span_a = sc["anamax"] - sc["anamin"]
                unit = str(sc.get("anaunit", "uV")).strip().lower()
                if unit == "nv":
                    span_a /= 1e3
                elif unit == "mv":
                    span_a *= 1e3
                step = float(span_a) / float(span_d)
                offset = float(sc["anamin"]) - step * float(sc["digmin"])
                steps.append(step)
                offsets.append(offset)
                seen = True
                continue
            steps.append(_FALLBACK_STEP_UV)
            offsets.append(0.0)
        self._scale_step = np.asarray(steps, dtype=np.float32)
        self._scale_offset = np.asarray(offsets, dtype=np.float32)
        if seen:
            med = float(np.median(self._scale_step))
            self._log(f"[eeg:blackrock] scaling: {med:.4g} uV/LSB (median, "
                      "来自设备 cbSCALING)")
        else:
            self._log("[eeg:blackrock] scaling: 设备未上报换算,用 "
                      f"{_FALLBACK_STEP_UV} uV/LSB", level="WARNING")

    # ------------------------------------------------------------------
    # SDK callbacks (run on SDK threads; list appends are atomic enough)
    # ------------------------------------------------------------------

    def _on_group_batch(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        if self._tracker is None:
            return
        start, gap = self._tracker.feed(timestamps)
        if gap:
            self._log(f"[eeg:blackrock] 丢 {gap} 样本(设备时钟间隙)",
                      level="WARNING")
        block = (samples.astype(np.float32) * self._scale_step
                 + self._scale_offset).astype(np.float32)
        self._on_block(start, block)

    def _on_digital_event(self, header, data) -> None:
        if self._tracker is None:
            return
        raw = event_word(data)
        # 事件码先过 mask:剥掉数字口空闲电平基线(实测 NSP 数字口空闲
        # 0xF988,码叠在 0x88 上 —— mask=0x77 还原出可与 marker 配对的码)
        word = raw & int(self.config.digital_mask)
        try:
            ts = int(header.time)
        except (AttributeError, TypeError, ValueError):
            ts = 0
        # 原始事件留档:真机上数字输入包的载荷布局没有文档,事后可据此重解
        self._acc("eeg_dig_event_ts", int(ts))
        self._acc("eeg_dig_event_word", raw)
        try:
            self._acc("eeg_dig_event_chid", int(header.chid))
        except (AttributeError, TypeError, ValueError):
            pass
        if word == self._last_dig_word:
            return
        if word != 0:      # 上升沿(或码间直切):新码即事件码
            if self._tracker.period > 0.0:
                self._on_event(word, self._tracker.index_of(ts))
            else:
                # 连续数据还没来,绝对位置无从谈起 —— 只留原始档
                self._log("[eeg:blackrock] 首批数据前收到数字输入事件,"
                          f"word={word},跳过对齐用副本", level="WARNING")
        self._last_dig_word = word

    def _on_sdk_error(self, msg: str) -> None:
        self._sdk_errors += 1
        if self._sdk_errors <= 5 or self._sdk_errors % 100 == 0:
            self._log(f"[eeg:blackrock] sdk error #{self._sdk_errors}: {msg}",
                      level="WARNING")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> bool:
        if not _HAVE_PYCBSDK:
            return self._fail(f"pycbsdk 不可用 — pip install pycbsdk "
                              f"({ _IMPORT_ERROR})")
        try:
            device_type = DeviceType[self.config.device_type.strip().upper()]
        except KeyError:
            return self._fail(f"未知 device_type: {self.config.device_type!r} "
                              f"(可选: {', '.join(d.name for d in DeviceType)})")
        try:
            session = Session(device_type=device_type,
                              callback_queue_depth=self.config.callback_queue_depth)
        except Exception as exc:  # noqa: BLE001 - cbsdk.dll 缺运行库、发现超时等
            return self._fail(
                f"Session 创建失败 — {type(exc).__name__}: {exc}。排查:"
                " ① NSP 电源/网线,PC 网卡需与 NSP 同段(默认 192.168.137.x,"
                "NSP 在 .128);② 或本机 Central 是否在运行;"
                "③ pycbsdk 走 UDP 广播发现,防火墙需放行;"
                "④ 换 device_type(NSP/HUB1/HUB2/HUB3/NPLAY)")
        self._session = session

        t0 = time.time()
        while not session.running and time.time() - t0 < self.config.open_timeout:
            time.sleep(0.1)
        if not session.running:
            return self._fail("设备未发现/未连接 — 检查 NSP 网络"
                              "(默认 192.168.137.x)或本机 Central")
        self._log(f"[eeg:blackrock] session running: {device_type.name}, "
                  f"protocol={session.protocol_version.name}, "
                  f"sysfreq={session.sysfreq}")
        session.on_error(self._on_sdk_error)

        try:
            auto = int(self.config.auto_enable_group)
        except (TypeError, ValueError):
            auto = 0
        if auto:
            group = _RATE_HZ_TO_GROUP.get(auto)
            if group is None:
                return self._fail(f"auto_enable_group={auto} 不是合法速率"
                                  " (500/1000/2000/10000/30000)")
            try:
                session.set_sample_group(None, ChannelType.FRONTEND,
                                         SampleRate(group))
                session.sync(timeout=5.0)
                self._log(f"[eeg:blackrock] 已把全部前端通道设到 "
                          f"{auto} Hz 采样组(组 {group})")
            except Exception as exc:  # noqa: BLE001
                return self._fail(f"auto_enable_group 配置失败 — "
                                  f"{type(exc).__name__}: {exc}")

        try:
            group, chids = self._pick_group(session)
        except ConnectionError as exc:
            return self._fail(str(exc))
        if not chids:
            return self._fail(f"采样组 {group} 的通道列表为空")
        self._sample_rate = float(SampleRate(group).hz)
        self._build_scaling(session, chids)

        labels = []
        for chid in chids:
            try:
                label = (session.get_channel_label(chid) or "").strip()
            except Exception:  # noqa: BLE001
                label = ""
            labels.append(label or f"chan{chid:03d}")
        self._channel_labels = labels

        wrap_mod = (1 << 32 if session.protocol_version.name
                    in ("V3_11", "V4_0") else 0)
        self._tracker = GroupIndexTracker(wrap_mod)
        session.on_group_batch(SampleRate(group))(self._on_group_batch)
        session.on_event(ChannelType.DIGITAL_IN)(self._on_digital_event)

        # 首帧闸门:批回调到达才算 open 成功
        t0 = time.time()
        while self._total_samples == 0:
            if time.time() - t0 > self.config.open_timeout:
                return self._fail(f"{self.config.open_timeout:g}s 内没有收到"
                                  " 连续数据 — 采样组没配、设备未运行或数字"
                                  "输入口无事件(TTL 事件只在 run 时产生)")
            time.sleep(0.05)
        self._log(f"[eeg:blackrock] group {group} @ {self._sample_rate:g} Hz, "
                  f"{len(chids)} ch ({labels[0]}..{labels[-1]}); "
                  f"trigger = NSP digital-in")
        return True

    def _poll(self, ts: float) -> None:
        # 数据经 SDK 回调线程到达,_poll 只监视 SDK 丢包计数
        session = self._session
        if session is None:
            return
        try:
            dropped = int(session.stats.packets_dropped)
        except Exception:  # noqa: BLE001 - 统计失败不影响采集
            return
        if dropped and dropped != self._last_dropped:
            self._last_dropped = dropped
            self._log(f"[eeg:blackrock] SDK 回调队列丢包 {dropped}",
                      level="WARNING")

    def _heartbeat_stats(self, elapsed: float) -> str:
        extra = super()._heartbeat_stats(elapsed)
        if self._tracker is not None and self._tracker.total_dropped:
            extra += f" dropped={self._tracker.total_dropped}"
        if self._sdk_errors:
            extra += f" sdk_err={self._sdk_errors}"
        return extra

    def _reset_stream_state(self) -> None:
        # 索引跟踪器重建:统一开录后的第一批数据从样本 0 重新计
        if self._tracker is not None:
            self._tracker = GroupIndexTracker(self._tracker.wrap_mod)
        self._last_dig_word = -1

    def _close(self) -> None:
        # 先关 session(停掉 SDK 线程,回调不再进来),再做对齐/落盘
        self._teardown_session()
        self._log(f"[eeg:blackrock] stopped (samples={self._total_samples}, "
                  f"events={len(self._buf.get('eeg_event_code', []))}, "
                  f"dropped={self._tracker.total_dropped if self._tracker else 0})")
        super()._close()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_output(self) -> dict[str, np.ndarray]:
        out = super()._build_output()
        out["eeg_scale_uv_per_lsb"] = self._scale_step
        return out
