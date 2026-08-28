"""Abstract EEG base.

Bookkeeping (blocks / events / channel info) plus the EEG<->PC alignment,
which runs in ``_close()``: Curry events carry the marker code with the amp
sample index (``latency``), and the marker recorder saved the same codes
with PC-clock times in its npz (``marker_t_local_recv``).  Both streams run
live over a socket and start/stop with the session, so codes are unique per
run; pairing them by code lookup and fitting ``t_pc = a * t_eeg + b`` maps
every EEG sample onto the PC clock.

Alignment is all-or-nothing: receive timestamps are never recorded, and when
the fit cannot be trusted no ``eeg_timestamps_pc`` is saved at all — the QC
step reports the missing field as an error rather than silently trusting a
fallback.

Concrete recorders feed data through::

    self._on_block(start_sample, block)      # (n, n_channels) f32
    self._on_event(code, latency)

and must call ``super()._close()`` after releasing their hardware.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..base import BaseRecorder


# =============================================================================
# Event alignment (EEG events vs marker bus records)
# =============================================================================

def _align_code_sequences(
    eeg_code: np.ndarray, eeg_t_s: np.ndarray,
    marker_code: np.ndarray, marker_t_pc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pair the two event streams by their codes.

    Both streams are pulled live over a socket and the recorder processes
    start and stop with the session, so there is no "superset" case any more
    — a superset could only come from mixing runs, which never happens here.
    What DOES happen is loss: a dropped UDP packet or a missed TTL removes
    one code from one side.  A lookup by code tolerates that on either side;
    the only things that must hold are that codes are unique within a run
    (otherwise the lookup is ambiguous) and that the surviving codes appear
    in the same order on both clocks (otherwise the pairing is wrong).

    Returns ``(eeg_t_s, pc_t_s)`` matched by event, or None.
    """
    eeg_code = np.asarray(eeg_code, dtype=np.int64).ravel()
    # eeg_t_s 已是秒(float):事件时刻是分数秒,取整会毁掉拟合
    eeg_t_s = np.asarray(eeg_t_s, dtype=np.float64).ravel()
    marker_code = np.asarray(marker_code, dtype=np.int64).ravel()
    marker_t_pc = np.asarray(marker_t_pc, dtype=np.float64).ravel()
    if eeg_code.size == 0 or marker_code.size == 0:
        return None
    # 重码会让"哪个是哪个"无法判断 — 直接拒绝,不猜测
    if np.unique(eeg_code).size != eeg_code.size \
            or np.unique(marker_code).size != marker_code.size:
        return None
    eeg_pos = {int(c): i for i, c in enumerate(eeg_code)}
    marker_pos = {int(c): i for i, c in enumerate(marker_code)}
    # 按 EEG 侧的到达顺序取交集 —— 绝不能按码值排序,事件顺序与码值无关
    shared = [int(c) for c in eeg_code if int(c) in marker_pos]
    if len(shared) < 2:
        return None
    eeg_idx = np.asarray([eeg_pos[c] for c in shared], dtype=np.int64)
    marker_idx = np.asarray([marker_pos[c] for c in shared], dtype=np.int64)
    # EEG 侧按构造同序;marker 侧也必须同序 — 否则配对本身是错的
    if not np.all(np.diff(marker_idx) > 0):
        return None
    return (eeg_t_s[eeg_idx], marker_t_pc[marker_idx])


# =============================================================================
# Robust linear fit (EEG amp clock -> PC clock)
# =============================================================================

# 斜率 = 两个晶振的频率比,物理上必须 ≈1。越界说明样本缺失或配对错误,
# 拟合结果不可信 — 此时不提供任何时间戳,让 QC 报错。
# ±1%:晶振漂移通常在 0.05% 内,但 dummy 场景下两端进程收尾时刻差可达
# ~15ms —— 对 5 秒会话就是 0.3%;真正的故障(缺样本、错配对)在 2%+ 量级。
SLOPE_LO, SLOPE_HI = 0.99, 1.01
SLOPE_COARSE = (0.9, 1.1)    # 候选斜率的粗滤(剔除单个被延迟的 marker)
RESID_MAX_S = 0.25           # 单个事件允许的抖动(内点线)


def _fit_eeg_to_pc(eeg_t_s: np.ndarray, marker_t_pc: np.ndarray) -> dict:
    """Fit ``t_pc = a * t_eeg + b``; ``fitted=False`` unless it holds up.

    One grossly delayed marker would distort a plain least-squares fit, so
    the slope is first estimated from the median of plausible pairwise
    slopes, gross outliers are rejected, and the consensus is refit.  The
    final slope then has to land inside [SLOPE_LO, SLOPE_HI] — two crystals
    cannot differ by half a percent, so a slope outside the band means the
    events do not span the samples they claim to (missing data or a wrong
    pairing), and any timestamps built on it would be smooth but wrong.
    """
    def _fail(reason: str) -> dict:
        return {"fitted": False, "reason": reason}

    x = np.asarray(eeg_t_s, dtype=np.float64)
    y = np.asarray(marker_t_pc, dtype=np.float64)
    if x.size < 2:
        return _fail("fewer than two matched events")
    x0, y0 = float(x[0]), float(y[0])
    xr, yr = x - x0, y - y0

    slopes = []
    for i in range(xr.size):
        for j in range(i + 1, xr.size):
            dx = xr[j] - xr[i]
            if dx < 0.5:
                continue
            s = (yr[j] - yr[i]) / dx
            if SLOPE_COARSE[0] <= s <= SLOPE_COARSE[1]:
                slopes.append(s)
    if not slopes:
        return _fail("no plausible slope between any two events")

    a0 = float(np.median(slopes))
    b0 = float(np.median(yr - a0 * xr))
    inliers = np.abs(yr - (a0 * xr + b0)) <= RESID_MAX_S
    if int(inliers.sum()) < 2:
        return _fail("no consistent pair of events survives outlier rejection")

    a, b = np.polyfit(xr[inliers], yr[inliers], 1)
    if not (SLOPE_LO <= a <= SLOPE_HI):
        return _fail(f"slope {a:.5f} outside [{SLOPE_LO},{SLOPE_HI}] — "
                     "missing samples or mispaired events")

    resid = yr - (a * xr + b)
    fit_resid = resid[inliers]
    return {
        "fitted": True,
        "slope_pc_per_eeg": float(a),
        "intercept_s_at_first_marker": float(b),
        "eeg_t0_s": x0,
        "pc_t0_s": y0,
        "resid_max_ms": float(np.abs(fit_resid).max() * 1000.0),
        "resid_rms_ms": float(np.sqrt(np.mean(fit_resid ** 2)) * 1000.0),
        "n": int(x.size),
        "n_inliers": int(inliers.sum()),
        "n_outliers": int((~inliers).sum()),
    }


# =============================================================================
# BaseEegRecorder
# =============================================================================

class BaseEegRecorder(BaseRecorder):
    name = "eeg"
    output_dir = "eeg"
    # Curry/dummy/Intan 的 eeg_data 最后一列是 Trigger(数字输入字);Blackrock
    # 的事件走独立的数字输入包,数据里没有 Trigger 列 —— eeg_n_eeg_channels
    # 据此决定减不减一。
    _has_trigger_channel: bool = True

    def __init__(self, config):
        super().__init__(config)
        self._blocks: list[np.ndarray] = []   # (n, n_channels) float32, C-order
        self._total_samples = 0
        self._sample_rate = 0.0
        self._channel_labels: list[str] = []
        self._fit: dict | None = None          # set by _close()
        self._timestamps_pc: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Feeding (concrete recorders)
    # ------------------------------------------------------------------

    def _on_block(self, start_sample: int, block: np.ndarray) -> None:
        """``block``: (n, n_channels) float32 C-order (samples x channels).

        No receive timestamps are kept: the amplifier's own sample clock is
        the only trustworthy time base, and PC alignment comes from the
        marker fit — never from arrival times.
        """
        self._blocks.append(np.asarray(block, dtype=np.float32))
        self._total_samples += block.shape[0]
        self._acc("eeg_block_start", start_sample)
        self._acc("eeg_block_n", block.shape[0])

    def _on_event(self, code: int, latency: int) -> None:
        self._acc("eeg_event_code", code)
        self._acc("eeg_event_latency", latency)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """统一开录时刻(launcher 的 go 之后)才调用:清掉 open 首帧闸门与
        ready 等待期漏进来的数据 —— 否则开得快的 recorder(如 Blackrock 的
        SDK 回调)会把等待慢 recorder 的几秒也录进去,各流起点参差。"""
        super()._setup()
        self._blocks.clear()
        self._total_samples = 0
        for key in ("eeg_block_start", "eeg_block_n",
                    "eeg_event_code", "eeg_event_latency",
                    "eeg_dig_event_ts", "eeg_dig_event_word",
                    "eeg_dig_event_chid"):
            self._buf.pop(key, None)
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        """子类重置各自的流锚点(时间戳原点/索引跟踪器),配合 _setup 清缓冲。"""

    # ------------------------------------------------------------------
    # Alignment (runs in _close, before _save)
    # ------------------------------------------------------------------

    def _close(self) -> None:
        self._align()

    def _align(self) -> None:
        """Pair EEG events with the marker stream and fit amp clock -> PC.

        All or nothing: when the fit cannot be trusted, NO timestamps are
        saved at all.  A receive-time fallback would be non-monotonic at
        every jittery block boundary, and worse, it would silently dress up
        "could not align" as "aligned" — the QC step reports the missing
        timestamps as an error instead.
        """
        try:
            self._fit = self._fit_from_markers()
        except Exception as exc:
            self._fit = {"fitted": False,
                         "reason": f"align error: {type(exc).__name__}: {exc}"}
        if self._fit.get("fitted"):
            f = self._fit
            self._log(
                f"[eeg] align: slope={f['slope_pc_per_eeg']:.7f} "
                f"resid_rms={f['resid_rms_ms']:.2f}ms "
                f"resid_max={f['resid_max_ms']:.2f}ms "
                f"n={f['n']} (inliers={f['n_inliers']})", echo=False)
        else:
            self._timestamps_pc = None
            self._log(f"[eeg] 对齐失败 — {self._fit.get('reason', '?')};"
                      "不保存时间戳,QC 将报错")

    def _fit_from_markers(self) -> dict:
        if self._total_samples == 0:
            return {"fitted": False, "reason": "no EEG samples recorded"}
        if len(self._buf.get("eeg_event_code", [])) < 2:
            return {"fitted": False, "reason": "fewer than two EEG events"}
        markers = self._load_markers()
        if markers is None:
            return {"fitted": False,
                    "reason": "no marker*/*.npz with marker_code found"}
        m_code = markers.get("marker_code")
        # 发送端时间戳是权威(无接收抖动);旧数据没有该字段时回退接收时刻
        m_t_pc = (markers.get("marker_t_sent_pc")
                  if markers.get("marker_t_sent_pc") is not None
                  else markers.get("marker_t_local_recv"))
        if m_code is None or m_t_pc is None or len(m_code) == 0:
            return {"fitted": False, "reason": "markers.npz has no markers"}

        eeg_code = np.asarray(self._buf["eeg_event_code"], dtype=np.int64)
        eeg_lat = np.asarray(self._buf["eeg_event_latency"], dtype=np.int64)
        fs = self._sample_rate or 1.0
        pair = _align_code_sequences(eeg_code, eeg_lat / fs,
                                     np.asarray(m_code, dtype=np.int64),
                                     np.asarray(m_t_pc, dtype=np.float64))
        if pair is None:
            return {"fitted": False,
                    "reason": "EEG 事件与 marker 事件无法按码配对"}

        eeg_t_matched, pc_t_matched = pair
        fit = _fit_eeg_to_pc(eeg_t_matched, pc_t_matched)
        if fit.get("fitted"):
            # t_pc(i) = pc_t0 + a * (t_eeg(i) - eeg_t0) + b; rows are capture
            # order, so the amp index is a plain ramp from the first block
            a = fit["slope_pc_per_eeg"]
            b = fit["intercept_s_at_first_marker"]
            start = self._buf["eeg_block_start"][0]
            t_eeg = (start + np.arange(self._total_samples)) / fs
            self._timestamps_pc = (
                fit["pc_t0_s"] + a * (t_eeg - fit["eeg_t0_s"]) + b)
        return fit

    def _load_markers(self) -> dict | None:
        """Poll for the marker recorder's npz for ``marker_wait_s`` seconds.

        Found by *content*, not by a fixed path: the marker recorder's output
        directory is its sensor slot name, which the launcher assigns at
        runtime (``marker/marker.npz``, not the class default
        ``markers/markers.npz``).  Hard-coding the latter meant the fit
        silently never ran and every session fell back to per-block receive
        times — which is what put the non-monotonic timestamps in eeg.npz.
        """
        if not self.config.session_dir:
            return None
        root = Path(self.config.session_dir)
        deadline = time.time() + float(
            getattr(self.config, "marker_wait_s", 10.0))
        while True:
            for npz in sorted(root.glob("marker*/*.npz")):
                try:
                    with np.load(npz, allow_pickle=False) as d:
                        if "marker_code" in d.files:
                            return {k: d[k] for k in d.files}
                except (OSError, ValueError):
                    pass  # mid-write by the marker recorder; retry
            if time.time() >= deadline:
                return None
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _heartbeat_stats(self, elapsed: float) -> str:
        fit = "pending" if self._fit is None else (
            "ok" if self._fit.get("fitted") else "skipped")
        return (f"eeg_samples={self._total_samples} "
                f"blocks={len(self._blocks)} "
                f"events={len(self._buf.get('eeg_event_code', []))} "
                f"fit={fit}")

    def _build_output(self) -> dict[str, np.ndarray]:
        out = super()._build_output()
        n_ch = self._blocks[0].shape[1] if self._blocks else 1
        out["eeg_data"] = (np.concatenate(self._blocks, axis=0)
                           if self._blocks
                           else np.zeros((0, n_ch), dtype=np.float32))
        out["eeg_sample_rate"] = np.asarray(self._sample_rate)
        out["eeg_channel_names"] = np.asarray(self._channel_labels)
        out["eeg_n_samples"] = np.asarray(self._total_samples)
        out["eeg_n_channels"] = np.asarray(n_ch)
        out["eeg_n_eeg_channels"] = np.asarray(
            max(0, n_ch - 1) if self._has_trigger_channel else n_ch)
        out["eeg_start_amp_sample"] = np.asarray(
            self._buf["eeg_block_start"][0]
            if self._buf.get("eeg_block_start") else 0)
        if self._fit is not None:
            for key in ("fitted", "slope_pc_per_eeg",
                        "intercept_s_at_first_marker", "eeg_t0_s", "pc_t0_s",
                        "resid_max_ms", "resid_rms_ms", "n", "n_inliers",
                        "n_outliers"):
                if key in self._fit:
                    out[f"eeg_fit_{key}"] = np.asarray(self._fit[key])
            if not self._fit.get("fitted"):
                out["eeg_fit_reason"] = np.asarray(
                    str(self._fit.get("reason", "?")))
        if self._timestamps_pc is not None:
            out["eeg_timestamps_pc"] = self._timestamps_pc
        return out
