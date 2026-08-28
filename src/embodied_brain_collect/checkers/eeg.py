"""EEG checker — Curry / NeuroScan amplifier stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput, Span
from .checks import (DeadChannel, MadOutlier, NanFraction, WindowSampleCount,
                     ts_checks)


@dataclass(frozen=True)
class BlockContinuity(BaseCheck):
    """The amplifier hands over fixed-size blocks with absolute sample
    numbers.  A discontinuity between one block's end and the next block's
    start is data the amplifier produced and the link dropped — invisible in
    the sample array itself, which just concatenates whatever arrived."""

    def applies(self, ctx: CheckContext) -> bool:
        return (ctx.arr("eeg_block_start") is not None
                and ctx.arr("eeg_block_n") is not None)

    def run(self, ctx: CheckContext) -> CheckOutput:
        starts = np.asarray(ctx.arr("eeg_block_start"), dtype=np.int64).ravel()
        sizes = np.asarray(ctx.arr("eeg_block_n"), dtype=np.int64).ravel()
        out = CheckOutput()
        if starts.size < 2:
            return out

        expected = starts[:-1] + sizes[:-1]
        lost = starts[1:] - expected
        breaks = np.flatnonzero(lost != 0)
        out.stats.update({"n_blocks": int(starts.size),
                          "n_breaks": int(breaks.size),
                          "lost_samples": int(lost[breaks].sum())})
        if breaks.size:
            # 断点的时刻:优先用对齐后的 PC 时间戳(块边界=样本流中的位置),
            # 旧数据没有时间戳时才退回块接收时刻(仅作展示,不再生成)。
            ts_pc = ctx.arr("eeg_timestamps_pc")
            spans = []
            if ts_pc is not None:
                tr = np.asarray(ts_pc, dtype=np.float64).ravel()
                ends = np.cumsum(sizes)
                spans = [Span(float(tr[min(int(ends[i]), tr.size - 1)]), 0.0,
                             f"缺 {int(lost[i])} 样本")
                         for i in breaks[:50]]
            else:
                t_recv = ctx.arr("eeg_block_t_recv")
                if t_recv is not None:
                    tr = np.asarray(t_recv, dtype=np.float64).ravel()
                    spans = [Span(float(tr[i + 1]), 0.0,
                                  f"缺 {int(lost[i])} 样本")
                             for i in breaks[:50] if i + 1 < tr.size]
            out.findings.append(self.finding(
                "WARN",
                f"{breaks.size} 处数据块不连续,共缺失 {int(lost[breaks].sum())} 个样本",
                field="eeg_block_start", observed=float(breaks.size),
                spans=spans))

        # A block at the maximum size means the receiver was catching up.
        at_cap = int(np.sum(sizes == sizes.max()))
        if sizes.max() > np.median(sizes) and at_cap:
            out.stats["blocks_at_max"] = at_cap
            out.findings.append(self.finding(
                "INFO", f"{at_cap} 个数据块达到最大尺寸 {int(sizes.max())} — "
                        "可能存在接收积压",
                field="eeg_block_n", observed=float(at_cap)))
        return out


@dataclass(frozen=True)
class MetadataMatch(BaseCheck):
    """Header sample count against the array actually saved."""

    def applies(self, ctx: CheckContext) -> bool:
        return (ctx.arr("eeg_n_samples") is not None
                and ctx.arr("eeg_data") is not None)

    def run(self, ctx: CheckContext) -> CheckOutput:
        claimed = int(np.asarray(ctx.arr("eeg_n_samples")).ravel()[0])
        actual = int(len(ctx.arr("eeg_data")))
        out = CheckOutput()
        out.stats.update({"n_samples_claimed": claimed, "n_samples_actual": actual})
        if claimed != actual:
            out.findings.append(self.finding(
                "WARN", f"元数据声明 {claimed} 个样本,实际 {actual} 个",
                field="eeg_n_samples", observed=float(abs(claimed - actual))))
        return out


@dataclass(frozen=True)
class ClockAlign(BaseCheck):
    """Alignment state: did the amplifier clock get fitted to the PC clock?

    The recorder's policy is all-or-nothing — a failed fit saves no
    timestamps, and the missing field itself is reported by
    ``TimestampSanity``.  This check reports the fit verdict directly.
    """

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.arr("eeg_fit_fitted") is not None

    def run(self, ctx: CheckContext) -> CheckOutput:
        fitted = bool(np.asarray(ctx.arr("eeg_fit_fitted")).ravel()[0])
        out = CheckOutput(stats={"fit_fitted": fitted})
        if fitted:
            return out
        reason = ctx.arr("eeg_fit_reason")
        tail = ""
        if reason is not None:
            tail = f": {str(np.asarray(reason).ravel()[0])}"
        out.findings.append(self.finding(
            "ERROR", f"EEG 时钟拟合失败{tail}",
            field="eeg_fit_fitted"))
        return out


class EegChecker(BaseChecker):
    """Curry amplifier.  Its expected rate comes from the file itself."""

    name = "eeg"
    matches = ("eeg",)
    default_series = "eeg"

    checks = [
        ts_checks("eeg"),
        WindowSampleCount(),
        ClockAlign(),
        BlockContinuity(),
        MetadataMatch(),
        NanFraction("eeg_data", frac_warn=0.0),
        DeadChannel("eeg_data"),
        MadOutlier("eeg_data"),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        rate = ctx.arr("eeg_sample_rate")
        ctx.add_series("eeg", key="eeg_timestamps_pc",
                       expected_rate=float(np.asarray(rate).ravel()[0])
                       if rate is not None else None)
