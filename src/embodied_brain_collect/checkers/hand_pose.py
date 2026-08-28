"""Hand pose checker — Manus data gloves, ergonomics + skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput, Span
from .checks import MadOutlier, NanFraction, ValueJump, ts_checks


@dataclass(frozen=True)
class SkeletonDropout(BaseCheck):
    """Windows where most of the skeleton is NaN — a glove that disconnected.

    The overall NaN fraction can stay low while a glove is out for several
    seconds, so the interesting question is *when* it was bad, not how much
    of the recording was.
    """

    field: str = "skeleton_positions"
    window_s: float = 1.0
    bad_frac: float = 0.5      # this much of a window NaN = the glove was out

    def applies(self, ctx: CheckContext) -> bool:
        a = ctx.arr(self.field)
        return a is not None and a.size > 0 and a.dtype.kind == "f"

    def run(self, ctx: CheckContext) -> CheckOutput:
        a = np.asarray(ctx.arr(self.field))
        s = self.target(ctx)
        out = CheckOutput()
        if s is None or s.n == 0:
            return out

        per_sample = np.isnan(a).reshape(len(a), -1).mean(axis=1)
        mask = ctx.mask(s.raw)
        if mask is not None and len(mask) == len(per_sample):
            per_sample = per_sample[mask]
        if len(per_sample) != s.n:
            return out

        edges = np.arange(s.t[0], s.t[-1] + self.window_s, self.window_s)
        if edges.size < 2:
            return out
        idx = np.clip(np.searchsorted(edges, s.t, side="right") - 1,
                      0, edges.size - 2)
        bad = []
        for w in range(edges.size - 1):
            sel = per_sample[idx == w]
            if sel.size and float(sel.mean()) > self.bad_frac:
                bad.append((float(edges[w]), float(sel.mean())))

        out.stats["n_bad_windows"] = len(bad)
        if bad:
            out.findings.append(self.finding(
                "WARN",
                f"{len(bad)} 个 {self.window_s:g}s 窗口内骨骼 NaN 占比 "
                f">{self.bad_frac:.0%} — 手套断连",
                field=self.field, threshold=self.bad_frac,
                observed=max(f for _, f in bad),
                spans=[Span(t, self.window_s, f"NaN {f:.0%}")
                       for t, f in bad[:50]]))
        return out


class HandPoseChecker(BaseChecker):
    """Manus gloves.  Ergonomics and skeleton share one timeline, and the
    poll loop runs far above the gloves' own update rate, so duplicate
    timestamps here are expected rather than a fault."""

    name = "hand_pose"
    matches = ("hand_pose",)
    default_series = "samples"

    checks = [
        ts_checks("samples"),
        MadOutlier("ergo_data"),
        NanFraction("skeleton_positions"),
        SkeletonDropout(),
        ValueJump("skeleton_positions", thr=0.05, per_node=True),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        # Both streams are stamped together; ergo is the primary, but a
        # skeleton-only recording still deserves a timeline.
        def load():
            got = ctx.arr("ergo_timestamps")
            return got if got is not None else ctx.arr("skeleton_timestamps")

        ctx.add_series("samples", loader=load)
