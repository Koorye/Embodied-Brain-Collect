"""Marker checker — the E-Prime event stream, and the run window it defines.

The markers are what make every other stream comparable: RUN_START and
RUN_END bracket the analysis window that all the other checkers clip to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..stim.marker_codes import RUN_END, RUN_START, name_of
from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput, StreamReport


def _marker_times(z) -> np.ndarray | None:
    """发送端时间戳优先(无接收抖动);旧数据无该字段时回退接收时刻。"""
    for key in ("marker_t_sent_pc", "marker_t_local_recv"):
        if key in z.files:
            return np.asarray(z[key], dtype=np.float64).ravel()
    return None


def find_run_window(root: Path) -> dict | None:
    """Locate the RUN_START -> RUN_END pair that bounds the analysis window.

    Returns ``{"t0", "t1", "n_markers", "file"}``, or None when the pair is
    missing — callers then fall back to the full data range.
    """
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if not d.name.startswith("marker"):
            continue
        for npz in sorted(d.glob("*.npz")):
            try:
                with np.load(npz, allow_pickle=False) as z:
                    if "marker_code" not in z.files:
                        continue
                    code = np.asarray(z["marker_code"]).ravel()
                    t = _marker_times(z)
                    if t is None or t.size != code.size:
                        continue
            except (OSError, ValueError, KeyError):
                continue
            starts = np.flatnonzero(code == RUN_START)
            if starts.size == 0:
                continue
            i0 = int(starts[0])
            ends = np.flatnonzero(code == RUN_END)
            ends = ends[ends >= i0]
            if ends.size == 0:
                continue
            i1 = int(ends[0])
            return {"t0": float(t[i0]), "t1": float(t[i1]),
                    "n_markers": int(code.size), "file": str(npz)}
    return None


@dataclass(frozen=True)
class MarkerPresence(BaseCheck):
    """A run with no markers cannot be aligned to anything else — an error,
    not a warning: without RUN_START/RUN_END no stream is alignable."""

    def run(self, ctx: CheckContext) -> CheckOutput:
        code = ctx.arr("marker_code")
        out = CheckOutput()
        n = 0 if code is None else int(np.asarray(code).size)
        out.stats["n_markers"] = n
        if n == 0:
            out.findings.append(self.finding("ERROR", "未记录任何标记"))
        return out


@dataclass(frozen=True)
class MarkerOrder(BaseCheck):
    """Markers must arrive in the order they were sent."""

    def applies(self, ctx: CheckContext) -> bool:
        t = self._times(ctx)
        return t is not None and t.size > 1

    def _times(self, ctx: CheckContext) -> np.ndarray | None:
        t = ctx.arr("marker_t_sent_pc")
        if t is None:
            t = ctx.arr("marker_t_local_recv")
        return None if t is None else np.asarray(t, dtype=np.float64).ravel()

    def run(self, ctx: CheckContext) -> CheckOutput:
        t = self._times(ctx)
        out = CheckOutput()
        n_back = int(np.sum(np.diff(t) < 0))
        out.stats["span_s"] = float(t[-1] - t[0])
        if n_back:
            out.findings.append(self.finding(
                "WARN", f"{n_back} 个标记时间倒序", field="marker_t_sent_pc",
                observed=float(n_back)))
        return out


class MarkerChecker(BaseChecker):
    """Event stream.  Declares no timestamp series: markers are sparse
    events, so rate and jitter statistics would be meaningless."""

    name = "marker"
    matches = ("marker",)

    checks = [MarkerPresence(), MarkerOrder()]

    def finalize(self, ctx: CheckContext, report: StreamReport) -> None:
        """List the markers themselves — the most useful thing in the report
        for this stream, and not a check."""
        code = ctx.arr("marker_code")
        t = ctx.arr("marker_t_sent_pc")
        if t is None:
            t = ctx.arr("marker_t_local_recv")
        if code is None or t is None:
            return
        code = np.asarray(code).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        tag = ctx.arr("marker_tag")

        mask = ctx.mask(t)
        keep = np.flatnonzero(mask) if mask is not None else np.arange(t.size)
        base = ctx.window["t0"] if ctx.window else (float(t[0]) if t.size else 0.0)

        report.stats["markers"] = {
            "n_total": int(code.size),
            "n_in_window": int(keep.size),
            "items": [{"code": int(code[i]), "name": str(name_of(int(code[i]))),
                       "tag": str(tag[i]) if tag is not None else "",
                       "t_offset": float(t[i] - base)} for i in keep],
        }
