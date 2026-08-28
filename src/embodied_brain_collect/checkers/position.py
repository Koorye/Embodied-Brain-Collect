"""Position checker — SteamVR / OpenVR 6-DOF trackers.

Several trackers share one recording, so most findings are per device and
carry the serial number in ``Finding.subject``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseCheck, BaseChecker, CheckContext, CheckOutput
from .checks import TimestampGap, ts_checks


def _devices(ctx: CheckContext):
    """(positions, valid, names) normalized to a device axis.

    Single-device recordings write flat ``(T, 3)``; give everything the same
    ``(T, D, ...)`` shape so the checks need only one code path.
    """
    pos = ctx.arr("positions_m")
    if pos is None:
        return None, None, []
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[:, None, :]

    valid = ctx.arr("valid")
    if valid is None:
        valid = ~np.isnan(pos).any(axis=-1)
    else:
        valid = np.asarray(valid)
        if valid.ndim == 1:
            valid = valid[:, None]

    ser = ctx.arr("serials")
    names = ([str(s) for s in np.asarray(ser).ravel()] if ser is not None
             else [f"dev{i}" for i in range(pos.shape[1])])

    # Clip to the run window alongside the timestamps.
    t = ctx.arr("timestamps_s")
    mask = ctx.mask(t)
    if mask is not None and len(mask) == pos.shape[0]:
        pos, valid = pos[mask], valid[mask]
    return pos, valid, names


@dataclass(frozen=True)
class ValidFraction(BaseCheck):
    """How much of the run each tracker was actually tracked for."""

    min_frac: float = 0.5

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.arr("positions_m") is not None

    def run(self, ctx: CheckContext) -> CheckOutput:
        pos, valid, names = ctx.artifact("devices", lambda: _devices(ctx))
        out = CheckOutput()
        if pos is None or valid.shape[0] == 0:
            return out
        fracs = valid.mean(axis=0)
        out.stats["valid_fraction"] = {n: float(f) for n, f in zip(names, fracs)}
        for name, frac in zip(names, fracs):
            if frac < self.min_frac:
                out.findings.append(self.finding(
                    "WARN", f"仅 {frac:.0%} 有效样本", subject=name,
                    field="valid", threshold=self.min_frac, observed=float(frac)))
        return out


@dataclass(frozen=True)
class TrackingGap(TimestampGap):
    """The same gap detection, fed the poses that are actually usable.

    A tracker that loses line of sight keeps producing samples — invalid
    ones — so the timeline has no hole and plain ``TimestampGap`` sees
    nothing.  Dropping the invalid samples puts the hole back where the
    inherited detection can find it, which keeps one definition of "a gap"
    and one threshold for both cases.
    """

    note: str = "(跟踪丢失)"

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.arr("positions_m") is not None

    def scan_targets(self, ctx: CheckContext):
        pos, valid, names = ctx.artifact("devices", lambda: _devices(ctx))
        s = self.target(ctx)
        if pos is None or s is None or s.n == 0 or len(valid) != s.n:
            return []
        med = s.interval_median or 1e-3
        return [(name, s.t[valid[:, i]], med) for i, name in enumerate(names)]


@dataclass(frozen=True)
class Teleport(BaseCheck):
    """A tracker jumping further between two samples than a body could move.

    Only valid samples are compared: a jump across a tracking dropout is the
    dropout's doing, not a teleport.
    """

    thr_m: float = 0.2

    def applies(self, ctx: CheckContext) -> bool:
        return ctx.arr("positions_m") is not None

    def run(self, ctx: CheckContext) -> CheckOutput:
        pos, valid, names = ctx.artifact("devices", lambda: _devices(ctx))
        out = CheckOutput()
        if pos is None:
            return out
        counts: dict[str, int] = {}
        for i, name in enumerate(names):
            v = valid[:, i]
            if int(v.sum()) < 3:
                counts[name] = 0
                continue
            disp = np.linalg.norm(np.diff(pos[v, i, :], axis=0), axis=-1)
            n = int(np.sum(disp > self.thr_m))
            counts[name] = n
            if n:
                out.findings.append(self.finding(
                    "WARN", f"{n} 次瞬移(相邻样本位移 > {self.thr_m:g} m)",
                    subject=name, field="positions_m", threshold=self.thr_m,
                    observed=float(disp.max())))
        out.stats["n_teleports"] = counts
        return out


class PositionChecker(BaseChecker):
    """VIVE trackers.  The poll loop free-runs far above the tracker's own
    update rate, so duplicate timestamps here are the norm."""

    name = "position"
    matches = ("position",)
    default_series = "combined"

    checks = [
        ts_checks("combined"),
        ValidFraction(),
        TrackingGap(),
        Teleport(),
    ]

    def prepare(self, ctx: CheckContext) -> None:
        ctx.add_series("combined", key="timestamps_s")
