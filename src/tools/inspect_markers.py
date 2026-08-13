"""inspect_markers.py -- pretty-print and validate markers.npz from sync_hub.

What it checks:
  - npz file exists and has the expected keys
  - all `code`s are in the known marker_codes book (warns on UNKNOWN)
  - inter-arrival distribution: min / max / median / mean (so you can spot
    if a `Sleep` got dropped or a packet went missing)
  - clock drift between E-Prime ms clock and PC time.time()
    (slope should be very close to 1.000 if both are running fine)
  - if `--neon-ip` was used during recording, t_neon_unix_ns should equal
    t_pc_recv * 1e9 within a microsecond (it's stamped by sync_hub itself
    just before send_event)

Run:
  python -m record.tools.inspect_markers <session_dir>/markers.npz
  python -m record.tools.inspect_markers markers.npz --csv markers.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sync.marker_codes import name_of, is_known


def fmt_dur(s: float) -> str:
    if s < 1.0:
        return f"{s*1000:6.2f} ms"
    if s < 60.0:
        return f"{s:7.3f} s "
    return f"{s/60:6.2f} min"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--csv", default=None,
                    help="optional: also dump as CSV at this path")
    ap.add_argument("--head", type=int, default=20,
                    help="how many events to print at the start (default 20)")
    ap.add_argument("--tail", type=int, default=10,
                    help="how many events to print at the end (default 10)")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"!!  no such file: {args.path}")
        return 1

    d = np.load(args.path, allow_pickle=False)
    expected = {"trial", "tag", "code", "t_eprime_ms", "t_pc_recv",
                "t_neon_unix_ns", "raw"}
    missing = expected - set(d.files)
    if missing:
        print(f"!!  npz is missing keys: {missing}")
        return 1

    n = len(d["trial"])
    if n == 0:
        print("(empty markers.npz)")
        return 0

    trial   = d["trial"]
    tag     = d["tag"]
    code    = d["code"]
    t_ep_ms = d["t_eprime_ms"]
    t_pc    = d["t_pc_recv"]
    t_neon  = d["t_neon_unix_ns"]

    t0 = float(t_pc[0])
    rel = t_pc - t0
    span = float(rel[-1])

    print(f"== {args.path}  ({n} events, span={fmt_dur(span)}) ==")
    print()

    def show(i: int) -> None:
        flag = " " if is_known(int(code[i])) else "!"
        neon = ""
        if t_neon[i] != 0:
            d_us = (t_neon[i] - int(t_pc[i] * 1e9)) / 1e3
            neon = f"  neon_dt={d_us:+.0f}us"
        print(f"  {flag}#{i:4d}  t={rel[i]:8.3f}s  trial={int(trial[i]):>4}  "
              f"0x{int(code[i]):02X} ({int(code[i]):3d})  "
              f"tag={str(tag[i]):<14s}  resolved={name_of(int(code[i])):<14s}{neon}")

    head = min(args.head, n)
    for i in range(head):
        show(i)
    if n > head + args.tail:
        print(f"  ... ({n - head - args.tail} skipped) ...")
        for i in range(max(head, n - args.tail), n):
            show(i)
    elif n > head:
        for i in range(head, n):
            show(i)

    print()

    print("-- per-tag counts --")
    uniq, cnts = np.unique(tag, return_counts=True)
    for u, c in sorted(zip(uniq, cnts), key=lambda kv: -int(kv[1])):
        print(f"   {str(u):<16s} {int(c):4d}")
    print()

    if n > 1:
        dt = np.diff(t_pc)
        print(f"-- inter-arrival (t_pc) --  "
              f"min={dt.min()*1000:.2f}ms  med={np.median(dt)*1000:.2f}ms  "
              f"mean={dt.mean()*1000:.2f}ms  max={dt.max()*1000:.2f}ms")

    valid = t_ep_ms > 0
    if valid.sum() >= 2:
        x = t_ep_ms[valid].astype(np.float64) / 1000.0
        y = t_pc[valid] - t_pc[valid][0]
        x = x - x[0]
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        print(f"-- clock fit eprime_s -> pc_s -- "
              f"slope={a:.6f}  intercept={b*1000:.2f}ms  "
              f"|resid|max={np.abs(resid).max()*1000:.2f}ms")
    else:
        print("-- clock fit -- skipped (no t_eprime_ms in packets)")

    nn = (t_neon != 0).sum()
    if nn > 0:
        diff_us = (t_neon[t_neon != 0] - (t_pc[t_neon != 0] * 1e9).astype(np.int64)) / 1e3
        print(f"-- neon stamps -- {nn}/{n} events forwarded; "
              f"|t_neon - t_pc| max={np.abs(diff_us).max():.1f}us "
              f"(should be < 100us; this is sync_hub overhead, not network)")

    unk = ~np.array([is_known(int(c)) for c in code])
    if unk.any():
        print(f"!!  {int(unk.sum())} events with unknown code(s):")
        for i in np.where(unk)[0]:
            print(f"     #{int(i):4d}  0x{int(code[i]):02X} tag={str(tag[i])}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "trial", "tag", "code_hex", "code_dec",
                        "t_eprime_ms", "t_pc_recv", "t_neon_unix_ns",
                        "resolved"])
            for i in range(n):
                w.writerow([i, int(trial[i]), str(tag[i]),
                            f"0x{int(code[i]):02X}", int(code[i]),
                            int(t_ep_ms[i]), float(t_pc[i]),
                            int(t_neon[i]), name_of(int(code[i]))])
        print(f"\nwrote csv -> {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
