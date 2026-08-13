"""Inspect Curry7 trigger channel from .dap/.dat or .cdt/.cdt.dpo files.

Validated ParallelBox TTL setup:
- Windows port: STMicroelectronics Virtual COM Port (COM14)
- Serial settings: 115200, 8N1
- Curry trigger raw value: 65280 + code, where code is the 8-bit marker byte.

Usage:
    python -m record.tools.inspect_curry_triggers "path/to/Acq.cdt.dpo"
    python -m record.tools.inspect_curry_triggers "path/to/Acquisition 08.dap"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tools.curry_io import decode_triggers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dap", type=Path, help="Curry .dap file; .dat must be next to it")
    parser.add_argument("--baseline", type=int, default=65280, help="ParallelBox idle raw value")
    parser.add_argument("--channel", type=int, default=None, help="1-based trigger channel; defaults to last channel")
    parser.add_argument("--min-duration-s", type=float, default=0.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None)
    args = parser.parse_args(argv)

    meta, runs = decode_triggers(args.dap, baseline=args.baseline, channel=args.channel, min_duration_s=args.min_duration_s)
    print("-- Curry trigger summary --")
    for key, value in meta.items():
        print(f"{key}: {value}")
    print("-- runs --")
    print(f"decoded_nonzero_runs: {len(runs)}")
    for row in runs:
        print(
            "sample {sample_start}-{last}  t={t_start_s:.3f}-{t_stop_s:.3f}s  code={code:3d}  raw={raw_value}  dur={duration_s:.3f}s".format(
                **row, last=int(row["sample_stop_exclusive"]) - 1
            )
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample_start", "sample_stop_exclusive", "t_start_s", "t_stop_s", "duration_s", "raw_value", "code"]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(runs)
        print(f"wrote {args.csv}")

    if args.npz:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.npz,
            sample_start=np.array([row["sample_start"] for row in runs], dtype=np.int64),
            sample_stop_exclusive=np.array([row["sample_stop_exclusive"] for row in runs], dtype=np.int64),
            t_start_s=np.array([row["t_start_s"] for row in runs], dtype=np.float64),
            t_stop_s=np.array([row["t_stop_s"] for row in runs], dtype=np.float64),
            duration_s=np.array([row["duration_s"] for row in runs], dtype=np.float64),
            raw_value=np.array([row["raw_value"] for row in runs], dtype=np.int64),
            code=np.array([row["code"] for row in runs], dtype=np.int32),
        )
        print(f"wrote {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
