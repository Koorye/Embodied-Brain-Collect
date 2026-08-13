"""Send marker bytes to ParallelBox TTL over a serial COM port.

Use this for Curry/EEG marker smoke tests before running E-Prime.
Example:
    python -m tools.send_parallelbox_markers --port COM14 --codes 241,17,33,81,82,97,98,113,114,242 --hold-s 0.05
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import serial


def _parse_codes(text: str) -> list[int]:
    codes = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        code = int(item, 0)
        if code < 0 or code > 255:
            raise ValueError(f'marker code must be 0..255, got {code}')
        codes.append(code)
    if not codes:
        raise ValueError('no marker codes provided')
    return codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='COM14')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--codes', default='241,17,33,81,82,97,98,113,114,242')
    ap.add_argument('--hold-s', type=float, default=0.05)
    ap.add_argument('--isi-s', type=float, default=0.45)
    ap.add_argument('--pre-clear-s', type=float, default=1.0)
    ap.add_argument('--trial', type=int, default=1)
    ap.add_argument('--tag-prefix', default='SMOKE')
    ap.add_argument('--udp-host', default=None)
    ap.add_argument('--udp-port', type=int, default=9999)
    ap.add_argument('--log', type=Path, default=None)
    args = ap.parse_args(argv)

    codes = _parse_codes(args.codes)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_host else None
    records = []

    ser = serial.Serial(args.port, args.baud, timeout=1)
    try:
        ser.write(bytes([0]))
        ser.flush()
        time.sleep(args.pre_clear_s)
        for idx, code in enumerate(codes, 1):
            tag = f'{args.tag_prefix}_{idx:02d}'
            t_wall = time.time()
            t0 = time.perf_counter()
            if udp is not None:
                msg = f'EVT|trial={args.trial}|tag={tag}|code={code}|t_eprime_ms={int(t_wall * 1000)}'
                udp.sendto(msg.encode('utf-8'), (args.udp_host, args.udp_port))
            else:
                msg = ''
            ser.write(bytes([code]))
            ser.flush()
            time.sleep(args.hold_s)
            ser.write(bytes([0]))
            ser.flush()
            t1 = time.perf_counter()
            row = {
                'idx': idx,
                'code': code,
                'tag': tag,
                'hold_s': args.hold_s,
                'isi_s': args.isi_s,
                't_wall': t_wall,
                't_perf_start': t0,
                't_perf_stop': t1,
                'udp_msg': msg,
            }
            records.append(row)
            print(f'SEND idx={idx:02d} code={code:3d} tag={tag} hold={t1 - t0:.4f}s')
            time.sleep(args.isi_s)
    finally:
        ser.write(bytes([0]))
        ser.flush()
        ser.close()
        if udp is not None:
            udp.close()

    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open('w', encoding='utf-8') as fh:
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'wrote {args.log}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
