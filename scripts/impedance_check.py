#!/usr/bin/env python3
"""Standalone Curry 9 NetStream impedance check tool (ACTIVE, no repo code).

Connects to 127.0.0.1:4455, triggers one impedance check (12), collects
DATA_Impedances (code 4, float32 per channel, ohms), stops the check (13),
then — before closing — waits for the EEG stream to resume.

Timing red lines learned on real hardware and honored here: after request
13 the amplifier falls back to its connected state and only resumes
reading after ~10s; disconnecting BEFORE the resume wedges the driver
(Device Error 5 — CURRY's own start button dies until restart), so this
tool always waits out the transition.  Never sends AmpConnect (10).

Usage:
    python impedance_check.py                 # trigger + print table
    python impedance_check.py --save z.npz
"""
import argparse
import socket
import struct
import sys
import time

import numpy as np

HOST, PORT = "127.0.0.1", 4455
HDR = struct.Struct(">4sHHIII")
CTRL = b"CTRL"
REQ_CHANNEL_INFO, REQ_BASIC_INFO = 3, 6
REQ_STREAM_START, REQ_STREAM_STOP = 8, 9
REQ_IMP_START, REQ_IMP_STOP = 12, 13
DATA_INFO, DATA_EEG, DATA_EVENTS, DATA_IMP = 1, 2, 3, 4


def send_req(sock, req):
    sock.sendall(HDR.pack(CTRL, 2, req, 0, 0, 0))


def recv_exact(sock, n, timeout):
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def read_pkt(sock, timeout=2.0):
    hdr = recv_exact(sock, 20, timeout)
    magic, code, rq, ss, size, us = HDR.unpack(hdr)
    body = recv_exact(sock, size, 5.0) if size else b""
    return magic, code, rq, ss, size, us, body


def wait_eeg(sock, seconds):
    """等 EEG 恢复流动(每秒补一次流请求);返回是否等到。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            send_req(sock, REQ_STREAM_START)
            m, c, rq, ss, size, us, body = read_pkt(sock, 1.0)
            if c == DATA_EEG:
                return True
        except (socket.timeout, ConnectionError, OSError):
            continue
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", default=None, help="把阻抗均值存成 npz")
    args = ap.parse_args()

    sock = socket.create_connection((HOST, PORT), timeout=2.0)
    sock.settimeout(2.0)

    send_req(sock, REQ_BASIC_INFO)
    m, c, rq, ss, size, us, body = read_pkt(sock)
    f = np.frombuffer(body, dtype="<u4")
    n_ch, fs = int(f[1]), int(f[2])
    print(f"Curry NetStream {HOST}:{PORT}: {n_ch} ch @ {fs} Hz")

    send_req(sock, REQ_CHANNEL_INFO)
    m, c, rq, ss, size, us, body = read_pkt(sock)
    stride = size // n_ch
    labels = []
    for i in range(n_ch):
        off = i * stride
        label = body[off + 4: off + stride].decode(
            "utf-16-le", errors="ignore").split("\x00", 1)[0].strip()
        labels.append(label or f"Ch{i + 1}")

    send_req(sock, REQ_STREAM_START)
    send_req(sock, REQ_IMP_START)
    print("requesting impedance check ...")
    snaps = []
    t_end = time.time() + 8.0
    while time.time() < t_end and len(snaps) < 3:
        try:
            m, c, rq, ss, size, us, body = read_pkt(sock, 1.0)
        except socket.timeout:
            continue
        if c != DATA_IMP:
            continue
        vals = np.frombuffer(body, dtype="<f4")
        if vals.size == n_ch and vals.any():
            snaps.append(vals)

    send_req(sock, REQ_IMP_STOP)

    if not snaps:
        # 没进阻抗态(检测被拒/无数据):不影响驱动,直接走
        send_req(sock, REQ_STREAM_STOP)
        sock.close()
        print("没有收到阻抗数据 — 需要先在 Curry 里点蓝色三角连上放大器。")
        return 1

    # 阻抗后回 connect 态,~10s 恢复读数;恢复前绝不断开(Error 5)
    print("等待放大器恢复读数(实测 ~10s;恢复前保持连接)...")
    resumed = wait_eeg(sock, 20.0) or wait_eeg(sock, 20.0)
    send_req(sock, REQ_STREAM_STOP)
    sock.close()
    if not resumed:
        print("放大器未恢复读数(已等 40s)— 先重启 Curry 再重连;"
              "未恢复前不要反复重连(报 Device Error)。")
        return 1

    z = np.stack(snaps).mean(axis=0)
    print(f"\n{len(snaps)} snapshots, mean impedance per channel:")
    print(f"{'ch':>4} {'label':>10} {'kOhm':>9}")
    for i, (lab, v) in enumerate(zip(labels, z)):
        flag = "  <-- check cap" if v > 50_000 else ""
        print(f"{i + 1:>4} {lab:>10} {v / 1000:>9.1f}{flag}")
    if args.save:
        np.savez(args.save, impedance_ohm=z, channel_names=np.asarray(labels),
                 sample_rate=np.asarray(fs))
        print(f"saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
