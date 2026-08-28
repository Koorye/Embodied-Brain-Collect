"""CurryEegRecorder — Neuroscan Curry NetStream TCP client (minimal).

Speaks the Curry NetStream wire format: 20-byte headers whose magic is
``CTRL`` on requests and ``DATA`` on responses (requests 6 = basic info,
3 = channel info, 8 = start streaming, 9 = stop).  After request 8 the host
pushes packets: code 2 = EEG data (channel-major float32, zlib-compressed
when the header sizes disagree; ``packet_size`` is the body size,
``uncompressed_size`` the raw size), code 3 = events (N x 536-byte structs:
i32 code @0:4 = marker code, i32 latency @4:8 = amp sample index), code 4 =
keepalive, code 1 = info (ignored).

A streaming request is sent before every read, which works against both
push-style hosts (which ignore it) and request-response hosts.  No
reconnect: if the connection fails during recording, the recorder logs it
and stops reading — the launcher owns session stop, and whatever was
captured is saved at teardown.
"""

from __future__ import annotations

import socket
import struct
import time
import zlib

import numpy as np

from .base_eeg_recorder import BaseEegRecorder
from .eeg_recorder_config import EegRecorderConfig

_HEADER = struct.Struct(">4sHHIII")
_REQ_MAGIC = b"CTRL"   # requests: client -> Curry
_RESP_MAGIC = b"DATA"  # responses: Curry -> client
_REQ_BASIC_INFO, _REQ_CHANNEL_INFO, _REQ_START, _REQ_STOP = 6, 3, 8, 9
_CODE_INFO, _CODE_EEG, _CODE_EVENT = 1, 2, 3
_EVENT_STRUCT_BYTES = 536
_MAX_BLOCK_BYTES = 64 * 1024 * 1024  # sanity cap per decompressed block


def _stream_request() -> bytes:
    return _HEADER.pack(_REQ_MAGIC, 2, _REQ_START, 0, 0, 0)


def _decode_block(body: bytes, uncompressed_size: int,
                  n_channels: int) -> np.ndarray | None:
    """Validate one EEG block -> (n_samples, n_channels) f32.

    Curry 9 sends raw channel-major float32; a header ``uncompressed_size``
    that disagrees with the body length means the body is zlib-compressed.
    """
    if 0 < uncompressed_size != len(body):
        if uncompressed_size > _MAX_BLOCK_BYTES:
            return None
        try:
            body = zlib.decompress(body)
        except zlib.error:
            return None
    elif len(body) > _MAX_BLOCK_BYTES:
        return None
    n_vals = len(body) // 4
    n_samples = n_vals // n_channels
    if n_samples <= 0 or n_vals % n_channels:
        return None
    flat = np.frombuffer(body[: n_samples * n_channels * 4],
                         dtype="<f4").copy()
    return flat.reshape((n_samples, n_channels))


def _parse_events(body: bytes) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for off in range(0, len(body) - 16 + 1, _EVENT_STRUCT_BYTES):
        code, latency = struct.unpack_from("<ii", body, off)
        out.append((code, latency))
    return out


class CurryEegRecorder(BaseEegRecorder):
    """Records EEG from Curry's TCP NetStream service."""

    config: EegRecorderConfig

    def __init__(self, config: EegRecorderConfig):
        super().__init__(config)
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> socket.socket:
        sock = socket.create_connection((self.config.host, self.config.port),
                                        timeout=1.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.5)
        return sock

    def _close_socket(self) -> None:
        sock, self._sock = self._sock, None   # null first: concurrent polls
        if sock is None:                      # see a closed recorder
            return
        try:
            sock.sendall(_HEADER.pack(_REQ_MAGIC, 2, _REQ_STOP, 0, 0, 0))
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _recv_exact(self, size: int, timeout: float | None = None) -> bytes:
        """Read exactly ``size`` bytes; ``timeout`` temporarily overrides the
        socket timeout and is restored afterwards."""
        assert self._sock is not None
        old = self._sock.gettimeout()
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            buf = bytearray()
            while len(buf) < size:
                chunk = self._sock.recv(size - len(buf))
                if not chunk:
                    raise ConnectionError("connection closed by peer")
                buf.extend(chunk)
            return bytes(buf)
        finally:
            self._sock.settimeout(old)

    def _request(self, req: int) -> tuple[int, bytes]:
        """One blocking request/response (handshake only): (code, body)."""
        assert self._sock is not None
        self._sock.sendall(_HEADER.pack(_REQ_MAGIC, 2, req, 0, 0, 0))
        hdr = self._recv_exact(20)
        magic, code, _rq, _ss, packet_size, _us = _HEADER.unpack(hdr)
        if magic not in (_REQ_MAGIC, _RESP_MAGIC):
            raise ConnectionError(f"bad magic {magic!r}")
        return code, self._recv_exact(packet_size)

    def _handshake(self) -> None:
        code, body = self._request(_REQ_BASIC_INFO)
        if code != _CODE_INFO:
            raise ConnectionError(f"basic info: unexpected code={code}")
        self._sample_rate = float(np.frombuffer(body[8:12], dtype="<u4")[0])
        n_channels = int(np.frombuffer(body[4:8], dtype="<u4")[0])
        data_size = int(np.frombuffer(body[12:16], dtype="<u4")[0])
        if data_size != 4:
            raise ConnectionError(f"unsupported data_size={data_size} "
                                  f"(only float32 supported)")
        self._n_channels = n_channels
        code, body = self._request(_REQ_CHANNEL_INFO)
        if code != _CODE_INFO:
            raise ConnectionError(f"channel info: unexpected code={code}")
        # per-channel block: u32 index @0:4, then a UTF-16LE label padded
        # with NULs to the block end; stride = packet_size // n_channels
        stride = (len(body) // n_channels
                  if n_channels and len(body) % n_channels == 0 else 120)
        labels: list[str] = []
        for i in range(n_channels):
            off = i * stride
            if off + 4 > len(body):
                break
            label = body[off + 4: off + stride].decode(
                "utf-16-le", errors="ignore").split("\x00", 1)[0].strip()
            labels.append(label or f"Ch{i + 1}")
        if len(labels) < n_channels:
            labels += [f"Ch{i + 1}" for i in range(len(labels), n_channels)]
        self._channel_labels = labels

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> bool:
        try:
            self._sock = self._connect()
            self._handshake()
        except (OSError, ConnectionError) as exc:
            self._close_socket()
            self._open_error = (f"cannot connect to Curry NetStream "
                                f"{self.config.host}:{self.config.port} — "
                                f"{type(exc).__name__}: {exc}")
            self._log(f"[eeg:curry] open failed — {self._open_error}")
            return False
        self._log(f"[eeg:curry] connected: {self._n_channels} ch @ "
                  f"{self._sample_rate:g} Hz float32; trigger = "
                  f"'{self._channel_labels[-1]}'")

        # first-data gate; events parsed on the way
        t0 = time.time()
        while self._total_samples == 0:
            if not self._read_packet():
                self._open_error = "stream failed before first data"
                self._log(f"[eeg:curry] open failed — {self._open_error}")
                self._close_socket()
                return False
            if time.time() - t0 > self.config.open_timeout:
                self._open_error = (f"no EEG block within "
                                    f"{self.config.open_timeout:g}s")
                self._log(f"[eeg:curry] open failed — {self._open_error}")
                self._close_socket()
                return False
        return True

    def _close(self) -> None:
        self._close_socket()
        self._log(f"[eeg:curry] stopped (samples={self._total_samples}, "
                  f"events={len(self._buf.get('eeg_event_code', []))})")
        super()._close()

    # ------------------------------------------------------------------
    # Packet stream
    # ------------------------------------------------------------------

    def _poll(self, ts: float) -> None:
        if self._sock is not None and not self._read_packet():
            # stream failed: log once and stop reading — like the other
            # recorders, the launcher owns session stop
            self._close_socket()

    def _read_packet(self) -> bool:
        """One streaming request + one complete packet; False on failure."""
        assert self._sock is not None
        try:
            self._sock.sendall(_stream_request())
            hdr = self._recv_exact(20)
        except socket.timeout:
            return True  # idle stream
        except OSError as exc:
            self._log(f"[eeg:curry] stream failed: {exc}")
            return False
        magic, code, _rq, start_sample, packet_size, usize = _HEADER.unpack(hdr)
        if magic not in (_REQ_MAGIC, _RESP_MAGIC):
            self._log("[eeg:curry] bad packet header; stopping reads",
                      level="ERROR")
            return False
        # body 可能是 532KB 的大块(1000 样本 x 133 通道):header 超时当 idle
        # 无害,但 body 读一半超时会丢半包导致后续流错位 —— 给足 5s,真超时
        # 则明确停读并报错,绝不在错位上继续解析。
        try:
            body = self._recv_exact(packet_size, timeout=5.0)
        except socket.timeout:
            self._log("[eeg:curry] body timeout (5s) — stream out of sync, "
                      "stopping reads", level="ERROR")
            return False
        except OSError as exc:
            self._log(f"[eeg:curry] stream failed: {exc}", level="ERROR")
            return False
        self._handle_packet(code, start_sample, usize, body)
        return True

    def _handle_packet(self, code: int, start_sample: int,
                       uncompressed_size: int, body: bytes) -> None:
        if code == _CODE_EEG:
            block = _decode_block(body, uncompressed_size, self._n_channels)
            if block is not None:
                self._on_block(start_sample, block)
            else:
                # 坏块会被 QC 的块连续性检查抓到,但日志里也要留痕
                self._log(f"[eeg:curry] 无法解码的块 @sample {start_sample} "
                          f"({len(body)}B, 未压缩 {uncompressed_size}B) — 丢弃",
                          level="WARNING")
        elif code == _CODE_EVENT:
            for ev_code, latency in _parse_events(body):
                self._on_event(ev_code, latency)
        # code 1 (info) / 4 (keepalive): ignore
