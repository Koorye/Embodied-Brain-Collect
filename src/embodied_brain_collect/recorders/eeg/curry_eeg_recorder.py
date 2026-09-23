"""CurryEegRecorder — Neuroscan Curry NetStream TCP client (minimal).

Speaks the Curry NetStream wire format: 20-byte headers whose magic is
``CTRL`` on requests and ``DATA`` on responses (requests 6 = basic info,
3 = channel info, 8 = start streaming, 9 = stop).  After request 8 the
host pushes packets: code 2 = EEG data (channel-major float32,
zlib-compressed when the header sizes disagree; ``packet_size`` is the
body size, ``uncompressed_size`` the raw size), code 3 = events (N x
536-byte structs: i32 code @0:4 = marker code, i32 latency @4:8 = amp
sample index), code 4 = impedance (one float32 per channel, ohms, only
while an impedance check is running), code 1 = info (ignored).

A streaming request is sent before every read, which works against both
push-style hosts (which ignore it) and request-response hosts.  No
reconnect: if the connection fails during recording, the recorder logs it
and stops reading — the launcher owns session stop, and whatever was
captured is saved at teardown.

Open-time impedance gate (``impedance_check``, default on): ``_open``
triggers one impedance check (12), averages a few DATA_Impedances packets,
records the per-channel values into the npz, and refuses to open unless
the channels outside a configurable edge list pass (< ``impedance_max_kohm``)
at a rate of ``impedance_pass_rate``.  ``impedance_check: false`` bypasses
the stage entirely — connect, handshake, record.

Two timing rules learned on real hardware and enforced here:

* after the check (13) the amplifier falls back to its connected state
  and only starts reading again after ~10s — the recorder waits for the
  EEG stream to actually resume (up to 40s) and NEVER disconnects before
  that: disconnecting in the post-impedance state wedges the driver
  (Device Error 5, CURRY's own start button stops working until restart);
* no AmpConnect (10) either — racing the amplifier's own restart caused
  the same wedge.

All gate-failure paths therefore wait out the transition before returning;
the disconnect that follows (open failed -> socket closed) then happens
from the normal data-flowing state, which is proven safe (every session
end does it).  Known side effect of the trigger: the amplifier's
digital-input event ingestion stops after a triggered check, so EEG<->PC
marker alignment fails (zero amplifier events) — impedance gating and
marker alignment currently do not mix; disable the gate for sessions that
need markers.
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
_REQ_IMP_START, _REQ_IMP_STOP = 12, 13   # need bAllowClientToControlAmp
# 主动触发阻抗的两条时序红线(实测得出,违反会把放大器驱动打成
# Device Error,连 Curry 里点三角都失效,只能重启 Curry/放大器):
#   1) 阻抗结束(13)后放大器回到 connect 态,~10s 才开始读数 —— 恢复
#      读数**之前绝不能断开连接**(本 recorder 的失败路径也一律先等恢复)。
#   2) 不要发 AmpConnect(10):与放大器自己的恢复过程相撞,同样出错。
_CODE_INFO, _CODE_EEG, _CODE_EVENT, _CODE_IMP = 1, 2, 3, 4
_EVENT_STRUCT_BYTES = 536
_MAX_BLOCK_BYTES = 64 * 1024 * 1024  # sanity cap per decompressed block

# 阻抗门禁:请求 12 后 Curry 在同一条流上推 code-4 包(每通道一个 float32,
# 单位 Ω)。首包可能是全零初值,取够若干帧有效包取均值;到点没取够按失败算。
_IMPEDANCE_SNAPSHOTS = 3
_IMPEDANCE_WINDOW_S = 12.0
_IMPEDANCE_RESUME_WAIT_S = 20.0   # 阻抗结束→读数恢复实测 ~10s,给两倍余量;
                                  # 两轮都等不到才按失败处理(共 40s)


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
        # open 时阻抗门禁的结果(进 npz + 决定 open 成败)
        self._impedance_ohm: np.ndarray | None = None
        self._impedance_checked: np.ndarray | None = None   # bool per channel
        self._impedance_pass_rate: float = 0.0
        self._impedance_pass: bool = False
        self._impedance_n: int = 0

    # ------------------------------------------------------------------
    # Impedance gate (open-time, ACTIVE trigger)
    # ------------------------------------------------------------------

    def _trigger_impedance(self) -> np.ndarray | None:
        """One client-triggered impedance check -> per-channel mean in ohms
        (None = no data).  Sends 12, collects DATA_Impedances packets (EEG
        test signal and events are discarded), then 13.  Only the mean is
        kept; snapshot count lands in ``self._impedance_n``.
        """
        assert self._sock is not None
        snaps: list[np.ndarray] = []
        self._impedance_n = 0
        try:
            self._sock.sendall(_stream_request())
            self._sock.sendall(
                _HEADER.pack(_REQ_MAGIC, 2, _REQ_IMP_START, 0, 0, 0))
            deadline = time.time() + _IMPEDANCE_WINDOW_S
            while len(snaps) < _IMPEDANCE_SNAPSHOTS and time.time() < deadline:
                try:
                    hdr = self._recv_exact(20, timeout=1.0)
                except socket.timeout:
                    continue          # nothing this second; keep waiting
                magic, code, _rq, _ss, size, _us = _HEADER.unpack(hdr)
                if magic not in (_REQ_MAGIC, _RESP_MAGIC):
                    return None
                try:
                    body = (self._recv_exact(size, timeout=5.0)
                            if size else b"")
                except socket.timeout:
                    return None
                if code == _CODE_IMP:
                    vals = np.frombuffer(body, dtype="<f4")
                    # 全零包是检测启动初值;有效包必有非零通道
                    if vals.size == self._n_channels and vals.any():
                        snaps.append(vals)
        finally:
            # 13 一定发:绝不把放大器留在阻抗模式
            try:
                self._sock.sendall(
                    _HEADER.pack(_REQ_MAGIC, 2, _REQ_IMP_STOP, 0, 0, 0))
            except OSError:
                pass
        if not snaps:
            return None
        self._impedance_n = len(snaps)
        return np.stack(snaps).mean(axis=0)

    def _wait_eeg(self, seconds: float) -> bool:
        """等 EEG 包恢复流动(每秒补一次流请求,顺带排掉通知/杂包)。"""
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                self._sock.sendall(_stream_request())
                hdr = self._recv_exact(20, timeout=1.0)
            except socket.timeout:
                continue               # 重构期静默:继续等
            magic, code, _rq, _ss, size, _us = _HEADER.unpack(hdr)
            if magic not in (_REQ_MAGIC, _RESP_MAGIC):
                continue
            try:
                if size:
                    self._recv_exact(size, timeout=5.0)
            except socket.timeout:
                continue
            if code == _CODE_EEG:
                return True
        return False

    def _impedance_gate(self) -> tuple[bool, str]:
        """主动阻抗门禁:触发一次阻抗检测、取均值做通过率检查。时序红线
        (见常量区):13 之后放大器回 connect 态、~10s 才恢复读数,恢复前
        断开连接会打坏驱动 —— 所以**无论门禁过不过,都先等 EEG 恢复再返
        回**,失败路径随后发生的断开就落在正常的"数据在流"状态(与每次
        会话结束时的断开相同,实测安全)。返回 (True, 摘要) 时
        ``self._impedance_*`` 已填好(随 npz 落盘);(False, 原因文案) 的
        文案直接给操作员看。边缘通道(耳后/头围等天然高阻位)不参与统计;
        Trigger 槽位是状态字不是阻抗值,同样排除。"""
        cfg = self.config
        z = self._trigger_impedance()
        if z is None:
            return False, ("未检测到阻抗检测(请求被拒或无数据)— 重试一次;"
                           "若反复出现,在 Curry 里手动做一次阻抗后重采")
        # 阻抗结束 → connect 态,实测 ~10s 恢复读数;给两轮共 40s 余量
        resumed = self._wait_eeg(_IMPEDANCE_RESUME_WAIT_S) \
            or self._wait_eeg(_IMPEDANCE_RESUME_WAIT_S)
        warn_tail = "" if resumed else (
            "。另:放大器尚未恢复读数 — 先重启 Curry 再重连(未恢复前反复"
            "重连会报 Device Error)")

        thr = cfg.impedance_max_kohm * 1000.0
        edge = {c.strip().upper() for c in cfg.impedance_edge_channels}
        labels = [lab.strip() for lab in self._channel_labels]
        checked = np.asarray([lab.strip().upper() not in edge
                              and lab.strip().upper() != "TRIGGER"
                              for lab in labels])
        checked_idx = np.flatnonzero(checked)
        if checked_idx.size == 0:
            return False, ("阻抗检查配置错误:参与统计的通道数为 0"
                           "(检查 impedance_edge_channels 是否覆盖了全部通道)"
                           + warn_tail)
        fail_idx = checked_idx[z[checked_idx] >= thr]
        rate = 1.0 - fail_idx.size / checked_idx.size
        self._impedance_ohm = z
        self._impedance_checked = checked
        self._impedance_pass_rate = float(rate)

        def _fmt(idx: np.ndarray, limit: int) -> str:
            items = [f"{labels[i]}({z[i] / 1000:.0f}k)" for i in idx]
            tail = f" …共{len(items)}个" if len(items) > limit else ""
            return " ".join(items[:limit]) + tail

        need = cfg.impedance_pass_rate
        if rate < need:
            self._impedance_pass = False
            return False, (
                f"阻抗检查未通过: {fail_idx.size}/{checked_idx.size} 通道 "
                f"≥ {cfg.impedance_max_kohm:g}kΩ,通过率 {rate:.1%} < "
                f"{need:.0%}(边缘通道已豁免)— 超标: "
                f"{_fmt(fail_idx, 15)};请整理电极/补充导电膏后重试"
                + warn_tail)
        self._impedance_pass = True
        if not resumed:
            return False, ("阻抗检查已通过,但放大器未恢复采集(已等 40s)— "
                           "先重启 Curry 再重连;未恢复前不要反复重连(报 "
                           "Device Error)")
        detail = (f"impedance ok: {checked_idx.size - fail_idx.size}/"
                  f"{checked_idx.size} 通道 < {cfg.impedance_max_kohm:g}kΩ "
                  f"(通过率 {rate:.1%} ≥ {need:.0%})")
        if fail_idx.size:
            detail += f";超标(未阻断): {_fmt(fail_idx, 15)}"
        return True, detail

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
        if self.config.impedance_check:
            try:
                gate_ok, detail = self._impedance_gate()
            except (OSError, ConnectionError) as exc:
                gate_ok = False
                detail = f"阻抗检测失败: {type(exc).__name__}: {exc}"
            level = "INFO" if gate_ok else "ERROR"
            self._log(f"[eeg:curry] {detail}", level=level)
            if not gate_ok:
                self._close_socket()
                self._open_error = detail
                return False
        # 不读首帧:流是否在发由 launcher 的确认阶段判断(_poll 每次读前
        # 都会带一个 streaming request)。阻抗门禁结束后 EEG 流要 ~10s 才
        # 恢复,由确认阶段兜底,这里不等待。
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
        # code 4 (阻抗,仅阻抗检测进行时出现) / code 1 (info):ignore —
        # 录制期间不该有人开阻抗检测;真发生了,注入的测试信号会进 EEG,
        # 属于操作问题,QC 的数据审查比这里拦截更合适
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_output(self) -> dict[str, np.ndarray]:
        out = super()._build_output()
        if self._impedance_ohm is not None:
            # 通道顺序/命名与 eeg_channel_names 一致;checked=False 的是
            # 边缘豁免通道与 Trigger 状态字槽位
            out["eeg_impedance_ohm"] = self._impedance_ohm
            out["eeg_impedance_checked"] = self._impedance_checked
            out["eeg_impedance_pass_rate"] = np.asarray(
                self._impedance_pass_rate)
            out["eeg_impedance_check_pass"] = np.asarray(
                self._impedance_pass)
            out["eeg_impedance_n_snapshots"] = np.asarray(self._impedance_n)
        return out
