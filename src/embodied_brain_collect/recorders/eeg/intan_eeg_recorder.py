"""IntanEegRecorder — RHD/RHS via the RHX software's TCP interface.

Speaks RHX's documented wire protocol directly (stdlib sockets + numpy, no
vendor library): an ASCII command channel and a binary waveform channel.

* Commands (default port 5000): ``set <param> <value>\\n`` / ``get <param>``
  / ``execute <cmd>\\n``; responses are ``Return: ...`` lines.  The recorder
  clears all TCP data outputs, enables the wide band for the requested
  amplifier channels plus one DIGITAL-IN channel, brings up the waveform
  server (``set tcpwaveformdata.status connect``) if it is not up yet, and
  flips run mode.  Prerequisite configured once in the RHX GUI: Settings →
  enable TCP Command Interface.

* Waveform data (default port 5001): blocks of ``<u4 magic 0x2ef07a08`` +
  128 frames; each frame is ``<u4 device sample counter`` + one ``<u2`` per
  enabled output word (our amplifier channels first, then the 16-bit
  digital-in word if enabled — RHX writes the full port word once).  Wide
  amplifier values convert as ``0.195 uV * (raw - 32768)``.

The digital-in word is the TTL path: the ParallelBox marker codes land in
it, edge detection turns them into EEG events (code = word value, latency =
device sample counter) for the close-time amp->PC clock fit — the same
contract Curry fulfils with its event packets.  The word is also appended
to ``eeg_data`` as the trailing Trigger column.
"""

from __future__ import annotations

import re
import socket
import time

import numpy as np

from .base_eeg_recorder import BaseEegRecorder
from .eeg_recorder_config import IntanEegRecorderConfig

_MAGIC = 0x2EF07A08          # RHX TCP waveform block magic (little-endian)
_FRAMES_PER_BLOCK = 128      # RHX writes fixed 128-frame blocks
_UV_PER_LSB = 0.195          # RHD/RHS wide 16-bit amplifier step
_ADC_ZERO = 32768

#: native 通道名:端口字母 + 三位编号(D-000 / A-001)
_CHANNEL_NAME_RE = re.compile(r"^([A-Ha-h])-(\d{1,3})$")


def wrap32(ts_new: int, ts_old: int) -> int:
    """Forward distance on the wrapping 32-bit device sample counter.

    u32 wraps after ~39.7 h @ 30 kHz; the frame timestamp is parsed as
    unsigned, so a negative delta means a wrap, not backwards time.
    """
    return (ts_new - ts_old) & 0xFFFFFFFF


class IntanStreamParser:
    """Byte-stream -> (start_index, block (n, W) f32, marker events).

    Frame layout = ``u32 timestamp + n_words * u16`` (words = amplifier
    channels in port/channel order, then the digital-in word last, if
    enabled).  Blocks not starting on the magic are resynchronised by
    search; a wrong ``n_words`` (headstage count mismatch) desynchronises
    every subsequent block, which the magic check surfaces immediately.
    """

    def __init__(self, n_amp: int, has_dig: bool, digital_mask: int = 0xFFFF,
                 digital_map: dict[int, int] | None = None):
        self.n_amp = int(n_amp)
        self.has_dig = bool(has_dig)
        self.digital_mask = int(digital_mask)
        #: 字(掩码后)→marker 码;None = 直接用字值。查不到的字不发事件。
        self.digital_map = ({int(k, 0) if isinstance(k, str) else int(k):
                             int(v) for k, v in digital_map.items()}
                            if digital_map else None)
        self.n_words = self.n_amp + (1 if self.has_dig else 0)
        self.frame_dtype = np.dtype(
            [("ts", "<u4"), ("v", "<u2", (self.n_words,))])
        self.frame_bytes = 4 + 2 * self.n_words
        self.block_bytes = 4 + _FRAMES_PER_BLOCK * self.frame_bytes
        self.ts0: int | None = None      # first frame counter -> index 0
        self.last_ts: int | None = None
        self.last_dig_word = -1          # 数字字边沿检测状态(跨块保持;映射后)
        self.resyncs = 0

    def reset(self) -> None:
        """统一开录时重置流锚点:时间戳原点/数字字状态/同步计数全部归零。"""
        self.ts0 = None
        self.last_ts = None
        self.last_dig_word = -1
        self.resyncs = 0

    def _index(self, ts: int) -> int:
        return wrap32(ts, self.ts0) if self.ts0 is not None else 0

    def feed(self, buf: bytearray) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Parse every complete block in ``buf``; caller keeps the remainder.

        Returns ``(start_index, block, events)`` triples where ``block`` is
        (128, n_amp + dig) float32 — amplifier columns in µV, trailing
        Trigger column = raw digital-in word — and ``events`` is the list of
        ``(code, latency)`` marker events found in this block (non-zero
        rising edges / code-to-code changes of the digital word).
        """
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        while True:
            if len(buf) < self.block_bytes:
                break
            if int.from_bytes(buf[0:4], "little") != _MAGIC:
                self._resync(buf)
                if len(buf) < self.block_bytes:
                    break
                continue
            frames = np.frombuffer(
                bytes(buf[: self.block_bytes]),
                dtype=self.frame_dtype, count=_FRAMES_PER_BLOCK, offset=4)
            ts = frames["ts"].astype(np.int64)
            v = frames["v"]
            if self.ts0 is None:
                self.ts0 = int(ts[0])
            start = self._index(int(ts[0]))
            block = np.empty(
                (_FRAMES_PER_BLOCK, self.n_words), dtype=np.float32)
            block[:, : self.n_amp] = (
                v[:, : self.n_amp].astype(np.int32) - _ADC_ZERO
            ).astype(np.float32) * np.float32(_UV_PER_LSB)
            events: list[tuple[int, int]] = []
            if self.has_dig:
                # Trigger 列存原始字;事件码 = 字 & mask 再查映射表
                block[:, self.n_amp] = v[:, self.n_amp].astype(np.float32)
                for i, raw in enumerate(v[:, self.n_amp].astype(np.int64)):
                    w = int(raw) & self.digital_mask
                    if self.digital_map is not None:
                        w = self.digital_map.get(w, 0)   # 查不到 → 不发事件
                    if w != self.last_dig_word:
                        if w != 0:    # 上升沿(或码间直切):新码即事件码
                            events.append((w, start + i))
                        self.last_dig_word = w
            self.last_ts = int(ts[-1])
            out.append((start, block, events))
            del buf[: self.block_bytes]
        return out

    def _resync(self, buf: bytearray) -> None:
        """Drop bytes up to the next magic (keep a 3-byte tail: the magic may
        straddle the read boundary)."""
        magic = _MAGIC.to_bytes(4, "little")
        pos = bytes(buf).find(magic)
        self.resyncs += 1
        if pos < 0:
            del buf[: max(0, len(buf) - 3)]
        else:
            del buf[:pos]


def parse_channel_spec(channels: str, ports: str,
                       channels_per_port: int, base: int = 1) -> list[str]:
    """Config -> native amplifier channel names (``A-001`` style).

    ``channels`` wins when given;token 形式三种:

    - 起止范围:``"D-000..D-127"``(同端口,含两端;显式给出时不做 0/1
      编号探测,写什么用什么)
    - 显式名列表:``"A-001,B-032"``
    - 裸端口字母:``"A,B"`` —— 按 ``channels_per_port`` 展开

    都没给时按 ``ports`` 展开。``base`` 是展开用的编号起点(在线探测)。
    """
    def expand(port: str) -> list[str]:
        port = port.strip().upper()
        return [f"{port}-{i + base:03d}"
                for i in range(max(0, int(channels_per_port)))]

    def expand_range(token: str) -> list[str]:
        first, _, last = token.partition("..")
        m1 = _CHANNEL_NAME_RE.match(first.strip())
        m2 = _CHANNEL_NAME_RE.match(last.strip())
        if not m1 or not m2 or m1.group(1) != m2.group(1):
            raise ValueError(
                f"通道范围 {token!r} 不合法 —— 应为同端口的名对,如 "
                '"D-000..D-127"')
        lo, hi = int(m1.group(2)), int(m2.group(2))
        if not lo <= hi or hi - lo + 1 > 1024:
            raise ValueError(f"通道范围 {token!r} 不合理({lo}..{hi})")
        port = m1.group(1)
        return [f"{port}-{n:03d}" for n in range(lo, hi + 1)]

    names: list[str] = []
    if channels and channels.strip():
        for token in channels.split(","):
            token = token.strip()
            if not token:
                continue
            if ".." in token:
                names.extend(expand_range(token))
            elif "-" in token:
                names.append(token.upper())
            else:
                names.extend(expand(token))
        return names
    for port in ports.split(","):
        port = port.strip()
        if port:
            names.extend(expand(port))
    return names


class IntanEegRecorder(BaseEegRecorder):
    """Records EEG/ECoG from Intan RHD/RHS hardware via the RHX software."""

    config: IntanEegRecorderConfig

    def __init__(self, config: IntanEegRecorderConfig):
        super().__init__(config)
        # socket 只在 _open() 里建 —— Windows spawn 下 recorder 会被 pickle
        self._cmd_sock: socket.socket | None = None
        self._data_sock: socket.socket | None = None
        self._parser: IntanStreamParser | None = None
        # 名字避开基类的 _buf(那是落盘数据 dict):这只是 socket 读缓冲
        self._rx_buf = bytearray()
        self._started_server = False
        self._changed_runmode = False

    # ------------------------------------------------------------------
    # Command channel
    #
    # RHX 对 get 回 "Return: ..."/"Error: ...";对 set,成功时静默,失败时
    # 也回一行 Error。因此:每次 get 前先排空积压的陈旧响应,否则会把前面
    # 一串 set 的错误行当成 get 的答案(真机上踩过:33 条 set 的错误堆在
    # 缓冲里,get 读到的是它们的拼接)。
    # ------------------------------------------------------------------

    def _drain(self) -> None:
        """丢掉命令口里积压的陈旧响应。"""
        assert self._cmd_sock is not None
        self._cmd_sock.setblocking(False)
        try:
            while True:
                try:
                    if not self._cmd_sock.recv(4096):
                        break
                except (BlockingIOError, ConnectionResetError, OSError):
                    break
        finally:
            self._cmd_sock.setblocking(True)
            self._cmd_sock.settimeout(1.0)

    def _cmd(self, line: str, timeout: float = 0.005) -> str:
        """发一条命令;顺带短读一下即时错误行(set 失败时 RHX 会回 Error)。"""
        assert self._cmd_sock is not None
        self._cmd_sock.sendall((line + "\n").encode())
        self._cmd_sock.settimeout(timeout)
        try:
            return self._cmd_sock.recv(4096).decode(errors="replace")
        except socket.timeout:
            return ""
        except OSError:
            return ""

    def _get(self, param: str, timeout: float = 0.5) -> str:
        self._drain()
        self._cmd_sock.sendall(f"get {param}\n".encode())
        self._cmd_sock.settimeout(timeout)
        try:
            return self._cmd_sock.recv(4096).decode(errors="replace").strip()
        except socket.timeout:
            return ""
        except OSError:
            return ""

    @staticmethod
    def _ok(resp: str) -> bool:
        """get 的响应是否为正常返回(而非 Error/无响应)。"""
        return resp.lower().startswith("return")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _fail(self, reason: str) -> bool:
        self._open_error = reason
        self._log(f"[eeg:intan] open failed — {reason}")
        self._close_sockets()
        return False

    def _close_sockets(self) -> None:
        cmd, self._cmd_sock = self._cmd_sock, None
        data, self._data_sock = self._data_sock, None
        for sock in (cmd, data):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _open(self) -> bool:
        cfg = self.config
        try:
            self._cmd_sock = socket.create_connection(
                (cfg.host, cfg.command_port), timeout=3.0)
        except OSError as exc:
            return self._fail(
                f"连不上 RHX 命令口 {cfg.host}:{cfg.command_port} — "
                f"{type(exc).__name__}: {exc};先在 RHX 软件的 Settings 里"
                "启用 TCP Command Interface")
        self._cmd_sock.settimeout(1.0)

        resp = self._get("runmode")
        if "Return:" not in resp:
            return self._fail(f"RHX 命令口无响应({resp!r})— 检查 RHX 软件"
                              "是否在运行、TCP Command Interface 是否启用")
        self._log(f"[eeg:intan] 命令口就绪: runmode={resp.split()[-1:]}")

        resp = self._get("sampleratehertz")
        try:
            self._sample_rate = float(resp.split()[-1])
        except (IndexError, ValueError):
            return self._fail(f"读不到采样率({resp!r})")
        if self._sample_rate <= 0:
            return self._fail(f"采样率异常: {self._sample_rate!r}")

        # 输出配置:清空全部 TCP 输出,使能放大器 wide + DIGITAL-IN。
        # 帧里的数字字只出现一次(整口 16 位字),使能 DIGITAL-IN-01 即可;
        # 注意 RHX 的数字通道名是两位补零(DIGITAL-IN-01,源码规定)。
        # 通道编号起点(0/1)因设备而异:显式 channels 清单按用户写的用;
        # 否则探测第一个端口的 -000 是否存在 —— 只有 0 起始的设备才有
        # XXX-000(1 起始的设备里 -001 存在但 -000 不存在,先探 001 会误判)。
        base = 1
        if not (cfg.channels and cfg.channels.strip()):
            probe_port = (cfg.ports.split(",")[0].strip().upper()
                          or "A")
            if self._ok(self._get(f"{probe_port}-000.tcpdataoutputenabled")):
                base = 0
                self._log(f"[eeg:intan] 通道编号从 0 起({probe_port}-000)")
        ch_names = parse_channel_spec(cfg.channels, cfg.ports,
                                      cfg.channels_per_port, base)
        if not ch_names:
            return self._fail("通道列表为空 — 检查 ports/channels_per_port"
                              "/channels 配置")
        self._cmd("execute clearalldataoutputs")
        confirmed: list[str] = []
        for name in ch_names:
            err = self._cmd(f"set {name}.tcpdataoutputenabled true")
            resp = self._get(f"{name}.tcpdataoutputenabled")
            if self._ok(resp) and "true" in resp.lower():
                confirmed.append(name)
            else:
                detail = (err.strip() or resp.strip()
                          or "无响应(通道不存在?)").splitlines()[-1]
                self._log(f"[eeg:intan] 通道 {name} 使能未确认 — {detail}",
                          level="WARNING")
        digital_ok = False
        if int(cfg.digital_in) > 0:
            self._cmd("set DIGITAL-IN-01.tcpdataoutputenabled true")
            resp = self._get("DIGITAL-IN-01.tcpdataoutputenabled")
            digital_ok = self._ok(resp) and "true" in resp.lower()
            if not digital_ok:
                self._log("[eeg:intan] DIGITAL-IN-01 TCP 输出使能未确认"
                          f"({(resp or '无响应')!r})— 帧里没有 TTL 字,"
                          "marker 对齐将失败", level="WARNING")
        if not confirmed:
            # 一个放大器通道都没使能上:探测 A-H 哪个口有头戴,给出可操作提示
            alive = [p for p in "ABCDEFGH"
                     if self._ok(self._get(f"{p}-001.tcpdataoutputenabled"))]
            hint = (f"检测到有通道的端口: {','.join(alive)}"
                    if alive else
                    "A-H 都没有通道 — 检查头戴连接/RHX 是否识别到控制器")
            return self._fail(
                f"请求的 {len(ch_names)} 个通道一个都没使能上"
                f"(请求了 {ch_names[0]}..{ch_names[-1]})。{hint};"
                '用 ports: "B"(或 channels: "B-001,...")配置实际端口')
        if len(confirmed) < len(ch_names):
            self._log(f"[eeg:intan] {len(ch_names) - len(confirmed)} 个通道"
                      "未确认,按实际使能的 "
                      f"{len(confirmed)} 路解析({confirmed[0]}.."
                      f"{confirmed[-1]})", level="WARNING")

        # 波形服务器:未启动就通过命令口拉起(端口不符则先改端口)。
        # 注意:就算 status 已是 connect 也照常连 —— 上一次运行若把服务器
        # 留在挂死状态,数据口会静默无流,由下面的自愈逻辑处理。
        if cfg.start_data_server and not self._data_connect():
            return False

        # 解析器按 RHX 确认使能的通道数建 —— 头戴通道数配错也不会失步
        self._channel_labels = confirmed + (["Trigger"] if digital_ok else [])
        self._parser = IntanStreamParser(
            len(confirmed), digital_ok, int(cfg.digital_mask),
            cfg.digital_map)

        if cfg.set_runmode:
            self._cmd("set runmode run")
            self._changed_runmode = True
        else:
            mode = self._get("runmode")
            if "run" not in mode.lower() and "record" not in mode.lower():
                return self._fail(f"RHX 未在采集({mode!r})且 set_runmode="
                                  "false — 在 RHX 里按 Run,或改配置")

        # 首帧闸门:3 秒无数据先尝试一次服务器重启(disconnect→connect,
        # 自愈挂死的 TCP 波形服务器),再等到闸门预算用尽。
        if not self._wait_first_block(3.0) and cfg.start_data_server:
            self._log("[eeg:intan] 3s 无波形数据 — 重启 RHX TCP 波形服务器"
                      "(disconnect→connect)后重试", level="WARNING")
            self._cmd("set tcpwaveformdata.status disconnect")
            self._close_data_sock()
            time.sleep(0.5)
            self._cmd("set tcpwaveformdata.status connect")
            self._started_server = True
            time.sleep(0.5)
            if not self._data_connect():
                return False
        # 闸门预算留 5s 提前量:让本处的明确报错跑在 launcher 的
        # open_timeout 强杀之前,不留一句含糊的 "open TIMEOUT"
        budget = max(5.0, float(self.config.open_timeout) - 5.0)
        if not self._wait_first_block(budget):
            return self._fail(
                f"{budget:g}s 内没有收到波形数据(含一次服务器重启)— RHX "
                "是否处于 Run 状态、通道使能是否与头戴一致、TCP Waveform "
                "Data 服务器是否被其他客户端占住")
        self._log(f"[eeg:intan] {len(confirmed)} ch @ "
                  f"{self._sample_rate:g} Hz (RHX TCP), trigger = "
                  + ("DIGITAL-IN word" if digital_ok else "无(使能失败)"))
        return True

    # ------------------------------------------------------------------
    # Data stream
    # ------------------------------------------------------------------

    def _data_connect(self) -> bool:
        """连数据口;start_data_server 时先确保服务器在 connect 状态。"""
        if self.config.start_data_server:
            status = self._get("tcpwaveformdata.status")
            if "connect" not in status.lower():
                port_resp = self._get("tcpwaveformdata.port")
                if f"{self.config.data_port}" not in port_resp:
                    self._cmd(
                        f"set tcpwaveformdata.port {self.config.data_port}")
                self._cmd("set tcpwaveformdata.status connect")
                self._started_server = True
                time.sleep(0.3)
        try:
            self._data_sock = socket.create_connection(
                (self.config.host, self.config.data_port), timeout=3.0)
        except OSError as exc:
            self._fail(f"连不上 RHX 波形数据口 {self.config.host}:"
                       f"{self.config.data_port} — {type(exc).__name__}: {exc};"
                       "检查 RHX 的 TCP Waveform Data 服务器状态/端口")
            return False
        self._data_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._data_sock.settimeout(0.05)
        return True

    def _close_data_sock(self) -> None:
        sock, self._data_sock = self._data_sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _wait_first_block(self, budget_s: float) -> bool:
        t0 = time.time()
        while self._total_samples == 0:
            try:
                self._poll_read()
            except (OSError, ConnectionError) as exc:
                self._log(f"[eeg:intan] 波形流断开 — {type(exc).__name__}: "
                          f"{exc}", level="WARNING")
                return False
            if time.time() - t0 > budget_s:
                return False
            time.sleep(0.02)
        return True

    def _reset_stream_state(self) -> None:
        if self._parser is not None:
            self._parser.reset()
        # socket 里积压的仍是统一开录之前的采样,一并丢弃
        self._rx_buf.clear()

    def _poll_read(self) -> None:
        assert self._data_sock is not None and self._parser is not None
        try:
            chunk = self._data_sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("waveform stream closed by RHX")
        except socket.timeout:
            return
        self._rx_buf.extend(chunk)
        for start, block, events in self._parser.feed(self._rx_buf):
            self._on_block(start, block)
            for code, latency in events:
                self._on_event(code, latency)

    def _poll(self, ts: float) -> None:
        if self._data_sock is not None:
            try:
                self._poll_read()
            except OSError as exc:
                self._log(f"[eeg:intan] 波形流断开: {exc} — 停止读取",
                          level="ERROR")
                try:
                    self._data_sock.close()
                except OSError:
                    pass
                self._data_sock = None

    def _heartbeat_stats(self, elapsed: float) -> str:
        extra = super()._heartbeat_stats(elapsed)
        if self._parser is not None and self._parser.resyncs:
            extra += f" resyncs={self._parser.resyncs}"
        return extra

    def _close(self) -> None:
        # 顺序很要紧:先关数据 socket,再断服务器 —— 反过来的话 RHX 被要求
        # disconnect 时客户端还挂着,TCP 波形服务器可能就此挂死(下次运行
        # 数据口静默无流,open 超时)。只回滚我们自己改过的状态。
        self._close_data_sock()
        if self._cmd_sock is not None:
            if self._changed_runmode:
                self._cmd("set runmode stop")
            if self._started_server:
                self._cmd("set tcpwaveformdata.status disconnect")
        self._close_sockets()
        self._log(f"[eeg:intan] stopped (samples={self._total_samples}, "
                  f"events={len(self._buf.get('eeg_event_code', []))}, "
                  f"resyncs={self._parser.resyncs if self._parser else 0})")
        super()._close()
