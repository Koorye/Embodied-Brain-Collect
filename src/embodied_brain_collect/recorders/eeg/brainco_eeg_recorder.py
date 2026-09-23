"""BrainCoEegRecorder — BCIGo 32 通道帽 (bcigo-sdk) + 软件同步。

BrainCo 设备没有 Curry 那种硬件 TTL 事件流。刺激与 EEG 的对齐走软件同步:

1. SDK 回调一到立刻盖 ``time.time()``(与 MarkerSender 的 ``t_sent_pc`` 同源)。
2. 按 LSL 惯例:本包**最后一样本** ≈ 包到达时刻,前面样本按 ``1/fs`` 回推。
3. 每个 EEG 包都是一个时钟观测点(远密于 stim marker)。收尾用与 Curry
   相同的稳健直线 ``t_pc = a * t_eeg + b`` 拟合包时钟。
4. 用拟合把 UDP marker 的 ``t_sent_pc`` 映射到 EEG 样本号,写成
   ``eeg_event_code`` / ``eeg_event_latency``,输出 schema 与 Curry 一致。

可选:刺激进程把同一份 UDP 再抄一份到 ``sync_udp_port``。录制中按最近
一包做线性插值,心跳就能看到事件数;收尾仍以全局包时钟 + markers.npz
为准(更稳,不受单包抖动支配)。
"""

from __future__ import annotations

import asyncio
import queue
import socket
import threading
import time
from typing import Any

import numpy as np

from .base_eeg_recorder import BaseEegRecorder, _fit_eeg_to_pc
from .eeg_recorder_config import BraincoEegRecorderConfig

BRAINCO_N_CHANNELS = 32
BRAINCO_CHANNEL_NAMES = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz",
    "P3", "P4", "P7", "P8", "Pz",
    "O1", "O2",
    "T7", "T8",
    "FC1", "FC2", "FC5", "FC6",
    "CP1", "CP2", "CP5", "CP6",
    "FT9", "FT10",
    "TP9", "TP10",
    "IO",
]
# 固件常在 32 路 EEG 后再附一路触发/状态,按 32 或 33 都能 reshape。
BRAINCO_N_CH_CANDIDATES = (BRAINCO_N_CHANNELS + 1, BRAINCO_N_CHANNELS)
BRAINCO_TRIGGER_NAME = "TRIG"

# get_eeg_buffer(take, clean): take 是样本个数。Python True → 1,再配
# clean=True 会丢掉同批其余点,实测只剩 ~30 Hz。一次尽量抽空缓冲。
EEG_BUFFER_TAKE = 8192

_FS_ATTR = {250: "SR_250Hz", 500: "SR_500Hz",
            1000: "SR_1000Hz", 2000: "SR_2000Hz"}
_GAIN_ATTR = {1: "GAIN_1", 2: "GAIN_2", 4: "GAIN_4", 6: "GAIN_6",
              8: "GAIN_8", 12: "GAIN_12", 24: "GAIN_24"}
_SIGNAL_ATTR = {
    "normal": "NORMAL",
    "test": "TEST_SIGNAL",
    "test_signal": "TEST_SIGNAL",
    "shorted": "SHORTED",
    "mvdd": "MVDD",
}

# 手动填写 host 但没写端口时的兜底;mDNS 会覆盖。
_DEFAULT_TCP_PORT = 8080


# =============================================================================
# Payload coercion
# =============================================================================

def _payload_from_cb(args: tuple, kwargs: dict) -> Any:
    if kwargs and not args:
        if "data" in kwargs:
            return kwargs
        return kwargs
    if len(args) == 1:
        return args[0]
    if not args:
        return kwargs
    return args


def coerce_eeg_payload(payload: Any) -> tuple[np.ndarray, dict]:
    """把 SDK 回调/缓冲区的多种形态收成 ``(n, n_ch)`` float32。

    官方文档只保证「批量 EEG + 序列号 + 时间戳 + 外部触发」,具体 Python
    对象形态随 SDK 版本变化。这里对 dict / 属性对象 / ndarray / 嵌套 list /
    变长 tuple 都做一次尽力解析;解析不出就返回空块,由调用方跳过。
    """
    extra: dict[str, Any] = {}
    data: Any = payload

    if isinstance(payload, dict):
        for k in ("seq", "sequence", "sn", "serial", "serial_number",
                  "timestamp", "ts", "time", "trigger", "trig",
                  "ext_trig", "ext_trigger"):
            if k in payload:
                extra[k] = payload[k]
        data = (payload.get("data") if payload.get("data") is not None
                else payload.get("eeg") if payload.get("eeg") is not None
                else payload.get("samples") if payload.get("samples") is not None
                else payload.get("values") if payload.get("values") is not None
                else payload.get("eeg_data"))
        if data is None:
            # 也许 dict 本身就是通道名 -> 序列
            vals = [payload[k] for k in BRAINCO_CHANNEL_NAMES if k in payload]
            if len(vals) >= 8:
                data = np.asarray(vals, dtype=np.float32).T
            else:
                return np.zeros((0, BRAINCO_N_CHANNELS), dtype=np.float32), extra

    elif not isinstance(payload, (np.ndarray, list, tuple, bytes, bytearray)):
        for k in ("seq", "sequence", "sn", "serial", "timestamp", "ts",
                  "time", "trigger", "trig"):
            if hasattr(payload, k):
                extra[k] = getattr(payload, k)
        for k in ("data", "eeg", "samples", "values", "eeg_data"):
            if hasattr(payload, k):
                data = getattr(payload, k)
                break

    if isinstance(data, tuple) and data:
        # (data,), (seq, data), (seq, ts, data), (seq, ts, trig, data) ...
        # 取「看起来最像波形」的那一项
        picked = None
        for item in reversed(data):
            if isinstance(item, (np.ndarray, list)):
                picked = item
                break
        if picked is None:
            picked = data[-1]
        if len(data) >= 2 and not isinstance(data[0], (np.ndarray, list)):
            extra.setdefault("seq", data[0])
        if len(data) >= 3 and isinstance(data[1], (int, float)):
            extra.setdefault("timestamp", data[1])
        data = picked

    try:
        arr = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return np.zeros((0, BRAINCO_N_CHANNELS), dtype=np.float32), extra

    if arr.size == 0:
        return np.zeros((0, BRAINCO_N_CHANNELS), dtype=np.float32), extra

    if arr.ndim == 1:
        for n_ch in BRAINCO_N_CH_CANDIDATES:
            if arr.size % n_ch == 0:
                arr = arr.reshape((-1, n_ch))
                break
        else:
            arr = arr.reshape((1, -1))
    elif arr.ndim == 2:
        known = set(BRAINCO_N_CH_CANDIDATES)
        if arr.shape[0] in known and arr.shape[1] not in known:
            arr = arr.T  # (n_ch, n) → (n, n_ch)
        elif (arr.shape[0] < arr.shape[1]
              and arr.shape[0] in known
              and arr.shape[1] not in known):
            arr = arr.T
    elif arr.ndim >= 3:
        arr = arr.reshape((-1, arr.shape[-1]))

    return np.ascontiguousarray(arr, dtype=np.float32), extra


def pull_eeg_buffer(sdk: Any) -> Any:
    """抽出 SDK 缓冲里当前所有 EEG 样本。失败返回 None。"""
    if not hasattr(sdk, "get_eeg_buffer"):
        return None
    try:
        return sdk.get_eeg_buffer(EEG_BUFFER_TAKE, True)
    except TypeError:
        try:
            return sdk.get_eeg_buffer()
        except Exception:
            return None
    except Exception:
        return None


def marker_times_to_latency(
    t_sent_pc: np.ndarray, fit: dict, fs: float,
) -> np.ndarray:
    """``t_pc = y0 + a * (t_eeg - x0) + b`` 的反函数 → 样本号。"""
    a = float(fit["slope_pc_per_eeg"])
    b = float(fit["intercept_s_at_first_marker"])
    x0 = float(fit["eeg_t0_s"])
    y0 = float(fit["pc_t0_s"])
    t = np.asarray(t_sent_pc, dtype=np.float64)
    t_eeg = x0 + (t - y0 - b) / a
    return np.rint(t_eeg * fs).astype(np.int64)


# =============================================================================
# Device discovery
# =============================================================================

async def discover_brainco_device(
    sdk: Any, *, host: str = "", port: int = 0,
    timeout_s: float = 10.0, device_sn: str = "",
) -> tuple[str, int]:
    """返回 ``(addr, port)``。``host`` 非空则跳过扫描。"""
    host = (host or "").strip()
    port = int(port or 0)
    if host:
        return host, port or _DEFAULT_TCP_PORT

    timeout_s = max(1.0, float(timeout_s))
    sn = (device_sn or "").strip() or None

    if hasattr(sdk, "scan_devices"):
        try:
            devices = await asyncio.wait_for(sdk.scan_devices(), timeout=timeout_s)
            if devices:
                d0 = devices[0]
                if isinstance(d0, (tuple, list)) and len(d0) >= 2:
                    return str(d0[0]), int(d0[1])
                if hasattr(d0, "addr"):
                    return str(d0.addr), int(d0.port)
        except (asyncio.TimeoutError, TypeError, RuntimeError):
            pass

    found: list[tuple[str, int]] = []

    def _on_mdns(result: Any) -> None:
        try:
            addr = str(result.addr)
            p = int(result.port)
        except Exception:
            return
        if sn and getattr(result, "sn", "") not in (sn, None, ""):
            return
        found.append((addr, p))

    if hasattr(sdk, "mdns_start_scan_multi"):
        maybe = sdk.mdns_start_scan_multi(_on_mdns)
        if asyncio.iscoroutine(maybe):
            await maybe
        deadline = time.time() + timeout_s
        while time.time() < deadline and not found:
            await asyncio.sleep(0.1)
        if hasattr(sdk, "mdns_stop_scan"):
            stop = sdk.mdns_stop_scan()
            if asyncio.iscoroutine(stop):
                await stop
        if found:
            return found[0]

    if hasattr(sdk, "mdns_start_scan"):
        try:
            maybe = sdk.mdns_start_scan(sn) if sn else sdk.mdns_start_scan()
            result = (await asyncio.wait_for(maybe, timeout=timeout_s)
                      if asyncio.iscoroutine(maybe) else maybe)
        except (asyncio.TimeoutError, TypeError, RuntimeError):
            result = None
        if result is not None:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return str(result[0]), int(result[1])
            if hasattr(result, "addr"):
                return str(result.addr), int(result.port)

    raise TimeoutError(
        f"未发现 BCIGo 设备(等了 {timeout_s:g}s)。"
        "确认帽子已开机并连上同一局域网,或在 recorders.yaml 填写 host/port。"
    )


def _sdk_enum(sdk: Any, cls_name: str, attr: str):
    cls = getattr(sdk, cls_name)
    return getattr(cls, attr)


# =============================================================================
# Recorder
# =============================================================================

class BrainCoEegRecorder(BaseEegRecorder):
    """Records EEG from a BrainCo BCIGo cap via ``bcigo_sdk``."""

    config: BraincoEegRecorderConfig
    _has_trigger_channel = False

    def __init__(self, config: BraincoEegRecorderConfig):
        super().__init__(config)
        self._amp_index = 0
        self._pkt_eeg_t: list[float] = []
        self._pkt_pc_t: list[float] = []
        self._last_sample = -1
        self._last_recv: float | None = None
        self._pkt_q: queue.SimpleQueue | None = None
        self._sdk_thread: threading.Thread | None = None
        self._sdk_stop = threading.Event()
        self._sdk_error = ""
        self._got_callback = False
        self._cb_logged = False
        self._rate_logged = False
        self._first_recv: float | None = None
        self._sync_sock: socket.socket | None = None
        self._n_ch = BRAINCO_N_CHANNELS
        self._dropped = 0

    # ------------------------------------------------------------------
    # SDK thread
    # ------------------------------------------------------------------

    def _sdk_thread_main(self) -> None:
        try:
            asyncio.run(self._sdk_async_main())
        except Exception as exc:
            self._sdk_error = f"{type(exc).__name__}: {exc}"
            self._log(f"[eeg:brainco] SDK 线程退出 — {self._sdk_error}",
                      level="ERROR")

    async def _sdk_async_main(self) -> None:
        try:
            import bcigo_sdk as sdk
        except ImportError as exc:
            self._sdk_error = "未安装 bcigo-sdk — pip install bcigo-sdk"
            raise RuntimeError(self._sdk_error) from exc

        fs_hz = int(round(float(self.config.sample_rate or 250)))
        fs_attr = _FS_ATTR.get(fs_hz, "SR_250Hz")
        gain_attr = _GAIN_ATTR.get(int(self.config.gain or 6), "GAIN_6")
        sig_attr = _SIGNAL_ATTR.get(
            str(self.config.signal or "normal").strip().lower(), "NORMAL")

        self._sample_rate = float(fs_hz)
        self._channel_labels = list(BRAINCO_CHANNEL_NAMES)
        self._n_ch = len(self._channel_labels)

        sdk.set_eeg_data_callback(self._on_sdk_eeg)
        # 原始包兜底:部分固件只走 received_data
        if hasattr(sdk, "set_received_data_callback"):
            sdk.set_received_data_callback(self._on_sdk_raw)

        addr, port = await discover_brainco_device(
            sdk,
            host=self.config.host,
            port=self.config.port,
            timeout_s=self.config.scan_timeout_s,
            device_sn=self.config.device_sn,
        )
        self._log(f"[eeg:brainco] 连接 {addr}:{port}  "
                  f"{self._n_ch} ch @ {self._sample_rate:g} Hz "
                  f"gain={gain_attr} signal={sig_attr}")

        client = sdk.BCIGoClient(addr, port)
        try:
            parser = sdk.MessageParser("bcigo", sdk.MsgType.BCIGo)
        except TypeError:
            parser = sdk.MessageParser()
        try:
            await client.start_stream(
                parser,
                fs=_sdk_enum(sdk, "EegSampleRate", fs_attr),
                gain=_sdk_enum(sdk, "EegSignalGain", gain_attr),
                signal=_sdk_enum(sdk, "EegSignalSource", sig_attr),
            )
        except TypeError:
            # 旧签名:位置参数
            await client.start_stream(parser)

        # start_stream 之后再显式切到 EEG 出数(有的固件停在阻抗/空闲,只握手不出波形)
        for name in ("start_eeg_stream", "enable_eeg_stream_mode"):
            fn = getattr(client, name, None)
            if fn is None:
                continue
            try:
                maybe = fn() if name == "start_eeg_stream" else fn(
                    fs=_sdk_enum(sdk, "EegSampleRate", fs_attr),
                    gain=_sdk_enum(sdk, "EegSignalGain", gain_attr),
                    signal=_sdk_enum(sdk, "EegSignalSource", sig_attr),
                )
                if asyncio.iscoroutine(maybe):
                    await maybe
                self._log(f"[eeg:brainco] {name}() ok", echo=False)
            except Exception as exc:
                self._log(f"[eeg:brainco] {name}() {exc}", echo=False)

        # 先给回调一点时间。回调一旦进数就不再抽缓冲,避免同一批被记两遍。
        # 回调若始终不来(上次实机如此),再按整批 take 抽 get_eeg_buffer。
        t_stream = time.time()
        callback_grace_s = 0.6
        while not self._sdk_stop.is_set():
            if not self._got_callback and (time.time() - t_stream) >= callback_grace_s:
                self._poll_sdk_buffer(sdk)
            await asyncio.sleep(0.008)

        try:
            if hasattr(sdk, "set_eeg_data_callback"):
                sdk.set_eeg_data_callback(None)
            if hasattr(sdk, "set_received_data_callback"):
                sdk.set_received_data_callback(None)
        except Exception:
            pass
        try:
            if hasattr(client, "disconnect_tcp_blocking"):
                client.disconnect_tcp_blocking()
            else:
                await client.disconnect_tcp()
        except Exception as exc:
            self._log(f"[eeg:brainco] disconnect: {exc}", level="WARNING")

    def _poll_sdk_buffer(self, sdk: Any) -> None:
        buf = pull_eeg_buffer(sdk)
        if buf is None:
            return
        self._enqueue_payload(buf, from_callback=False)

    def _on_sdk_eeg(self, *args, **kwargs) -> None:
        self._got_callback = True
        self._enqueue_payload(_payload_from_cb(args, kwargs), from_callback=True)

    def _on_sdk_raw(self, *args, **kwargs) -> None:
        if self._got_callback:
            return
        self._enqueue_payload(_payload_from_cb(args, kwargs), from_callback=False)

    def _enqueue_payload(self, payload: Any, *, from_callback: bool) -> None:
        t_recv = time.time()
        if self._pkt_q is None:
            return
        try:
            block, extra = coerce_eeg_payload(payload)
        except Exception:
            return
        if block.size == 0:
            return
        if not self._cb_logged:
            self._cb_logged = True
            src = "callback" if from_callback else "buffer"
            self._log(
                f"[eeg:brainco] 首包 via {src}: shape={block.shape} "
                f"extra={sorted(extra)} "
                f"min={float(np.min(block)):.3f} max={float(np.max(block)):.3f}",
                echo=False)
        # SDK 可能复用底层缓冲,必须拷走
        block = np.array(block, dtype=np.float32, copy=True)
        try:
            self._pkt_q.put_nowait((t_recv, block, extra))
        except Exception:
            self._dropped += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> bool:
        try:
            import bcigo_sdk as sdk  # noqa: F401
        except ImportError:
            self._open_error = "未安装 bcigo-sdk — pip install bcigo-sdk"
            self._log(f"[eeg:brainco] open failed — {self._open_error}")
            return False

        fs_hz = int(round(float(self.config.sample_rate or 250)))
        self._sample_rate = float(fs_hz)
        self._channel_labels = list(BRAINCO_CHANNEL_NAMES)
        self._n_ch = len(self._channel_labels)
        self._pkt_q = queue.SimpleQueue()
        self._sdk_stop.clear()
        self._sdk_error = ""
        self._got_callback = False
        self._cb_logged = False
        self._rate_logged = False
        self._first_recv = None
        self._sdk_thread = threading.Thread(
            target=self._sdk_thread_main, name="brainco-sdk", daemon=True)
        self._sdk_thread.start()

        self._open_sync_udp()

        t0 = time.time()
        timeout = float(self.config.open_timeout or 30.0)
        while self._total_samples == 0:
            if self._sdk_error:
                self._open_error = self._sdk_error
                self._log(f"[eeg:brainco] open failed — {self._open_error}")
                self._stop_sdk()
                return False
            if not self._sdk_thread.is_alive() and self._total_samples == 0:
                self._open_error = self._sdk_error or "SDK 线程在出数前退出"
                self._log(f"[eeg:brainco] open failed — {self._open_error}")
                self._stop_sdk()
                return False
            self._drain_packets(max_n=32)
            if self._total_samples > 0:
                break
            if time.time() - t0 > timeout:
                self._open_error = (f"no EEG block within {timeout:g}s"
                                    " — 帽子未开机/未入网/host 填错?")
                self._log(f"[eeg:brainco] open failed — {self._open_error}")
                self._stop_sdk()
                return False
            time.sleep(0.02)

        self._log(f"[eeg:brainco] 已出数: {self._n_ch} ch @ "
                  f"{self._sample_rate:g} Hz")
        return True

    def _open_sync_udp(self) -> None:
        port = int(getattr(self.config, "sync_udp_port", 0) or 0)
        if port <= 0:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            sock.close()
            self._log(f"[eeg:brainco] 软件触发口 127.0.0.1:{port} 绑定失败"
                      f" — {exc};仅用收尾离线映射", level="WARNING")
            return
        sock.settimeout(0.0)
        self._sync_sock = sock
        self._log(f"[eeg:brainco] 软件触发 UDP 127.0.0.1:{port}")

    def _stop_sdk(self) -> None:
        self._sdk_stop.set()
        th, self._sdk_thread = self._sdk_thread, None
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
        if self._sync_sock is not None:
            try:
                self._sync_sock.close()
            except OSError:
                pass
            self._sync_sock = None

    def _reset_stream_state(self) -> None:
        self._amp_index = 0
        self._pkt_eeg_t.clear()
        self._pkt_pc_t.clear()
        self._last_sample = -1
        self._last_recv = None
        self._first_recv = None
        self._rate_logged = False
        if self._pkt_q is not None:
            while True:
                try:
                    self._pkt_q.get_nowait()
                except queue.Empty:
                    break

    def _close(self) -> None:
        self._stop_sdk()
        n_ev = len(self._buf.get("eeg_event_code", []))
        self._log(f"[eeg:brainco] stopped (samples={self._total_samples}, "
                  f"pkts={len(self._pkt_eeg_t)}, live_events={n_ev}, "
                  f"dropped={self._dropped})")
        super()._close()

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def _poll(self, ts: float) -> None:
        self._drain_sync_udp()
        self._drain_packets(max_n=128)
        if self._sdk_error and self._sdk_thread is not None \
                and not self._sdk_thread.is_alive():
            self._log(f"[eeg:brainco] SDK 中途退出 — {self._sdk_error}",
                      level="ERROR")
            self._sdk_thread = None

    def _drain_packets(self, max_n: int = 64) -> None:
        if self._pkt_q is None:
            return
        n = 0
        while n < max_n:
            try:
                t_recv, block, extra = self._pkt_q.get_nowait()
            except queue.Empty:
                break
            self._ingest_packet(t_recv, block, extra)
            n += 1

    def _ingest_packet(self, t_recv: float, block: np.ndarray,
                       extra: dict) -> None:
        if block.ndim != 2 or block.shape[0] <= 0:
            return
        if block.shape[1] != self._n_ch:
            # 通道数与预设不符:以首包为准,避免硬崩
            if self._total_samples == 0:
                self._n_ch = int(block.shape[1])
                labels = list(BRAINCO_CHANNEL_NAMES)
                if self._n_ch == BRAINCO_N_CHANNELS + 1:
                    labels.append(BRAINCO_TRIGGER_NAME)
                elif self._n_ch > len(labels):
                    labels += [f"Ch{i + 1}" for i in range(len(labels), self._n_ch)]
                else:
                    labels = labels[:self._n_ch]
                self._channel_labels = labels
                self._log(f"[eeg:brainco] 通道数改为 {self._n_ch}",
                          level="WARNING")
            elif block.shape[1] > self._n_ch:
                block = block[:, :self._n_ch]
            else:
                pad = np.zeros((block.shape[0], self._n_ch), dtype=np.float32)
                pad[:, :block.shape[1]] = block
                block = pad

        n = int(block.shape[0])
        start = self._amp_index
        if self._first_recv is None:
            self._first_recv = float(t_recv)
        self._on_block(start, block)
        if (not self._rate_logged and self._first_recv is not None
                and self._total_samples >= 200):
            dt = max(time.time() - self._first_recv, 1e-3)
            hz = self._total_samples / dt
            self._rate_logged = True
            self._log(f"[eeg:brainco] 实测约 {hz:.1f} Hz "
                      f"(n={self._total_samples} in {dt:.2f}s, "
                      f"cfg={self._sample_rate:g})")
        last = start + n - 1
        fs = self._sample_rate or 1.0
        # LSL 惯例:最后一样本 ≈ 包到达时刻
        self._pkt_eeg_t.append(last / fs)
        self._pkt_pc_t.append(float(t_recv))
        self._last_sample = last
        self._last_recv = float(t_recv)
        self._amp_index = last + 1

        trig = extra.get("trigger", extra.get("trig", extra.get("ext_trig")))
        if trig is not None:
            self._maybe_hw_trigger(trig, start, n)

    def _maybe_hw_trigger(self, trig: Any, start: int, n: int) -> None:
        """设备若带外部触发状态,边沿写入事件(软件同步仍是主路径)。"""
        try:
            arr = np.asarray(trig).reshape(-1)
        except (TypeError, ValueError):
            try:
                code = int(trig)
            except (TypeError, ValueError):
                return
            if code:
                self._on_event(code, start)
            return
        if arr.size == 1:
            code = int(arr.flat[0])
            if code:
                self._on_event(code, start)
            return
        prev = 0
        limit = min(int(arr.size), n)
        for i in range(limit):
            code = int(arr[i])
            if code and code != prev:
                self._on_event(code, start + i)
            prev = code

    def _drain_sync_udp(self) -> None:
        """在线软件 TTL:用最近一包把 t_sent_pc 插到样本号(心跳用)。"""
        sock = self._sync_sock
        if sock is None:
            return
        fs = self._sample_rate or 1.0
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError, OSError):
                return
            evt = _parse_marker_pkt(data)
            if evt is None or self._last_recv is None or self._last_sample < 0:
                continue
            t_sent = evt["t_sent_pc"]
            if t_sent is None:
                t_sent = time.time()
            latency = int(round(
                self._last_sample + (float(t_sent) - self._last_recv) * fs))
            latency = max(0, min(latency, max(self._amp_index - 1, 0)))
            self._on_event(int(evt["code"]), latency)

    # ------------------------------------------------------------------
    # Alignment — 包时钟为主,marker 映射为事件
    # ------------------------------------------------------------------

    def _align(self) -> None:
        try:
            self._fit = self._fit_from_packet_clock()
        except Exception as exc:
            self._fit = {"fitted": False,
                         "reason": f"align error: {type(exc).__name__}: {exc}"}
        if not self._fit.get("fitted"):
            # 包时钟不够时,退回「在线软件 TTL 事件 ↔ marker」的 Curry 路径
            self._log(f"[eeg:brainco] 包时钟拟合失败 — "
                      f"{self._fit.get('reason', '?')};尝试 marker 配对")
            super()._align()
            if self._fit and self._fit.get("fitted"):
                self._fit["sync_method"] = "software_live_udp"
            return

        f = self._fit
        self._log(
            f"[eeg:brainco] 包时钟: slope={f['slope_pc_per_eeg']:.7f} "
            f"resid_rms={f['resid_rms_ms']:.2f}ms "
            f"resid_max={f['resid_max_ms']:.2f}ms "
            f"n_pkts={f['n']} (inliers={f['n_inliers']})", echo=False)

        markers = self._load_markers()
        n_mapped = self._inject_software_events(markers)
        self._fit["sync_method"] = "software_packet_clock"
        self._fit["n_software_events"] = n_mapped
        if n_mapped < 2:
            self._log("[eeg:brainco] marker 不足 2 个,时间戳仍按包时钟写出;"
                      "刺激对齐请确认 UDP marker 在跑")

    def _fit_from_packet_clock(self) -> dict:
        if self._total_samples == 0:
            return {"fitted": False, "reason": "no EEG samples recorded"}
        if len(self._pkt_eeg_t) < 2:
            return {"fitted": False, "reason": "fewer than two EEG packets"}
        eeg_t = np.asarray(self._pkt_eeg_t, dtype=np.float64)
        pc_t = np.asarray(self._pkt_pc_t, dtype=np.float64)
        fit = _fit_eeg_to_pc(eeg_t, pc_t)
        if fit.get("fitted"):
            fs = self._sample_rate or 1.0
            a = fit["slope_pc_per_eeg"]
            b = fit["intercept_s_at_first_marker"]
            start = self._buf["eeg_block_start"][0]
            t_eeg = (start + np.arange(self._total_samples)) / fs
            self._timestamps_pc = (
                fit["pc_t0_s"] + a * (t_eeg - fit["eeg_t0_s"]) + b)
        return fit

    def _inject_software_events(self, markers: dict | None) -> int:
        """按包时钟把 marker 的 t_sent_pc 映射成 EEG 样本号。

        在线 UDP 插值事件先清掉,避免与离线映射混在一起(码会重复,配对拒绝)。
        """
        for key in ("eeg_event_code", "eeg_event_latency"):
            self._buf.pop(key, None)
        if markers is None:
            return 0
        m_code = markers.get("marker_code")
        m_t_pc = (markers.get("marker_t_sent_pc")
                  if markers.get("marker_t_sent_pc") is not None
                  else markers.get("marker_t_local_recv"))
        if m_code is None or m_t_pc is None or len(m_code) == 0:
            return 0
        if not self._fit or not self._fit.get("fitted"):
            return 0
        fs = self._sample_rate or 1.0
        lat = marker_times_to_latency(
            np.asarray(m_t_pc, dtype=np.float64), self._fit, fs)
        start = int(self._buf["eeg_block_start"][0]
                    if self._buf.get("eeg_block_start") else 0)
        lo, hi = start, start + max(self._total_samples - 1, 0)
        n = 0
        for code, latency in zip(np.asarray(m_code).ravel(), lat):
            sample = int(np.clip(int(latency), lo, hi))
            self._on_event(int(code), sample)
            n += 1
        if n and self._timestamps_pc is not None:
            # marker 落在拟合时间轴上的残差 = 软件同步质量
            idx = np.clip(lat - start, 0, self._total_samples - 1)
            pred = self._timestamps_pc[idx]
            resid_ms = (pred - np.asarray(m_t_pc, dtype=np.float64)) * 1000.0
            self._fit["marker_resid_rms_ms"] = float(
                np.sqrt(np.mean(resid_ms ** 2)))
            self._fit["marker_resid_max_ms"] = float(np.max(np.abs(resid_ms)))
            self._log(
                f"[eeg:brainco] 软件事件 {n} 个; "
                f"marker↔包时钟 resid_rms="
                f"{self._fit['marker_resid_rms_ms']:.2f}ms "
                f"resid_max={self._fit['marker_resid_max_ms']:.2f}ms")
        return n

    def _build_output(self) -> dict[str, np.ndarray]:
        out = super()._build_output()
        method = "software_packet_clock"
        if self._fit and self._fit.get("sync_method"):
            method = str(self._fit["sync_method"])
        out["eeg_sync_method"] = np.asarray(method)
        out["eeg_pkt_n"] = np.asarray(len(self._pkt_eeg_t))
        if self._fit:
            for key in ("n_software_events", "marker_resid_rms_ms",
                        "marker_resid_max_ms"):
                if key in self._fit:
                    out[f"eeg_fit_{key}"] = np.asarray(self._fit[key])
        return out


def _parse_marker_pkt(data: bytes) -> dict | None:
    """与 UdpMarkerRecorder 同一套 EVT| 文本。"""
    try:
        text = data.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if not text.startswith("EVT|"):
        return None
    fields: dict[str, str] = {}
    for tok in text.split("|")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        fields[k.strip()] = v.strip()
    try:
        sent = fields.get("t_sent_pc")
        return {
            "code": int(fields.get("code", -1)),
            "t_sent_pc": float(sent) if sent else None,
        }
    except ValueError:
        return None
