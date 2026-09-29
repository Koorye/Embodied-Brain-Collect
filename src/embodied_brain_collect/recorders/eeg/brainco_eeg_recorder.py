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

开录阻抗门禁(``impedance_check``,默认开;参考 CurryEegRecorder 的主动
触发版):``_open`` 阶段经 SDK ``enable_impedance_detection_mode`` 触发一
次 leadoff 阻抗检测(SDK 内部对激励波形做正弦拟合算阻抗,并逐 chip 轮询,
4 chip x 8 通道 = 32 路),``set_imp_data_callback`` 收满若干窗口后每通
道取末值做通过率检查,不过则拒绝 open 并提示超标数量(通道名与固件
槽位的映射未核实,不点名);结果与 Curry 同
schema 随 npz
落盘(``eeg_impedance_*``),打包器透传成 ``observation.eeg_impedance``。
收尾必发 ``disable_impedance_detection_mode`` —— SDK 自己会重启 EEG 流,
没有 Curry 那条"恢复前断开打坏驱动"的时序红线,但 ``_open`` 仍等读数恢
复才返回。回调的官方签名未公开,``coerce_imp_payload`` 对 tuple/dict/
属性对象做尽力解析;阻抗值 SDK 原始单位按 kΩ 处理(与上位机同量级;
若实机末值对照上位机差 1000 倍,改 ``IMP_VALUE_TO_KOHM`` 一处),经它
换算成 kΩ 后再做阈值比较、展示与落盘(``eeg_impedance_kohm``)。
"""

from __future__ import annotations

import asyncio
import queue
import re
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

# ---- 开录阻抗门禁(参考 CurryEegRecorder 的常量区)----
# BCIGo 的 32 路电极由 4 个 leadoff 芯片各管 8 路:chip N 覆盖
# BRAINCO_CHANNEL_NAMES 的 [8N, 8N+8)(顺序映射为实机默认假设,实机
# 首跑时对照上位机的阻抗表核一遍)。SDK 收到 leadoff 波形后自己做正弦
# 拟合(pyd 日志 "impedance waveform fit"),算出的阻抗值经
# set_imp_data_callback 推给 Python —— 实机一窗推 chip 号 + 32 通道值。
# 数值单位按 kΩ(IMP_VALUE_TO_KOHM = 1):SDK 原值与 BCIGo 上位机读数
# 同量级(上位机"个位数~几千",原值 313~4869)。读数本身另有一个坑:
# 2026-09-29 实机,旧门禁只收每通道前两窗做均值,连续两轮都钉在
# 313~4869 窄带、不随电极整理变化(同期上位机已是"几路绿 + 大片红")——
# 收集太短,读到的是固件上一轮阻抗会话的陈值/未收敛值。现改法:多读
# 几窗、每通道取末值(最近一窗),收尾日志并排打 raw 与 kΩ;若实机对照
# 上位机差 1000 倍(上位机 14.9 而原值≈14900),把 IMP_VALUE_TO_KOHM
# 改 1e-3 一处即可。阈值(impedance_max_kohm)、展示、落盘统一用换算后
# 的 kΩ。
IMP_VALUE_TO_KOHM = 1
BRAINCO_CH_PER_CHIP = 8
BRAINCO_N_CHIPS = BRAINCO_N_CHANNELS // BRAINCO_CH_PER_CHIP
# leadoff 激励参数走 config(impedance_freq / impedance_current),这里
# 只作 config 为空时的兜底。BCIGo 上位机的激励配置未知 —— 读数与它对
# 不上时先换激励试(如 Cur6uA);SDK 未公开 schema/单位,外部参考实现
# 也只透传原始载荷不做解释,所以首回调 repr 与 SDK 版本都落日志留证。
_IMP_FREQ_ATTR = "Ac31p2hz"    # leadoff 激励频率(SDK 默认 31.2Hz AC)
_IMP_CURRENT_ATTR = "Cur6nA"   # 激励电流(SDK 默认;干电极高阻可试 Cur6uA)
# 收集节奏对齐 BCIGo 上位机并加长(2026-09-29 实机:每通道 3 窗/10s 仍
# 与上位机有差,读数收敛慢):固件每窗只推若干通道,轮询到"所有通道都
# 读到"后继续多轮读数,每通道 ≥5 窗且收集 ≥20s 才提前收工,聚合只取每
# 通道末值(最近一窗);上限 45s,到点没收齐按失败算。
_IMPEDANCE_SNAPSHOTS = 4        # 首轮全覆盖 + 多轮稳定(对齐上位机节奏)
_IMPEDANCE_MIN_WINDOW_S = 25.0  # 最短收集时长:给陈值/慢收敛留足刷新时间
_IMPEDANCE_WINDOW_S = 40.0      # 收集窗口上限
_IMPEDANCE_CMD_TIMEOUT_S = 10.0  # enable/disable 命令的等待上限


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


def _chip_index(chip: Any) -> int | None:
    """``LeadOffChip`` 枚举 / int / 字符串 → 0 基 chip 序号(0..3)。

    枚举优先看 ``name``("Chip2" → 1);裸整数按枚举判别值处理(NONE=0,
    Chip1..Chip4 = 1..4,1 基)。认不出的返回 None。
    """
    if isinstance(chip, bool):
        return None
    name = getattr(chip, "name", "")
    if isinstance(name, str):
        m = re.search(r"(?i)chip\s*(\d+)", name)
        if m:
            idx = int(m.group(1)) - 1
            return idx if 0 <= idx < BRAINCO_N_CHIPS else None
    v: Any = None
    if isinstance(chip, int):
        v = chip
    else:
        for attr in ("value", "int_value"):
            cand = getattr(chip, attr, None)
            if isinstance(cand, int) and not isinstance(cand, bool):
                v = cand
                break
    if not isinstance(v, int):
        m = re.search(r"(?i)chip\s*(\d+)", str(chip))
        if m:
            idx = int(m.group(1)) - 1
            return idx if 0 <= idx < BRAINCO_N_CHIPS else None
        return None
    if 1 <= v <= BRAINCO_N_CHIPS:
        return v - 1
    return None   # 0 = LeadOffChip.NONE 等无效值


def coerce_imp_payload(payload: Any) -> tuple[int | None, np.ndarray] | None:
    """把 ``set_imp_data_callback`` 的多种形态收成 ``(chip_idx, 值向量)``。

    官方未公开回调签名。实机固件推 ``(chip, 32 值整帽窗口)`` —— 每个
    chip 一窗,只填自己那 8 路、其余补 0;也有每窗只推 8 值的形态。
    两种都收:8 值形态必须带有效 chip 才能落位;32 值形态 chip 解释
    不了也收(chip 记 None,靠 0 值判"该窗未报")。解析不出或长度
    对不上返回 None,调用方记一次 repr 便于实机对格式。
    """
    chip: Any = None
    vals: Any = None
    if isinstance(payload, dict):
        chip = payload.get("chip", payload.get("chip_id",
                                               payload.get("lead_off_chip")))
        for k in ("values", "imp", "impedance", "impedances",
                  "imp_data", "data"):
            if payload.get(k) is not None:
                vals = payload[k]
                break
    elif isinstance(payload, (tuple, list)) and payload:
        if len(payload) == 2:
            chip, vals = payload
        elif len(payload) == BRAINCO_CH_PER_CHIP + 1:
            # (chip, v0..v7) 铺平形态
            chip, vals = payload[0], payload[1:]
        else:
            # (seq, ts, values) 等变长形态:取最像数值列表的一项,
            # 首个裸整数当 chip/序号
            for item in payload:
                if isinstance(item, (list, np.ndarray)):
                    vals = item
                    break
            for item in payload:
                if isinstance(item, int) and not isinstance(item, bool):
                    chip = item
                    break
    elif payload is not None and not isinstance(
            payload, (np.ndarray, bytes, bytearray)):
        chip = getattr(payload, "chip", getattr(payload, "chip_id", None))
        for k in ("values", "imp", "impedance", "impedances",
                  "imp_data", "data"):
            v = getattr(payload, k, None)
            if v is not None:
                vals = v
                break

    idx = _chip_index(chip) if chip is not None else None
    if vals is None:
        return None
    try:
        arr = np.asarray(vals, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == BRAINCO_N_CHANNELS:
        return idx, arr   # 整帽窗口:chip 缺失也收(落位不靠 chip)
    if arr.size == BRAINCO_CH_PER_CHIP and idx is not None:
        return idx, arr   # 经典 per-chip 窗口
    return None


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
        # open 时阻抗门禁的结果(进 npz + 决定 open 成败);与 Curry 同 schema
        self._imp_windows: list[tuple[int | None, np.ndarray]] = []
        self._imp_cb_logged = False
        self._imp_raw_logged = False
        self._imp_logged_chips: set[int | None] = set()
        self._impedance_kohm: np.ndarray | None = None
        self._impedance_checked: np.ndarray | None = None   # bool per channel
        self._impedance_pass_rate: float = 0.0
        self._impedance_pass: bool = False
        self._impedance_n: int = 0
        self._impedance_error: str = ""   # 非空 = 门禁拒绝开录的原因文案
        # 门禁完成信号:_open 据此判断"样本已到"是否发生在门禁结束后
        # (start_stream 到 enable 阻抗之间会有预热样本,不能只看样本数)
        self._imp_gate_done = threading.Event()

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
                  f"gain={gain_attr} signal={sig_attr} "
                  f"sdk={getattr(sdk, '__version__', '?')}")

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

        # 开录阻抗门禁(参考 Curry):enable 阻抗模式会停 EEG 流,disable
        # 再由 SDK 恢复 —— 门禁结束时 EEG 才真正出数,结果决定 open 成败。
        if self.config.impedance_check:
            try:
                detail = await self._impedance_gate(sdk, client)
            finally:
                self._imp_gate_done.set()
            self._log(f"[eeg:brainco] {detail}",
                      level="ERROR" if self._impedance_error else "INFO")

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
            if hasattr(sdk, "set_imp_data_callback"):
                sdk.set_imp_data_callback(None)
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
    # Impedance gate (open-time, ACTIVE trigger — 参照 CurryEegRecorder)
    # ------------------------------------------------------------------

    def _on_sdk_imp(self, *args, **kwargs) -> None:
        """``set_imp_data_callback``:实机固件每窗推 chip 号 + 32 通道值
        (本 chip 没覆盖的通道填 0)。从 SDK 线程调;append 在 GIL 下原子,
        读方只做快照。"""
        payload = _payload_from_cb(args, kwargs)
        if not self._imp_raw_logged:
            # schema/单位 SDK 未公开(外部参考实现同样只透传原始载荷):
            # 首回调原样落日志,给实机对格式/单位留证据
            self._imp_raw_logged = True
            self._log(f"[eeg:brainco] imp 原始载荷: {payload!r:.200}",
                      echo=False)
        parsed = coerce_imp_payload(payload)
        if parsed is None:
            if not self._imp_cb_logged:
                self._imp_cb_logged = True
                self._log(f"[eeg:brainco] imp 回调格式未识别 — "
                          f"repr={payload!r:.200}", level="WARNING")
            return
        chip, vals = parsed
        if chip not in self._imp_logged_chips:
            # 每个 chip 号各记首窗:轮询是否在跑、各片读数一眼可见
            self._imp_logged_chips.add(chip)
            where = f"chip={chip + 1}" if chip is not None else "chip=?"
            self._log(f"[eeg:brainco] imp 首窗 {where}: "
                      f"kΩ≈{np.round(vals * IMP_VALUE_TO_KOHM, 1).tolist()}",
                      echo=False)
        self._imp_windows.append((chip, vals))

    def _imp_channel_counts(self) -> np.ndarray:
        """每通道已收到的有效读数窗数。0 视为该窗未报:整帽窗口里别的
        chip 没覆盖的通道、per-chip 窗口里拟合失败的通道都填 0,而真实
        阻抗不可能是精确 0(那等于短路)。"""
        counts = np.zeros(BRAINCO_N_CHANNELS, dtype=np.int64)
        for chip, vals in self._imp_windows:
            if vals.size >= BRAINCO_N_CHANNELS:
                counts += (vals[:BRAINCO_N_CHANNELS] != 0.0)
            elif chip is not None and vals.size == BRAINCO_CH_PER_CHIP:
                sl = slice(chip * BRAINCO_CH_PER_CHIP,
                           (chip + 1) * BRAINCO_CH_PER_CHIP)
                counts[sl] += (vals != 0.0)
        return counts

    def _imp_last_raw(self) -> np.ndarray:
        """每通道取最近一次有效窗口的原始值(SDK 原单位)。0 视为该窗未
        报,不覆盖已有值;固件 4 chip 轮询,各通道末窗时刻天然不对齐。"""
        last = np.zeros(BRAINCO_N_CHANNELS, dtype=np.float64)
        for chip, vals in self._imp_windows:
            if vals.size >= BRAINCO_N_CHANNELS:
                v = vals[:BRAINCO_N_CHANNELS]
                nz = v != 0.0
                last[nz] = v[nz]
            elif chip is not None and vals.size == BRAINCO_CH_PER_CHIP:
                sl = slice(chip * BRAINCO_CH_PER_CHIP,
                           (chip + 1) * BRAINCO_CH_PER_CHIP)
                nz = vals != 0.0
                last[sl][nz] = vals[nz]
        return last

    def _impedance_last(
            self) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """收拢窗口 → ``(32 通道末值 kΩ, 原值, reason)``。

        每通道取最近一窗而非均值:上位机同款节奏 —— 固件每窗只推若干
        通道,首轮全覆盖后读数还要再续约 2 轮才稳,会话早窗是陈值/未收
        敛值(实机:读数一度不随电极整理变化),平均会把旧状态掺进结果。
        按通道计覆盖:全部通道都有有效读数才出值,缺的点名;完全没有数
        据时给排查文案。原值随 kΩ 一起返回,供日志并排核对单位。"""
        if not self._imp_windows:
            return None, None, (
                "未收到阻抗数据 — 确认帽子固件支持 leadoff 阻抗"
                "检测(可在 BCIGo 上位机手动做一次阻抗后重试),"
                "或 impedance_check: false 跳过门禁")
        counts = self._imp_channel_counts().astype(np.float64)
        missing = np.flatnonzero(counts == 0.0)
        if missing.size:
            # 通道名与固件槽位的映射未核实,文案不点名只报数量
            return None, None, (
                f"阻抗数据不完整:{missing.size}/"
                f"{BRAINCO_N_CHANNELS} 通道无有效读数 — 检查电极佩戴/"
                "导联,反复出现说明固件阻抗检测覆盖不全,可 "
                "impedance_check: false 跳过门禁并人工核查")
        raw = self._imp_last_raw()
        z = raw * IMP_VALUE_TO_KOHM   # kΩ,与阈值/日志/落盘同单位
        self._impedance_n = len(self._imp_windows)
        return z, raw, ""

    def _evaluate_impedance(self, z: np.ndarray) -> tuple[bool, str]:
        """与 Curry 同口径:末值(kΩ)vs 阈值(kΩ),边缘通道豁免,通过
        率门槛。顺带把 ``self._impedance_kohm / _checked / _pass_rate /
        _pass`` 填好(随 npz 落盘)。"""
        cfg = self.config
        thr = cfg.impedance_max_kohm
        edge = {c.strip().upper() for c in cfg.impedance_edge_channels}
        labels = [lab.strip() for lab in self._channel_labels]
        checked = np.asarray([lab.strip().upper() not in edge
                              and lab.strip().upper() != "TRIG"
                              for lab in labels])
        checked_idx = np.flatnonzero(checked)
        if checked_idx.size == 0:
            return False, ("阻抗检查配置错误:参与统计的通道数为 0"
                           "(检查 impedance_edge_channels 是否覆盖了全部通道)")
        fail_idx = checked_idx[z[checked_idx] >= thr]
        rate = 1.0 - fail_idx.size / checked_idx.size
        self._impedance_kohm = z
        self._impedance_checked = checked
        self._impedance_pass_rate = float(rate)
        # 通道名与固件槽位的映射未核实(实测恒定 MΩ 级的槽位已按别名
        # 豁免):文案只报数量不点名 —— 点名会误导操作员整理错误的电极。
        need = cfg.impedance_pass_rate
        if rate < need:
            self._impedance_pass = False
            return False, (
                f"阻抗检查未通过: {fail_idx.size}/{checked_idx.size} 通道 "
                f"≥ {cfg.impedance_max_kohm:g}kΩ,通过率 {rate:.1%} < "
                f"{need:.0%}(边缘通道已豁免)— 请整理/浸湿电极后重试")
        self._impedance_pass = True
        return True, (f"impedance ok: {checked_idx.size - fail_idx.size}/"
                      f"{checked_idx.size} 通道 < "
                      f"{cfg.impedance_max_kohm:g}kΩ "
                      f"(通过率 {rate:.1%} ≥ {need:.0%})")

    async def _impedance_gate(self, sdk: Any, client: Any) -> str:
        """主动阻抗门禁:enable → 收窗口 → 评估 →(finally)disable。

        disable 一定发 —— SDK 会顺带 start_eeg_stream 恢复读数,绝不把
        帽子留在阻抗模式;也没有 Curry 那条"恢复前断开打坏驱动"的红线
        (模式切换全部由 SDK 命令兜底)。评估在 disable 之前完成,
        ``_impedance_error`` 先于 EEG 恢复就位,``_open`` 的等待循环据此
        快速失败。返回给操作员看的文案;拒绝开录时同时存进
        ``self._impedance_error``。"""
        self._imp_windows = []
        self._imp_cb_logged = False
        self._imp_raw_logged = False
        self._imp_logged_chips = set()
        self._impedance_error = ""
        registered = False
        try:
            if hasattr(sdk, "set_imp_data_callback"):
                sdk.set_imp_data_callback(self._on_sdk_imp)
                registered = True
            if not hasattr(client, "enable_impedance_detection_mode"):
                raise RuntimeError(
                    "bcigo-sdk 没有 enable_impedance_detection_mode — "
                    "SDK 版本过旧,升级 SDK 或 impedance_check: false 跳过门禁")
            freq_attr = (str(getattr(self.config, "impedance_freq", "")
                             or "").strip() or _IMP_FREQ_ATTR)
            cur_attr = (str(getattr(self.config, "impedance_current", "")
                            or "").strip() or _IMP_CURRENT_ATTR)
            try:
                # 签名只有 (loop_check, freq, current),没有 chip 参数 ——
                # SDK 内部自己注册多 chip 轮询。传 chip= 会 TypeError 落到
                # 下面的无参兜底,loop_check 变 False:固件只 sweep chip1,
                # 实机就只剩 FP1..C3 那 8 路有读数(踩过的坑)。外部参考
                # 实现裸调(SDK 全默认)—— 学不得,默认 loop_check 只
                # sweep 单 chip;激励参数走 config 才是可实验的路线。
                maybe = client.enable_impedance_detection_mode(
                    loop_check=True,
                    freq=_sdk_enum(sdk, "LeadOffFreq", freq_attr),
                    current=_sdk_enum(sdk, "LeadOffCurrent", cur_attr),
                )
            except TypeError:
                # 旧签名:无参(SDK 默认值)
                maybe = client.enable_impedance_detection_mode()
                freq_attr = cur_attr = "?"
            if asyncio.iscoroutine(maybe):
                await asyncio.wait_for(maybe,
                                       timeout=_IMPEDANCE_CMD_TIMEOUT_S)
            self._log(f"[eeg:brainco] 阻抗检测开始 loop_check=True "
                      f"freq={freq_attr} current={cur_attr}", echo=False)
            # 窗口由 SDK 线程推;这里等"最短收集时长 + 每通道收够"两条都
            # 满足才提前收工,到点没收齐交给 _impedance_last 报缺哪几路
            t_collect = time.time()
            deadline = t_collect + _IMPEDANCE_WINDOW_S
            while time.time() < deadline:
                if (time.time() - t_collect >= _IMPEDANCE_MIN_WINDOW_S
                        and np.all(self._imp_channel_counts()
                                   >= _IMPEDANCE_SNAPSHOTS)):
                    break
                await asyncio.sleep(0.05)
            z, raw, reason = self._impedance_last()
            if z is None:
                self._impedance_error = reason
                return reason
            # raw 与 kΩ 并排落日志,带窗数/时长:实机对照上位机绿色通道
            # 核单位,拿窗数/时长判断收敛是否足够
            self._log(f"[eeg:brainco] 阻抗末值({len(self._imp_windows)}窗/"
                      f"{time.time() - t_collect:.0f}s,每通道取最近一窗): "
                      f"raw≈{np.round(raw, 0).astype(np.int64).tolist()} → "
                      f"kΩ≈{np.round(z, 1).tolist()}", echo=False)
            ok, detail = self._evaluate_impedance(z)
            if not ok:
                self._impedance_error = detail
            return detail
        except Exception as exc:
            self._impedance_error = (f"阻抗检测失败: {type(exc).__name__}: "
                                     f"{exc}")
            return self._impedance_error
        finally:
            try:
                maybe = client.disable_impedance_detection_mode()
                if asyncio.iscoroutine(maybe):
                    await asyncio.wait_for(maybe,
                                           timeout=_IMPEDANCE_CMD_TIMEOUT_S)
            except Exception as exc:
                # 失败不覆盖主失败文案,但要留痕:帽子可能仍停在阻抗模式
                self._log(f"[eeg:brainco] disable_impedance 失败: "
                          f"{type(exc).__name__}: {exc} — 确认上位机能否"
                          f"恢复 EEG 流,必要时重连帽子", level="WARNING")
                if not self._impedance_error:
                    self._impedance_error = (
                        "阻抗结束命令失败 — 确认 BCIGo 上位机能否恢复 "
                        "EEG 流,必要时重连帽子")
            if registered:
                try:
                    sdk.set_imp_data_callback(None)
                except Exception:
                    pass

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

        # 开录阻抗门禁:门禁出结论前不算 open 成功 —— start_stream 到
        # enable 阻抗之间会有预热样本,只看样本数会在门禁评估前放行
        gate = self._imp_gate_done if self.config.impedance_check else None
        if gate is not None:
            gate.clear()

        t0 = time.time()
        timeout = float(self.config.open_timeout or 30.0)
        while (self._total_samples == 0
               or (gate is not None and not gate.is_set())):
            if self._impedance_error:
                self._open_error = self._impedance_error
                self._log(f"[eeg:brainco] open failed — {self._open_error}")
                self._stop_sdk()
                return False
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
            if self._total_samples > 0 and (gate is None or gate.is_set()):
                break
            if time.time() - t0 > timeout:
                if gate is None or gate.is_set():
                    self._open_error = (f"no EEG block within {timeout:g}s"
                                        " — 帽子未开机/未入网/host 填错?")
                else:
                    self._open_error = (
                        f"阻抗检测未在 {timeout:g}s 内完成 — 帽子未开机/"
                        "未入网/host 填错,或固件不支持阻抗检测")
                self._log(f"[eeg:brainco] open failed — {self._open_error}")
                self._stop_sdk()
                return False
            time.sleep(0.02)

        if self._impedance_error:
            # 循环顶部的兜底:门禁拒绝但 disable 已恢复 EEG 时,样本会
            # 先到 —— 有错误文案就不能放行
            self._open_error = self._impedance_error
            self._log(f"[eeg:brainco] open failed — {self._open_error}")
            self._stop_sdk()
            return False

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
            raise RuntimeError(
                f"BrainCo SDK 线程中途退出 — {self._sdk_error}")

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
        if self._impedance_kohm is not None:
            # 通道顺序/命名与 eeg_channel_names 一致;33 路固件在门禁后才
            # 确定通道数时,给末位 TRIG 槽位补零并标为不检查(同 Curry)
            n = len(self._channel_labels)
            kohm = np.asarray(self._impedance_kohm,
                              dtype=np.float32).ravel()
            checked = np.asarray(self._impedance_checked,
                                 dtype=bool).ravel()
            if kohm.size < n:
                kohm = np.concatenate(
                    [kohm, np.zeros(n - kohm.size, dtype=np.float32)])
                checked = np.concatenate(
                    [checked, np.zeros(n - checked.size, dtype=bool)])
            # checked=False 的是边缘豁免通道(与 Trigger 状态字槽位)
            out["eeg_impedance_kohm"] = kohm[:n]   # kΩ,与阈值同单位
            out["eeg_impedance_checked"] = checked[:n]
            out["eeg_impedance_pass_rate"] = np.asarray(
                self._impedance_pass_rate)
            out["eeg_impedance_check_pass"] = np.asarray(
                self._impedance_pass)
            out["eeg_impedance_n_snapshots"] = np.asarray(self._impedance_n)
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
