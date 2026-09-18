"""MarkerSender — ParallelBox TTL ×N + UDP marker emitter for stim scripts.

Replaces E-Prime's InLine marker code in pure Python.  Sends the same byte to:
  - one or more ParallelBoxes over serial COM (EEG trigger, hardware path —
    e.g. one box wired to one amplifier's TTL port, another to a second
    amplifier; every amplifier then gets its own copy of the codes for its
    own clock fit)
  - UDP to sync_hub / UdpMarkerRecorder (software path)

The UDP packet uses the wire format understood by sync_hub::

    EVT|trial=<int>|tag=<NAME>|code=<int>|t_eprime_ms=<int>|t_sent_pc=<float>

``t_sent_pc`` is the SENDER's wall-clock stamp — the moment the marker went
out.  Receivers timestamp by it rather than by their own arrival time, which
removes UDP queuing and receiver-scheduling jitter from the marker timeline
(and therefore from the EEG alignment fit).  ``t_eprime_ms`` stays for the
E-Prime-relative clock, which is not comparable to anything else.

Multiple COM ports: pass ``port`` as a comma-separated string
(``"COM5,COM14"``) or a list (``["COM5", "COM14"]``).  A port that fails to
open or write is dropped with a loud warning — the remaining boxes keep
working; only when NOTHING can be opened does construction raise.
"""

from __future__ import annotations

import socket
import time
from contextlib import contextmanager

try:
    import serial  # pyserial
except Exception:  # pragma: no cover
    serial = None  # type: ignore


def ports_from_spec(spec) -> list[str]:
    """COM 配置 -> 去重后的端口列表。

    接受单个口(``"COM5"``)、逗号/分号/空白分隔串(``"COM5, COM14"``)或
    列表;顺序保持,重复剔除,空项丢弃。
    """
    if spec is None:
        return []
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    out: list[str] = []
    for item in items:
        for part in str(item).replace(";", ",").split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


class MarkerSender:
    """Send markers to N ParallelBoxes (hardware) and sync_hub (UDP)."""

    def __init__(
        self,
        port: str | list[str] = "",   # 空 = 没配串口;serial 开着会在构造时报错
        baud: int = 115200,
        udp_host: str = "127.0.0.1",
        udp_port: int = 9999,
        hold_s: float = 0.020,
        pre_clear_s: float = 0.5,
        enable_serial: bool = True,
        enable_udp: bool = True,
        verbose: bool = False,
    ) -> None:
        self.port = port                     # 原始配置(兼容旧引用)
        self.ports = ports_from_spec(port)   # 规范化后的端口列表
        self.baud = baud
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.hold_s = hold_s
        self.enable_serial = enable_serial and serial is not None
        self.enable_udp = enable_udp
        self.verbose = verbose

        self._sers: list = []        # 打开成功的串口(与 self.ports 子集对应)
        self._sock = None
        self._trial = 0
        self._t0_perf = time.perf_counter()

        if self.enable_serial:
            if not self.ports:
                raise ValueError(
                    "enable_serial=True 但没有配置任何 COM 口 — "
                    'port 传 "COM5" / "COM5,COM14" / ["COM5", "COM14"]')
            errors: list[str] = []
            for p in self.ports:
                try:
                    ser = serial.Serial(p, self.baud, timeout=0.5)  # type: ignore[union-attr]
                    ser.write(bytes([0]))
                    ser.flush()
                    self._sers.append(ser)
                except Exception as exc:  # noqa: BLE001 - 单口失败不拖垮其余
                    errors.append(f"{p}: {exc}")
            if errors and self._sers:
                print(f"[marker] ⚠ {len(errors)} 个 TTL 串口打不开,继续用 "
                      f"{[s.port for s in self._sers]} — {'; '.join(errors)}")
            elif errors:            # 一个都没开成:与旧版单口行为一致,硬失败
                raise serial.SerialException(  # type: ignore[union-attr]
                    f"TTL 串口全部打开失败 — {'; '.join(errors)}")
            time.sleep(pre_clear_s)

        if self.enable_udp:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ------------------------------------------------------------------
    # Serial fan-out
    # ------------------------------------------------------------------

    def _serial_write_all(self, data: bytes, what: str) -> None:
        """同一个字节写给全部 TTL 口;单口写失败只告警,拔线不再影响其他口。"""
        for ser in list(self._sers):
            try:
                ser.write(data)
                ser.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"[marker] ⚠ 串口 {ser.port} 写{what}失败: {exc} — 移除")
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
                self._sers.remove(ser)

    def set_trial(self, trial: int) -> None:
        self._trial = int(trial)

    def next_trial(self) -> int:
        self._trial += 1
        return self._trial

    def mark(self, code: int, tag: str) -> float:
        """Emit one marker on all paths, return the wall-clock emit time."""
        if not 0 <= code <= 255:
            raise ValueError(f"code must be 0..255, got {code}")
        t_wall = time.time()
        t_eprime_ms = int((time.perf_counter() - self._t0_perf) * 1000.0)

        if self._sock is not None:
            msg = (
                f"EVT|trial={self._trial}|tag={tag}|code={code}"
                f"|t_eprime_ms={t_eprime_ms}|t_sent_pc={t_wall!r}"
            )
            try:
                self._sock.sendto(msg.encode("utf-8"), (self.udp_host, self.udp_port))
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"[marker] UDP send failed: {exc}")

        if self._sers:
            # 全部口先打码,共享同一个保持窗口,再一起清零 —— 时序与单口一致
            self._serial_write_all(bytes([code]), f" code={code} ")
            time.sleep(self.hold_s)
            self._serial_write_all(bytes([0]), "清零 ")

        if self.verbose:
            print(f"[marker] trial={self._trial} {tag:<12} code={code:3d} hex=0x{code:02X}"
                  + (f"  ttl={','.join(s.port for s in self._sers)}"
                     if self._sers else ""))
        return t_wall

    def close(self) -> None:
        for ser in self._sers:
            try:
                ser.write(bytes([0]))
                ser.flush()
                ser.close()
            except Exception:  # noqa: BLE001
                pass
        self._sers = []
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:  # noqa: BLE001
                pass
            self._sock = None

    def __enter__(self) -> "MarkerSender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def open_sender(**kwargs):
    s = MarkerSender(**kwargs)
    try:
        yield s
    finally:
        s.close()
