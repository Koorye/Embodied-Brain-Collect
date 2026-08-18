"""MarkerSender — ParallelBox TTL + UDP marker emitter for stim scripts.

Replaces E-Prime's InLine marker code in pure Python.  Sends the same byte to:
  - ParallelBox over serial COM (EEG trigger, hardware path)
  - UDP to sync_hub / UdpMarkerRecorder (software path)

The UDP packet uses the wire format understood by sync_hub::

    EVT|trial=<int>|tag=<NAME>|code=<int>|t_eprime_ms=<int>

Both writes share one PC timestamp; the spread between hardware pulse and UDP
send is typically <1 ms on the same machine, well below EEG sampling period.
"""

from __future__ import annotations

import socket
import time
from contextlib import contextmanager

try:
    import serial  # pyserial
except Exception:  # pragma: no cover
    serial = None  # type: ignore


class MarkerSender:
    """Send markers to ParallelBox (hardware) and sync_hub (UDP)."""

    def __init__(
        self,
        port: str = "COM14",
        baud: int = 115200,
        udp_host: str = "127.0.0.1",
        udp_port: int = 9999,
        hold_s: float = 0.020,
        pre_clear_s: float = 0.5,
        enable_serial: bool = True,
        enable_udp: bool = True,
        verbose: bool = False,
    ) -> None:
        self.port = port
        self.baud = baud
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.hold_s = hold_s
        self.enable_serial = enable_serial and serial is not None
        self.enable_udp = enable_udp
        self.verbose = verbose

        self._ser = None
        self._sock = None
        self._trial = 0
        self._t0_perf = time.perf_counter()

        if self.enable_serial:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)  # type: ignore[union-attr]
            self._ser.write(bytes([0]))
            self._ser.flush()
            time.sleep(pre_clear_s)

        if self.enable_udp:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_trial(self, trial: int) -> None:
        self._trial = int(trial)

    def next_trial(self) -> int:
        self._trial += 1
        return self._trial

    def mark(self, code: int, tag: str) -> float:
        """Emit one marker on both paths, return the wall-clock emit time."""
        if not 0 <= code <= 255:
            raise ValueError(f"code must be 0..255, got {code}")
        t_wall = time.time()
        t_eprime_ms = int((time.perf_counter() - self._t0_perf) * 1000.0)

        if self._sock is not None:
            msg = (
                f"EVT|trial={self._trial}|tag={tag}|code={code}"
                f"|t_eprime_ms={t_eprime_ms}"
            )
            try:
                self._sock.sendto(msg.encode("utf-8"), (self.udp_host, self.udp_port))
            except Exception as exc:
                if self.verbose:
                    print(f"[marker] UDP send failed: {exc}")

        if self._ser is not None:
            try:
                self._ser.write(bytes([code]))
                self._ser.flush()
                time.sleep(self.hold_s)
                self._ser.write(bytes([0]))
                self._ser.flush()
            except Exception as exc:
                if self.verbose:
                    print(f"[marker] serial write failed for code={code}: {exc}")

        if self.verbose:
            print(f"[marker] trial={self._trial} {tag:<12} code={code:3d} hex=0x{code:02X}")
        return t_wall

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.write(bytes([0]))
                self._ser.flush()
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
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
