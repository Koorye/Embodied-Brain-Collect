"""Global base classes for all recorders.

Each recorder implements ``run() -> int``.  The base provides:
  - ``_ts()`` / ``_setup()`` / ``_teardown()`` — lifecycle
  - ``_acc(key, val)`` / ``_acc_arr(key, arr)`` / ``_acc_ts(cam, ts)`` — data routing
  - ``_save()`` / ``_build_output()`` — NPZ output
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


# =============================================================================
# Signal
# =============================================================================

def install_signal_handlers() -> None:
    for name in ("SIGBREAK", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.default_int_handler)
        except (ValueError, OSError):
            pass


# =============================================================================
# Config
# =============================================================================

@dataclass
class BaseRecorderConfig:
    enabled: bool = True
    session_dir: str = ""
    duration: float = 0.0
    open_timeout: float = 30.0  # max seconds for _open() before launcher aborts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseRecorderConfig":
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in names})


# =============================================================================
# BaseRecorder
# =============================================================================

class BaseRecorder(ABC):
    name: ClassVar[str] = "base"
    output_dir: ClassVar[str] = "base"

    def __init__(self, config: BaseRecorderConfig):
        self.config = config
        self._buf: dict[str, list[Any]] = {}
        self._arr_buf: dict[str, list[np.ndarray]] = {}
        self._ts_buf: dict[str, list[float]] = {}
        self._open_error = ""   # specific failure reason when _open() returns False

    # ==================================================================
    # run()
    # ==================================================================

    def run(self) -> int:
        try:
            ok = self._open()
        except Exception as exc:
            self._log(f"[{self.name}] open error: {type(exc).__name__}: {exc}")
            return 1
        if not ok:
            self._log(f"[{self.name}] open failed"
                      + (f" — {self._open_error}" if self._open_error else ""))
            return 1
        self._setup()
        try:
            self._loop()
        except KeyboardInterrupt:
            self._log(f"\n[{self.name}] Ctrl+C")
        except Exception as exc:
            self._log(f"\n[{self.name}] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1
        finally:
            self._teardown()
        return 0

    # ==================================================================
    # Subclass contract
    # ==================================================================

    @abstractmethod
    def _open(self) -> bool:
        """Open the device.  Return True when ready, False on failure.

        On failure, set ``self._open_error`` to a specific reason and log it
        via ``self._log()`` before returning False.
        """
    @abstractmethod
    def _close(self) -> None: ...

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _ts() -> float:
        return time.time()

    def _log(self, msg: str, echo: bool = True) -> None:
        """Print one log line and tee it into the recorder's output directory.

        The file copy lands next to the recorder's own NPZ, i.e.
        ``{session_dir}/{output_dir}/{output_dir}.log``, and gets a
        wall-clock timestamp so logs from different recorders can be compared.
        Console output stays exactly as before (callers include their own
        ``[name]`` prefixes); pass ``echo=False`` to write the file only.
        """
        if echo:
            print(msg)
        if not self.config.session_dir:
            return
        try:
            out_dir = Path(self.config.session_dir) / (self.output_dir or self.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = (self.output_dir or self.name or "recorder").replace(":", "_")
            with (out_dir / f"{fname}.log").open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except OSError:
            pass  # logging must never break recording

    def _setup(self) -> None:
        install_signal_handlers()
        self._log(
            f"[{self.name}] "
            f"recording (Ctrl+C to stop, duration={self.config.duration}s) ..."
        )

    def _teardown(self) -> None:
        self._close()
        self._save()

    def _should_stop(self, elapsed: float) -> bool:
        return 0 < self.config.duration <= elapsed

    # ==================================================================
    # Loop
    # ==================================================================

    def _loop(self) -> None:
        t0 = self._ts()
        last = t0

        while True:
            ts = self._ts()
            elapsed = ts - t0
            if self._should_stop(elapsed):
                break
            self._poll(ts)
            if ts - last > 0.5:
                last = ts
                self._heartbeat(elapsed)
            time.sleep(0.001)

    def _poll(self, ts: float) -> None:
        pass

    # ==================================================================
    # Heartbeat
    # ==================================================================

    def _heartbeat(self, elapsed: float) -> None:
        extra = self._heartbeat_stats(elapsed)
        parts = [f"\r[{self.name}] t={elapsed:5.1f}s"]
        if extra:
            parts.append(extra)
        sys.stdout.write("  ".join(parts).ljust(110))
        sys.stdout.flush()

    def _heartbeat_stats(self, elapsed: float) -> str:
        return ""

    # ==================================================================
    # Data routing
    # ==================================================================

    def _acc(self, key: str, value: Any) -> None:
        self._buf.setdefault(key, []).append(value)

    def _acc_arr(self, key: str, arr: np.ndarray) -> None:
        self._arr_buf.setdefault(key, []).append(arr)

    def _acc_ts(self, cam: str, ts: float) -> None:
        self._ts_buf.setdefault(cam, []).append(ts)

    # ==================================================================
    # Output
    # ==================================================================

    def _npz_path(self) -> Path | None:
        if not self.config.session_dir:
            return None
        d = Path(self.config.session_dir) / self.output_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.output_dir}.npz"

    def _save(self) -> None:
        out = self._npz_path()
        if out is None:
            return
        payload = self._build_output()
        if not payload:
            self._log(f"[{self.name}] nothing to save.")
            return
        np.savez(out, **payload)
        self._log(f"[{self.name}] saved {out} ({out.stat().st_size/1e6:.1f} MB)")

    def _build_output(self) -> dict[str, np.ndarray]:
        out = {}
        for key in self._buf:
            vals = self._buf[key]
            out[key] = np.asarray(vals) if vals else np.zeros(0, dtype=np.float64)
        for key in self._arr_buf:
            vals = self._arr_buf[key]
            out[key] = np.stack(vals) if vals else np.zeros((0, 1))
        for cam, ts_list in self._ts_buf.items():
            out[f"{cam}_timestamps"] = (
                np.array(ts_list, dtype=np.float64)
                if ts_list else np.zeros(0, dtype=np.float64))
        return out
