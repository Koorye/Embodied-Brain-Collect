"""Global base classes for all recorders.

Each recorder implements ``run() -> int``.  The base provides:
  - ``_ts()`` / ``_setup()`` / ``_teardown()`` — lifecycle
  - ``_acc(key, val)`` / ``_acc_arr(key, arr)`` / ``_acc_ts(cam, ts)`` — data routing
  - ``_save()`` / ``_build_output()`` — NPZ output
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger as _loguru


# =============================================================================
# Process-wide loguru setup + uncaught-exception hooks
# =============================================================================

# One console sink for the whole process (recorder name is already part of
# every message via the existing "[name]" prefixes); messages logged with
# ``bind(console=False)`` (file-only) are filtered out here.
_loguru.remove()
_loguru.add(
    sys.stderr,
    format=("<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "<level>{message}</level>"),
    colorize=True,
    filter=lambda r: r["extra"].get("console", True),
)


def _install_exception_hooks() -> None:
    """Route uncaught exceptions to loguru with full tracebacks."""
    if getattr(_install_exception_hooks, "_done", False):
        return
    _install_exception_hooks._done = True

    def _hook(exc_type, exc_value, tb):
        _loguru.opt(exception=(exc_type, exc_value, tb)).error(
            "Uncaught exception (main thread)")

    def _thread_hook(args):
        _loguru.opt(exception=(args.exc_type, args.exc_value,
                               args.exc_traceback)).error(
            f"Uncaught exception in thread {args.thread.name}")

    sys.excepthook = _hook
    threading.excepthook = _thread_hook


_install_exception_hooks()


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
        self.stop_event = threading.Event()  # launcher sets this to request a
                                             # graceful stop (run-style recorders)
        # loguru: per-recorder bound logger; a file sink in the recorder's
        # output dir is added when a session_dir is known.
        self.logger = _loguru.bind(name=self.name)
        self._log_sink_id: int | None = None
        # Heartbeat rate window: counts at the previous _heartbeat_stats call.
        self._hb_prev: dict[str, int] = {}
        self._hb_prev_at: float = 0.0
        self._hb_queue: object = None  # launcher sets this (mp.Queue) to route
                                       # heartbeats to the parent process
        if config.session_dir:
            try:
                out_dir = Path(config.session_dir) / (self.output_dir or self.name)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = (self.output_dir or self.name or "recorder").replace(":", "_")
                self._log_sink_id = _loguru.add(
                    out_dir / f"{fname}.log",
                    format="{time:HH:mm:ss.SSS} | {level: <7} | {message}",
                    filter=lambda r: r["extra"].get("name") == self.name,
                    mode="a", encoding="utf-8")
            except OSError:
                pass  # logging must never break recording

    # ==================================================================
    # run() / _record()
    # ==================================================================

    def run(self) -> int:
        """Standalone entry: open (first-data gate) → record.

        ALL failures are caught here — full traceback goes to the console
        and the recorder's session log file."""
        try:
            ok = self._open()
        except Exception as exc:
            self.logger.opt(exception=True).error(
                f"[{self.name}] open crashed: {type(exc).__name__}: {exc}")
            return 1
        if not ok:
            self.logger.error(f"[{self.name}] open failed"
                              + (f" — {self._open_error}"
                                 if self._open_error else ""))
            return 1
        try:
            self._record()
        except Exception as exc:
            self.logger.opt(exception=True).error(
                f"[{self.name}] crashed: {type(exc).__name__}: {exc}")
            return 1
        return 0

    def _record(self) -> None:
        """setup → loop → teardown, errors fully logged.

        The launcher's child process calls this after the open gate;
        asyncio-style recorders override it with their own event loop."""
        self._setup()
        try:
            self._loop()
        except KeyboardInterrupt:
            self.logger.info(f"[{self.name}] Ctrl+C")
        finally:
            self._teardown()

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
        """Log one line: console (unless ``echo=False``) + the recorder's
        session log file ``{session_dir}/{output_dir}/{output_dir}.log``.
        """
        if echo:
            self.logger.info(msg)
        else:
            self.logger.bind(console=False).info(msg)

    def _log_exc(self, msg: str = "error") -> None:
        """Log an error WITH a full traceback (call from an except block)."""
        self.logger.opt(exception=True).error(msg)

    def _wait_first_sample(self, poll_fn, what: str, timeout: float) -> bool:
        """First-data gate for ``_open``: poll until ``poll_fn()`` returns
        True, else fail the open with a specific reason."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if poll_fn():
                return True
            time.sleep(0.05)
        self._open_error = f"no {what} within {timeout:g}s"
        self.logger.error(f"[{self.name}] open failed — {self._open_error}")
        return False

    def _setup(self) -> None:
        install_signal_handlers()
        self._log(
            f"[{self.name}] "
            f"recording (Ctrl+C to stop, duration={self.config.duration}s) ..."
        )

    def _teardown(self) -> None:
        self._close()
        self._save()
        if self._log_sink_id is not None:
            _loguru.remove(self._log_sink_id)
            self._log_sink_id = None

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
            if self._should_stop(elapsed) or self.stop_event.is_set():
                break
            self._poll(ts)
            if ts - last > 0.5:
                last = ts
                self._heartbeat(elapsed)
            # time.sleep(0.001)

    def _poll(self, ts: float) -> None:
        pass

    # ==================================================================
    # Heartbeat
    # ==================================================================

    def _heartbeat(self, elapsed: float) -> None:
        extra = self._heartbeat_stats(elapsed)
        line = (f"t={elapsed:5.1f}s  {extra}" if extra
                else f"t={elapsed:5.1f}s")
        if self._hb_queue is not None:
            # Running under the launcher: report to the parent process,
            # which aggregates one combined status line (send stats only —
            # the parent adds its own t= header and the recorder name).
            try:
                self._hb_queue.put_nowait(("hb", self.name, extra))
            except Exception:
                pass
            return
        sys.stdout.write(f"\r[{self.name}] {line}".ljust(110))
        sys.stdout.flush()

    def _heartbeat_stats(self, elapsed: float) -> str:
        """Every recorded stream: length, latest array's shape, and the
        CURRENT rate (samples received since the previous heartbeat call,
        normalized to per-second regardless of the call cadence)."""
        now = time.time()
        dt = now - self._hb_prev_at if self._hb_prev_at else 1.0
        parts: list[str] = []

        def _rate(key: str, n: int) -> str:
            prev = self._hb_prev.get(key, n)
            rate = (n - prev) / dt if dt > 0 else 0.0
            return f"{rate:.0f}/s"

        for key, vals in self._buf.items():
            parts.append(f"{key}={len(vals)}({_rate(key, len(vals))})")
        for key, arrs in self._arr_buf.items():
            shape = tuple(arrs[-1].shape) if arrs else "empty"
            parts.append(f"{key}={len(arrs)}{shape}({_rate(key, len(arrs))})")
        for cam, ts_list in self._ts_buf.items():
            parts.append(f"{cam}_ts={len(ts_list)}({_rate(cam + '_ts', len(ts_list))})")

        self._hb_prev = {
            **{k: len(v) for k, v in self._buf.items()},
            **{k: len(v) for k, v in self._arr_buf.items()},
            **{k + "_ts": len(v) for k, v in self._ts_buf.items()},
        }
        self._hb_prev_at = now
        return "  ".join(parts)

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
            if not vals:
                out[key] = np.zeros((0, 1))
                continue
            try:
                out[key] = np.stack(vals)
            except ValueError:
                # Ragged frames (e.g. skeleton node count changes when a
                # glove connects/disconnects mid-session): pad to the largest
                # shape so the session still saves.
                out[key] = self._pad_stack(vals)
        for cam, ts_list in self._ts_buf.items():
            out[f"{cam}_timestamps"] = (
                np.array(ts_list, dtype=np.float64)
                if ts_list else np.zeros(0, dtype=np.float64))
        return out

    @staticmethod
    def _pad_stack(vals: list[np.ndarray]) -> np.ndarray:
        """Stack ragged arrays by padding to the largest shape.

        Missing rows are NaN (False for bool arrays) so downstream code can
        mask them out by ``np.isnan`` / ``~valid``.
        """
        ndim = vals[0].ndim
        shape = (len(vals),) + tuple(
            max(v.shape[d] for v in vals) for d in range(ndim))
        dtype = np.result_type(*vals)
        fill = False if dtype == np.bool_ else np.nan
        padded = np.full(shape, fill, dtype=dtype)
        for i, v in enumerate(vals):
            padded[(i,) + tuple(slice(0, s) for s in v.shape)] = v
        return padded
