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

# probe_data_flow 的最小测速窗口(秒):数据一到就回报会把 1 次 poll 的
# 延迟当成速率,算出天文数字。
_RATE_WINDOW = 0.5


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
    hz: float = 1000.0          # poll ceiling: a poll that finishes faster than
                                # 1/hz is padded; a slower one is never sped up

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
    #: False = 该 slot 由 stim 驱动(如 UDP marker),stim 启动前不可能有数据,
    #: launcher 的"录制确认"对其直接放行,不等缓冲增长。
    data_before_stim: ClassVar[bool] = True

    def __init__(self, config: BaseRecorderConfig):
        self.config = config
        # 槽位名 = 数据目录名(launcher 经 set_output_dir 指到具体传感器,
        # 如 cam_head / emg_left);display_name 是可选的设备显示名,进 meta
        self.slot = self.output_dir or self.name
        self.display_name: str = ""
        self._buf: dict[str, list[Any]] = {}
        self._arr_buf: dict[str, list[np.ndarray]] = {}
        self._ts_buf: dict[str, list[float]] = {}
        # 每帧的 perf_counter,与 _buf/_arr_buf 同 key 等长 —— 非时间戳字段
        # 的心跳间隔统计(min/max/mean)由此而来;录制结束即释放
        self._frame_ts: dict[str, list[float]] = {}
        self._open_error = ""   # specific failure reason when _open() returns False
        self.stop_event = threading.Event()  # launcher sets this to request a
                                             # graceful stop (run-style recorders)
        # 两段式启动:open 后立即开录(预热),launcher 在所有 slot 确认
        # 正常产数后广播 commit —— 预热数据丢弃、会话时钟从 commit 起算
        self._commit_evt = threading.Event()
        self._committed = False
        self._loop_t0: float | None = None
        # launcher 子进程置 True:预热期 duration 兜底不生效(预热时长由
        # launcher 的确认看门狗约束),commit 后 duration 才开始起算
        self._launch_mode = False
        # abort 路径:录了但不落盘(launcher 中止时丢弃预热数据)
        self._save_enabled = True
        # loguru: per-recorder bound logger; a file sink in the recorder's
        # output dir is added when a session_dir is known.
        self._init_logging()
        # Heartbeat rate window: counts at the previous _heartbeat_stats call.
        self._hb_prev: dict[str, int] = {}
        self._hb_prev_at: float = 0.0
        self._hb_queue: object = None  # launcher sets this (mp.Queue) to route
                                       # heartbeats to the parent process

    def _init_logging(self) -> None:
        """(Re)build the bound logger + session file sink.

        Called from ``__init__`` and again from ``__setstate__``: under the
        Windows launcher (spawn start method) the recorder is pickled into
        its child process, where a sink handle from the parent's loguru
        registry means nothing and must be re-added.
        """
        self.logger = _loguru.bind(name=self.name)
        self._log_sink_id: int | None = None
        if self.config.session_dir:
            try:
                out_dir = Path(self.config.session_dir) / (self.output_dir or self.name)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = (self.output_dir or self.name or "recorder").replace(":", "_")
                self._log_sink_id = _loguru.add(
                    out_dir / f"{fname}.log",
                    format="{time:HH:mm:ss.SSS} | {level: <7} | {message}",
                    filter=lambda r: r["extra"].get("name") == self.name,
                    mode="a", encoding="utf-8")
            except OSError:
                pass  # logging must never break recording

    def set_output_dir(self, name: str) -> None:
        """Point this recorder at a concrete sensor slot (``cam_head``).

        ``_init_logging`` already opened a sink under the generic modality
        directory during ``__init__``, which is what used to leave an empty
        ``camera/`` holding a 0-byte log next to the real ``cam_head/``.  So
        tear that down — sink, file, and directory — before re-wiring.
        """
        if not name or name == self.output_dir:
            return
        stale_dir = (Path(self.config.session_dir) / self.output_dir
                     if self.config.session_dir else None)
        stale_log = (self.output_dir or self.name or "recorder").replace(":", "_")

        if self._log_sink_id is not None:
            try:
                _loguru.remove(self._log_sink_id)
            except ValueError:
                pass
            self._log_sink_id = None
        if stale_dir is not None:
            try:
                log = stale_dir / f"{stale_log}.log"
                if log.is_file() and log.stat().st_size == 0:
                    log.unlink()
                if stale_dir.is_dir() and not any(stale_dir.iterdir()):
                    stale_dir.rmdir()
            except OSError:
                pass  # housekeeping must never break recording

        self.output_dir = name
        self.slot = name
        self._init_logging()

    # ==================================================================
    # Pickling (needed on Windows, where the launcher uses "spawn")
    # ==================================================================

    def __getstate__(self) -> dict[str, Any]:
        """State that can cross a process boundary.

        ``logger`` is a bound loguru object and ``_log_sink_id`` refers to
        the parent process's loguru registry — both are rebuilt by
        ``__setstate__``.  Plain ``threading.Event``s are not picklable;
        they are stripped here and rebuilt fresh by ``__setstate__``.
        They only ever synchronize threads inside one process, so a fresh
        copy is always correct — this covers ``stop_event`` as well as
        per-recorder events like the eye recorder's ``_ready_evt`` /
        ``_start_evt`` / first-sample gates.  The launcher swaps
        ``stop_event`` for a picklable multiprocessing Event before
        spawning — that one passes through untouched.
        """
        state = self.__dict__.copy()
        state.pop("logger", None)
        state.pop("_log_sink_id", None)
        stripped = [k for k, v in state.items()
                    if type(v) is threading.Event]
        for k in stripped:
            state.pop(k)
        state["_stripped_events"] = stripped
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        stripped = state.pop("_stripped_events", [])
        self.__dict__.update(state)
        for k in stripped:
            self.__dict__[k] = threading.Event()
        self._init_logging()

    # ==================================================================
    # run() / _record()
    # ==================================================================

    def run(self) -> int:
        """Standalone entry: open (connect + configure) → record.

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

    def request_commit(self) -> None:
        """Launcher marks the session's true start (all slots streaming, stim
        about to launch).  The next loop iteration drops everything captured
        before this point and restarts the session clock, so duration and the
        saved timeline both anchor on the commit, not on ``_open``."""
        self._commit_evt.set()

    def discard_recording(self) -> None:
        """Abort path: recording happened but the launch was aborted — tear
        down without writing an NPZ."""
        self._save_enabled = False

    def _maybe_commit(self) -> bool:
        """Consume a pending commit inside the record loop.  True exactly on
        the iteration that commits (so callers can reset loop-local state)."""
        if self._committed or not self._commit_evt.is_set():
            return False
        self._committed = True
        self._loop_t0 = self._ts()
        self._discard_preface()
        self._log(f"[{self.name}] commit — 预热数据已丢弃,会话时钟从现在起算",
                  echo=False)
        return True

    def _discard_preface(self) -> None:
        """Drop everything buffered so far and let subclasses reset their
        stream anchors (device-side sample indices, UTC windows, ...)."""
        self._buf.clear()
        self._arr_buf.clear()
        self._ts_buf.clear()
        self._frame_ts.clear()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        """Commit-time anchor reset for callback/driver-buffered recorders —
        clear device-side indices/windows so the saved timeline starts at the
        commit.  Buffers are cleared generically; only anchors live here."""

    def _stop_by_duration(self, elapsed: float) -> bool:
        """duration 兜底是否到点。launcher 两段式下,预热期不启用 ——
        duration 属于 commit 之后的正式会话;预热时长由 launcher 的
        open/确认看门狗约束。独立 run() 无 commit,duration 照旧从开录起算。"""
        if self._launch_mode and not self._committed:
            return False
        return self._should_stop(elapsed)

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

    def _log(self, msg: str, echo: bool = True,
             level: str = "INFO") -> None:
        """Log one line at ``level`` (INFO/WARNING/ERROR/DEBUG).

        ``echo=True`` also goes to the console; the recorder's session log
        file ``{session_dir}/{output_dir}/{output_dir}.log`` gets everything.
        """
        logger = self.logger if echo else self.logger.bind(console=False)
        logger.log(level, msg)

    def _log_exc(self, msg: str = "error") -> None:
        """Log an error WITH a full traceback (call from an except block)."""
        self.logger.opt(exception=True).error(msg)

    # ==================================================================
    # Data-flow probe (preflight)
    # ==================================================================

    def _recording_signal(self) -> int:
        """单调产量计数,launcher 的录制确认阶段用它判断"数据在流动"。
        默认 = 缓冲样本总数;数据不落这三类缓冲的 recorder(如相机走
        异步写盘管线)自行覆盖。"""
        return self._count_samples()

    def _count_samples(self) -> int:
        """当前缓冲的样本总数(跨全部字段) —— preflight 用来测增量。"""
        n = 0
        for values in self._buf.values():
            n += len(values)
        for arrays in self._arr_buf.values():
            n += len(arrays)
        for values in self._ts_buf.values():
            n += len(values)
        return n

    def probe_data_flow(self, timeout: float = 3.0) -> tuple[bool, str]:
        """持续数据探测:``timeout`` 秒内轮询 ``_poll``,看样本是否持续新增。

        preflight 专用 —— ``_open`` 只负责连接与配置,数据健康全部由这里
        判断(launcher 的确认阶段用的也是同一思路)。速率至少用
        ``_RATE_WINDOW`` 的窗口算,首个 poll 立刻命中时不会出现"每秒百万
        样本"的假速率。自带事件循环的 recorder(如 neon_eye_async)覆盖
        此方法用自己的队列判断。
        """
        n0 = self._count_samples()
        t_first = None
        t0 = time.time()
        while True:
            dt = time.time() - t0
            if dt >= timeout:
                return False, f"{timeout:g}s 内没有新样本(数据未持续到达)"
            try:
                self._poll(time.time())
            except Exception as exc:                # noqa: BLE001
                return False, f"轮询崩溃: {type(exc).__name__}: {exc}"
            if self._count_samples() > n0:
                if t_first is None:
                    t_first = time.time()
                # 检测到数据后再观察一小段,速率才有意义
                if time.time() - t_first >= _RATE_WINDOW or dt + 1e-9 >= timeout:
                    n = self._count_samples() - n0
                    span = max(time.time() - t_first, 1e-3)
                    return True, (f"{span:.1f}s 内新增 {n} 个样本"
                                  f"(≈{n / span:.1f}/s)")
            time.sleep(0.02)

    def _setup(self) -> None:
        install_signal_handlers()
        # 回调式 recorder(如 Blackrock)在 open 期间就会往 _frame_ts 攒
        # 心跳帧时刻 —— 开录时清一次,统计只反映录制期
        self._frame_ts.clear()
        self._log(
            f"[{self.name}] "
            f"recording (Ctrl+C to stop, duration={self.config.duration}s) ..."
        )

    def _teardown(self) -> None:
        # close 与 save 分开兜底:close 崩溃不能连累落盘,save 崩溃也要
        # 留下完整 traceback —— 两条路径的异常都进 recorder 自己的 .log。
        try:
            self._close()
        except Exception:
            self._log_exc(f"[{self.name}] close failed — 继续尝试保存已采数据")
        if self._save_enabled:
            try:
                self._save()
            except Exception:
                self._log_exc(f"[{self.name}] save failed — 数据可能不完整")
        if self._log_sink_id is not None:
            _loguru.remove(self._log_sink_id)
            self._log_sink_id = None

    def _should_stop(self, elapsed: float) -> bool:
        return 0 < self.config.duration <= elapsed

    # ==================================================================
    # Loop
    # ==================================================================

    def _loop(self) -> None:
        self._loop_t0 = self._ts()
        last = self._loop_t0
        period = 1.0 / self.config.hz if self.config.hz > 0 else 0.0

        while True:
            if self._maybe_commit():
                last = self._loop_t0    # commit 即会话零点,心跳同步重置
            ts = self._ts()
            elapsed = ts - self._loop_t0
            if self._stop_by_duration(elapsed) or self.stop_event.is_set():
                break
            poll_t0 = time.perf_counter()
            self._poll(ts)
            if ts - last > 0.5:
                last = ts
                self._heartbeat(elapsed)
            # Throttle, never accelerate: pad only the part of the period
            # the poll did not already consume.  Recorders whose _poll
            # blocks or sleeps (serial timeouts, dummy sleep, VR/Manus
            # polling) already run slower than 1000 Hz, so this is a no-op
            # for them — it only reins in free-spinning loops.
            if period > 0:
                time.sleep(max(0.0, period - (time.perf_counter() - poll_t0)))

    def _poll(self, ts: float) -> None:
        pass

    # ==================================================================
    # Heartbeat
    # ==================================================================

    def _heartbeat(self, elapsed: float) -> None:
        # 详细块先于 stats:此时 _hb_prev 还是上一次的值,速率计算才正确
        self._heartbeat_detail(elapsed)
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

    def _heartbeat_detail(self, elapsed: float) -> None:
        """每个心跳把全部字段的计数/shape/速率/采样间隔写进文件日志。

        控制台与 launcher 只收一行摘要;完整细节只在 .log 里 —— 事后排查
        丢包/停流/字段异常时,这是唯一能看到"当时每个数组长什么样"的地方。
        时间戳类字段额外给窗口内样本间隔的 min/max/mean(ms),其余字段给
        窗口平均间隔。
        """
        lines = [f"[hb t={elapsed:6.1f}s]  "
                 f"fields={len(self._buf) + len(self._arr_buf) + len(self._ts_buf)}"]
        for key, vals in sorted(self._buf.items()):
            lines.append(f"    buf  {key:<24} n={len(vals):>9}"
                         f"  rate={self._hb_rate(key, len(vals)):>8.1f}/s"
                         f"  {self._hb_interval(key, len(vals))}")
        for key, arrs in sorted(self._arr_buf.items()):
            shape = str(arrs[-1].shape) if arrs else "-"
            lines.append(f"    arr  {key:<24} n={len(arrs):>9}"
                         f"  shape={shape:<14}"
                         f"  rate={self._hb_rate(key, len(arrs)):>8.1f}/s"
                         f"  {self._hb_interval(key, len(arrs))}")
        for cam, ts_list in sorted(self._ts_buf.items()):
            key = f"{cam}_ts"
            lines.append(f"    ts   {key:<24} n={len(ts_list):>9}"
                         f"  rate={self._hb_rate(key, len(ts_list)):>8.1f}/s"
                         f"  {self._hb_interval(key, len(ts_list))}")
        self._log("\n".join(lines), echo=False)

    def _hb_interval(self, key: str, n: int) -> str:
        """窗口内采样间隔统计:min/max/mean(ms)。

        时间戳类字段直接从值序列算 diff;其余字段只有"每心跳新增了多少
        样本",给窗口平均间隔。窗口内不足 2 个新样本时没有统计意义。
        """
        window_n = n - self._hb_prev.get(key, n)
        if window_n < 2:
            return "dt: -"
        dt_s = time.time() - self._hb_prev_at if self._hb_prev_at else 0.0
        mean_ms = dt_s / max(window_n, 1) * 1e3

        is_ts = ("timestamp" in key or "time" in key
                 or key.endswith("_ts"))
        if is_ts:
            container = (self._buf.get(key)
                         or self._arr_buf.get(key)
                         or self._ts_buf.get(key))
            if container is not None:
                try:
                    vals = np.asarray(container[self._hb_prev.get(key, 0):],
                                      dtype=np.float64)
                    d = np.diff(vals)
                    pos = d[d > 0]
                    if pos.size:
                        return (f"dt: min={pos.min() * 1e3:.2f}ms "
                                f"max={pos.max() * 1e3:.2f}ms "
                                f"mean={pos.mean() * 1e3:.2f}ms")
                except (TypeError, ValueError):
                    pass
        # 非时间戳字段:用逐帧 perf_counter 的窗口段算真实的 min/max/mean
        frames = self._frame_ts.get(key)
        if frames is not None:
            window = frames[self._hb_prev.get(key, 0):]
            if len(window) >= 2:
                d = np.diff(window) * 1e3
                return (f"dt: min={d.min():.2f}ms max={d.max():.2f}ms "
                        f"mean={d.mean():.2f}ms")
        return f"dt: mean={mean_ms:.2f}ms (窗口平均)"


    def _hb_rate(self, key: str, n: int) -> float:
        """该字段自上次心跳以来的速率(样本/秒)。"""
        prev = self._hb_prev.get(key, n)
        dt = time.time() - self._hb_prev_at if self._hb_prev_at else 1.0
        return (n - prev) / dt if dt > 0 else 0.0

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
        self._frame_ts.setdefault(key, []).append(time.perf_counter())

    def _acc_arr(self, key: str, arr: np.ndarray) -> None:
        self._arr_buf.setdefault(key, []).append(arr)
        self._frame_ts.setdefault(key, []).append(time.perf_counter())

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
        # 最终字段清单:每个字段的 shape/dtype/体积写进文件日志 —— 采集结束
        # 后核对数据完整性(该有的字段是否都在、大小是否合理)只看这里。
        self._log(f"[save] {len(payload)} 个字段:", echo=False)
        for key in sorted(payload):
            arr = payload[key]
            shape = str(arr.shape) if hasattr(arr, "shape") else "scalar"
            dtype = getattr(arr, "dtype", type(arr).__name__)
            nbytes = getattr(arr, "nbytes", 0)
            self._log(f"    {key:<24} shape={shape:<18} dtype={dtype} "
                      f"{nbytes / 1e6:.3f} MB", echo=False)
        self._frame_ts.clear()   # 帧时刻只服务于心跳统计,落盘后即释放
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
