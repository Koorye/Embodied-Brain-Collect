"""EGO headband — wired TCP receiver (4 fisheye JPEG cameras + 2 IMUs).

Speaks the device's "wired ROS TCP" protocol (reference client:
``C:\\Users\\31454\\wired_tcp``).  Every record on the wire is::

    uint32 meta_len | uint32 payload_len | UTF-8 JSON meta | raw payload

(network byte order).  The SERVER drives a stateful handshake before any data::

    client -> {"kind":"hello","version":1}
    server -> {"kind":"time_probe","id":i,"t1":ns}     (xN; client must reply)
    client -> {"kind":"time_reply","id":i,"t2":ns,"t3":ns}
    server -> {"kind":"sync_result","synced":bool,...}
    server -> {"kind":"stream_start","topics":{name:type},"missing_publishers":[..]}
    server -> {"kind":"data","topic":..,"header":{..},...} + payload   (repeating)
    server -> {"kind":"status"|"error",...}

The server sets ITS clock to this machine's during the probe, so the ROS
``header.stamp_ns`` capture time shares the session's time base (``time.time()``)
— that, not the receive time, is what we stamp each sample with.

Cameras (``sensor_msgs/CompressedImage``) carry a JPEG payload: decoded with
cv2 and fed to the shared HEVC writer (``{name}.mp4`` + 1:1 timestamps, the
name from ``camera_names``).  IMUs
(``sensor_msgs/Imu``) carry no payload; angular_velocity / linear_acceleration
come straight from the JSON.  A mid-session disconnect stops the loop and still
saves everything received so far.

The launcher opens every recorder before starting ANY of them, so between
``_open`` (stream_start) and the first ``_poll`` this socket would otherwise
sit unread for seconds while the device keeps streaming — long enough for TCP
flow control to stall the server's sends and get the connection dropped before
a single sample lands.  A keep-alive drain thread therefore reads (and discards
the pre-roll) from stream_start until ``_poll`` takes over, so all four cameras
and both IMUs survive the wait and record once the session actually starts.

.. note::
   ``meta['imu']['angular_velocity' | 'linear_acceleration']`` is read as a
   ROS vector3 (``{x,y,z}`` or a 3-sequence).  If the device names these
   differently, ``_xyz`` is the single place to adjust.
"""

import ctypes
import json
import queue
import socket
import struct
import threading
import time

import numpy as np

from .base_ego_headband_recorder import BaseEgoHeadbandRecorder
from .ego_headband_recorder_config import EgoHeadbandRecorderConfig

_HEADER = struct.Struct("!II")
_MAX_META = 65536
_MAX_PAYLOAD = 32 * 1024 * 1024
_VERSION = 1
_POLL_TIMEOUT = 0.2          # idle recv wait during recording (s)
_DRAIN_MAX = 64              # bounded burst drain per poll


# Precise wall clock the device syncs against (matches the reference client):
# GetSystemTimePreciseAsFileTime on Windows, time_ns elsewhere.
if hasattr(ctypes, "WinDLL"):
    _precise = ctypes.WinDLL("kernel32",
                             use_last_error=True).GetSystemTimePreciseAsFileTime
    _precise.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    _precise.restype = None

    def _wall_ns() -> int:
        v = ctypes.c_ulonglong()
        _precise(ctypes.byref(v))
        return (v.value - 116444736000000000) * 100
else:
    def _wall_ns() -> int:
        return time.time_ns()


def _xyz(v) -> np.ndarray:
    """(3,) float32 from a ROS vector3 — a ``{x,y,z}`` dict or a 3-sequence."""
    if isinstance(v, dict):
        return np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)],
                        dtype=np.float32)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return np.asarray(v[:3], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


class NetEgoHeadbandRecorder(BaseEgoHeadbandRecorder):
    """Real EGO headband over the wired ROS TCP protocol."""

    config: EgoHeadbandRecorderConfig

    def __init__(self, config: EgoHeadbandRecorderConfig):
        super().__init__(config)
        # Everything here must stay picklable: the launcher spawns this
        # recorder into its own process, and the socket / cv2 module are
        # created in _open() there, never in the parent.
        self._sock: socket.socket | None = None
        self._cv2 = None
        self._rx = bytearray()                 # TCP reassembly buffer
        # Keep-alive drain (open -> record handoff): the thread is created in
        # _open (child process, never pickled); the Event is stripped/rebuilt
        # by BaseRecorder.__getstate__/__setstate__ like the other events.
        self._drain_thread: threading.Thread | None = None
        self._drain_stop = threading.Event()
        self._preroll = 0                      # pre-roll data frames discarded
        self._cam_index: dict[str, int] = {}   # topic -> cam slot
        self._imu_index: dict[str, int] = {}   # topic -> imu slot
        self._cam_count: dict[int, int] = {}
        self._imu_count: dict[int, int] = {}
        self._cam_decode_err: dict[int, int] = {}
        self._cam_empty: dict[int, int] = {}
        self._imu_warned: set[int] = set()
        self._unknown: set[str] = set()
        self._last_seq: dict[str, int] = {}
        self._gaps: dict[str, int] = {}
        self._topics: dict[str, str] = {}
        self._synced: bool | None = None
        self._ended = False
        self._audio_available = False
        self._decode_queues = {}
        self._decode_threads = {}
        self._decode_stop = threading.Event()
        self._cam_decode_dropped = {}

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # Under Windows spawn the parent pickles this instance and the child
        # NEVER runs __init__ — it is rebuilt from ``state`` alone.  If the
        # parent's loaded copy predates the keep-alive drain (e.g. a session
        # runner still holding the old module in sys.modules while the child
        # re-imports the new one from disk), these attrs are absent and the
        # child's _open() -> _start_drain() crashes.  Guarantee them so a
        # parent/child version skew can never break the open.
        self.__dict__.setdefault("_drain_thread", None)
        self.__dict__.setdefault("_drain_stop", threading.Event())
        self.__dict__.setdefault("_preroll", 0)
        self.__dict__.setdefault("_audio_writer", None)
        self.__dict__.setdefault("_audio_error", "")
        self.__dict__.setdefault("_audio_available", False)
        self.__dict__.setdefault("_decode_queues", {})
        self.__dict__.setdefault("_decode_threads", {})
        self.__dict__.setdefault("_decode_stop", threading.Event())
        self.__dict__.setdefault("_cam_decode_dropped", {})

    # ---- lifecycle ------------------------------------------------------

    def _open(self) -> bool:
        cfg = self.config
        import cv2                    # in the child process, not __init__
        self._cv2 = cv2
        try:
            self._sock = socket.create_connection(
                (cfg.host, cfg.port), timeout=cfg.connect_timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as exc:
            self._open_error = (f"cannot connect tcp://{cfg.host}:{cfg.port}"
                                f" — {type(exc).__name__}: {exc}")
            self._log(f"[ego_headband:net] open failed — {self._open_error}")
            return False

        try:
            self._send({"kind": "hello", "version": _VERSION,
                        "capabilities": ["pcm_audio_v1"] if cfg.audio_enabled else []})
            if not self._handshake():
                self._close_sock()
                return False
        except (OSError, EOFError, ValueError) as exc:
            self._open_error = f"handshake failed — {type(exc).__name__}: {exc}"
            self._log(f"[ego_headband:net] open failed — {self._open_error}")
            self._close_sock()
            return False

        self._log(f"[ego_headband:net] streaming tcp://{cfg.host}:{cfg.port} — "
                  f"{len(self._cam_index)} cams {sorted(self._cam_index.values())}"
                  f" + {len(self._imu_index)} IMUs "
                  f"{sorted(self._imu_index.values())} (synced={self._synced})")
        # Keep the socket drained until recording actually starts, or the
        # device drops us during the launcher's "open everything, then start"
        # barrier (see _start_drain).
        self._start_drain()
        return True

    def _record(self) -> None:
        # Hand the socket from the keep-alive drain to the poll loop before
        # recording (idempotent — _poll guards too, for the live harness that
        # polls directly without ever entering _record).
        self._stop_drain()
        super()._record()

    def _close(self) -> None:
        self._stop_drain()
        self._close_sock()
        self._stop_camera_decoders()
        self._close_microphone()
        if self.config.audio_enabled and self._audio_available and self._audio_writer is None:
            self._audio_error = self._audio_error or "microphone advertised but no audio recorded"
            self._log(f"[ego_headband:net] {self._audio_error}", level="WARNING")
        cams = sum(self._cam_count.values())
        imus = sum(self._imu_count.values())
        extra = ""
        if self._cam_decode_err:
            extra += f" decode_err={dict(sorted(self._cam_decode_err.items()))}"
        if self._cam_empty:
            extra += f" empty_payload={dict(sorted(self._cam_empty.items()))}"
        if self._cam_decode_dropped:
            extra += f" decode_queue_dropped={self._cam_decode_dropped}"
        if self._gaps:
            extra += f" stream_gaps={dict(sorted(self._gaps.items()))}"
        self._log(f"[ego_headband:net] stopped (cam_frames={cams} imu={imus}"
                  f"{extra})")

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ---- keep-alive drain (open -> record handoff) ----------------------

    def _start_drain(self) -> None:
        """Read (and discard) from stream_start until ``_poll`` takes over.

        The launcher runs every recorder's ``_open`` before starting ANY of
        them, so without this the socket would sit unread for seconds while
        the device keeps blasting 4x1920x1088 JPEGs: the TCP receive buffer
        fills, flow control stalls the server's sends, and it drops the
        connection before a single sample is recorded (cameras/IMUs then come
        back empty or missing).  Pre-roll frames are discarded — they precede
        the session's t0 and would misalign with stim — but ``time_probe`` is
        still answered so the device clock stays synced.
        """
        if self._drain_thread is not None:
            return
        self._drain_stop.clear()
        t = threading.Thread(target=self._drain_loop,
                             name=f"{self.name}-keepalive", daemon=True)
        self._drain_thread = t
        t.start()

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set() and not self._ended:
            got = self._recv(_POLL_TIMEOUT)
            if got is None:
                return                       # stream ended; _end_stream fired
            if got:
                for _ in range(_DRAIN_MAX):
                    if not self._recv(0.0):
                        break
                self._drain_discard()

    def _drain_discard(self) -> None:
        """Drop queued pre-roll frames; only ``time_probe`` is acted on."""
        while True:
            try:
                frame = self._take_frame()
            except ValueError as exc:
                self._end_stream(f"invalid frame during keep-alive: {exc}")
                self._rx.clear()
                return
            if frame is None:
                return
            meta, _ = frame
            kind = meta.get("kind")
            if kind == "data":
                self._preroll += 1
            elif kind == "time_probe":
                try:
                    self._send({"kind": "time_reply", "id": meta.get("id"),
                                "t2": _wall_ns(), "t3": _wall_ns()})
                except OSError:
                    pass

    def _stop_drain(self) -> None:
        """Stop+join the drain (idempotent); hand ``_rx`` back to ``_poll``.

        Deliberately does NOT clear ``_rx``: a partial frame left there is
        finished by ``_poll`` on the same reassembly buffer, so the handoff
        neither loses a frame nor desyncs the frame boundary.
        """
        self._drain_stop.set()
        t = self._drain_thread
        if t is None:
            return
        t.join(timeout=2.0)
        self._drain_thread = None
        if self._preroll:
            self._log(f"[ego_headband:net] keep-alive drained {self._preroll} "
                      f"pre-roll frames while waiting to start (connection "
                      f"held open)")
            self._preroll = 0

    # ---- handshake (server-driven, blocking) ---------------------------

    def _handshake(self) -> bool:
        """Reply to time probes until the server opens the stream."""
        cfg = self.config
        budget = max(cfg.connect_timeout, (cfg.open_timeout or 30.0) - 2.0)
        deadline = time.time() + budget
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._open_error = "handshake timed out before stream_start"
                self._log(f"[ego_headband:net] open failed — {self._open_error}")
                return False
            meta, _ = self._recv_frame(remaining)
            kind = meta.get("kind")
            if kind == "time_probe":
                self._send({"kind": "time_reply", "id": meta.get("id"),
                            "t2": _wall_ns(), "t3": _wall_ns()})
            elif kind == "sync_result":
                self._synced = bool(meta.get("synced"))
                self._log(f"[ego_headband:net] clock sync: "
                          f"synced={self._synced} "
                          f"system_clock_set={meta.get('system_clock_set')}")
                if not self._synced and cfg.require_synced:
                    self._open_error = "device clock not synchronized"
                    self._log(f"[ego_headband:net] open failed — "
                              f"{self._open_error}")
                    return False
            elif kind == "stream_start":
                self._on_stream_start(meta)
                if cfg.audio_enabled and cfg.require_audio and not self._audio_available:
                    self._open_error = "server did not advertise requested microphone audio/PCM"
                    return False
                return True
            elif kind == "error":
                self._open_error = f"server error: {meta.get('message')}"
                self._log(f"[ego_headband:net] open failed — {self._open_error}")
                return False
            # 'status' or anything else before stream_start: keep waiting.

    def _on_stream_start(self, meta: dict) -> None:
        self._topics = dict(meta.get("topics") or {})
        self._audio_available = (self.config.audio_enabled and
                                 self._topics.get(self.config.audio_topic) == "audio/PCM")
        missing = meta.get("missing_publishers") or []
        cam_topics = list(self.config.camera_topics)
        imu_topics = list(self.config.imu_topics)
        for topic in self._topics:
            if topic in cam_topics:
                i = cam_topics.index(topic)
                self._cam_index[topic] = i
                self._cam_count.setdefault(i, 0)
            elif topic in imu_topics:
                j = imu_topics.index(topic)
                self._imu_index[topic] = j
                self._imu_count.setdefault(j, 0)
            elif self._audio_available and topic == self.config.audio_topic:
                pass
            else:
                self._unknown.add(topic)
        note = f" missing_publishers={missing}" if missing else ""
        ign = f" ignoring_unconfigured={sorted(self._unknown)}" if self._unknown else ""
        self._log(f"[ego_headband:net] stream_start: "
                  f"{len(self._topics)} topics ->"
                  f" cams={sorted(self._cam_index.values())}"
                  f" imus={sorted(self._imu_index.values())}"
                  f" microphone={self._audio_available}{note}{ign}")

    # ---- wire framing ---------------------------------------------------

    def _send(self, meta: dict) -> None:
        raw = json.dumps(meta, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        self._sock.sendall(_HEADER.pack(len(raw), 0) + raw)

    def _recv(self, timeout: float):
        """One recv into ``_rx``.  True=data, False=timeout, None=ended."""
        try:
            self._sock.settimeout(timeout)
            chunk = self._sock.recv(65536)
        except (socket.timeout, BlockingIOError):
            return False
        except OSError as exc:
            self._end_stream(f"{type(exc).__name__}: {exc}")
            return None
        if not chunk:
            self._end_stream("peer closed connection")
            return None
        self._rx.extend(chunk)
        return True

    def _take_frame(self):
        """Pop one complete frame from ``_rx``, or None if it is incomplete."""
        buf = self._rx
        if len(buf) < _HEADER.size:
            return None
        m, n = _HEADER.unpack_from(buf, 0)
        if not 0 < m <= _MAX_META or n > _MAX_PAYLOAD:
            raise ValueError(f"invalid frame length (meta={m} payload={n})")
        total = _HEADER.size + m + n
        if len(buf) < total:
            return None
        meta = json.loads(bytes(buf[_HEADER.size:_HEADER.size + m]))
        payload = bytes(buf[_HEADER.size + m:total])
        del buf[:total]
        return meta, payload

    def _recv_frame(self, timeout: float):
        """Block until one whole frame arrives (handshake only)."""
        deadline = time.time() + timeout
        while True:
            frame = self._take_frame()
            if frame is not None:
                return frame
            r = self._recv(max(0.01, deadline - time.time()))
            if r is None:
                raise EOFError("connection ended during handshake")

    # ---- recording poll -------------------------------------------------

    def _poll(self, ts: float) -> None:
        # First poll takes over from the keep-alive drain started in _open:
        # stop+join it so there is only ever ONE reader of the socket / _rx
        # (no-op afterwards).  Also covers the live-visualization harness,
        # which calls _poll directly and never enters _record.
        self._stop_drain()
        if self._sock is None or self._ended:
            return
        got = self._recv(_POLL_TIMEOUT)
        if got is None:
            return                       # stream ended; stop_event already set
        if got:
            # Drain a burst so a run of large JPEGs cannot outrun the
            # hz-limited loop; bounded so we still return to check stop.
            for _ in range(_DRAIN_MAX):
                if not self._recv(0.0):
                    break
        self._drain_frames()

    def _drain_frames(self) -> None:
        while True:
            try:
                frame = self._take_frame()
            except ValueError as exc:
                self._end_stream(str(exc))
                self._rx.clear()
                return
            if frame is None:
                return
            meta, payload = frame
            try:
                self._handle(meta, payload)
            except Exception as exc:                # noqa: BLE001
                self._log(f"[ego_headband:net] bad frame skipped — "
                          f"{type(exc).__name__}: {exc}", level="WARNING")

    def _handle(self, meta: dict, payload: bytes) -> None:
        kind = meta.get("kind")
        if kind == "data":
            self._on_data(meta, payload)
        elif kind == "time_probe":
            # The server may re-probe mid-session; keep the reply honest.
            try:
                self._send({"kind": "time_reply", "id": meta.get("id"),
                            "t2": _wall_ns(), "t3": _wall_ns()})
            except OSError:
                pass
        elif kind == "error":
            self._end_stream(f"server error: {meta.get('message')}")
        # 'status' and anything else carry nothing to store.

    # ---- routing --------------------------------------------------------

    def _on_data(self, meta: dict, payload: bytes) -> None:
        if self._ended:
            return
        topic = meta.get("topic")
        header = meta.get("header") or {}
        stamp_ns = header.get("stamp_ns")
        # Capture time on the session clock (the device synced to us); fall
        # back to arrival time if a stamp is missing.
        t = (float(stamp_ns) / 1e9) if stamp_ns else time.time()

        if topic in self._cam_index:
            self._queue_camera(self._cam_index[topic], payload, t)
        elif topic in self._imu_index:
            self._on_imu(self._imu_index[topic], meta, t)
        elif self._audio_available and topic == self.config.audio_topic:
            try:
                self._write_microphone(meta, payload)
            except (ValueError, OSError, EOFError) as exc:
                self._audio_error = str(exc)
                self._end_stream(f"microphone recording failed: {exc}")
                return
        elif topic not in self._unknown:
            self._unknown.add(topic)
            self._log(f"[ego_headband:net] ignoring unconfigured topic "
                      f"{topic!r}", level="WARNING")

        seq = meta.get("stream_seq")
        if seq is not None:
            seq = int(seq)
            prev = self._last_seq.get(topic)
            if prev is not None and seq != prev + 1:
                self._gaps[topic] = self._gaps.get(topic, 0) + abs(seq - prev - 1)
            self._last_seq[topic] = seq

    def _queue_camera(self, i: int, payload: bytes, t: float) -> None:
        # Keep JPEG decode/color conversion off the socket loop: PCM and IMUs
        # must keep draining even when full-resolution video is expensive.
        if i not in self._decode_queues:
            q = queue.Queue(maxsize=8)
            self._decode_queues[i] = q
            worker = threading.Thread(target=self._decode_camera_loop,
                                      args=(i, q), name=f"headband-cam{i}-decode",
                                      daemon=True)
            self._decode_threads[i] = worker
            worker.start()
        q = self._decode_queues[i]
        try:
            q.put_nowait((payload, t))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait((payload, t))
            self._cam_decode_dropped[i] = self._cam_decode_dropped.get(i, 0) + 1

    def _decode_camera_loop(self, i: int, q) -> None:
        while True:
            try:
                payload, stamp = q.get(timeout=0.1)
            except queue.Empty:
                if self._decode_stop.is_set():
                    return
                continue
            try:
                self._on_camera(i, payload, stamp)
            except Exception as exc:
                self._cam_decode_err[i] = self._cam_decode_err.get(i, 0) + 1
                self._log(f"[ego_headband:net] cam{i} decode failed: {exc}", level="WARNING")

    def _stop_camera_decoders(self) -> None:
        self._decode_stop.set()
        # Bounded queues are drained before the video writers are finalized.
        for worker in self._decode_threads.values():
            worker.join()
        self._decode_threads.clear()
        self._decode_queues.clear()

    def _on_camera(self, i: int, payload: bytes, t: float) -> None:
        if not payload:
            self._cam_empty[i] = self._cam_empty.get(i, 0) + 1
            return
        img = self._cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                                 self._cv2.IMREAD_COLOR)     # -> BGR HxWx3
        if img is None:
            self._cam_decode_err[i] = self._cam_decode_err.get(i, 0) + 1
            return
        # Record RGB like the other cameras; arr_video queues it for the
        # async HEVC writer and stamps the stream only for frames written.
        self.arr_video(self._cam_stem(i), t,
                       self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB))
        self._cam_count[i] = self._cam_count.get(i, 0) + 1

    def _on_imu(self, j: int, meta: dict, t: float) -> None:
        imu = meta.get("imu") or {}
        if not imu and j not in self._imu_warned:
            self._imu_warned.add(j)
            self._log(f"[ego_headband:net] imu{j}: no 'imu' object in metadata "
                      f"(keys={sorted(meta)}) — check the field mapping",
                      level="WARNING")
        self._acc(f"imu{j}_ts", t)
        self._acc_arr(f"imu{j}_gyro", _xyz(imu.get("angular_velocity")))
        self._acc_arr(f"imu{j}_accel", _xyz(imu.get("linear_acceleration")))
        self._imu_count[j] = self._imu_count.get(j, 0) + 1

    # ---- stream end -----------------------------------------------------

    def _end_stream(self, reason: str) -> None:
        if self._ended:
            return
        self._ended = True
        self._log(f"[ego_headband:net] stream ended — {reason}; stopping and "
                  f"saving what was received", level="WARNING")
        self.stop_event.set()          # _loop breaks -> _teardown saves
