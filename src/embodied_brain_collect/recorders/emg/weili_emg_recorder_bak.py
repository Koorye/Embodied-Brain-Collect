"""Weili (WAVELETECH) 8-channel EMG — real serial port, 29-byte frames.

Frame layout (29 bytes, packed back-to-back, no padding)::

    3-byte header D2 D2 D2 | 1-byte type | 1-byte seq | 24-byte payload

EMG (0xAA) payload: 8 channels x int24 (big-endian).
IMU (0xBB) payload: gyro xyz + accel xyz (int16, scaled).
EMG and IMU frames share ONE sequence number.

The 0xD2 header bytes also occur inside payload data, so the parser
validates each header candidate by its type byte AND by frame cadence (a
real frame is immediately followed by the next frame's header byte), and
re-syncs byte-by-byte on false matches so a real header hiding inside a
rejected window is never consumed.
"""

import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

from .base_emg_recorder import BaseEmgRecorder
from .emg_recorder_config import EmgRecorderConfig

HEADER = b"\xD2\xD2\xD2"
FRAME_LEN = 29
EMG_TYPE = 0xAA
IMU_TYPE = 0xBB
GYRO_SCALE = 0.0012
ACC_SCALE = 0.0005978

# First-frame gate: the armband's RF link can take ~10 s to come up after
# the dongle is plugged in, so 5 s was too short and caused spurious
# "no EMG frame" open failures right after re-plugging.
OPEN_GATE_TIMEOUT = 20.0

# USB tags that identify the Weili armband dongles (CH343 is the one seen
# in the field; CP210x kept for older dongle revisions).
_DEVICE_TAGS = ("VID:PID=1A86:55D3", "CH343",
                "VID:PID=10C4", "CP210", "Silicon Labs")


def _s24_be(b):
    v = (b[0] << 16) | (b[1] << 8) | b[2]
    return v - (1 << 24) if v & 0x800000 else v


def _s16_be(b):
    return struct.unpack(">h", b)[0]


def _auto_detect_ports() -> list[str]:
    """All serial ports that look like a Weili dongle (insertion order)."""
    return [p.device for p in list_ports.comports()
            if any(t in (p.hwid or "").upper() or t in (p.description or "")
                   for t in _DEVICE_TAGS)]


def _auto_detect_port() -> str | None:
    """First candidate port, or None (kept for callers wanting one name)."""
    ports = _auto_detect_ports()
    return ports[0] if ports else None


# ---- cross-process port claiming -------------------------------------------
# Each recorder runs in its OWN process (launcher), and two armbands would
# otherwise both auto-detect the same first port.  A lock file per port,
# held for the recorder's lifetime, lets siblings skip an already-claimed
# port and take the next candidate.

def _port_lock_path(session_dir: str, port: str) -> Path:
    base = Path(session_dir) if session_dir else Path(tempfile.gettempdir())
    safe = port.replace("/", "_").replace("\\", "_").replace(":", "_")
    return base / f".ebc_emg_{safe}.lock"


def _lock_port(path: Path):
    """Non-blocking exclusive lock on *path*; returns an open handle that
    holds the lock, or None if another process holds it."""
    try:
        fh = open(path, "a+b")
    except OSError:
        return None
    try:
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0, 2)
            if fh.tell() == 0:
                fh.write(b"\0")
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fh.close()
        except OSError:
            pass
        return None
    return fh


class WeiliEmgRecorder(BaseEmgRecorder):
    """Real WAVELETECH EMG armband over CH343/CP210x USB-UART."""

    config: EmgRecorderConfig

    def __init__(self, config: EmgRecorderConfig):
        super().__init__(config)
        self._ser: serial.Serial | None = None
        self._last_sn: int | None = None
        self._dropped = 0
        self._resyncs = 0          # false-header rejections (parser noise)
        self._raw_bytes_in = 0     # diagnostics: bytes seen since open
        self._port = ""
        self._lock = None          # (lock_path, handle) holding the port claim
        self._raw_buf = bytearray()

    # ---- lifecycle ----------------------------------------------------------

    def _open(self) -> bool:
        if self.config.port:
            candidates = [self.config.port]
        else:
            candidates = _auto_detect_ports()
        if not candidates:
            self._open_error = ("no supported EMG serial adapter found "
                                "(expected a CH343/CP210x dongle)")
            self._log(f"[emg:weili] open failed — {self._open_error}. "
                      f"Serial ports:")
            for p in list_ports.comports():
                self._log(f"  - {p.device}: {p.description} [{p.hwid}]")
            return False

        # Claim the first candidate not already taken by another recorder
        # process (two armbands auto-detect into distinct ports this way).
        port = None
        for cand in candidates:
            lock_path = _port_lock_path(self.config.session_dir, cand)
            fh = _lock_port(lock_path)
            if fh is not None:
                port, self._lock = cand, (lock_path, fh)
                break
        if port is None:
            self._open_error = (f"all candidate EMG ports already in use "
                                f"({', '.join(candidates)}) — is another "
                                f"recorder running?")
            return self._fail_open()
        if len(candidates) > 1:
            self._log(f"[emg:weili] candidate ports: {', '.join(candidates)}")

        try:
            self._ser = serial.Serial(port, self.config.baud, timeout=0.005)
        except Exception as exc:
            self._open_error = f"cannot open {port}: {type(exc).__name__}: {exc}"
            return self._fail_open()
        self._port = port
        self._ser.reset_input_buffer()
        self._raw_bytes_in = 0

        self._log(f"[emg:weili] open {port} @ {self.config.baud} — "
                  f"waiting for first frame ...")
        t0 = time.time()
        last_log = t0
        try:
            while time.time() - t0 < OPEN_GATE_TIMEOUT:
                self._poll(time.time())
                if self._buf.get("emg_sn"):
                    break
                if time.time() - last_log > 3.0:
                    last_log = time.time()
                    self._log(f"[emg:weili] still waiting "
                              f"({time.time() - t0:.0f}s, "
                              f"{self._raw_bytes_in} bytes) — the armband "
                              f"link can take ~10s after plug-in")
            else:
                self._open_error = (
                    f"no EMG frame within {OPEN_GATE_TIMEOUT:g}s — check the "
                    f"armband is on and the dongle RF link is up; if the "
                    f"dongle was just plugged in, wait a few seconds and "
                    f"re-run")
                return self._fail_open()
        except (OSError, serial.SerialException) as exc:
            self._open_error = f"{port}: {type(exc).__name__}: {exc}"
            return self._fail_open()

        # Gate samples used wall-clock ts; clear so the session timeline
        # starts clean from the launcher's t0, and reset gate-time counters.
        self._buf.clear()
        self._arr_buf.clear()
        self._raw_buf.clear()
        self._dropped = 0
        self._resyncs = 0
        self._log("[emg:weili] first frame received — ready")
        return True

    def _fail_open(self) -> bool:
        """Log the recorded failure, release the port, return False."""
        self._log(f"[emg:weili] open failed — {self._open_error}")
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._release_port_lock()
        return False

    def _release_port_lock(self) -> None:
        if self._lock is not None:
            _, fh = self._lock
            try:
                fh.close()   # closing releases flock / the msvcrt region
            except OSError:
                pass
            self._lock = None

    def _poll(self, ts):
        if self._ser is None:
            return
        chunk = self._ser.read(4096)
        if chunk:
            self._raw_buf.extend(chunk)
            self._raw_bytes_in += len(chunk)

        frames: list[dict] = []
        buf = self._raw_buf
        while len(buf) >= FRAME_LEN:
            idx = buf.find(HEADER)
            if idx < 0:
                # No header anywhere: keep only a possible trailing partial
                # header (<= 2 bytes) so garbage can't accumulate forever.
                if len(buf) > 2:
                    del buf[:-2]
                break
            if idx + FRAME_LEN > len(buf):
                break  # incomplete frame at end of buffer, wait for more data

            ptype = buf[idx + 3]
            # Cadence check: frames are packed back-to-back, so a real frame
            # is immediately followed by the next frame's header byte 0xD2.
            # A 0xD2 run inside payload data looks like a header but fails
            # either this check or the type check.
            next_ok = (idx + FRAME_LEN >= len(buf)
                       or buf[idx + FRAME_LEN] == HEADER[0])
            if ptype not in (EMG_TYPE, IMU_TYPE) or not next_ok:
                # False header match: advance ONE byte and re-scan, so a
                # real header hiding inside this window is never consumed.
                self._resyncs += 1
                del buf[:idx + 1]
                continue

            f = bytes(buf[idx:idx + FRAME_LEN])
            del buf[:idx + FRAME_LEN]

            sn = f[4]
            payload = f[5:]

            if self._last_sn is not None:
                expect = (self._last_sn + 1) & 0xFF
                if sn != expect:
                    self._dropped += (sn - expect) & 0xFF
            self._last_sn = sn

            if ptype == EMG_TYPE:
                frames.append({
                    "type": "emg", "ts": ts, "sn": sn,
                    "data": [_s24_be(payload[3 * c:3 * c + 3]) for c in range(8)],
                })
            else:
                frames.append({
                    "type": "imu", "ts": ts, "sn": sn,
                    "gyro": (
                        _s16_be(payload[0:2]) * GYRO_SCALE,
                        _s16_be(payload[2:4]) * GYRO_SCALE,
                        _s16_be(payload[4:6]) * GYRO_SCALE,
                    ),
                    "accel": (
                        _s16_be(payload[6:8]) * ACC_SCALE,
                        _s16_be(payload[8:10]) * ACC_SCALE,
                        _s16_be(payload[10:12]) * ACC_SCALE,
                    ),
                })

        for f in frames:
            if f["type"] == "emg":
                self._acc("timestamps", ts)
                self._acc("emg_sn", f["sn"])
                print(np.array(f["data"], dtype=np.int32))
                self._acc_arr("emg_data", np.array(f["data"], dtype=np.int32))
            else:
                self._acc("timestamps", ts)
                self._acc("imu_sn", f["sn"])
                self._acc_arr("imu_gyro", np.array(f["gyro"], dtype=np.float32))
                self._acc_arr("imu_accel", np.array(f["accel"], dtype=np.float32))

    def _close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._release_port_lock()

    def _heartbeat_stats(self, elapsed: float) -> str:
        return (super()._heartbeat_stats(elapsed)
                + f"  drop={self._dropped} resync={self._resyncs}")
