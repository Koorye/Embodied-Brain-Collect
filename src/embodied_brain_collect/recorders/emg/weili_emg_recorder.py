"""Weili (WAVELETECH) 8-channel EMG — real serial port, 29-byte frames.

EMG and IMU arrive as separate frame types on one wire, interleaved and
sharing a single 8-bit rolling sequence number.

Output (under <session>/<emg_left|emg_right>/*.npz)::

    emg_timestamps          (N_emg,)     float64  per-frame, fitted from *_sn
    emg_arrival_timestamps  (N_emg,)     float64  PC clock at the carrying read
    emg_sn                  (N_emg,)     int      wire sequence number
    emg_data                (N_emg, 8)   int32
    imu_timestamps          (N_imu,)     float64
    imu_arrival_timestamps  (N_imu,)     float64
    imu_sn                  (N_imu,)     int
    imu_gyro                (N_imu, 3)   float32
    imu_accel               (N_imu, 3)   float32

``*_arrival_timestamps`` are what the poll loop saw: one ``Serial.read()``
returns every frame the driver has buffered, so all of them share that read's
timestamp.  The wire rate is far above the read cadence, so those series are
mostly runs of identical values, each frame carrying up to a batch-period of
latency.  ``*_timestamps`` are per-frame times fitted from the sequence number
at close — see ``timestamp_rebuild``.  If the fit is refused, both series hold
arrival times and the ``*_arrival_*`` fields are absent.

Both frame types share one sequence number that advances once per transmitted
frame, so consecutive EMG frames legitimately step by 2 when an IMU frame sits
between them.  Counting every step != 1 as a gap is wrong; unwrap one stream's
``*_sn`` to get the frames the device sent, and subtract ``N_emg + N_imu`` to
get the frames actually lost.
"""

import struct
import time

import numpy as np
import serial
from serial.tools import list_ports

from .base_emg_recorder import BaseEmgRecorder
from .emg_recorder_config import EmgRecorderConfig
from .timestamp_rebuild import rebuild

HEADER = b"\xD2\xD2\xD2"
FRAME_LEN = 29
EMG_TYPE = 0xAA
IMU_TYPE = 0xBB
GYRO_SCALE = 0.0012
ACC_SCALE = 0.0005978


def _s24_be(b):
    v = (b[0] << 16) | (b[1] << 8) | b[2]
    return v - (1 << 24) if v & 0x800000 else v


def _s16_be(b):
    return struct.unpack(">h", b)[0]


def _auto_detect_port() -> str | None:
    for p in list_ports.comports():
        h = (p.hwid or "").upper()
        d = p.description or ""
        tags = ("VID:PID=10C4", "CP210", "Silicon Labs",
                "VID:PID=1A86:55D3", "CH343")
        if any(t in h or t in d for t in tags):
            return p.device
    


class WeiliEmgRecorder(BaseEmgRecorder):
    """Real WAVELETECH EMG armband over CP210x USB-UART."""

    config: EmgRecorderConfig

    def __init__(self, config: EmgRecorderConfig):
        super().__init__(config)
        self._ser: serial.Serial | None = None
        self._last_sn: int | None = None
        self._dropped = 0
        self._raw_buf = bytearray()

    # ---- lifecycle ----------------------------------------------------------

    def _open(self) -> bool:
        port = self.config.port or _auto_detect_port()
        if not port:
            self._open_error = "no supported EMG serial adapter found"
            self._log(f"[emg:weili] open failed — {self._open_error}")
            return False

        self._log(f"[emg:weili] open {port} @ {self.config.baud} — "
                  f"waiting for first frame ...")
        self._ser = serial.Serial(port, self.config.baud, timeout=0.005)
        self._ser.reset_input_buffer()

        def _try_poll() -> bool:
            self._poll(time.time())
            return bool(self._buf.get("emg_sn"))

        if not self._wait_first_sample(_try_poll, "EMG frame", timeout=5.0):
            self._ser.close()
            self._ser = None
            return False
        # Gate samples used wall-clock ts; clear so the session timeline
        # starts clean from the launcher's t0.
        self._buf.clear()
        self._arr_buf.clear()
        self._raw_buf.clear()
        self._log("[emg:weili] first frame received — ready")
        return True

    def _setup(self) -> None:
        # The port opens seconds before the launcher starts the run (device
        # discovery, other recorders' open gates), and the armband streams
        # the whole time with nobody reading.  By now the driver buffer holds
        # a backlog of pre-run frames — and has already overflowed, dropping
        # the oldest.  Keeping it would file seconds of stale signal under
        # the run's first few milliseconds, right where RUN_START lands.
        super()._setup()
        if self._ser is None:
            return
        pending = self._ser.in_waiting
        self._ser.reset_input_buffer()
        self._raw_buf.clear()
        self._last_sn = None  # resync after the flush is not a dropped frame
        if pending:
            self._log(f"[emg:weili] discarded {pending} B "
                      f"(~{pending // FRAME_LEN} frames) buffered before the run")

    def _poll(self, ts):
        assert self._ser is not None
        chunk = self._ser.read(4096)
        if chunk:
            self._raw_buf.extend(chunk)

        frames: list[dict] = []
        while len(self._raw_buf) >= FRAME_LEN:
            idx = self._raw_buf.find(HEADER)
            if idx < 0:
                break
            if idx + FRAME_LEN > len(self._raw_buf):
                break  # incomplete frame at end of buffer, wait for more data
            f = bytes(self._raw_buf[idx:idx + FRAME_LEN])
            del self._raw_buf[:idx + FRAME_LEN]

            ptype = f[3]
            sn = f[4]
            payload = f[5:]
            if ptype not in (EMG_TYPE, IMU_TYPE):
                continue

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
                self._acc("emg_timestamps", ts)
                self._acc("emg_sn", f["sn"])
                self._acc_arr("emg_data", np.array(f["data"], dtype=np.int32))
            else:
                self._acc("imu_timestamps", ts)
                self._acc("imu_sn", f["sn"])
                self._acc_arr("imu_gyro", np.array(f["gyro"], dtype=np.float32))
                self._acc_arr("imu_accel", np.array(f["accel"], dtype=np.float32))

    def _close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._rebuild_timestamps()

    def _rebuild_timestamps(self) -> None:
        """Swap in per-frame timestamps fitted from the sequence number,
        keeping the read-arrival series alongside.

        Here rather than per-poll on purpose: the fit wants the whole run's
        lever arm (3 ppm over 25 s, versus ~40 ppm from a few seconds), and a
        per-batch refit lets the line wander between batches — which trades
        duplicate timestamps for *backwards* ones.  Runs after the port is
        closed and before ``_save``; a refusal leaves the arrival series in
        place, since a cosmetic fix must never cost a recording.
        """
        arrival_emg = self._buf.get("emg_timestamps")
        if not arrival_emg:
            return
        arrival_imu = self._buf.get("imu_timestamps") or []
        r = rebuild(arrival_emg, self._buf.get("emg_sn") or [],
                    arrival_imu, self._buf.get("imu_sn") or [])
        self._log(f"[emg:weili] {r.summary()}",
                  level="INFO" if r.ok else "WARNING")
        if not r.ok:
            return

        self._buf["emg_arrival_timestamps"] = arrival_emg
        self._buf["emg_timestamps"] = r.emg_timestamps.tolist()
        if arrival_imu:
            self._buf["imu_arrival_timestamps"] = arrival_imu
            self._buf["imu_timestamps"] = r.imu_timestamps.tolist()

    def _heartbeat_stats(self, elapsed: float) -> str:
        return super()._heartbeat_stats(elapsed) + f"  drop={self._dropped}"
