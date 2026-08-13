"""Weili (WAVELETECH) 8-channel EMG — real serial port, 29-byte frames."""

import struct

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
            self._log(f"[emg:weili] open failed — {self._open_error}. Ports:")
            for p in list_ports.comports():
                self._log(f"  - {p.device}: {p.description} [{p.hwid}]")
            return False

        self._log(f"[emg:weili] open {port} @ {self.config.baud}")
        self._ser = serial.Serial(port, self.config.baud, timeout=0.005)
        self._ser.reset_input_buffer()
        return True

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
                self._acc("timestamps", ts)
                self._acc("emg_sn", f["sn"])
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

    def _heartbeat_stats(self, elapsed: float) -> str:
        n_emg = len(self._buf.get("emg_sn", []))
        n_imu = len(self._buf.get("imu_sn", []))
        return (
            f"emg={n_emg:>5} ({n_emg/elapsed:.1f}/s)  "
            f"imu={n_imu:>5}  drop={self._dropped}"
        )
