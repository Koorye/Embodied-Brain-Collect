"""Byte-level parser for the physiological wristband BLE protocol V1.5.

The BLE notification boundary is not an application-frame boundary: one
60-byte 0x15 frame may be split across notifications, and one notification may
also contain several frames.  ``FrameStreamDecoder`` therefore treats incoming
notifications as an arbitrary byte stream before the typed parsers below are
called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


FRAME_HEADER = 0xAA
FRAME_TAIL = 0xCC
DATA_FRAME_TYPE = 0x15
STATUS_FRAME_TYPE = 0x17
TIME_FRAME_TYPE = 0x20

DATA_FRAME_LENGTH = 60
SHORT_FRAME_LENGTH = 14
FRAME_LENGTHS = {
    DATA_FRAME_TYPE: DATA_FRAME_LENGTH,
    STATUS_FRAME_TYPE: SHORT_FRAME_LENGTH,
    TIME_FRAME_TYPE: SHORT_FRAME_LENGTH,
}

INVALID_TIMESTAMP_MS = 0xFFFFFFFFFFFFFFFF
INVALID_TEMPERATURE_RAW = 0xFFFF
INVALID_PERCENT_RAW = 0xFF

# The firmware engineer supplied these fixed-range conversion formulae.
# Raw values are always stored as well, so a future range change is recoverable.
ACCEL_SCALE_M_S2 = 9.807 / 16384.0
GYRO_SCALE_RAD_S = math.pi / (128.0 * 180.0)


@dataclass(frozen=True)
class DataFrame:
    """One complete 0x15 application frame."""

    pressure_raw: tuple[int, int, int]
    ppg_raw: tuple[tuple[int, int, int], tuple[int, int, int]]
    accel_raw: tuple[int, int, int]
    gyro_raw: tuple[int, int, int]
    temperature_raw: int
    spo2_raw: int
    timestamp_ms: int
    sequence: int
    raw: bytes

    @property
    def timestamp_valid(self) -> bool:
        # A broad fixed range catches all-FF/uninitialised values without tying
        # parsing to the host computer's current clock.
        return 946_684_800_000 <= self.timestamp_ms < 4_102_444_800_000

    @property
    def temperature_c(self) -> float | None:
        if self.temperature_raw == INVALID_TEMPERATURE_RAW:
            return None
        return self.temperature_raw / 100.0

    @property
    def spo2_percent(self) -> float | None:
        if self.spo2_raw == INVALID_PERCENT_RAW:
            return None
        return float(self.spo2_raw)

    @property
    def accel_m_s2(self) -> tuple[float, float, float]:
        return tuple(v * ACCEL_SCALE_M_S2 for v in self.accel_raw)

    @property
    def gyro_rad_s(self) -> tuple[float, float, float]:
        return tuple(v * GYRO_SCALE_RAD_S for v in self.gyro_raw)


@dataclass(frozen=True)
class StatusFrame:
    """One complete 0x17 battery/storage frame."""

    battery_raw: int
    storage_remaining_raw: int
    raw: bytes

    @property
    def battery_percent(self) -> float | None:
        if self.battery_raw == INVALID_PERCENT_RAW:
            return None
        return float(self.battery_raw)

    @property
    def storage_remaining_percent(self) -> float | None:
        if self.storage_remaining_raw == INVALID_PERCENT_RAW:
            return None
        return float(self.storage_remaining_raw)


@dataclass(frozen=True)
class TimeResponseFrame:
    """A 0x20 response.  Success is checked against the sent command."""

    raw: bytes

    @property
    def is_failure(self) -> bool:
        return self.raw[2:12] == b"\xFF" * 10


ParsedFrame = DataFrame | StatusFrame | TimeResponseFrame


class FrameStreamDecoder:
    """Reassemble fixed-length V1.5 frames from arbitrary byte chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.invalid_frames = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.discarded_bytes = 0
        self.invalid_frames = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[bytes]:
        if chunk:
            self.buffer.extend(chunk)

        frames: list[bytes] = []
        while True:
            header_index = self.buffer.find(bytes((FRAME_HEADER,)))
            if header_index < 0:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if header_index:
                self.discarded_bytes += header_index
                del self.buffer[:header_index]

            if len(self.buffer) < 2:
                break

            frame_length = FRAME_LENGTHS.get(self.buffer[1])
            if frame_length is None:
                self.invalid_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < frame_length:
                break

            candidate = bytes(self.buffer[:frame_length])
            if candidate[-1] != FRAME_TAIL:
                # The apparent 0xAA was inside corrupt/noisy data.  Move one
                # byte and search again; payload bytes may themselves be 0xAA.
                self.invalid_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue

            frames.append(candidate)
            del self.buffer[:frame_length]

        return frames


def parse_frame(frame: bytes) -> ParsedFrame:
    """Parse a complete frame and reject an invalid length/header/tail."""

    if len(frame) < 2:
        raise ValueError("frame is shorter than header + type")
    frame_type = frame[1]
    expected_length = FRAME_LENGTHS.get(frame_type)
    if expected_length is None:
        raise ValueError(f"unsupported frame type 0x{frame_type:02X}")
    if len(frame) != expected_length:
        raise ValueError(
            f"0x{frame_type:02X} frame length {len(frame)} != {expected_length}"
        )
    if frame[0] != FRAME_HEADER or frame[-1] != FRAME_TAIL:
        raise ValueError("invalid frame header or tail")

    if frame_type == DATA_FRAME_TYPE:
        return _parse_data_frame(frame)
    if frame_type == STATUS_FRAME_TYPE:
        return StatusFrame(
            battery_raw=frame[2],
            storage_remaining_raw=frame[3],
            raw=frame,
        )
    return TimeResponseFrame(raw=frame)


def _parse_data_frame(frame: bytes) -> DataFrame:
    pressure = tuple(
        int.from_bytes(frame[offset : offset + 3], "big", signed=True)
        for offset in (2, 5, 8)
    )

    ppg_values = [
        int.from_bytes(frame[offset : offset + 4], "big", signed=True)
        for offset in range(11, 35, 4)
    ]
    ppg = (
        tuple(ppg_values[0:3]),
        tuple(ppg_values[3:6]),
    )

    imu_values = [
        int.from_bytes(frame[offset : offset + 2], "big", signed=True)
        for offset in range(35, 47, 2)
    ]
    sequence = frame[58]
    if sequence > 249:
        raise ValueError(f"0x15 sequence out of range: {sequence}")

    return DataFrame(
        pressure_raw=pressure,
        ppg_raw=ppg,
        accel_raw=tuple(imu_values[0:3]),
        gyro_raw=tuple(imu_values[3:6]),
        temperature_raw=int.from_bytes(frame[47:49], "big", signed=False),
        spo2_raw=frame[49],
        timestamp_ms=int.from_bytes(frame[50:58], "little", signed=False),
        sequence=sequence,
        raw=frame,
    )


def build_time_sync_frame(
    now: datetime | None = None,
    sequence: int = 0,
) -> bytes:
    """Build the 14-byte 0x20 UTC-setting command defined by protocol V1.5."""

    if not 0 <= sequence <= 249:
        raise ValueError("time-command sequence must be in 0..249")
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        raise ValueError("time sync datetime must be timezone-aware")
    now = now.astimezone(timezone.utc)

    year_offset = now.year - 1970
    if not 0 <= year_offset <= 255:
        raise ValueError("year cannot be represented as uint8(year - 1970)")
    milliseconds = now.microsecond // 1000

    return bytes(
        (
            FRAME_HEADER,
            TIME_FRAME_TYPE,
            year_offset,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            (milliseconds >> 8) & 0xFF,
            milliseconds & 0xFF,
            0x00,
            0x00,
            sequence,
            FRAME_TAIL,
        )
    )
