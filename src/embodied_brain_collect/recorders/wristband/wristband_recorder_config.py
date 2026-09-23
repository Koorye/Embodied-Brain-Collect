"""Configuration for the physiological wristband recorder."""

from dataclasses import dataclass

from ..base import BaseRecorderConfig


# Nordic UART Service (NUS) UUIDs supplied by the mobile App engineer.
DEFAULT_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


@dataclass
class WristbandRecorderConfig(BaseRecorderConfig):
    """Connection knobs for one wristband.

    ``address`` is the safest selector when several wristbands are nearby.
    On macOS Bleak exposes a CoreBluetooth identifier rather than a public MAC
    address; copying the identifier printed by the hardware test is supported.
    If neither address nor name is supplied, discovery matches the advertised
    service UUID.
    """

    device_name: str = ""
    address: str = ""
    service_uuid: str = DEFAULT_SERVICE_UUID
    write_uuid: str = DEFAULT_WRITE_UUID
    notify_uuid: str = DEFAULT_NOTIFY_UUID

    scan_timeout: float = 10.0
    connect_timeout: float = 10.0

    # 扫描→连接→0x20 对时作为整体在 open 阶段最多尝试的次数(含首次)。
    # 手环常处于 RSSI 边缘,建立中的链路掉一次后设备几秒内就会重新广播;
    # 不重试的话一次瞬时掉线就废掉整条录制任务。
    open_attempts: int = 3

    # The device transmits the previous second's 50 frames as one 1 Hz burst.
    # Keep listening briefly after stop so the burst covering the session's
    # final second actually arrives before disconnect.
    tail_wait: float = 1.5
