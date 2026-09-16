"""EGO headband recorder config.

The headband streams over the network and carries 4 cameras + 2 IMUs.
``host``/``port``/``transport`` describe the local endpoint this recorder
binds (UDP server) or connects (TCP client) to.
"""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class EgoHeadbandRecorderConfig(BaseRecorderConfig):
    # ---- network ----
    host: str = "0.0.0.0"      # bind/connect address for the incoming stream
    port: int = 5555           # device stream port
    transport: str = "udp"     # "udp" | "tcp"

    # ---- cameras (4) ----
    n_cameras: int = 4
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: float = 30.0

    # ---- IMUs (2) ----
    n_imus: int = 2
    imu_rate_hz: float = 200.0
