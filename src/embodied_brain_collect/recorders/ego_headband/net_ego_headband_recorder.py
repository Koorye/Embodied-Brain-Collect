"""EGO headband — network receiver (4 cameras + 2 IMUs).

The headband streams over the network (UDP by default).  This recorder owns
the socket lifecycle and demultiplexes incoming packets into per-camera
frames and per-IMU samples, routing them through the BaseRecorder
accumulators.

.. warning::
   The wire protocol (packet framing, byte order, image encoding) is
   device-specific.  ``_parse_packet()`` is the SINGLE place to implement it
   — everything else (socket handling, accumulation, NPZ output, heartbeat)
   is generic and should not need changes.

Output (under ``<session>/ego_headband/ego_headband.npz``)::

    cam{i}_timestamps   (T_i,)         float64   host receive clock
    cam{i}_frames       (T_i, H, W, 3) uint8     decoded RGB frame
    imu{j}_ts           (M_j,)         float64
    imu{j}_gyro         (M_j, 3)       float32   rad/s
    imu{j}_accel        (M_j, 3)       float32   m/s^2
"""

import socket
import numpy as np

from .base_ego_headband_recorder import BaseEgoHeadbandRecorder
from .ego_headband_recorder_config import EgoHeadbandRecorderConfig


class NetEgoHeadbandRecorder(BaseEgoHeadbandRecorder):
    """Real EGO headband over UDP/TCP.

    TODO(protocol): implement ``_parse_packet`` for the device's wire format.
    """

    config: EgoHeadbandRecorderConfig

    def __init__(self, config: EgoHeadbandRecorderConfig):
        super().__init__(config)
        self._sock: socket.socket | None = None
        self._rx = bytearray()                      # TCP reassembly buffer
        self._cam_count = [0] * config.n_cameras
        self._imu_count = [0] * config.n_imus

    # ---- lifecycle ------------------------------------------------------

    def _open(self) -> bool:
        cfg = self.config
        transport = (cfg.transport or "udp").lower()
        try:
            if transport == "udp":
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.bind((cfg.host, cfg.port))
            elif transport == "tcp":
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.connect((cfg.host, cfg.port))
            else:
                self._open_error = f"unsupported transport {cfg.transport!r}"
                self._log(f"[ego_headband:net] open failed — {self._open_error}")
                return False
        except OSError as exc:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
            self._open_error = (f"cannot open {transport}://{cfg.host}:{cfg.port}"
                                f" — {type(exc).__name__}: {exc}")
            self._log(f"[ego_headband:net] open failed — {self._open_error}")
            return False

        self._sock.settimeout(0.005)
        self._log(f"[ego_headband:net] listening on {transport}://"
                  f"{cfg.host}:{cfg.port} "
                  f"({cfg.n_cameras} cams + {cfg.n_imus} IMUs)")
        return True

    def _close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        cams = sum(self._cam_count)
        imus = sum(self._imu_count)
        self._log(f"[ego_headband:net] stopped (cam_frames={cams} imu={imus})")

    # ---- polling --------------------------------------------------------

    def _poll(self, ts: float) -> None:
        assert self._sock is not None
        transport = (self.config.transport or "udp").lower()
        try:
            if transport == "udp":
                data, _ = self._sock.recvfrom(65535)
                if data:
                    self._handle_packet(data, ts)
            else:  # tcp — byte stream, needs framing
                chunk = self._sock.recv(65535)
                if not chunk:
                    return
                self._rx.extend(chunk)
                for pkt in self._split_tcp_frames():
                    self._handle_packet(pkt, ts)
        except (socket.timeout, BlockingIOError):
            pass
        except OSError:
            pass

    def _split_tcp_frames(self) -> list[bytes]:
        """TODO(protocol): extract complete packets from ``self._rx``.

        Split on the device's framing (length prefix / delimiter), delete the
        consumed bytes from ``self._rx`` and return the complete packets.
        Placeholder returns nothing until the framing is known.
        """
        return []

    # ---- wire format (device-specific) ----------------------------------

    def _parse_packet(self, data: bytes) -> list[dict]:
        """Decode one network packet into a list of events.

        Each event is a dict::

            {"kind": "cam", "index": int, "frame": np.ndarray(H, W, 3) uint8}
            {"kind": "imu", "index": int,
             "gyro": (gx, gy, gz), "accel": (ax, ay, az)}

        TODO(protocol): implement the real decode here.  The default drops
        every packet so the recorder can be wired up and smoke-tested before
        the format is finalised.
        """
        return []

    # ---- routing --------------------------------------------------------

    def _handle_packet(self, data: bytes, ts: float) -> None:
        for ev in self._parse_packet(data):
            kind = ev.get("kind")
            if kind == "cam":
                self._route_cam(ev, ts)
            elif kind == "imu":
                self._route_imu(ev, ts)

    def _route_cam(self, ev: dict, ts: float) -> None:
        i = int(ev.get("index", 0))
        if not (0 <= i < self.config.n_cameras):
            return
        self._acc_ts(f"cam{i}", ts)
        frame = ev.get("frame")
        if isinstance(frame, np.ndarray):
            self._acc_arr(f"cam{i}_frames", frame)
        # TODO(protocol): if the device sends JPEG/H.264, decode to an ndarray
        # above (cv2.imdecode) or store the encoded bytes instead.
        self._cam_count[i] += 1

    def _route_imu(self, ev: dict, ts: float) -> None:
        j = int(ev.get("index", 0))
        if not (0 <= j < self.config.n_imus):
            return
        self._acc(f"imu{j}_ts", ts)
        self._acc_arr(f"imu{j}_gyro",
                      np.asarray(ev.get("gyro", (0.0, 0.0, 0.0)), dtype=np.float32))
        self._acc_arr(f"imu{j}_accel",
                      np.asarray(ev.get("accel", (0.0, 0.0, 0.0)), dtype=np.float32))
        self._imu_count[j] += 1

    # ---- heartbeat ------------------------------------------------------

    def _heartbeat_stats(self, elapsed: float) -> str:
        cams = sum(self._cam_count)
        imus = sum(self._imu_count)
        fps = cams / elapsed if elapsed > 0 else 0
        return (f"cam={cams:>5} ({fps:.1f}/s)  imu={imus:>5}  "
                f"percam={self._cam_count} perimu={self._imu_count}")
