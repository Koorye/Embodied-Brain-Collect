"""UdpMarkerRecorder — UDP listener, saves markers/markers.npz.

Parses the same wire format sync_hub understands (see ``src/sync/sync_hub``)::

    EVT|trial=<int>|tag=<NAME>|code=<int>|t_eprime_ms=<int>
"""

import socket
from .base_marker_recorder import BaseMarkerRecorder
from .marker_recorder_config import MarkerRecorderConfig


def _parse_evt(data: bytes) -> dict | None:
    """Parse an ``EVT|...`` marker packet into a dict, or None if invalid."""
    try:
        text = data.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if not text.startswith("EVT|"):
        return None

    fields: dict[str, str] = {}
    for tok in text.split("|")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        fields[k.strip()] = v.strip()

    try:
        sent = fields.get("t_sent_pc")
        return {
            "trial": int(fields.get("trial", -1)),
            "tag": fields.get("tag", "?"),
            "code": int(fields.get("code", -1)),
            "t_eprime_ms": int(fields.get("t_eprime_ms", -1)),
            # 发送端 PC 时刻(新 sender);旧包没有则 None,由调用方回退
            "t_sent_pc": float(sent) if sent else None,
        }
    except ValueError:
        return None


class UdpMarkerRecorder(BaseMarkerRecorder):
    """Listens on UDP for markers from stim, saves as NPZ."""

    config: MarkerRecorderConfig

    def __init__(self, config: MarkerRecorderConfig):
        super().__init__(config)
        self._sock: socket.socket | None = None

    def _open(self) -> bool:
        cfg = self.config
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((cfg.host, cfg.port))
        except OSError as exc:
            self._sock.close()
            self._sock = None
            self._open_error = (f"cannot bind udp://{cfg.host}:{cfg.port} — "
                                f"{type(exc).__name__}: {exc}")
            self._log(f"[marker:udp] open failed — {self._open_error}")
            return False
        self._sock.settimeout(0.001)
        self._log(f"[marker:udp] listening on udp://{cfg.host}:{cfg.port}")
        return True

    def _close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None
        n = len(self._buf.get("marker_tag", []))
        self._log(f"[marker:udp] stopped ({n} markers)")

    def probe_data_flow(self, timeout: float = 1.0) -> tuple[bool, str]:
        # UDP 只有 stim 发 marker 才有数据,预检时发送端不在 —— 端口绑定
        # 成功即为就绪,不做数据流探测。
        return True, "端口绑定成功(marker 由 stim 发送,预检不验证数据流)"

    def _poll(self, ts: float) -> None:
        assert self._sock is not None
        try:
            data, _ = self._sock.recvfrom(4096)
            evt = _parse_evt(data)
            if evt is None:
                return
            self._acc("marker_trial", evt["trial"])
            self._acc("marker_tag", evt["tag"])
            self._acc("marker_code", evt["code"])
            self._acc("marker_t_eprime_ms", evt["t_eprime_ms"])
            # 发送端时间戳是权威;缺省(旧 sender 包)时用接收时刻填充,
            # 保证数组与 marker_code 等长。
            self._acc("marker_t_sent_pc",
                      evt["t_sent_pc"] if evt["t_sent_pc"] is not None else ts)
            self._acc("marker_t_local_recv", ts)
        except (socket.timeout, BlockingIOError):
            pass
