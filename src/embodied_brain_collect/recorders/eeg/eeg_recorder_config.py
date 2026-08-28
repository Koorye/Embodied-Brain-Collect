"""EEG recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class EegRecorderConfig(BaseRecorderConfig):
    host: str = "127.0.0.1"      # Curry NetStream TCP host
    port: int = 4455             # Curry NetStream TCP port
    marker_wait_s: float = 10.0  # poll markers/markers.npz this long in _close
                                 # for the EEG<->PC alignment
    dummy_events: str = ""       # dummy 模式的事件节奏:"" = 默认 1 码/秒;
                                 # "sync_test" = 精确复刻 sync_test 的节奏
