"""EMG recorder config."""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class EmgRecorderConfig(BaseRecorderConfig):
    port: str = ""
    baud: int = 921600
    link_timeout: float = 1.0  # 链路静默判死:出过字节后连续这么久一个字节都
                               # 没收到 → 连接已死(臂环断电/USB 松动时 serial
                               # read 只是返回空,不会有任何异常)。首次字节到
                               # 达前不武装(慢启动交给 launcher 确认阶段)。
                               # 0 = 关闭
