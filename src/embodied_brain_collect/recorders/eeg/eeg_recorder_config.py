"""EEG recorder configs."""
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


@dataclass
class BlackrockEegRecorderConfig(BaseRecorderConfig):
    """Blackrock Cerebus/NeuroPort via pycbsdk (CereLink)。

    连接地址不可配:pycbsdk 10.x 的 Session 走 cbSDK 标准自动发现
    (NSP 默认 192.168.137.x 子网,或本机 Central)。
    """
    device_type: str = "LEGACY_NSP"  # LEGACY_NSP / NSP / HUB1 / HUB2 / HUB3 / NPLAY
    sample_group: int = 0        # 0 = 自动选前端通道最多的采样组;
                                 # 1-6 = 指定组(=500/1k/2k/10k/30k/raw)
    auto_enable_group: int = 0   # 0 = 不改设备配置(缺组时 open 失败并提示);
                                 # 设为 500/1000/2000/10000/30000 时,把全部
                                 # 前端通道设到该速率组(等效在 Central 里配置)
    digital_mask: int = 0xFFFF     # 事件码 = 数字输入字 & mask;用于剥掉
                                   # 空闲电平基线(如 NSP 数字口的 0xF988,
                                   # 码叠在 0x88 上 → mask=0x0077)
    marker_wait_s: float = 10.0
    callback_queue_depth: int = 16384  # pycbsdk 回调队列深度(包数)


@dataclass
class IntanEegRecorderConfig(BaseRecorderConfig):
    """Intan RHD/RHS via RHX 软件的 TCP 接口。

    前提:RHX 软件里已启用 TCP Command Interface(Settings 菜单,一次性);
    波形数据服务器可以由 recorder 通过命令口自行拉起。
    """
    host: str = "127.0.0.1"
    command_port: int = 5000     # RHX TCP Command Interface 端口
    data_port: int = 5001        # RHX TCP 波形数据端口
    ports: str = "A"             # 放大器端口字母,如 "A" / "AB" / "ABCD"
    channels_per_port: int = 32  # 每端口使能的放大器通道数(由头戴决定,如 32/64)
    channels: str = ""           # 显式通道列表,如 "A-001,A-002,...";给出时
                                 # 覆盖 ports/channels_per_port
    digital_in: int = 1          # 使能的 DIGITAL-IN 通道数;>0 时帧尾附带完整
                                 # 16 位数字输入字(TTL/marker 通路,ParallelBox 接这里)
    digital_mask: int = 0xFFFF   # 事件码 = 数字字 & mask;剥空闲基线用
                                 # (接线只通部分位时码会碰撞,mask 救不了)
    digital_map: dict = None     # 字→码映射表(掩码后再查表;查不到不发事件)。
                                 # 接线只通个别位时用它把字型翻译回 marker 码,
                                 # 如 {"0x4000": 16, "0x2000": 32}
    set_runmode: bool = True     # 录制时 set runmode run,收尾 stop——会一并
                                 # 停掉 RHX 自身的 Record,慎改
    start_data_server: bool = True  # 允许 recorder 通过命令口启动 TCP 波形服务器
    marker_wait_s: float = 10.0
