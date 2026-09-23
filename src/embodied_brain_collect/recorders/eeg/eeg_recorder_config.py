"""EEG recorder configs."""
from dataclasses import dataclass, field
from ..base import BaseRecorderConfig


def _intan_default_map() -> dict:
    """猴台架接线的字型映射,目标码取 markers.yaml 码表的边界对当前值。"""
    from ...stim import marker_codes as M
    return {"0x4000": M.RUN_START, "0x2000": M.RUN_END}


@dataclass
class EegRecorderConfig(BaseRecorderConfig):
    """Curry NetStream(以及共用协议形状的 dummy)。"""
    host: str = "127.0.0.1"      # Curry NetStream TCP host
    port: int = 4455             # Curry NetStream TCP port
    marker_wait_s: float = 10.0  # poll markers/markers.npz this long in _close
                                 # for the EEG<->PC alignment
    dummy_events: str = ""       # dummy 模式的事件节奏:"" = 默认 1 码/秒;
                                 # "sync_test" = 精确复刻 sync_test 的节奏
    # ---- 开录阻抗门禁(主动触发;Curry 专用,dummy 忽略)----
    # open 时经 NetStream 12/13 触发一次阻抗检测,取均值做通过率检查。
    # 时序红线(实测违反会把放大器驱动打成 Device Error,只能重启
    # Curry/放大器):阻抗结束后放大器回 connect 态、~10s 才恢复读数,
    # 恢复前绝不能断开连接 —— recorder 的所有失败路径也都先等恢复再断开;
    # 也不要发 AmpConnect(10),会与放大器自己的恢复相撞。
    impedance_check: bool = True   # open 时触发一次阻抗检测并检查;不达标
                                   # 直接拒开并提示超标通道
    impedance_max_kohm: float = 100.0  # 单通道通过阈值:阻抗 < 该值算通过
    impedance_pass_rate: float = 0.87  # 通过率下限;低于则 open 失败并提示
    impedance_edge_channels: list[str] = field(default_factory=lambda: [
        # 头围/耳后/下颌等天然高阻的边缘电极,不参与通过率统计
        "M1", "P9", "PO9", "Cb1", "Cbz", "Cb2", "PO10", "P10", "M2",
        "FT10", "FT12", "F10", "F12", "FT11", "FT9", "F11", "F9",
    ])


@dataclass
class BraincoEegRecorderConfig(BaseRecorderConfig):
    """BrainCo BCIGo 专用(软件同步,无硬件 TTL/阻抗概念)。"""
    host: str = "127.0.0.1"      # 设备发现/连接地址(mDNS 扫描的过滤 hint)
    port: int = 13800            # 设备数据端口
    sample_rate: float = 250.0   # 250 / 500 / 1000 / 2000 Hz
    gain: int = 6                # 1, 2, 4, 6, 8, 12, 24
    signal: str = "normal"       # normal / test_signal
    scan_timeout_s: float = 10.0  # mDNS/scan 发现设备的等待时间
    sync_udp_port: int = 0       # >0 时绑定该口收 stim 抄送的软件触发
                                 # (在线心跳插值;落盘仍以包时钟拟合为准),
                                 # 与 stim.yaml 的 sync_udp_port 一致
    device_sn: str = ""          # 非空则只连该序列号的帽子


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
    # 字→码映射表(掩码后再查表;查不到不发事件)。接线只通个别位时用它把
    # 字型翻译回 marker 码。默认 = 猴台架实测接线(box bit4→DIN14,
    # bit5→DIN13)映射到 **markers.yaml 码表的 RUN_START/RUN_END 当前值**
    # —— 两线接法请把 markers.yaml 的 run_start/run_end 也设成单比特码
    # (16/32),两边才自洽;接线改了或换台架在 recorders.yaml 里覆盖,
    # 全 8 位接线时删掉此表即可。
    digital_map: dict = field(
        default_factory=lambda: _intan_default_map())
    set_runmode: bool = True     # 录制时 set runmode run,收尾 stop——会一并
                                 # 停掉 RHX 自身的 Record,慎改
    start_data_server: bool = True  # 允许 recorder 通过命令口启动 TCP 波形服务器
    marker_wait_s: float = 10.0
