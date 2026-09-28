"""Position tracker config."""
from dataclasses import dataclass, field
from ..base import BaseRecorderConfig


@dataclass
class PositionRecorderConfig(BaseRecorderConfig):
    device_classes: str = "tracker"  # comma-separated device classes
    expected_devices: int = 3        # open 时已连接 tracker 少于该数 → open 失败,
                                     # 录制不开始(少连一台是最常见的漏采)
    role_serial_map: dict = field(default_factory=dict)
    # 角色→设备绑定,如 {"left_wrist": "61-BH3702186", ...}。匹配值两种写法
    # 任选:OpenVR 序列号(61-BH…,推荐,永不变化)或 VIVE Hub 里设置的
    # 角色(vive_tracker_chest / vive_tracker_left_elbow …,在 Hub 里改角色
    # 会跟着变)。注意 VIVE Hub 显示的设备 ID(FA61…)不是序列号,匹配不到。
    # 配置后 npz 的列序 = 此映射的书写顺序(会话间稳定),开录时逐台校验:
    # 缺一台、多一台未绑定、两角色配同一设备,都直接拒绝开录。序列号在
    # open 日志设备清单行里(serial=…)。不配 = 按枚举顺序排列 —— 那个顺序
    # 跟设备开机先后走,会话间会漂移,左右手可能互换。
    require_all_valid: bool = True
    # 录制中所有 tracker 必须始终有效:任一 tracker 出现无效 pose(遮挡/
    # 掉线/断光塔)即运行期错误 —— 录制立即收摊(launcher 端按异常退出
    # 处理:其余模态落盘、跳过 QC 直接进 n/r/f/q 选择)。launcher 预热段
    # (commit 前)豁免 —— 那段数据本来就会被丢弃。QC 侧 ValidAlways
    # 用同一口径复核落盘数据。false = 关闭看门狗(不推荐)。
