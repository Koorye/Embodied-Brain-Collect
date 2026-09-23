"""Recorder 工厂 —— ``kind``(recorders.yaml)→ 类 的唯一注册表。

每个条目是 ``(子模块, Recorder 类名, Config 类名)``,**按需导入**:SDK
类(neon/openvr/manus/pycbsdk/pyrealsense2/depthai…)只在对应 kind 被构造
时才 import —— 没装某个 SDK 的机器不影响其他模态的构建与预检。

构建本身不需要每模态一个函数:Config dataclass 的字段默认值就是默认
配置,yaml 里写了的键覆盖、没写的键走 dataclass 默认,直接
``Config(session_dir=..., duration=..., **yaml_params)`` 即可。
dict → tuple 之类的归一化归 Config 自己(``__post_init__``)管。
本模块同时提供 yaml 编排入口(:func:`build_recorder` /
:func:`get_production_recorders` / :func:`get_dummy_recorders`)——
按 recorders.yaml 批量构建、enabled 过滤、槽位命名与 dummy 捆绑。
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from .base import BaseRecorder

#: kind -> (recorders 下子模块路径, Recorder 类名, Config 类名)
REGISTRY: dict[str, tuple[str, str, str]] = {
    # 相机
    "opencv_camera":   ("camera.opencv_camera_recorder",
                        "OpencvCameraRecorder", "OpencvCameraConfig"),
    "depthai_camera":  ("camera.depthai_camera_recorder",
                        "DepthaiCameraRecorder", "DepthaiCameraConfig"),
    "realsense_camera": ("camera.realsense_camera_recorder",
                         "RealsenseCameraRecorder", "RealsenseCameraConfig"),
    "dummy_camera":    ("camera.dummy_camera_recorder",
                        "DummyCameraRecorder", "CameraRecorderConfig"),
    # EEG
    "curry_eeg":       ("eeg.curry_eeg_recorder",
                        "CurryEegRecorder", "EegRecorderConfig"),
    "brainco_eeg":     ("eeg.brainco_eeg_recorder",
                        "BrainCoEegRecorder", "BraincoEegRecorderConfig"),
    "blackrock_eeg":   ("eeg.blackrock_eeg_recorder",
                        "BlackrockEegRecorder", "BlackrockEegRecorderConfig"),
    "intan_eeg":       ("eeg.intan_eeg_recorder",
                        "IntanEegRecorder", "IntanEegRecorderConfig"),
    "dummy_eeg":       ("eeg.dummy_eeg_recorder",
                        "DummyEegRecorder", "EegRecorderConfig"),
    # EMG
    "weili_emg":       ("emg.weili_emg_recorder",
                        "WeiliEmgRecorder", "EmgRecorderConfig"),
    "dummy_emg":       ("emg.dummy_emg_recorder",
                        "DummyEmgRecorder", "EmgRecorderConfig"),
    # 眼动
    "neon_eye":        ("eye.neon_eye_recorder",
                        "NeonEyeRecorder", "EyeRecorderConfig"),
    "neon_eye_async":  ("eye.neon_eye_async_recorder",
                        "NeonEyeAsyncRecorder", "EyeRecorderConfig"),
    "dummy_eye":       ("eye.dummy_eye_recorder",
                        "DummyEyeRecorder", "EyeRecorderConfig"),
    # 手部姿态
    "manus_hand_pose": ("hand_pose.manus_hand_pose_recorder",
                        "ManusHandPoseRecorder", "HandPoseRecorderConfig"),
    "dummy_hand_pose": ("hand_pose.dummy_hand_pose_recorder",
                        "DummyHandPoseRecorder", "HandPoseRecorderConfig"),
    # 位置
    "openvr_position": ("position.openvr_position_recorder",
                        "OpenvrPositionRecorder", "PositionRecorderConfig"),
    "dummy_position":  ("position.dummy_position_recorder",
                        "DummyPositionRecorder", "PositionRecorderConfig"),
    # marker
    "udp_marker":      ("marker.udp_marker_recorder",
                        "UdpMarkerRecorder", "MarkerRecorderConfig"),
    # 触觉
    "touchtronix_tactile": ("tactile.touchtronix_tactile_recorder",
                            "TouchtronixTactileRecorder",
                            "TactileRecorderConfig"),
    "dummy_tactile":   ("tactile.dummy_tactile_recorder",
                        "DummyTactileRecorder", "TactileRecorderConfig"),
    # 腕带
    "wristband":       ("wristband.wristband_recorder",
                        "WristbandRecorder", "WristbandRecorderConfig"),
    "dummy_wristband": ("wristband.dummy_wristband_recorder",
                        "DummyWristbandRecorder", "WristbandRecorderConfig"),
    # EGO 头环
    "net_ego_headband": ("ego_headband.net_ego_headband_recorder",
                         "NetEgoHeadbandRecorder", "EgoHeadbandRecorderConfig"),
    "dummy_ego_headband": ("ego_headband.dummy_ego_headband_recorder",
                           "DummyEgoHeadbandRecorder",
                           "EgoHeadbandRecorderConfig"),
}


def resolve_kind(kind: str) -> tuple[type, type]:
    """kind → (Recorder 类, Config 类);按需 import,未知 kind 报错。"""
    if kind not in REGISTRY:
        raise ValueError(f"未知 recorder kind: {kind!r} (可用: "
                         f"{sorted(REGISTRY)})")
    module_name, cls_name, cfg_name = REGISTRY[kind]
    module = import_module(f"embodied_brain_collect.recorders.{module_name}")
    return getattr(module, cls_name), getattr(module, cfg_name)


def build(kind: str, params: dict, session_dir: str | Path,
          duration: float = 0.0) -> BaseRecorder:
    """从 yaml 风格参数字典构建一条 recorder。

    ``params`` 是该槽位 yaml 里除 ``kind``/``enabled``/``name`` 外的全部
    键 —— 写了的覆盖 Config dataclass 默认,没写的走 dataclass 默认
    (含 ``default_factory``);dict→tuple 等归一化由 Config 的
    ``__post_init__`` 负责。
    """
    cls, cfg_cls = resolve_kind(kind)
    cfg = cfg_cls(session_dir=str(session_dir), duration=duration, **params)
    return cls(cfg)


# ======================================================================
# yaml 编排:recorders.yaml 的槽位 → recorder
# ======================================================================

#: 顶层配置键 —— 属于编排层,不进 Config dataclass
_META_KEYS = ("kind", "enabled", "name")

#: dummy 捆绑:槽位名 -> dummy kind("marker" 用真实 UDP listener 收 stim 码)
_DUMMY_SLOT_KINDS = {
    "camera": "dummy_camera",
    "eeg": "dummy_eeg",
    "emg": "dummy_emg",
    "eye": "dummy_eye",
    "hand_pose": "dummy_hand_pose",
    "position": "dummy_position",
    "marker": "udp_marker",
    "wristband": "dummy_wristband",
    "ego_headband": "dummy_ego_headband",
}


def _with_sensor_name(name: str, rec: BaseRecorder) -> BaseRecorder:
    """把 recorder 的输出目录指到槽位名:每个传感器落在
    ``{session_dir}/<name>/`` 而不是通用模态目录(``camera/``、``emg/``)。"""
    rec.set_output_dir(name)
    return rec


def build_recorder(slot: str, cfg: dict, session_dir: str | Path,
                   duration: float = 0.0) -> BaseRecorder:
    """一个 ``configs/recorders.yaml`` 槽位 → recorder。

    ``cfg`` 是 ``{kind, name?, enabled?, ...params}``;除 ``kind`` /
    ``enabled`` / ``name`` 外的键全部进 Config dataclass(写了覆盖默认,
    没写走 dataclass 默认)。``name`` 是可选的设备显示名,落进 session 的
    meta.yaml;slot 键仍是目录/进程身份。
    """
    kind = cfg.get("kind", "")
    params = {k: v for k, v in cfg.items() if k not in _META_KEYS}
    rec = build(kind, params, session_dir, duration)
    rec.display_name = str(cfg["name"]) if cfg.get("name") else ""
    return _with_sensor_name(slot, rec)


def get_production_recorders(session_dir: str | Path, duration: float = 0.0,
                             slots: list[str] | None = None,
                             ) -> dict[str, BaseRecorder]:
    """严格按 ``configs/recorders.yaml`` 构建生产记录器。

    ``slots`` 选子集;选择发生在构造之前 —— recorder 在 ``__init__`` 里
    就会建输出目录,先构造再过滤会留下孤儿目录。
    """
    from ..session.config import load_recorders
    config = load_recorders()
    if slots is not None:
        unknown = [s for s in slots if s not in config]
        if unknown:
            raise ValueError(f"recorders.yaml 没有这些 slot: {unknown}")
    recs = {}
    for slot, cfg in config.items():
        if slots is not None and slot not in slots:
            continue
        if not cfg.get("enabled", True):
            if slots is not None:
                # 操作员点名了停用槽位:必须喊出来,否则少一路数据毫无线索
                print(f"[factory] 槽位 '{slot}' 在 recorders.yaml 里 "
                      "enabled: false — 保持关闭(要启用请改 yaml,命令行"
                      "无法唤回)")
            continue
        recs[slot] = build_recorder(slot, cfg, session_dir, duration)
    if not recs:
        raise ValueError("configs/recorders.yaml 没有任何启用的 recorder")
    return recs


def get_dummy_recorders(session_dir: str | Path, duration: float = 0.0,
                        slots: list[str] | None = None,
                        stim: str | None = None) -> dict[str, BaseRecorder]:
    """dummy 全模态捆绑;``slots`` 只建其中一部分。

    选择同样发生在构造之前。``stim="sync_test"`` 时 dummy EEG 复刻
    sync_test 的事件节奏。
    """
    names = slots if slots is not None else list(_DUMMY_SLOT_KINDS)
    unknown = [n for n in names if n not in _DUMMY_SLOT_KINDS]
    if unknown:
        raise ValueError(f"未知 dummy recorder: {unknown}")
    recs = {}
    for name in names:
        kind = _DUMMY_SLOT_KINDS[name]
        params: dict = {}
        if name == "eeg" and stim == "sync_test":
            params["dummy_events"] = "sync_test"   # 精确复刻 sync_test 节奏
        recs[name] = _with_sensor_name(
            name, build(kind, params, session_dir, duration))
    return recs
