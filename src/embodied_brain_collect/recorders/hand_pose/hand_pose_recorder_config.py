"""Hand pose recorder config."""
from dataclasses import dataclass, field
from ..base import BaseRecorderConfig


@dataclass
class HandPoseRecorderConfig(BaseRecorderConfig):
    # MANUS-specific
    hand_motion: str = "NoMotion"    # "NoMotion" | "IMU" | "Tracker" | "Auto"
    calibration_dir: str = ""        # dir with .mcal files; default: ~/.cache/manus_glove
    left_calibration: str = "LeftMetaglovePro.mcal"
    right_calibration: str = "RightMetaglovePro.mcal"
    lib_path: str = ""               # SDK lib path (libManusSDK_Integrated.so on Linux, ManusSDK.dll on Windows); "" = auto-resolve
