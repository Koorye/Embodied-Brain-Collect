"""Recorder factory functions — import what you need, compose freely.

Each ``get_<variant>_<modality>`` returns a fully configured recorder instance.
Mix and match::

    from src.session.recorder_presets import (
        get_weili_emg, get_manus_hand_pose, get_dummy_camera, get_dummy_marker,
    )
    from src.session.launcher import launch

    launch({
        "emg":       get_weili_emg("./sessions/run1", duration=120),
        "hand_pose": get_manus_hand_pose("./sessions/run1", duration=120),
        "camera":    get_dummy_camera("./sessions/run1", duration=120),
        "marker":    get_dummy_marker("./sessions/run1", duration=120),
    })
"""

from __future__ import annotations

from src.recorders.base import BaseRecorder

# ---- EMG ----------------------------------------------------------------

from src.recorders.emg import (
    DummyEmgRecorder, WeiliEmgRecorder, EmgRecorderConfig,
)


def get_dummy_emg(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_weili_emg(session_dir: str, duration: float = 0.0,
                  port: str = "", baud: int = 921600) -> BaseRecorder:
    return WeiliEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration, port=port, baud=baud))


# ---- Hand Pose ----------------------------------------------------------

from src.recorders.hand_pose import (
    DummyHandPoseRecorder, ManusHandPoseRecorder, HandPoseRecorderConfig,
)


def get_dummy_hand_pose(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_manus_hand_pose(session_dir: str, duration: float = 0.0,
                        hand_motion: str = "NoMotion",
                        lib_path: str = "",
                        calibration_dir: str = "",
                        ) -> BaseRecorder:
    return ManusHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration,
        hand_motion=hand_motion, lib_path=lib_path,
        calibration_dir=calibration_dir))


# ---- Camera -------------------------------------------------------------

from src.recorders.camera import (
    DummyCameraRecorder, OpencvCameraRecorder, DepthaiCameraRecorder,
    CameraRecorderConfig,
)


def get_dummy_camera(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyCameraRecorder(CameraRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_opencv_camera(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return OpencvCameraRecorder(CameraRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_depthai_camera(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DepthaiCameraRecorder(CameraRecorderConfig(
        session_dir=session_dir, duration=duration))


# ---- Eye -----------------------------------------------------------------

from src.recorders.eye import (
    DummyEyeRecorder, NeonEyeRecorder, EyeRecorderConfig,
)


def get_dummy_eye(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_neon_eye(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return NeonEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration))


# ---- Position -----------------------------------------------------------

from src.recorders.position import (
    DummyPositionRecorder, OpenvrPositionRecorder, PositionRecorderConfig,
)


def get_dummy_position(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_openvr_position(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return OpenvrPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration))


# ---- Marker -------------------------------------------------------------

from src.recorders.marker import (
    DummyMarkerRecorder, UdpMarkerRecorder, MarkerRecorderConfig,
)


def get_dummy_marker(session_dir: str, duration: float = 0.0) -> BaseRecorder:
    return DummyMarkerRecorder(MarkerRecorderConfig(
        session_dir=session_dir, duration=duration))


def get_udp_marker(session_dir: str, duration: float = 0.0,
                   host: str = "127.0.0.1", port: int = 9999) -> BaseRecorder:
    """UDP marker listener — receives markers sent by stim's MarkerSender."""
    return UdpMarkerRecorder(MarkerRecorderConfig(
        session_dir=session_dir, duration=duration, host=host, port=port))


# ---- Convenience bundles ------------------------------------------------

def get_dummy_recorders(session_dir: str, duration: float = 0.0
                        ) -> dict[str, BaseRecorder]:
    return {
        "camera":    get_dummy_camera(session_dir, duration),
        "emg":       get_dummy_emg(session_dir, duration),
        "eye":       get_dummy_eye(session_dir, duration),
        "hand_pose": get_dummy_hand_pose(session_dir, duration),
        "position":  get_dummy_position(session_dir, duration),
        # "marker":    get_dummy_marker(session_dir, duration),
        "marker":    get_udp_marker(session_dir, duration),
    }


def get_production_recorders(session_dir: str, duration: float = 0.0
                             ) -> dict[str, BaseRecorder]:
    return {
        "camera":    get_opencv_camera(session_dir, duration),
        "emg":       get_weili_emg(session_dir, duration),
        "eye":       get_neon_eye(session_dir, duration),
        "hand_pose": get_manus_hand_pose(session_dir, duration),
        "position":  get_openvr_position(session_dir, duration),
        "marker":    get_udp_marker(session_dir, duration),
    }
