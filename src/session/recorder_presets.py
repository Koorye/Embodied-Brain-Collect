"""Recorder factory functions — import what you want, compose freely.

Every ``get_<variant>_<modality>`` passes through ALL parameters of the
recorder's config dataclass (plus the common ``session_dir`` / ``duration`` /
``open_timeout``), so a caller can tune anything without constructing
configs by hand::

    from src.session.recorder_presets import get_weili_emg, get_realsense_camera
    from src.session.launcher import launch

    launch({
        "emg":     get_weili_emg("./sessions/run1", duration=120),
        "camera":  get_realsense_camera("./sessions/run1", depth=True, crf=20),
    })
"""

from __future__ import annotations

from src.recorders.base import BaseRecorder

# ---- EMG ----------------------------------------------------------------

from src.recorders.emg import (
    DummyEmgRecorder, WeiliEmgRecorder, EmgRecorderConfig,
)


def get_dummy_emg(session_dir: str, duration: float = 0.0,
                  open_timeout: float = 30.0) -> BaseRecorder:
    return DummyEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_weili_emg(session_dir: str, duration: float = 0.0,
                  port: str = "", baud: int = 921600,
                  open_timeout: float = 30.0) -> BaseRecorder:
    return WeiliEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration,
        port=port, baud=baud, open_timeout=open_timeout))


# ---- Hand Pose ----------------------------------------------------------

from src.recorders.hand_pose import (
    DummyHandPoseRecorder, ManusHandPoseRecorder, HandPoseRecorderConfig,
)


def get_dummy_hand_pose(session_dir: str, duration: float = 0.0,
                        open_timeout: float = 30.0) -> BaseRecorder:
    return DummyHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_manus_hand_pose(session_dir: str, duration: float = 0.0,
                        hand_motion: str = "NoMotion",
                        lib_path: str = "",
                        calibration_dir: str = "",
                        left_calibration: str = "LeftMetaglovePro.mcal",
                        right_calibration: str = "RightMetaglovePro.mcal",
                        open_timeout: float = 30.0) -> BaseRecorder:
    return ManusHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration,
        hand_motion=hand_motion, lib_path=lib_path,
        calibration_dir=calibration_dir,
        left_calibration=left_calibration,
        right_calibration=right_calibration,
        open_timeout=open_timeout))


# ---- Camera -------------------------------------------------------------

from src.recorders.camera import (
    DummyCameraRecorder, OpencvCameraRecorder, DepthaiCameraRecorder,
    RealsenseCameraRecorder,
    CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
    RealsenseCameraConfig,
)


def get_dummy_camera(session_dir: str, duration: float = 0.0,
                     crf: int = 23, preset: str = "medium",
                     open_timeout: float = 30.0) -> BaseRecorder:
    return DummyCameraRecorder(CameraRecorderConfig(
        session_dir=session_dir, duration=duration,
        crf=crf, preset=preset, open_timeout=open_timeout))


def get_opencv_camera(session_dir: str, duration: float = 0.0,
                      idx: int = 0, width: int = 640, height: int = 480,
                      fps: int = 30, fourcc: str = "MJPG",
                      warmup: float = 1.0,
                      crf: int = 23, preset: str = "medium",
                      open_timeout: float = 30.0) -> BaseRecorder:
    return OpencvCameraRecorder(OpencvCameraConfig(
        session_dir=session_dir, duration=duration,
        idx=idx, width=width, height=height, fps=fps, fourcc=fourcc,
        warmup=warmup, crf=crf, preset=preset, open_timeout=open_timeout))


def get_depthai_camera(session_dir: str, duration: float = 0.0,
                       cam_w: int = 1280, cam_h: int = 720,
                       cam_fps_hint: float = 30.0,
                       crf: int = 23, preset: str = "medium",
                       open_timeout: float = 30.0) -> BaseRecorder:
    return DepthaiCameraRecorder(DepthaiCameraConfig(
        session_dir=session_dir, duration=duration,
        cam_w=cam_w, cam_h=cam_h, cam_fps_hint=cam_fps_hint,
        crf=crf, preset=preset, open_timeout=open_timeout))


def get_realsense_camera(session_dir: str, duration: float = 0.0,
                         serial: str = "", width: int = 640,
                         height: int = 480, fps: int = 30,
                         depth: bool = False,
                         crf: int = 23, preset: str = "medium",
                         open_timeout: float = 30.0) -> BaseRecorder:
    return RealsenseCameraRecorder(RealsenseCameraConfig(
        session_dir=session_dir, duration=duration,
        serial=serial, width=width, height=height, fps=fps, depth=depth,
        crf=crf, preset=preset, open_timeout=open_timeout))


# ---- Eye -----------------------------------------------------------------

from src.recorders.eye import (
    DummyEyeRecorder, NeonEyeRecorder, EyeRecorderConfig, NeonEyeAsyncRecorder,
)


def get_dummy_eye(session_dir: str, duration: float = 0.0,
                  open_timeout: float = 30.0) -> BaseRecorder:
    return DummyEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_neon_eye(session_dir: str, duration: float = 0.0,
                 no_scene_video: bool = False,
                 crf: int = 23, preset: str = "medium",
                 open_timeout: float = 30.0) -> BaseRecorder:
    """Neon via the simple (synchronous) API — device auto-discovered."""
    return NeonEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        no_scene_video=no_scene_video, crf=crf, preset=preset,
        open_timeout=open_timeout))


def get_neon_eye_async(session_dir: str, duration: float = 0.0,
                       no_scene_video: bool = False,
                       crf: int = 23, preset: str = "medium",
                       open_timeout: float = 30.0) -> BaseRecorder:
    """Neon via the async full-rate API — device auto-discovered."""
    return NeonEyeAsyncRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        no_scene_video=no_scene_video, crf=crf, preset=preset,
        open_timeout=open_timeout))


# ---- Position -----------------------------------------------------------

from src.recorders.position import (
    DummyPositionRecorder, OpenvrPositionRecorder, PositionRecorderConfig,
)


def get_dummy_position(session_dir: str, duration: float = 0.0,
                       open_timeout: float = 30.0) -> BaseRecorder:
    return DummyPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_openvr_position(session_dir: str, duration: float = 0.0,
                        device_classes: str = "tracker",
                        open_timeout: float = 30.0) -> BaseRecorder:
    return OpenvrPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration,
        device_classes=device_classes, open_timeout=open_timeout))


# ---- Marker -------------------------------------------------------------

from src.recorders.marker import (
    DummyMarkerRecorder, UdpMarkerRecorder, MarkerRecorderConfig,
)


def get_dummy_marker(session_dir: str, duration: float = 0.0,
                     open_timeout: float = 30.0) -> BaseRecorder:
    return DummyMarkerRecorder(MarkerRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_udp_marker(session_dir: str, duration: float = 0.0,
                   host: str = "127.0.0.1", port: int = 9999,
                   open_timeout: float = 30.0) -> BaseRecorder:
    """UDP marker listener — receives markers sent by stim's MarkerSender."""
    return UdpMarkerRecorder(MarkerRecorderConfig(
        session_dir=session_dir, duration=duration,
        host=host, port=port, open_timeout=open_timeout))


# ---- Tactile ------------------------------------------------------------

from src.recorders.tactile import (
    DummyTactileRecorder, TouchtronixTactileRecorder, TactileRecorderConfig,
)


def get_dummy_tactile(session_dir: str, duration: float = 0.0,
                      open_timeout: float = 30.0) -> BaseRecorder:
    return DummyTactileRecorder(TactileRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


def get_touchtronix_tactile(session_dir: str, duration: float = 0.0,
                            open_timeout: float = 30.0) -> BaseRecorder:
    """TouchTronix tactile glove — NOT IMPLEMENTED (open always fails)."""
    return TouchtronixTactileRecorder(TactileRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout))


# ---- Convenience bundles ------------------------------------------------

def _with_sensor_name(name: str, rec: BaseRecorder) -> BaseRecorder:
    """Save this recorder's output under its sensor slot name.

    The dict key in the bundles below is the sensor's concrete name (e.g.
    ``cam_third``, ``emg_left``); the recorder's ``output_dir`` is pointed at
    it so each sensor lands in ``{session_dir}/<name>/`` instead of the
    generic modality directory (``camera/``, ``emg/`` ...).
    """
    rec.output_dir = name
    return rec


def get_dummy_recorders(session_dir: str, duration: float = 0.0
                        ) -> dict[str, BaseRecorder]:
    recs = {
        "camera":    get_dummy_camera(session_dir, duration),
        "emg":       get_dummy_emg(session_dir, duration),
        "eye":       get_dummy_eye(session_dir, duration),
        "hand_pose": get_dummy_hand_pose(session_dir, duration),
        "position":  get_dummy_position(session_dir, duration),
        "marker":    get_udp_marker(session_dir, duration),
    }
    return {name: _with_sensor_name(name, rec) for name, rec in recs.items()}


def get_production_recorders(session_dir: str, duration: float = 0.0
                             ) -> dict[str, BaseRecorder]:
    recs = {
        "cam_head": get_depthai_camera(session_dir, duration),
        "cam_left_wrist": get_opencv_camera(session_dir, duration, idx=2),
        "cam_right_wrist": get_opencv_camera(session_dir, duration, idx=3),
        "cam_third":    get_realsense_camera(session_dir, duration),
        "emg_left":       get_weili_emg(session_dir, duration),
        "emg_right": get_weili_emg(session_dir, duration),
        "eye":       get_neon_eye_async(session_dir, duration),
        "hand_pose": get_manus_hand_pose(session_dir, duration),
        "position":  get_openvr_position(session_dir, duration),
        "marker":    get_udp_marker(session_dir, duration),
    }
    return {name: _with_sensor_name(name, rec) for name, rec in recs.items()}
