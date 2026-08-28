"""Recorder factory functions — import what you want, compose freely.

Every ``get_<variant>_<modality>`` passes through ALL parameters of the
recorder's config dataclass (plus the common ``session_dir`` / ``duration`` /
``open_timeout``), so a caller can tune anything without constructing
configs by hand::

    from embodied_brain_collect.session.recorder_presets import get_weili_emg, get_realsense_camera
    from embodied_brain_collect.session.launcher import launch

    launch({
        "emg":     get_weili_emg("./sessions/run1", duration=120),
        "camera":  get_realsense_camera("./sessions/run1", depth=True, crf=20),
    })
"""

from __future__ import annotations

from embodied_brain_collect.recorders.base import BaseRecorder

# ---- EMG ----------------------------------------------------------------

from embodied_brain_collect.recorders.emg import (
    DummyEmgRecorder, WeiliEmgRecorder, EmgRecorderConfig,
)


def get_dummy_emg(session_dir: str, duration: float = 0.0,
                  open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


def get_weili_emg(session_dir: str, duration: float = 0.0,
                  port: str = "", baud: int = 921600,
                  open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return WeiliEmgRecorder(EmgRecorderConfig(
        session_dir=session_dir, duration=duration,
        port=port, baud=baud, open_timeout=open_timeout, hz=hz))


# ---- Hand Pose ----------------------------------------------------------

from embodied_brain_collect.recorders.hand_pose import (
    DummyHandPoseRecorder, ManusHandPoseRecorder, HandPoseRecorderConfig,
)


def get_dummy_hand_pose(session_dir: str, duration: float = 0.0,
                        open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


def get_manus_hand_pose(session_dir: str, duration: float = 0.0,
                        hand_motion: str = "NoMotion",
                        lib_path: str = "",
                        calibration_dir: str = "",
                        left_calibration: str = "LeftMetaglovePro.mcal",
                        right_calibration: str = "RightMetaglovePro.mcal",
                        open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return ManusHandPoseRecorder(HandPoseRecorderConfig(
        session_dir=session_dir, duration=duration,
        hand_motion=hand_motion, lib_path=lib_path,
        calibration_dir=calibration_dir,
        left_calibration=left_calibration,
        right_calibration=right_calibration,
        open_timeout=open_timeout, hz=hz))


# ---- Camera -------------------------------------------------------------

from embodied_brain_collect.recorders.camera import (
    DummyCameraRecorder, OpencvCameraRecorder, DepthaiCameraRecorder,
    RealsenseCameraRecorder,
    CameraRecorderConfig, OpencvCameraConfig, DepthaiCameraConfig,
    RealsenseCameraConfig,
)


def get_dummy_camera(session_dir: str, duration: float = 0.0,
                     crf: int = 23, preset: str = "medium",
                     open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyCameraRecorder(CameraRecorderConfig(
        session_dir=session_dir, duration=duration,
        crf=crf, preset=preset, open_timeout=open_timeout, hz=hz))


def get_opencv_camera(session_dir: str, duration: float = 0.0,
                      idx: int = 0, width: int = 640, height: int = 480,
                      fps: int = 30, fourcc: str = "MJPG",
                      warmup: float = 1.0,
                      crf: int = 23, preset: str = "medium",
                      open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return OpencvCameraRecorder(OpencvCameraConfig(
        session_dir=session_dir, duration=duration,
        idx=idx, width=width, height=height, fps=fps, fourcc=fourcc,
        warmup=warmup, crf=crf, preset=preset, open_timeout=open_timeout, hz=hz))


def get_depthai_camera(session_dir: str, duration: float = 0.0,
                       cam_w: int = 1280, cam_h: int = 720,
                       cam_fps_hint: float = 30.0,
                       crf: int = 23, preset: str = "medium",
                       open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DepthaiCameraRecorder(DepthaiCameraConfig(
        session_dir=session_dir, duration=duration,
        cam_w=cam_w, cam_h=cam_h, cam_fps_hint=cam_fps_hint,
        crf=crf, preset=preset, open_timeout=open_timeout, hz=hz))


def get_realsense_camera(session_dir: str, duration: float = 0.0,
                         serial: str = "", width: int = 640,
                         height: int = 480, fps: int = 30,
                         depth: bool = False,
                         crf: int = 23, preset: str = "medium",
                         open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return RealsenseCameraRecorder(RealsenseCameraConfig(
        session_dir=session_dir, duration=duration,
        serial=serial, width=width, height=height, fps=fps, depth=depth,
        crf=crf, preset=preset, open_timeout=open_timeout, hz=hz))


# ---- Eye -----------------------------------------------------------------

from embodied_brain_collect.recorders.eye import (
    DummyEyeRecorder, EyeRecorderConfig,
)
# Neon 的两个 recorder 顶层 import pupil_labs SDK —— 延迟到工厂内导入,
# 没装该 SDK 的机器(开发机/部分部署)仍可用其余模态。


def get_dummy_eye(session_dir: str, duration: float = 0.0,
                  open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


def get_neon_eye(session_dir: str, duration: float = 0.0,
                 no_scene_video: bool = False,
                 crf: int = 23, preset: str = "medium",
                 open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    """Neon via the simple (synchronous) API — device auto-discovered."""
    from embodied_brain_collect.recorders.eye import NeonEyeRecorder
    return NeonEyeRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        no_scene_video=no_scene_video, crf=crf, preset=preset,
        open_timeout=open_timeout, hz=hz))


def get_neon_eye_async(session_dir: str, duration: float = 0.0,
                       no_scene_video: bool = False,
                       crf: int = 23, preset: str = "medium",
                       open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    """Neon via the async full-rate API — device auto-discovered."""
    from embodied_brain_collect.recorders.eye import NeonEyeAsyncRecorder
    return NeonEyeAsyncRecorder(EyeRecorderConfig(
        session_dir=session_dir, duration=duration,
        no_scene_video=no_scene_video, crf=crf, preset=preset,
        open_timeout=open_timeout, hz=hz))


# ---- Position -----------------------------------------------------------

from embodied_brain_collect.recorders.position import (
    DummyPositionRecorder, PositionRecorderConfig,
)
# OpenVR recorder 顶层 import openvr SDK —— 延迟到工厂内导入,理由同 eye。


def get_dummy_position(session_dir: str, duration: float = 0.0,
                       open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


def get_openvr_position(session_dir: str, duration: float = 0.0,
                        device_classes: str = "tracker",
                        open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    from embodied_brain_collect.recorders.position import OpenvrPositionRecorder
    return OpenvrPositionRecorder(PositionRecorderConfig(
        session_dir=session_dir, duration=duration,
        device_classes=device_classes, open_timeout=open_timeout, hz=hz))


# ---- Marker -------------------------------------------------------------

from embodied_brain_collect.recorders.marker import (
    UdpMarkerRecorder, MarkerRecorderConfig,
)


def get_udp_marker(session_dir: str, duration: float = 0.0,
                   host: str = "127.0.0.1", port: int = 9999,
                   open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    """UDP marker listener — receives markers sent by stim's MarkerSender."""
    return UdpMarkerRecorder(MarkerRecorderConfig(
        session_dir=session_dir, duration=duration,
        host=host, port=port, open_timeout=open_timeout, hz=hz))


# ---- EEG ----------------------------------------------------------------

from embodied_brain_collect.recorders.eeg import (
    DummyEegRecorder, CurryEegRecorder, EegRecorderConfig,
)


def get_dummy_eeg(session_dir: str, duration: float = 0.0,
                  open_timeout: float = 30.0, hz: float = 1000.0,
                  dummy_events: str = "") -> BaseRecorder:
    return DummyEegRecorder(EegRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz, dummy_events=dummy_events))


def get_curry_eeg(session_dir: str, duration: float = 0.0,
                  host: str = "127.0.0.1", port: int = 4455,
                  open_timeout: float = 30.0, hz: float = 1000.0,
                  marker_wait_s: float = 10.0) -> BaseRecorder:
    """Neuroscan Curry NetStream EEG (alignment runs in the recorder's
    close, against markers/markers.npz)."""
    return CurryEegRecorder(EegRecorderConfig(
        session_dir=session_dir, duration=duration,
        host=host, port=port, open_timeout=open_timeout, hz=hz,
        marker_wait_s=marker_wait_s))


# ---- Tactile ------------------------------------------------------------

from embodied_brain_collect.recorders.tactile import (
    DummyTactileRecorder, TouchtronixTactileRecorder, TactileRecorderConfig,
)


def get_dummy_tactile(session_dir: str, duration: float = 0.0,
                      open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    return DummyTactileRecorder(TactileRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


def get_touchtronix_tactile(session_dir: str, duration: float = 0.0,
                            open_timeout: float = 30.0, hz: float = 1000.0) -> BaseRecorder:
    """TouchTronix tactile glove — NOT IMPLEMENTED (open always fails)."""
    return TouchtronixTactileRecorder(TactileRecorderConfig(
        session_dir=session_dir, duration=duration,
        open_timeout=open_timeout, hz=hz))


# ---- Convenience bundles ------------------------------------------------

def _with_sensor_name(name: str, rec: BaseRecorder) -> BaseRecorder:
    """Save this recorder's output under its sensor slot name.

    The dict key in the bundles below is the sensor's concrete name (e.g.
    ``cam_third``, ``emg_left``); the recorder's ``output_dir`` is pointed at
    it so each sensor lands in ``{session_dir}/<name>/`` instead of the
    generic modality directory (``camera/``, ``emg/`` ...).
    """
    rec.set_output_dir(name)
    return rec


def get_dummy_recorders(session_dir: str, duration: float = 0.0,
                        slots: list[str] | None = None,
                        stim: str | None = None
                        ) -> dict[str, BaseRecorder]:
    """All dummy modalities; pass ``slots`` to build only some.

    Selection happens BEFORE construction — a recorder creates its output
    dir in ``__init__``, so constructing then filtering is what left empty
    ``camera/`` directories next to real recordings.
    """
    all_factories = {
        "camera":    lambda: get_dummy_camera(session_dir, duration),
        "eeg":       lambda: get_dummy_eeg(
            session_dir, duration,
            dummy_events=stim if stim == "sync_test" else ""),
        "emg":       lambda: get_dummy_emg(session_dir, duration),
        "eye":       lambda: get_dummy_eye(session_dir, duration),
        "hand_pose": lambda: get_dummy_hand_pose(session_dir, duration),
        "position":  lambda: get_dummy_position(session_dir, duration),
        "marker":    lambda: get_udp_marker(session_dir, duration),
    }
    names = slots if slots is not None else list(all_factories)
    unknown = [n for n in names if n not in all_factories]
    if unknown:
        raise ValueError(f"未知 dummy recorder: {unknown}")
    return {n: _with_sensor_name(n, all_factories[n]()) for n in names}


#: kind (configs/recorders.yaml) -> factory.  Explicit, mirrors house style.
FACTORY_BY_KIND = {
    "depthai_camera":   get_depthai_camera,
    "opencv_camera":    get_opencv_camera,
    "realsense_camera": get_realsense_camera,
    "curry_eeg":        get_curry_eeg,
    "weili_emg":        get_weili_emg,
    "neon_eye":         get_neon_eye,
    "neon_eye_async":   get_neon_eye_async,
    "manus_hand_pose":  get_manus_hand_pose,
    "openvr_position":  get_openvr_position,
    "udp_marker":       get_udp_marker,
    "dummy_camera":     get_dummy_camera,
    "dummy_eeg":        get_dummy_eeg,
    "dummy_emg":        get_dummy_emg,
    "dummy_eye":        get_dummy_eye,
    "dummy_hand_pose":  get_dummy_hand_pose,
    "dummy_position":   get_dummy_position,
}


def build_recorder(slot: str, cfg: dict, session_dir: str,
                   duration: float = 0.0) -> BaseRecorder:
    """One recorder from a ``configs/recorders.yaml`` entry.

    ``cfg`` is ``{kind, ...params}``; every key besides ``kind`` is passed
    to the matching factory, so the TOML carries the full per-slot
    configuration (camera indices, COM ports, baud, hz) instead of the code.
    """
    kind = cfg.get("kind", "")
    factory = FACTORY_BY_KIND.get(kind)
    if factory is None:
        raise ValueError(f"未知 recorder kind: {kind!r} (slot={slot})")
    params = {k: v for k, v in cfg.items() if k != "kind"}
    return _with_sensor_name(slot, factory(session_dir, duration, **params))


def get_production_recorders(session_dir: str, duration: float = 0.0,
                             slots: list[str] | None = None,
                             ) -> dict[str, BaseRecorder]:
    """Recorders exactly as ``configs/recorders.yaml`` declares them.

    ``slots`` selects a subset; selection happens before construction for
    the same reason as in ``get_dummy_recorders`` — no orphan directories.
    """
    from embodied_brain_collect.config.load import load_recorders
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
            continue
        recs[slot] = build_recorder(slot, cfg, session_dir, duration)
    if not recs:
        raise ValueError("configs/recorders.yaml 没有任何启用的 recorder")
    return recs
