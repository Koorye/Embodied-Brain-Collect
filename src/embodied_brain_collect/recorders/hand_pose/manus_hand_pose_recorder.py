"""MANUS Quantum Metaglove recorder via manus_glove SDK."""

import time

import numpy as np

from .base_hand_pose_recorder import BaseHandPoseRecorder
from .hand_pose_recorder_config import HandPoseRecorderConfig


class ManusHandPoseRecorder(BaseHandPoseRecorder):
    """Real MANUS Quantum Metaglove / Metaglove Pro.

    Connects to Manus Core via the SDK and records per-hand ergonomics
    (40-ch finger joint angles) plus skeleton node positions/quaternions.
    """

    config: HandPoseRecorderConfig

    def __init__(self, config: HandPoseRecorderConfig):
        super().__init__(config)
        self._pub = None          # ManusDataPublisher
        self._glove_ids: list[int] = []
        self._glove_sides: dict[int, int] = {}   # glove_id → Side enum
        self._last_glove_check = 0.0

    # ---- lifecycle ----------------------------------------------------------

    def _open(self) -> bool:
        from manus_glove import ManusDataPublisher, HandMotion

        hm = getattr(HandMotion, self.config.hand_motion, HandMotion.NoMotion)
        kwargs = dict(
            hand_motion=hm,
            debug=False,
        )
        if self.config.lib_path:
            kwargs["lib_path"] = self.config.lib_path
        if self.config.calibration_dir:
            kwargs["calibration_dir"] = self.config.calibration_dir
        if self.config.left_calibration:
            kwargs["left_calibration_file"] = self.config.left_calibration
        if self.config.right_calibration:
            kwargs["right_calibration_file"] = self.config.right_calibration

        self._pub = ManusDataPublisher(**kwargs)
        self._pub.Initialize()
        self._log("[hand_pose:manus] SDK initialized, connecting to Manus Core ...")
        self._pub.Connect()

        # Wait for initial landscape
        t0 = time.time()
        while self._pub.GetLandscape() is None:
            if time.time() - t0 > 15:
                self._open_error = "timed out waiting for landscape (15 s)"
                self._log(f"[hand_pose:manus] open failed — {self._open_error}")
                return False
            time.sleep(0.1)

        # Both gloves must be paired before recording starts — the recorder
        # records a fixed two-hand layout and does not track devices that
        # connect/disconnect later.
        self._log("[hand_pose:manus] waiting for BOTH gloves (2/2) ...")
        deadline = time.time() + float(self.config.open_timeout or 30.0)
        last_count = -1
        while len(self._glove_ids) < 2:
            if time.time() > deadline:
                self._open_error = (
                    f"only {len(self._glove_ids)}/2 gloves found after "
                    f"{self.config.open_timeout:g}s — pair both gloves in "
                    "Manus Core and retry")
                self._log(f"[hand_pose:manus] open failed — {self._open_error}")
                return False
            self._refresh_gloves()
            if self._pub.GetGloveIds():
                self._refresh_gloves()
            if len(self._glove_ids) != last_count:
                last_count = len(self._glove_ids)
                self._log(f"[hand_pose:manus] gloves detected: "
                          f"{last_count}/2 (ids={self._glove_ids})")
            time.sleep(0.5)

        self._pub.LoadCalibrationFiles()
        self._log(f"[hand_pose:manus] both gloves connected: "
                  f"ids={self._glove_ids} — waiting for first glove data ...")

        def _try_poll() -> bool:
            self._poll(time.time())
            return (bool(self._buf.get("ergo_timestamps"))
                    or bool(self._buf.get("skeleton_timestamps")))

        if not self._wait_first_sample(_try_poll, "glove data", timeout=10.0):
            return False
        # Gate samples used wall-clock ts; clear so the session timeline
        # starts clean from the launcher's t0.
        self._buf.clear()
        self._arr_buf.clear()
        self._log("[hand_pose:manus] first glove data received — ready")
        return True

    def _poll(self, ts) -> None:
        if self._pub is None:
            return

        # One combined frame per poll tick: left hand fills 0-19, right fills
        # 20-39.  Appending one frame per glove would interleave left/right
        # rows (each zeroed on the other hand's half) and make per-channel
        # time series sawtooth between value and zero.
        # The glove list is fixed at open time (both gloves required).
        ergo_flat = np.zeros(40, dtype=np.float32)
        have_ergo = False
        skel_parts: list[tuple[np.ndarray, np.ndarray]] = []
        have_skel = False

        for gid in self._glove_ids:
            data = self._pub.GetGloveData(gid)
            if data is None:
                continue

            # ---- ergonomics (finger joint angles) ----
            ergo = data.get("ergonomics")
            if ergo:
                have_ergo = True
                # entry["type"] is side-agnostic (e.g. "ThumbMCPSpread"),
                # and the snapshot only holds this glove's own side — use
                # the glove side to place left at 0-19, right at 20-39.
                offset = _ERGO_SIDE_OFFSET.get(data.get("side", "Left"), 0)
                for entry in ergo:
                    idx = _ERGO_INDEX.get(entry["type"], -1)
                    if idx >= 0:
                        ergo_flat[offset + idx] = entry["value"]

            # ---- skeleton ----
            nodes = data.get("raw_nodes")
            if nodes:
                have_skel = True
                pos = np.array([n["position"] for n in nodes], dtype=np.float32)
                rot = np.array([n["rotation"] for n in nodes], dtype=np.float32)
                skel_parts.append((pos, rot))

        if have_ergo:
            self._acc("ergo_timestamps", ts)
            self._acc_arr("ergo_data", ergo_flat)
        if have_skel:
            self._acc("skeleton_timestamps", ts)
            self._acc_arr("skeleton_positions",
                          np.concatenate([p for p, _ in skel_parts], axis=0))
            self._acc_arr("skeleton_rotations",
                          np.concatenate([r for _, r in skel_parts], axis=0))

    def _refresh_gloves(self) -> None:
        """Refresh glove list from landscape (handles late connections)."""
        landscape = self._pub.GetLandscape()
        if landscape is None:
            return
        gloves = landscape.get("gloves", [])
        new_ids = [g["id"] for g in gloves]
        new_sides = {g["id"]: g["side"] for g in gloves}

        # Log newly connected / disconnected gloves
        added = set(new_ids) - set(self._glove_ids)
        removed = set(self._glove_ids) - set(new_ids)
        for gid in added:
            side = new_sides.get(gid, "?")
            bat = next((g.get("batteryPercentage", "?") for g in gloves if g["id"] == gid), "?")
            self._log(f"[hand_pose:manus] glove connected: id={gid} side={side} battery={bat}%")
            self._pub.LoadCalibrationFiles()
        for gid in removed:
            self._log(f"[hand_pose:manus] glove disconnected: id={gid}")

        self._glove_ids = new_ids
        self._glove_sides = new_sides

    def _close(self) -> None:
        if self._pub is not None:
            self._pub.ShutDown()
            self._pub = None
            self._log("[hand_pose:manus] SDK shut down")

    def _heartbeat_stats(self, elapsed: float) -> str:
        return super()._heartbeat_stats(elapsed)


# Map ErgonomicsDataType string (side-agnostic, as returned by
# GetGloveData) → base index within one hand's 20 channels.
_ERGO_INDEX = {
    "ThumbMCPSpread":  0,  "ThumbMCPStretch":  1,
    "ThumbPIPStretch": 2,  "ThumbDIPStretch":  3,
    "IndexSpread":     4,  "IndexMCPStretch":  5,
    "IndexPIPStretch": 6,  "IndexDIPStretch":  7,
    "MiddleSpread":    8,  "MiddleMCPStretch": 9,
    "MiddlePIPStretch":10, "MiddleDIPStretch":11,
    "RingSpread":     12,  "RingMCPStretch":  13,
    "RingPIPStretch": 14,  "RingDIPStretch":  15,
    "PinkySpread":    16,  "PinkyMCPStretch": 17,
    "PinkyPIPStretch":18,  "PinkyDIPStretch": 19,
}
# Side → flat 40-ch offset (left hand occupies 0-19, right 20-39).
_ERGO_SIDE_OFFSET = {"Left": 0, "Right": 20}
