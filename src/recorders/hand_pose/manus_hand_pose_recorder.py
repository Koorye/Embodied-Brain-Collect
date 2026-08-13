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

        # Wait up to 8 s for device detection (landscape arrives before devices,
        # and gloves are discovered asynchronously one-by-one).
        self._log("[hand_pose:manus] waiting for glove detection ...")
        t0 = time.time()
        first_seen = 0.0
        while time.time() - t0 < 8:
            prev_ids = set(self._glove_ids)
            self._refresh_gloves()
            # Also check skeleton stream for glove IDs
            if self._pub.GetGloveIds():
                self._refresh_gloves()
            new_ids = set(self._glove_ids) - prev_ids
            if new_ids and not first_seen:
                first_seen = time.time()
            # After first glove appears, keep waiting up to 3 s for more
            if first_seen and time.time() - first_seen > 3:
                break
            time.sleep(0.2)

        if not self._glove_ids:
            self._log("[hand_pose:manus] WARNING: no gloves found after 8 s. "
                      "Ensure gloves are paired and Manus Core is running.")
            self._log("[hand_pose:manus] Will keep checking for gloves while recording ...")
        else:
            self._pub.LoadCalibrationFiles()
        return True

    def _poll(self, ts) -> None:
        if self._pub is None:
            return

        # Periodically refresh landscape to detect newly connected gloves
        if ts - self._last_glove_check > 2.0:
            self._last_glove_check = ts
            self._refresh_gloves()

        for gid in self._glove_ids:
            data = self._pub.GetGloveData(gid)
            if data is None:
                continue

            # ---- ergonomics (finger joint angles) ----
            ergo = data.get("ergonomics")
            if ergo is not None:
                # Accumulate only when we see new data (check first value)
                if ergo:
                    self._acc("ergo_timestamps", ts)
                    flat = np.zeros(40, dtype=np.float32)
                    for entry in ergo:
                        idx = _ERGO_INDEX.get(entry["type"], -1)
                        if idx >= 0:
                            flat[idx] = entry["value"]
                    self._acc_arr("ergo_data", flat)

            # ---- skeleton ----
            nodes = data.get("raw_nodes")
            if nodes:
                self._acc("skeleton_timestamps", ts)
                pos = np.array([n["position"] for n in nodes], dtype=np.float32)
                rot = np.array([n["rotation"] for n in nodes], dtype=np.float32)
                self._acc_arr("skeleton_positions", pos)
                self._acc_arr("skeleton_rotations", rot)

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
        n_ergo = len(self._buf.get("ergo_timestamps", []))
        n_skel = len(self._buf.get("skeleton_timestamps", []))
        return (
            f"ergo={n_ergo:>5} ({n_ergo/elapsed:.1f}/s)  "
            f"skel={n_skel:>5}"
        )


# Map ErgonomicsDataType string → flat 40-ch index (0-19 left, 20-39 right).
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
# Build right-hand indices (20-39) using same names.
_RIGHT = {}
for _name, _idx in list(_ERGO_INDEX.items()):
    _RIGHT[_name] = _idx + 20
_ERGO_INDEX.update(_RIGHT)
