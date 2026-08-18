"""Dummy hand pose — 40-channel ergonomics data @ 60 Hz."""

import time
import numpy as np
from .base_hand_pose_recorder import BaseHandPoseRecorder
from .hand_pose_recorder_config import HandPoseRecorderConfig


class DummyHandPoseRecorder(BaseHandPoseRecorder):
    config: HandPoseRecorderConfig

    def _open(self) -> bool:
        self._log("[hand_pose:dummy] simulated 40-ch ergonomics @ 60Hz")
        return True

    def _close(self) -> None:
        self._log(f"[hand_pose:dummy] stopped")

    def _poll(self, ts):
        time.sleep(0.016)
        data = (
            np.sin(2 * np.pi * 2 * ts + np.arange(40) * 0.1).astype(np.float32)
            + np.random.randn(40).astype(np.float32) * 0.005)
        self._acc("ergo_timestamps", ts)
        self._acc_arr("ergo_data", data)
