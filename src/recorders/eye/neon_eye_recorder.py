from .base_eye_recorder import BaseEyeRecorder
from .eye_recorder_config import EyeRecorderConfig

import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device


class NeonEyeRecorder(BaseEyeRecorder):
    config: EyeRecorderConfig

    def __init__(self, config: EyeRecorderConfig):
        super().__init__(config)
        self.device = None

    def _open(self) -> bool:
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("[neon eye] no device found.")
            return False
        print("[neon eye] open, get first data.")
        self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        self.device.receive_imu_datum()
        print("[neon eye] got.")
        return True

    def _poll(self, ts):
        matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        scene = matched.scene.bgr_pixels[:, :, ::-1]
        x, y = matched.gaze.x, matched.gaze.y
        imu = self.device.receive_imu_datum()
        self._acc("scene_timestamps", matched.scene.timestamp_unix_seconds)
        self._acc("gaze_timestamps", matched.gaze.timestamp_unix_seconds)
        self._acc("imu_timestamps", imu.timestamp_unix_seconds)
        self._acc_arr("scene_frames", scene)
        self._acc_arr("gaze_xy", np.array([x, y], dtype=np.float32))
        self._acc_arr("imu_gyro", np.array([imu.gyro_data.x, imu.gyro_data.y, imu.gyro_data.z], dtype=np.float32))
        self._acc_arr("imu_accel", np.array([imu.accel_data.x, imu.accel_data.y, imu.accel_data.z], dtype=np.float32))

    def _close(self):
        print("[neon eye] closed.")
    
    def _heartbeat_stats(self, elapsed):
        return super()._heartbeat_stats(elapsed)