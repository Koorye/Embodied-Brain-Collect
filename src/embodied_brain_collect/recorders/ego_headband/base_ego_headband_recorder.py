"""Abstract EGO headband base — N fisheye cameras (mp4) + M IMUs (npz).

Cameras reuse the camera recorder's async HEVC write pipeline: one
``{name}.mp4`` per stream (``name`` from ``config.camera_names``), its
timestamps recorded 1:1 with the frames the writer actually committed.  The
IMUs accumulate into arrays.  ``_save`` stops the writers (finalizing every mp4
and flushing their timestamp buffers) and then writes ONE npz holding each
camera's ``{name}_timestamps`` plus every IMU's ``imu{j}_ts`` /
``imu{j}_gyro`` / ``imu{j}_accel``.

Output (under ``<session>/<output_dir>/``)::

    {name}.mp4                 one per camera topic that actually streamed
    <output_dir>.npz           {name}_timestamps   (N_i,) float64  (1:1 w/ mp4)
                               imu{j}_ts         (M_j,) float64
                               imu{j}_gyro       (M_j, 3) float32  rad/s
                               imu{j}_accel      (M_j, 3) float32  m/s^2
"""
from pathlib import Path
import numpy as np
from .microphone_writer import MicrophoneWriter
from .ffmpeg_jpeg_writer import FFmpegJpegWriter

from ..camera.base_camera_recorder import BaseCameraRecorder


class BaseEgoHeadbandRecorder(BaseCameraRecorder):
    name = "ego_headband"
    output_dir = "ego_headband"

    def __init__(self, config):
        super().__init__(config)
        self._audio_writer = None
        self._audio_error = ""

    def _cam_stem(self, i: int) -> str:
        """Output stem for camera slot ``i`` — its configured friendly name
        (``left``/``right``/... -> ``left.mp4`` + ``left_timestamps``), falling
        back to ``cam{i}`` only for slots beyond the configured names."""
        names = getattr(self.config, "camera_names", ()) or ()
        return str(names[i]) if i < len(names) else f"cam{i}"

    def _write_microphone(self, meta: dict, payload: bytes) -> None:
        # 两段式启动:commit 前的预热音频不落盘。wav 是流式直写磁盘的,
        # 预热段一旦写入便无法像视频缓冲那样在 commit 时丢弃 —— 不拦的话
        # wav 会比 mp4 和 npz 元数据多出整段预热音频(wav_sample_offset
        # 不从 0 起,时间轴也无法与视频对齐)。
        if self._launch_mode and not self._committed:
            return
        if self._audio_writer is None:
            path = (Path(self.config.session_dir) / self.output_dir / "microphone.wav"
                    if self.config.session_dir else None)
            self._audio_writer = MicrophoneWriter(path)
        row = self._audio_writer.write(meta, payload)
        for key, value in row.items():
            self._acc(key, value)

    def _close_microphone(self) -> None:
        if self._audio_writer is not None:
            self._audio_writer.close()

    def _build_output(self) -> dict:
        result = super()._build_output()
        if self._audio_writer is not None or self._audio_error:
            result.update(
                microphone_sample_rate=np.asarray(16000, dtype=np.int64),
                microphone_channels=np.asarray(1, dtype=np.int64),
                microphone_sample_width=np.asarray(2, dtype=np.int64),
                microphone_timestamp_basis=np.asarray(
                    (self._audio_writer.timestamp_basis or "unknown") if self._audio_writer else "unknown"),
                microphone_timestamp_reference=np.asarray(
                    "first_sample" if self._audio_writer and
                    self._audio_writer.timestamp_basis == MicrophoneWriter.CAPTURE_BASIS else "read_complete"),
                microphone_timestamp_clock=np.asarray("CLOCK_REALTIME"),
                # Unknown physical accuracy is explicit; software checks do not certify AV alignment.
                microphone_timestamp_accuracy_ns=np.asarray(-1, dtype=np.int64),
                microphone_av_sync_validated=np.asarray(False),
                microphone_error=np.asarray(self._audio_error),
            )
        return result

    def _write_fps(self) -> float:
        # The fisheye streams run at cam_fps; BaseCameraRecorder looks for
        # cfg.fps / cfg.cam_fps_hint, which this config names cam_fps.
        return float(getattr(self.config, "cam_fps", 0.0) or 30.0)

    def _make_writer(self, path, fps, data):
        # Motion-JPEG frames (the net headband's cameras) go through the GPU
        # transcode writer: ffmpeg decodes (mjpeg_cuvid) + encodes (hevc_nvenc)
        # with no CPU cv2.imdecode — four 1920x1088@30 fisheye streams saturate
        # the CPU otherwise.  Raw frames (the dummy's synthetic RGB) fall back
        # to the shared libx265 writer via super().
        if isinstance(data, (bytes, bytearray)):
            w = FFmpegJpegWriter(
                path, fps,
                encoder=str(getattr(self.config, "encoder", "auto")),
                crf=int(self.config.crf), preset=str(self.config.preset))
            self._log(f"[{self.name}] {path.stem} mp4 -> {path} "
                      f"(jpeg -> {w.encoder_used})")
            return w
        return super()._make_writer(path, fps, data)

    def _save(self) -> None:
        # The writer threads own the mp4s AND feed the camera timestamp
        # buffers (a frame's _acc_ts runs only once it is written) — stop them
        # before reading those buffers, or the npz and the mp4s disagree.
        self._close_microphone()
        self._stop_write_worker()
        if not self.config.session_dir:
            return
        # _build_output already emits {name}_timestamps (from _ts_buf) plus
        # every imu{j}_* array (from _buf / _arr_buf) — exactly this layout.
        payload = self._build_output()
        if payload:
            out = self._npz_path()
            np.savez(out, **payload)
            self._log(f"[{self.name}] saved {out} "
                      f"({out.stat().st_size / 1e6:.1f} MB, "
                      f"{len(payload)} fields)")
        for key, path in sorted(self._video_paths.items()):
            size = path.stat().st_size / 1e6 if path.is_file() else 0.0
            self._log(f"[{self.name}] {key} -> {path} "
                      f"({self._written.get(key, 0)} frames, {size:.1f} MB)")
        if not payload and not self._written:
            self._log(f"[{self.name}] nothing to save.")
