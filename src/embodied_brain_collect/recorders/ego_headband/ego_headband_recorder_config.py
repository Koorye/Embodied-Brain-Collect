"""EGO headband recorder config.

The headband is a network device that streams over TCP using the "wired ROS"
protocol (length-prefixed JSON metadata + binary payload; see
``net_ego_headband_recorder``).  It carries four fisheye cameras (each frame a
JPEG payload) and two IMUs (angular_velocity / linear_acceleration in JSON).

``camera_topics`` / ``imu_topics`` map the device's named ROS topics onto the
output slots IN ORDER, so a slot stays tied to the same physical sensor across
sessions even when the server reports some publishers missing.  Each camera's
output stem comes from ``camera_names`` (parallel to ``camera_topics``): it is
written to ``{name}.mp4`` with ``{name}_timestamps`` in the NPZ (the shared
HEVC pipeline); ``encoder="auto"`` puts that encode on the GPU when available.
IMUs stay numbered ``imu{j}``.
"""
from dataclasses import dataclass
from ..base import BaseRecorderConfig

# Default device topics (wired_tcp server).  Order defines the camera slots.
DEFAULT_CAMERA_TOPICS = (
    "/fisheye/left/image_raw/compressed",
    "/fisheye/right/image_raw/compressed",
    "/fisheye/bleft/image_raw/compressed",
    "/fisheye/bright/image_raw/compressed",
)
# Output stem per camera slot, parallel to DEFAULT_CAMERA_TOPICS: the mp4 is
# ``{name}.mp4`` and its timestamps ``{name}_timestamps`` (no cam{i} numbering).
DEFAULT_CAMERA_NAMES = ("left", "right", "bleft", "bright")
DEFAULT_IMU_TOPICS = (
    "/imu_data_raw",
    "/imu_data_filterd",      # device spelling (as the ROS topic is published)
)


@dataclass
class EgoHeadbandRecorderConfig(BaseRecorderConfig):
    # ---- network (this recorder is the TCP client; the device is server) ----
    host: str = "192.168.55.6"   # wired_tcp server address on the device
    port: int = 5577             # device stream port
    connect_timeout: float = 10.0
    require_synced: bool = False  # True = fail open if clock sync didn't verify

    # ---- topic -> slot mapping (order defines the camera slots / imu{j}) ----
    camera_topics: tuple = DEFAULT_CAMERA_TOPICS
    camera_names: tuple = DEFAULT_CAMERA_NAMES   # output stem per slot (mp4/npz)
    imu_topics: tuple = DEFAULT_IMU_TOPICS

    # ---- camera video encoding ({name}.mp4, like the other cameras) ----
    crf: int = 23
    preset: str = "medium"
    encoder: str = "auto"        # "auto" = GPU (hevc_nvenc) else CPU (libx265)
    cam_fps: float = 30.0        # mp4 container rate for the fisheye streams

    # ---- frame size + dummy synthesis ----
    # The device's four fisheye streams are 1920x1088.  The NET recorder
    # ignores cam_width/cam_height (it encodes whatever each decoded JPEG
    # actually is); the dummy synthesizes at this size to match real data.
    n_cameras: int = 4
    cam_width: int = 1920
    cam_height: int = 1088
    n_imus: int = 2
    imu_rate_hz: float = 200.0

    # Microphone capability negotiated with wired_tcp; PCM is stored in WAV.
    audio_enabled: bool = False
    require_audio: bool = True  # fail open if requested microphone is unavailable
    audio_topic: str = "/microphone/audio"
