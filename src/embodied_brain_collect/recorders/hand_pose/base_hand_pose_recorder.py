"""Abstract hand pose base."""
from ..base import BaseRecorder


class BaseHandPoseRecorder(BaseRecorder):
    name = "hand_pose"
    output_dir = "hand_pose"
