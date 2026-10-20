"""Stateful HAL Annex-B decoder. One baseline access unit yields one RGB frame.

Keep one instance per camera from the first subscription IDR through pre-roll
and commit. Never decode P frames independently or reset at commit.
"""
import re


class H264Decoder:
    @staticmethod
    def check_available():
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("HAL H.264 reception requires PyAV (pip install av)") from exc
        return av

    def __init__(self):
        self.av = self.check_available()
        self.codec = self.av.CodecContext.create("h264", "r")
        self.codec.thread_count = 1
        self.codec.thread_type = "SLICE"
        self.started = False

    def decode(self, payload, keyframe):
        if not payload or not payload.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")):
            raise ValueError("expected nonempty Annex-B H.264 access unit")
        nals = {part[0] & 31 for part in re.split(b"\x00\x00\x00?\x01", payload) if part}
        if bool(keyframe) != (5 in nals):
            raise ValueError("H.264 keyframe flag does not match IDR payload")
        if not self.started and not (keyframe is True and {5, 7, 8} <= nals):
            raise ValueError("first H.264 access unit requires IDR + SPS/PPS")
        frames = self.codec.decode(self.av.Packet(payload))
        # The sender contract is baseline/no B-frames, one full frame per packet.
        # Reject buffered/reordered streams instead of attaching the wrong stamp.
        if len(frames) != 1 or frames[0].pict_type == self.av.video.frame.PictureType.B:
            raise ValueError("expected one immediate baseline frame per H.264 packet")
        self.started = True
        return frames[0].to_ndarray(format="rgb24")
