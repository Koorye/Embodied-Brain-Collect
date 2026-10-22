"""Pure Annex-B -> MP4 muxing; no decoder or encoder is created.

The container carries nominal CFR timestamps (``pts = i * 90000/fps``) so the
downstream GOP-level stream-copy cut applies; the authoritative per-frame
capture time lives in the NPZ ``{name}_timestamps``.  Capture order is still
validated — a non-increasing capture timestamp means the stream itself is
corrupt, even though it no longer shapes the container timeline.
"""
from fractions import Fraction
import re


def nals(payload, keyframe):
    if not payload or not payload.startswith((b'\x00\x00\x01', b'\x00\x00\x00\x01')):
        raise ValueError('expected Annex-B H.264 access unit')
    parts = [p for p in re.split(b'\x00\x00\x00?\x01', payload) if p]
    kinds = {p[0] & 31 for p in parts}
    if type(keyframe) is not bool or keyframe != (5 in kinds):
        raise ValueError('H.264 keyframe flag does not match IDR')
    if not kinds.intersection((1, 5)):
        raise ValueError('H.264 packet has no video slice')
    for p in parts:
        if p[0] & 31 in (1, 5):
            bits = Bits(p[1:]); bits.ue()
            if bits.ue() % 5 == 1:
                raise ValueError('B frames are unsupported: DTS must equal PTS')
    return parts


class Bits:
    def __init__(self, data):
        self.data = re.sub(b'\x00\x00\x03', b'\x00\x00', data)
        self.pos = 0

    def get(self, count=1):
        if self.pos + count > len(self.data) * 8:
            raise ValueError('truncated H.264 header')
        value = 0
        for _ in range(count):
            value = (value << 1) | ((self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1)
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while not self.get():
            zeros += 1
            if zeros > 31:
                raise ValueError('invalid H.264 Exp-Golomb header')
        return (1 << zeros) - 1 + self.get(zeros)

    def se(self):
        n = self.ue()
        return (n + 1) // 2 if n & 1 else -(n // 2)


def dimensions(sps):
    b = Bits(sps[1:])
    if b.get(8) != 66:
        raise ValueError('expected baseline H.264 from camd')
    b.get(16); b.ue(); b.ue()
    poc = b.ue()
    if poc == 0:
        b.ue()
    elif poc == 1:
        b.get(); b.se(); b.se()
        for _ in range(b.ue()):
            b.se()
    elif poc != 2:
        raise ValueError('invalid H.264 POC type')
    b.ue(); b.get()
    width, height = (b.ue() + 1) * 16, (b.ue() + 1) * 16
    if b.get() != 1:
        raise ValueError('interlaced H.264 unsupported')
    b.get()
    if b.get():
        left, right, top, bottom = [b.ue() for _ in range(4)]
        width -= 2 * (left + right); height -= 2 * (top + bottom)
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError('invalid H.264 dimensions')
    return width, height


class H264MuxWriter:
    @staticmethod
    def check_available():
        import av
        if not hasattr(av.container.OutputContainer, 'add_mux_stream'):
            raise RuntimeError('H.264 passthrough requires PyAV with add_mux_stream (stay uses av 18.1)')
        return av

    def __init__(self, path, fps, first_payload):
        self.av = self.check_available()
        parts = nals(first_payload, True)
        sps = next((p for p in parts if p[0] & 31 == 7), None)
        if sps is None or not any(p[0] & 31 == 8 for p in parts):
            raise ValueError('recording must start with IDR + SPS/PPS')
        self.width, self.height = dimensions(sps)
        self.sps = sps
        fps_i = int(fps)
        if fps_i <= 0 or float(fps_i) != float(fps) or 90000 % fps_i:
            raise ValueError('H.264 CFR mux requires an integer fps dividing 90000 '
                             f'(e.g. 30); got cam_fps={fps}')
        self.time_base = Fraction(1, 90000)
        self.ticks = 90000 // fps_i
        self.index = 0
        self.last_stamp_ns = None
        self.duration = self.ticks
        self.container = self.av.open(str(path), 'w', format='mp4')
        try:
            self.stream = self.container.add_mux_stream('h264', rate=Fraction(str(fps_i)),
                width=self.width, height=self.height, time_base=self.time_base)
        except BaseException:
            self.container.close()
            raise

    def write(self, payload, stamp_ns, keyframe):
        parts = nals(payload, keyframe)
        for p in parts:
            if p[0] & 31 == 7 and p != self.sps:
                raise ValueError('SPS changed during recording')
        if self.index == 0:
            if not keyframe:
                raise ValueError('first saved packet must be IDR')
        if (self.last_stamp_ns is not None
                and int(stamp_ns) <= self.last_stamp_ns):
            raise ValueError('non-increasing H.264 capture timestamp')
        self.last_stamp_ns = int(stamp_ns)
        pts = self.index * self.ticks
        packet = self.av.Packet(payload)
        packet.stream = self.stream
        packet.time_base = self.time_base
        packet.pts = packet.dts = pts
        packet.duration = self.duration
        packet.is_keyframe = keyframe
        self.container.mux(packet)
        self.index += 1

    def close(self):
        if self.container is not None:
            container, self.container = self.container, None
            container.close()
