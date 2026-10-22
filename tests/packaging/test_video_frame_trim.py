"""Frame-exact packaging regression, including H.264 microsecond/VFR input."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('pack_daily_trim_test', ROOT / 'scripts/pack_daily.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
from embodied_brain_collect.recorders.ego_headband.h264_mux_writer import (
    H264MuxWriter, dimensions, nals)


def mux_capture_pts(path, packets, keys, stamps_us, fps=30):
    """旧录制语义:PTS = 采集时刻(微秒、可变帧率)。

    2026-10-22 之前的头环直存文件就是这种容器;保留这条封装路径,让
    "VFR 历史数据仍按帧序号截段"的回归不会因为录制端改 CFR 而失覆盖。
    """
    parts = nals(packets[0], True)
    sps = next(p for p in parts if p[0] & 31 == 7)
    width, height = dimensions(sps)
    tb = Fraction(1, 1000000)
    with av.open(str(path), 'w', format='mp4') as container:
        stream = container.add_mux_stream('h264', rate=Fraction(str(fps)),
                                          width=width, height=height,
                                          time_base=tb)
        for payload, key, pts in zip(packets, keys, stamps_us):
            packet = av.Packet(payload)
            packet.stream = stream
            packet.time_base = tb
            packet.pts = packet.dts = int(pts)
            packet.duration = round(1000000 / fps)
            packet.is_keyframe = key
            container.mux(packet)


def source(path, variable=False):
    enc = av.CodecContext.create('libx264', 'w')
    enc.width, enc.height = 64, 48
    enc.pix_fmt = 'yuv420p'; enc.time_base = Fraction(1, 30)
    enc.framerate = Fraction(30, 1)
    enc.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline',
                   'x264-params': 'keyint=30:bframes=0:repeat-headers=1:scenecut=0'}
    encoded = []
    for i in range(60):
        f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 3, np.uint8), format='rgb24'); f.pts=i
        encoded.extend(enc.encode(f))
    encoded.extend(enc.encode(None))
    payloads = [(bytes(p), p.is_keyframe) for p in encoded]

    ts = 1700000000.0 + np.arange(60) / 30
    origin = 1700000000000000000
    stamps = origin + np.round(np.arange(60) * 1e9 / 30).astype(np.int64)
    if variable:
        # 采集侧真实抖动:第 25 帧起晚 2 帧 → 容器 PTS 间距不等(VFR)
        stamps = stamps + np.where(np.arange(60) >= 25, 66666667, 0)
        ts[25:] += 2 / 30
        mux_capture_pts(path, [p for p, _k in payloads], [k for _p, k in payloads],
                        ((stamps - stamps[0]) // 1000).tolist())
    else:
        w = H264MuxWriter(path, 30, payloads[0][0])
        for i, (payload, key) in enumerate(payloads):
            w.write(payload, int(stamps[i]), key)
        w.close()
    return ts


class FrameExactCutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.cfr = cls.dir/'source.mp4'; cls.vfr = cls.dir/'source_vfr.mp4'
        cls.ts = source(cls.cfr); cls.vts = source(cls.vfr, variable=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def assert_clip(self, path, first, count):
        with av.open(str(path)) as c:
            frames=list(c.decode(video=0))
        self.assertEqual(len(frames), count)
        self.assertLess(abs(float(frames[0].to_ndarray(format='rgb24').mean())-first*3),4)
        self.assertLess(abs(float(frames[-1].to_ndarray(format='rgb24').mean())-(first+count-1)*3),4)
        np.testing.assert_allclose([float(f.pts*f.time_base) for f in frames],np.arange(count)/30,atol=1e-6)

    def test_h264_millisecond_seek_regression(self):
        out=self.dir/'rounding.mp4'
        self.assertEqual(pack.write_video_ffmpeg(self.cfr,out,17/30,43/30,30),43)
        self.assert_clip(out,17,43)

    def test_variable_capture_pts_preserve_frame_ordinals(self):
        out=self.dir/'vfr_trim.mp4'
        n=pack.cut_video_aligned(self.vfr,out,self.vts,(self.vts[17],self.vts[59]),self.vts[17],'headband_right')
        self.assertEqual(n,43);self.assert_clip(out,17,43)

    def test_frame_starts_either_side_of_gop(self):
        for i0 in (29,30,31):
            with self.subTest(i0=i0):
                out=self.dir/f'gop{i0}.mp4'
                self.assertEqual(pack.write_video_ffmpeg(self.cfr,out,i0/30,5/30,30),5)
                self.assert_clip(out,i0,5)

    def test_cfr_h264_source_uses_gop_stream_copy(self):
        """名义 CFR 的 H.264(新录制端语义)必须走整 GOP 流拷贝,不二次编码。"""
        out = self.dir / 'copied.mp4'
        pack._smart_cut(self.cfr, out, 17, 43, 30)

        def nals_of(path):
            """容器包 → NAL 字节列表(屏蔽 length 前缀/起始码差异)。"""
            result = []
            with av.open(str(path)) as c:
                for pkt in c.demux(video=0):
                    if not pkt.size:
                        continue
                    raw, units = bytes(pkt), []
                    if raw[:4] == b'\x00\x00\x00\x01':
                        for part in re.split(b'\x00\x00\x00?\x01', raw)[1:]:
                            if part:
                                units.append(part)
                    else:
                        while raw:
                            size = int.from_bytes(raw[:4], 'big')
                            units.append(raw[4:4 + size]); raw = raw[4 + size:]
                    result.append(units)
            return result

        src, dst = nals_of(self.cfr), nals_of(out)
        self.assertEqual(len(dst), 43)
        head = 30 - 17                      # i0 到下一个关键帧的边角重编码段
        self.assertGreater(head, 0)
        tail_src, tail_dst = src[30:], dst[head:]
        self.assertEqual(len(tail_src), len(tail_dst))
        for a, b in zip(tail_src, tail_dst):
            # 拷贝段逐 NAL 一致;关键帧包只允许多出 TS 拼接带的参数集
            self.assertTrue(set(a) <= set(b) or a == b,
                            msg=f"拷贝段被重编码:{[x[:1] for x in a]} vs {[x[:1] for x in b]}")

    def test_short_source_fails_and_does_not_publish(self):
        out=self.dir/'short.mp4'
        with self.assertRaisesRegex(RuntimeError,'43/44'):
            pack.write_video_ffmpeg(self.cfr,out,17/30,44/30,30)
        self.assertFalse(out.exists());self.assertFalse(out.with_suffix('.tmp.mp4').exists())

    def test_empty_window_is_failure(self):
        with self.assertRaisesRegex(RuntimeError,'fewer than two'):
            pack.cut_video_aligned(self.cfr,self.dir/'empty.mp4',self.ts,(self.ts[0],self.ts[0]),self.ts[0],'right')

    def test_wrong_count_is_failure_not_skipped_video(self):
        out=self.dir/'wrong_count.mp4'
        with patch.object(pack,'write_video_ffmpeg',return_value=42),patch.object(pack,'_decode_frame_at',return_value=np.zeros((2,2))),patch.object(pack,'_decode_frames',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'42/43'):
                pack.cut_video_aligned(self.cfr,out,self.ts,(self.ts[17],self.ts[-1]),self.ts[17],'right')

    def test_no_decodable_output_is_failure(self):
        with patch.object(pack,'write_video_ffmpeg',return_value=43),patch.object(pack,'_decode_frame_at',side_effect=IndexError('truncated')):
            with self.assertRaisesRegex(RuntimeError,'no decodable'):
                pack.cut_video_aligned(self.cfr,self.dir/'bad.mp4',self.ts,(self.ts[17],self.ts[-1]),self.ts[17],'right')


if __name__ == '__main__': unittest.main()
