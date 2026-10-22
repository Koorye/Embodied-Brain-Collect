"""Frame-exact packaging regression, including H.264 microsecond/VFR input."""
from fractions import Fraction
import importlib.util
from pathlib import Path
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
from embodied_brain_collect.recorders.ego_headband.h264_mux_writer import H264MuxWriter


def source(path, variable=False):
    enc = av.CodecContext.create('libx264', 'w')
    enc.width, enc.height = 64, 48
    enc.pix_fmt = 'yuv420p'; enc.time_base = Fraction(1, 30)
    enc.framerate = Fraction(30, 1)
    enc.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline',
                   'x264-params': 'keyint=30:bframes=0:repeat-headers=1:scenecut=0'}
    payloads = []
    for i in range(60):
        f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 3, np.uint8), format='rgb24'); f.pts=i
        payloads.extend(enc.encode(f))
    payloads.extend(enc.encode(None))
    w = H264MuxWriter(path, 30, bytes(payloads[0]))
    ts = 1700000000.0 + np.arange(60) / 30
    if variable: ts[25:] += 2 / 30
    for i, p in enumerate(payloads):
        w.write(bytes(p), 1700000000000000000 + round(i*1e9/30) + (66666667 if variable and i>=25 else 0), p.is_keyframe)
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
