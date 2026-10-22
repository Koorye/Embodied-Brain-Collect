"""HAL mux and CUDA preview tests; no headband connection."""
from fractions import Fraction
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import av
import numpy as np

from embodied_brain_collect.recorders.ego_headband.h264_decoder import H264Decoder
from embodied_brain_collect.recorders.ego_headband.h264_mux_writer import H264MuxWriter
from embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder import NetEgoHeadbandRecorder
from embodied_brain_collect.recorders.ego_headband.ego_headband_recorder_config import EgoHeadbandRecorderConfig


def packets(count=12):
    codec = av.CodecContext.create('libx264', 'w')
    codec.width, codec.height = 64, 48
    codec.pix_fmt = 'yuv420p'
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline',
                     'x264-params': 'keyint=4:scenecut=0:bframes=0:repeat-headers=1'}
    result = []
    for i in range(count):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 15, dtype=np.uint8), format='rgb24')
        frame.pts = i
        result.extend(codec.encode(frame))
    result.extend(codec.encode(None))
    assert len(result) == count
    return [(bytes(p), p.is_keyframe) for p in result]


def recorder(directory=None):
    r = NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig(session_dir=directory, encoder='cpu'))
    topic = r.config.camera_topics[0]
    r._on_stream_start({'source': 'hal', 'topics': {topic: 'h264/annexb', '/imu_data_raw': 'sensor_msgs/Imu'}})
    return r, topic


def metadata(topic, seq, keyframe):
    return {'kind': 'data', 'topic': topic, 'type': 'h264/annexb', 'stream_seq': seq,
            'keyframe': keyframe, 'header': {'stamp_ns': 1000000000 + seq * 33333333}}


class TestHalH264(unittest.TestCase):
    def test_continuous_p_frames(self):
        decoder = H264Decoder()
        self.assertTrue(decoder.codec.is_hwaccel)
        data = packets()
        for i, (payload, keyframe) in enumerate(data):
            rgb = decoder.decode(payload, keyframe)
            self.assertEqual(rgb.shape, (48, 64, 3))
            self.assertLess(abs(float(rgb.mean()) - i * 15), 5)
        with self.assertRaises(ValueError):
            H264Decoder().decode(data[1][0], False)

    def test_bad_frame_and_flag(self):
        with self.assertRaises(ValueError):
            H264Decoder().decode(b'not h264', True)
        with self.assertRaises(ValueError):
            H264Decoder().decode(packets(1)[0][0], False)

    def test_hal_one_imu_and_jpeg_compatibility(self):
        r, topic = recorder()
        self.assertEqual(r._imu_index, {'/imu_data_raw': 0})
        self.assertEqual(r._h264_cameras, {0})
        old = NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig())
        old._on_stream_start({'topics': {topic: 'sensor_msgs/CompressedImage'}})
        with patch.object(old, '_queue_camera') as enqueue:
            old._on_data({'topic': topic}, b'jpeg')
            enqueue.assert_called_once()
        self.assertFalse(old._h264_cameras)

    def test_gap_stops_before_enqueue(self):
        r, topic = recorder()
        with patch.object(r, '_queue_h264') as enqueue:
            r._on_data(metadata(topic, 0, True), b'dummy')
            r._on_data(metadata(topic, 2, False), b'dummy')
            self.assertTrue(r._ended)
            self.assertEqual(enqueue.call_count, 1)

    def test_missing_stamp_stops(self):
        r, topic = recorder()
        meta = metadata(topic, 0, True)
        meta['header'] = {}
        r._on_data(meta, b'dummy')
        self.assertTrue(r._ended)

    def test_decode_queue_full_never_replaces(self):
        r, _ = recorder()
        q = queue.Queue(maxsize=1)
        q.put('original')
        r._decode_queues[0] = q
        with self.assertRaises(RuntimeError):
            r._queue_h264(0, b'p', 1.0, True, False)
        self.assertEqual(q.get_nowait(), 'original')

    def test_cuda_no_software_fallback(self):
        with patch('av.codec.hwaccel.HWAccel', side_effect=RuntimeError('GPU unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback disabled'):
                H264Decoder()

    def test_pure_mux_packet_pts_and_payload(self):
        data = packets(12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'direct.mp4'
            w = H264MuxWriter(path, 30, data[0][0])
            self.assertIsNone(w.stream.codec_context)  # no codec created at all
            origin = 1700000000000000000
            for i, (payload, key) in enumerate(data):
                w.write(payload, origin + i * 33333333, key)
            w.close()
            with av.open(str(path)) as c:
                stream = c.streams.video[0]
                self.assertEqual(stream.codec_context.name, 'h264')
                # 名义 CFR 时基:打包端据此走整 GOP 流拷贝
                self.assertEqual(stream.time_base, Fraction(1, 90000))
                demuxed = [p for p in c.demux(video=0) if p.size]
                self.assertEqual(len(demuxed), len(data))
                for i, p in enumerate(demuxed):
                    self.assertEqual(p.pts, i * 3000)
                    self.assertAlmostEqual(float(p.pts * p.time_base), i / 30, places=5)
                    # MP4 rewrites start codes into lengths, but slice bytes are unchanged.
                    from embodied_brain_collect.recorders.ego_headband.h264_mux_writer import nals
                    raw = bytes(p); parts = []
                    while raw:
                        size = int.from_bytes(raw[:4], 'big'); parts.append(raw[4:4+size]); raw=raw[4+size:]
                    self.assertEqual([x for x in parts if x[0]&31 in (1,5)],
                                     [x for x in nals(*data[i]) if x[0]&31 in (1,5)])
            with av.open(str(path)) as c:
                self.assertEqual(len(list(c.decode(video=0))),12)

    def test_real_mux_commit_mp4_npz_tail_without_decoder(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            r, topic = recorder(tmp); r._launch_mode = True
            data = packets(12)
            with patch('embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder.H264Decoder',
                       side_effect=AssertionError('recording must not decode')):
                for i,(payload,key) in enumerate(data[:5]):
                    r._on_h264_data(metadata(topic,i,key),payload,record=False)
                deadline=time.monotonic()+3
                while not r._ready_to_commit() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue(r._ready_to_commit())
                r._committed=True
                # Commit at P frame 5: skip 5,6,7; next independent IDR is frame 8.
                for i,(payload,key) in enumerate(data[5:],5):
                    r._on_h264_data(metadata(topic,i,key),payload,record=True)
                r._stop_camera_decoders(); r._save()
                from loguru import logger
                if r._log_sink_id is not None:
                    logger.remove(r._log_sink_id); r._log_sink_id = None
            self.assertFalse(r._ended,r._ended_reason)
            out=Path(tmp)/'ego_headband'
            with np.load(out/'ego_headband.npz') as z:
                np.testing.assert_allclose(z['left_timestamps'],1+np.arange(8,12)*33333333/1e9)
                self.assertEqual(int(z['left_h264_start_skipped_frames']),3)
                self.assertEqual(str(z['video_storage']),'h264_passthrough_cfr')
            with av.open(str(out/'left.mp4')) as c:
                frames=list(c.decode(video=0)); self.assertEqual(len(frames),4)
                self.assertLess(abs(float(frames[0].to_ndarray(format='rgb24').mean())-120),6)

    def test_preview_failure_does_not_stop_mux(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,topic=recorder(tmp)
            with patch.object(r,'_preview_ports',return_value={'left':12345}), patch(
                'embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder.H264Decoder',
                side_effect=RuntimeError('GPU unavailable')):
                for i,(payload,key) in enumerate(packets(4)):
                    r._on_h264_data(metadata(topic,i,key),payload,record=True)
                r._stop_camera_decoders(); r._save()
                from loguru import logger
                if r._log_sink_id is not None:
                    logger.remove(r._log_sink_id); r._log_sink_id = None
            self.assertFalse(r._ended)
            self.assertEqual(r._written['left'],4)
            self.assertIn('left_preview_error',r._static)

    def test_gpu_preview_reaches_udp(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sink:
            sink.bind(('127.0.0.1',0)); sink.settimeout(2)
            r,topic=recorder(); r._save_enabled=False
            with patch.object(r,'_preview_ports',return_value={'left':sink.getsockname()[1]}):
                for i,(payload,key) in enumerate(packets(4)):
                    r._on_h264_data(metadata(topic,i,key),payload,record=False)
                r._stop_camera_decoders()
            self.assertFalse(r._ended,r._ended_reason)
            self.assertTrue(r._h264_decoders[0].codec.is_hwaccel)
            self.assertTrue(sink.recv(65536).startswith(b'\xff\xd8'))
            r._close_preview()

    def test_mux_rejects_non_monotonic_stamp(self):
        data=packets(2)
        with tempfile.TemporaryDirectory() as tmp:
            w=H264MuxWriter(Path(tmp)/'t.mp4',30,data[0][0]); w.write(data[0][0],1000000,True)
            with self.assertRaisesRegex(ValueError,'non-increasing'): w.write(data[1][0],1000000,False)
            w.close()


if __name__ == '__main__':
    unittest.main()
