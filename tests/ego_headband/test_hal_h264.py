"""Offline HAL regression tests: no headband connection or hardware access."""
from fractions import Fraction
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from embodied_brain_collect.recorders.ego_headband.h264_decoder import H264Decoder
from embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder import NetEgoHeadbandRecorder
from embodied_brain_collect.recorders.ego_headband.ego_headband_recorder_config import EgoHeadbandRecorderConfig


def packets(count=12):
    codec = av.CodecContext.create('libx264', 'w')
    codec.width, codec.height = 64, 48
    codec.pix_fmt = 'yuv420p'
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline',
                     'x264-params': 'keyint=60:bframes=0:repeat-headers=1'}
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

    def test_warm_packet_cannot_leak_after_commit(self):
        r, _ = recorder()
        r._committed = True
        r._write_queues['left'] = queue.Queue(maxsize=1)
        with patch.object(r, '_start_write_worker'):
            r._enqueue_h264_rgb('left', 1.0, np.zeros((2, 2, 3)), False)
        self.assertTrue(r._write_queues['left'].empty())
        self.assertEqual(r._frames_precommit, 1)

    def test_write_queue_full_fails(self):
        r, _ = recorder()
        q = queue.Queue(maxsize=1)
        q.put('original')
        r._write_queues['left'] = q
        with patch.object(r, '_start_write_worker'), self.assertRaises(RuntimeError):
            r._enqueue_h264_rgb('left', 1.0, np.zeros((2, 2, 3)), True)
        self.assertEqual(q.get_nowait(), 'original')

    def test_real_decode_commit_mp4_npz_tail(self):
        # Real H.264 decode + existing HEVC writer + actual MP4 decode count.
        with tempfile.TemporaryDirectory() as tmp:
            r, topic = recorder(tmp)
            r._launch_mode = True
            data = packets(12)
            # Subscription IDR and warm-up P frames pass through keepalive.
            frames = iter([(metadata(topic, i, key), payload)
                           for i, (payload, key) in enumerate(data[:4])] + [None])
            with patch.object(r, '_take_frame', side_effect=lambda: next(frames)):
                r._drain_discard()
            # Drain pre-roll completely, keeping decoder state intact.
            r._decode_stop.set()
            r._stop_camera_decoders()
            decoder = r._h264_decoders[0]
            r._committed = True
            # Feed later P frames through the same stateful decoder.
            for i, (payload, key) in enumerate(data[4:], start=4):
                rgb = decoder.decode(payload, key)
                r._enqueue_h264_rgb('left', i / 30, rgb, True)
            r._save()
            from loguru import logger
            if r._log_sink_id is not None:
                logger.remove(r._log_sink_id)
                r._log_sink_id = None
            out = Path(tmp) / 'ego_headband'
            with np.load(out / 'ego_headband.npz') as saved:
                np.testing.assert_allclose(saved['left_timestamps'], np.arange(4, 12) / 30)
            with av.open(str(out / 'left.mp4')) as video:
                decoded = list(video.decode(video=0))
            self.assertEqual(len(decoded), 8)
            self.assertLess(abs(float(decoded[0].to_ndarray(format='rgb24').mean()) - 60), 6)


if __name__ == '__main__':
    unittest.main()
