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
                r.request_commit()
                self.assertTrue(r._maybe_commit())
                # Commit at P frame 5: retain IDR 4 and every subsequent P frame.
                for i,(payload,key) in enumerate(data[5:],5):
                    r._on_h264_data(metadata(topic,i,key),payload,record=True)
                r._stop_camera_decoders(); r._save()
                from loguru import logger
                if r._log_sink_id is not None:
                    logger.remove(r._log_sink_id); r._log_sink_id = None
            self.assertFalse(r._ended,r._ended_reason)
            out=Path(tmp)/'ego_headband'
            with np.load(out/'ego_headband.npz') as z:
                np.testing.assert_allclose(z['left_timestamps'],1+np.arange(4,12)*33333333/1e9)
                self.assertEqual(int(z['left_h264_start_skipped_frames']),0)
                self.assertEqual(int(z['left_video_start_ns']), metadata(topic,4,True)['header']['stamp_ns'])
                self.assertEqual(str(z['video_storage']),'h264_passthrough_cfr')
            with av.open(str(out/'left.mp4')) as c:
                frames=list(c.decode(video=0)); self.assertEqual(len(frames),8)
                for i, frame in enumerate(frames,4):
                    self.assertLess(abs(float(frame.to_ndarray(format='rgb24').mean())-i*15),6)

    def test_task_window_cuts_cached_dependencies_from_real_recording(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'pack_daily_gop_test', Path(__file__).resolve().parents[2]/'scripts'/'pack_daily.py')
        pack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pack)
        with tempfile.TemporaryDirectory() as tmp:
            r, topic = recorder(tmp)
            r._launch_mode = True
            data = packets(12)
            try:
                for seq, (payload, key) in enumerate(data[:5]):
                    r._on_h264_data(metadata(topic, seq, key), payload, record=False)
                r.request_commit()
                r._maybe_commit()
                for seq, (payload, key) in enumerate(data[5:], 5):
                    r._on_data(metadata(topic, seq, key), payload)
                r._save()
                self.assertFalse(r._ended, r._ended_reason)
                out = Path(tmp)/'ego_headband'
                with np.load(out/'ego_headband.npz') as z:
                    ts = z['left_timestamps']
                np.testing.assert_allclose(ts, 1+np.arange(4,12)*33333333/1e9)
                trimmed = Path(tmp)/'task.mp4'
                count = pack.cut_video_aligned(out/'left.mp4', trimmed, ts,
                                               (ts[1], ts[6]), ts[1], 'left')
                self.assertEqual(count, 6)
                with av.open(str(trimmed)) as c:
                    frames = list(c.decode(video=0))
                self.assertEqual(len(frames), 6)
                for seq, frame in enumerate(frames, 5):
                    self.assertLess(abs(float(frame.to_ndarray(format='rgb24').mean())-seq*15),6)
            finally:
                r._stop_camera_decoders()
                from loguru import logger
                if r._log_sink_id is not None:
                    logger.remove(r._log_sink_id)
                    r._log_sink_id = None

    def test_commit_queued_preroll_then_immediate_stop(self):
        # Hold workers until AFTER commit clears the timestamp buffers. Both
        # cameras have queued precommit packets, and no postcommit packet arrives.
        with tempfile.TemporaryDirectory() as tmp:
            r, topic = recorder(tmp)
            r._launch_mode = True
            other = r.config.camera_topics[1]
            r._cam_index[other] = 1
            r._h264_cameras.add(1)
            data = packets(7)
            for i in (0, 1):
                q = queue.Queue(maxsize=64)
                r._decode_queues[i] = q
                for seq, (payload, key) in enumerate(data):
                    r._on_h264_data(metadata(topic if i == 0 else other, seq, key),
                                    payload, record=False)
            r._acc_ts('left', -123.0)  # must be cleared before cached GOP writes
            r.request_commit()
            self.assertTrue(r._maybe_commit())
            self.assertFalse(r._maybe_commit())  # no duplicate commit marker
            r.stop_event.set()
            r._decode_stop.set()
            for i, q in r._decode_queues.items():
                worker = threading.Thread(target=r._decode_h264_loop, args=(i, q))
                r._decode_threads[i] = worker
                worker.start()
            try:
                r._save()
                self.assertFalse(r._ended, r._ended_reason)
                with np.load(Path(tmp)/'ego_headband'/'ego_headband.npz') as z:
                    for key in ('left', 'right'):
                        np.testing.assert_allclose(z[f'{key}_timestamps'],
                                                   1+np.arange(4,7)*33333333/1e9)
                        with av.open(str(Path(tmp)/'ego_headband'/f'{key}.mp4')) as c:
                            frames = list(c.decode(video=0))
                        self.assertEqual(len(frames), 3)
                        self.assertLess(abs(float(frames[0].to_ndarray(format='rgb24').mean())-60),6)
            finally:
                from loguru import logger
                if r._log_sink_id is not None:
                    logger.remove(r._log_sink_id)
                    r._log_sink_id = None

    def test_preroll_limits_stop_without_evicting_references(self):
        module = 'embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder'
        for limit, value in (('_H264_GOP_MAX_PACKETS', 2),
                             ('_H264_GOP_MAX_BYTES', 1),
                             ('_H264_GOP_MAX_NS', 1)):
            with self.subTest(limit=limit), patch(f'{module}.{limit}', value):
                r, topic = recorder()
                for seq, (payload, key) in enumerate(packets(3)):
                    r._on_h264_data(metadata(topic, seq, key), payload, record=False)
                r._stop_camera_decoders()
                self.assertTrue(r._ended)
                self.assertIn('GOP exceeds cache limit', r._ended_reason)
                self.assertFalse(r._written)

    def test_commit_queue_overflow_keeps_existing_packet(self):
        r, _ = recorder()
        q = queue.Queue(maxsize=1)
        q.put('original')
        r._decode_queues[0] = q
        r.request_commit()
        with self.assertRaisesRegex(RuntimeError, 'overflow at commit'):
            r._maybe_commit()
        self.assertEqual(q.get_nowait(), 'original')
        self.assertTrue(r._ended)

    def test_preroll_rejects_non_monotonic_timestamp(self):
        r, topic = recorder()
        for seq, (payload, key) in enumerate(packets(2)):
            meta = metadata(topic, seq, key)
            meta['header']['stamp_ns'] = 1000000000
            r._on_h264_data(meta, payload, record=False)
        r._stop_camera_decoders()
        self.assertTrue(r._ended)
        self.assertIn('non-increasing', r._ended_reason)

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
