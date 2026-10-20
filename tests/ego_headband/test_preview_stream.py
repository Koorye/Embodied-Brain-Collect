"""辅助员控制台预览流:头环四路鱼眼必须真的发出 UDP 包。

两条回归(现场症状都是控制台 headband 面板永远停在"等待预览"):

1. 设备升级到 HAL H.264(``stream_source=hal``)后,帧走
   ``_decode_h264_loop``,而推预览的调用只在旧 MJPEG 的 ``_on_camera``
   里 —— 新路径一帧都不推。
2. 旧 MJPEG 路径直接转发原始 JPEG:1920x1088 鱼眼帧 200-400KB,远超
   UDP 单包上限,``sendto`` 必然失败,而预览的失败是静默的。
"""
import socket
import unittest

import cv2
import numpy as np

from embodied_brain_collect.recorders.ego_headband.ego_headband_recorder_config import (
    EgoHeadbandRecorderConfig)
from embodied_brain_collect.recorders.ego_headband.net_ego_headband_recorder import (
    NetEgoHeadbandRecorder)


class PreviewSink:
    """本机 UDP 收包端:给 recorder 一个真实端口,收到什么就是控制台看到什么。"""

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.sock.close()

    def frames(self):
        out = []
        while True:
            try:
                out.append(self.sock.recv(1 << 16))
            except socket.timeout:
                return out


def jpeg(width=1920, height=1088, quality=95):
    """逼近真实鱼眼帧的 JPEG:随机噪声不可压缩,体积必然超过单包上限。"""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return enc.tobytes()


def recorder(port):
    return NetEgoHeadbandRecorder(EgoHeadbandRecorderConfig(
        preview_ports={"left": port}, encoder="cpu"))


class TestHeadbandPreview(unittest.TestCase):
    def test_h264_rgb_frame_reaches_console(self):
        with PreviewSink() as sink:
            r = recorder(sink.port)
            rgb = np.zeros((1088, 1920, 3), dtype=np.uint8)
            rgb[..., 0] = 200                      # 纯红(RGB 序)
            r.preview_rgb("left", 1.0, rgb)
            got = sink.frames()
            r._close_preview()

        self.assertEqual(len(got), 1)
        arr = cv2.imdecode(np.frombuffer(got[0], dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        self.assertIsNotNone(arr)
        # 控制台按 BGR 解码 → 红色必须落在 B/G 之上,否则红蓝互换
        b, g, red = arr.mean(axis=(0, 1))
        self.assertGreater(red, 150)
        self.assertLess(max(b, g), 60)

    def test_oversized_jpeg_is_downscaled_not_dropped(self):
        raw = jpeg()
        self.assertGreater(len(raw), 65507, "测试素材必须超过 UDP 单包上限")
        with PreviewSink() as sink:
            r = recorder(sink.port)
            r.preview_jpeg("left", 1.0, raw)
            got = sink.frames()
            r._close_preview()

        self.assertEqual(len(got), 1)
        self.assertLessEqual(len(got[0]), 65507)
        arr = cv2.imdecode(np.frombuffer(got[0], dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape[1], 640)        # preview_width

    def test_small_jpeg_is_forwarded_verbatim(self):
        raw = jpeg(320, 240, quality=60)
        self.assertLessEqual(len(raw), 65000)
        with PreviewSink() as sink:
            r = recorder(sink.port)
            r.preview_jpeg("left", 1.0, raw)
            got = sink.frames()
            r._close_preview()
        self.assertEqual(got, [raw])

    def test_unconfigured_camera_is_not_pushed(self):
        with PreviewSink() as sink:
            r = recorder(sink.port)
            r.preview_rgb("bright", 1.0, np.zeros((64, 64, 3), dtype=np.uint8))
            self.assertEqual(sink.frames(), [])
            r._close_preview()

    def test_h264_decode_loop_pushes_preview(self):
        """H.264 解码线程必须推预览 —— 这正是现场漏掉的那条路径。"""
        from unittest.mock import patch

        from .test_hal_h264 import packets

        with PreviewSink() as sink:
            r = recorder(sink.port)
            r._on_stream_start({
                "source": "hal",
                "topics": {r.config.camera_topics[0]: "h264/annexb"}})
            data = packets(3)
            for i, (payload, keyframe) in enumerate(data):
                r._queue_h264(0, payload, i / 30, False, keyframe)
            r._decode_stop.set()
            with patch.object(r, "_enqueue_h264_rgb"):
                r._stop_camera_decoders()
            got = sink.frames()
            r._close_preview()

        self.assertFalse(r._ended, r._ended_reason)
        # 限流 30Hz,三帧时间戳跨 2/30s → 第一帧必推,后两帧按闸门取舍
        self.assertGreaterEqual(len(got), 1)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(got[0], dtype=np.uint8),
                                          cv2.IMREAD_COLOR))


if __name__ == "__main__":
    unittest.main()
