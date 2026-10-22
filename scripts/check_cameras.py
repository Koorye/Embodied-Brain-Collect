#!/usr/bin/env python3
"""相机体检 —— 按 recorders.yaml 打开相机并实时显示画面,不经过 recorder。

    python scripts/check_cameras.py                 # 按 yaml 开窗显示相机
    python scripts/check_cameras.py --list          # 只列出,不开画面
    python scripts/check_cameras.py --seconds 15    # 显示 15 秒后自动退出
    python scripts/check_cameras.py --idx 2         # 忽略 yaml,手动看 USB 索引 2

只开 configs/recorders.yaml 里启用(enabled 不为 false)的相机槽位,
yaml 没配的 kind 一律不碰 —— 不扫全量 USB 索引,也不枚举 SDK 里的
全部设备,看到哪路就等于采集程序会录哪路:

  * opencv_camera —— 只开各槽位 ``idx`` 指定的索引(与采集同一路)
  * realsense_camera —— 配了 serial 精确开;没配与 recorder 同款,默认第一台
  * depthai_camera —— 配了 mxid 精确开;没配与 recorder 同款,默认第一台
  * net_ego_headband —— 连 yaml ego_headband 槽位的 host/port,走与
    net_ego_headband_recorder 相同的握手协议;每次连接校验三个标定文件,
    预览仅保存在内存,不生成录制 session;四路鱼眼各占一格

打开顺序 realsense/depthai 在前、opencv 收尾:RealSense 的彩色头在
Windows 上就是一个可枚举的 UVC 摄像头,若 DSHOW 先抢到它,librealsense
会 start 成功却永远等不到帧 —— 同一台相机被两套栈打开。UVC 流独占,
专用 SDK 先拿走后,opencv 打到该索引时读不到帧自动跳过。

显示时所有画面拼成一张网格大图(单窗口),每格标注来源、分辨率,以及
recorders.yaml 里对应的槽位名(对不上时显示 no-slot,配重了显示 a+b)。
按 q / Esc 退出;请退出体检后再开始正式采集,释放设备连接。

无法导入的 SDK 会标注「未安装」——那类相机在此机器上本来就录不了。
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.recorders.camera.opencv_camera_recorder import (  # noqa: E402
    preferred_backend,
)


@dataclass
class Source:
    """一个打开成功的相机流。"""
    kind: str          # opencv / realsense / depthai / headband
    label: str
    res: tuple[int, int] | None = None
    fps: float = 0.0
    handle: object = None
    pipeline: object = None
    q: object = None
    extra: dict = field(default_factory=dict)


# =============================================================================
# 探测
# =============================================================================

def _probe_opencv(idx: int, cv2, note_conflict: bool = False) -> Source | None:
    cap = cv2.VideoCapture(idx, preferred_backend(cv2))
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()          # 后端可能虚报 isOpened,读到一帧才算数
    if not ok:
        cap.release()
        if note_conflict:
            print(f"  [opencv] idx={idx} 能打开但读不到帧 — "
                  f"多半是被上面某路专用 SDK 占用的画面头,跳过")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return Source("opencv", f"USB idx={idx}", (w, h), fps, handle=cap,
                  extra={"idx": idx})


def _probe_realsense(rs, ctx, wanted: list[tuple[str, dict]]) -> list[Source]:
    """只开 yaml realsense_camera 槽位要的设备:配了 serial 精确开,
    没配 serial 的槽位与 recorder 同款 —— 默认拿枚举到的第一台。"""
    out = []
    try:
        devices = ctx.query_devices()
    except Exception:
        devices = []
    want_sn: list[str] = []            # yaml 点名的串号(去重,保持顺序)
    slot_of: dict[str, str] = {}
    for slot, c in wanted:
        sn = str(c.get("serial") or "")
        if sn and sn not in want_sn:
            want_sn.append(sn)
            slot_of[sn] = slot
    auto_left = 1 if any(not c.get("serial") for _, c in wanted) else 0
    for d in devices:
        try:
            sn = str(d.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
        if want_sn:
            if sn not in want_sn:
                continue               # yaml 点名了,这台不在名单,不开
        elif auto_left <= 0:
            continue
        try:
            pipe = rs.pipeline(ctx)
            cfg = rs.config()
            cfg.enable_device(sn)
            # 只显式使能 color 流(与 realsense_camera_recorder 一致):
            # 裸 config 会让 depth/IR 等默认流全部打开,抢带宽导致不出帧
            cfg.enable_stream(rs.stream.color, 640, 480,
                              rs.format.rgb8, 30)
            profile = pipe.start(cfg)
            s = profile.get_stream(rs.stream.color)
            intr = s.as_video_stream_profile().get_intrinsics()
            fps = float(s.as_video_stream_profile().fps())
            out.append(Source("realsense", f"RealSense {sn}",
                              (intr.width, intr.height), fps,
                              pipeline=pipe, extra={"serial": sn}))
        except Exception as exc:
            print(f"  [realsense] {sn} 打开失败: {exc}")
        if want_sn:
            want_sn.remove(sn)
        else:
            auto_left -= 1
    for sn in want_sn:                 # 循环后剩下的 = 没找到的串号
        print(f"  [yaml] {slot_of.get(sn, '?')} 配置的 serial={sn} 未发现设备")
    return out


def _probe_depthai(dai, wanted: list[tuple[str, dict]]) -> list[Source]:
    """只开 yaml depthai_camera 槽位要的设备:配了 mxid 精确开;没配
    mxid 时 recorder 也是默认拿第一台,这里同样只开一台。"""
    out = []
    try:
        infos = dai.Device.getAllAvailableDevices()
    except Exception:
        infos = []
    want_mx: list[str] = []            # yaml 点名的 mxid(去重,保持顺序)
    slot_of: dict[str, str] = {}
    for slot, c in wanted:
        mx = str(c.get("mxid") or "")
        if mx and mx not in want_mx:
            want_mx.append(mx)
            slot_of[mx] = slot
    auto_left = 1 if any(not c.get("mxid") for _, c in wanted) else 0
    for info in infos:
        name = str(getattr(info, "mxid", None) or info.name)
        if want_mx:
            if name not in want_mx:
                continue               # yaml 点名了,这台不在名单,不开
        elif auto_left <= 0:
            continue
        try:
            device = dai.Device(info)
            pipe = dai.Pipeline(device)      # 绑定到这台设备
            cam = pipe.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A)
            imgout = cam.requestOutput((640, 360), dai.ImgFrame.Type.NV12)
            q = imgout.createOutputQueue(maxSize=4, blocking=True)
            pipe.start()
            frame = q.get()
            img = frame.getCvFrame()
            out.append(Source("depthai", f"DepthAI {name}",
                              (img.shape[1], img.shape[0]), 30.0,
                              handle=device, pipeline=pipe, q=q,
                              extra={"mxid": name}))
        except Exception as exc:
            print(f"  [depthai] {name} 打开失败: {exc}")
        if want_mx:
            want_mx.remove(name)
        else:
            auto_left -= 1
    for mx in want_mx:                 # 循环后剩下的 = 没找到的 mxid
        print(f"  [yaml] {slot_of.get(mx, '?')} 配置的 mxid={mx} 未发现设备")
    return out


class _HeadbandFeed:
    """headband 的轻量 wired-TCP 客户端,只服务本脚本。

    与 net_ego_headband_recorder 说同一套协议(帧格式常量、时钟回复函数
    直接从 recorder 模块导入,协议变了这边跟着变):hello → 回 time_probe
    → 接收并校验三个标定文件 → calibration_ack → stream_start 后,
    常驻一个读线程**只收包、不解码**,每路 H.264 另有独立解码线程。

    解码和收包必须分开:4 路 1920x1200 H.264 全解在弱机上可能追不上
    120fps 的入流,若在读线程里同步解码,socket 就没人排空,积压会一路
    涨到设备端丢帧(实测超过 128MB),随后序号跳变让整条预览(四路)
    一起黑屏。现在解码落后只会丢**本路**的积压帧,并跳到下一个 IDR 重新
    起解 —— 预览永远只显示最新画面,内存有界。
    """

    #: 每路预览输出帧率上限。H.264 必须逐帧解码(参考帧不能跳),但
    #: GPU→CPU 拷贝 + BGR 转换只按这个节奏做;>=15fps 预览已足够顺滑。
    PREVIEW_FPS = 15.0
    #: 每路解码队列上限(帧)。约 1.5s @30fps;满了就丢积压、等下一个 IDR。
    H264_QUEUE_MAX = 45
    #: socket 接收缓冲:吸收开窗/后台负载造成的亚秒级突发。
    SOCK_RCVBUF = 8 * 1024 * 1024

    def __init__(self, host: str, port: int, camera_topics, connect_timeout: float):
        self.host = host
        self.port = port
        self.camera_topics = list(camera_topics)
        self.connect_timeout = connect_timeout
        self.synced: bool | None = None    # 设备时钟同步结果(仅展示用)
        self.advertised: dict = {}         # stream_start 报告的全部 topic
        self.error = ""                    # 连接级致命错误(整条 feed 不可用)
        self.calibrations: list[dict] = []  # 本次连接校验通过的标定清单
        self._net = None                   # net_ego_headband_recorder 模块
        self._sock: socket.socket | None = None
        self._rx = bytearray()
        self._latest: dict = {}            # topic -> 解码后的 BGR ndarray
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # ---- 解码侧(与收包线程分离) ----
        self._decode_stop = threading.Event()
        self._h264_q: dict[str, queue.Queue] = {}
        self._h264_threads: dict[str, threading.Thread] = {}
        self._h264_resync: set[str] = set()   # 等下一个 IDR 再起解
        self._topic_error: dict[str, str] = {}  # 单路解码/落后说明,不杀整条流
        self._jpeg_pending: dict[str, bytes] = {}  # JPEG 只留最新一帧
        self._jpeg_thread: threading.Thread | None = None
        self._last_seq: dict[str, int] = {}        # H.264 topic -> 上一帧序号

    # ---- 握手(与 recorder 的 _handshake 同流程,不要求同步成功)----

    def open(self) -> bool:
        self.close()
        alive = self._thread is not None and self._thread.is_alive()
        alive = alive or any(t.is_alive() for t in self._h264_threads.values())
        alive = alive or (self._jpeg_thread is not None
                          and self._jpeg_thread.is_alive())
        if alive:
            self.error = "上一次预览线程尚未退出,请稍后重试"
            return False
        self._stop.clear()
        self._decode_stop.clear()
        self._rx.clear()
        with self._lock:
            self._latest.clear()
        self._h264_q.clear()
        self._h264_threads.clear()
        self._h264_resync.clear()
        self._topic_error.clear()
        self._jpeg_pending.clear()
        self._last_seq.clear()
        self.error = ""
        self.synced = None
        self.advertised = {}
        self.calibrations = []
        try:
            from embodied_brain_collect.recorders.ego_headband import \
                net_ego_headband_recorder as net
            from embodied_brain_collect.recorders.ego_headband.calibration_receiver import (
                CAPABILITY, CalibrationReceiver,
            )
        except Exception as exc:
            self.error = f"recorder 模块导入失败 — {exc}"
            return False
        self._net = net
        calibrations = CalibrationReceiver()  # 每次连接重新接收,不复用旧标定
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 设备持续高速出流,收包线程偶尔被调度延迟时需要缓冲兜底。
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                      self.SOCK_RCVBUF)
            except OSError:
                pass
            # 必须支持标定传输;预览不请求麦克风。
            self._send({"kind": "hello", "version": net._VERSION,
                        "capabilities": [CAPABILITY]})
        except OSError as exc:
            self.error = f"tcp://{self.host}:{self.port} — {exc}"
            self._close_sock()
            return False
        deadline = time.monotonic() + self.connect_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.error = "握手超时(没等到 stream_start)"
                self._close_sock()
                return False
            try:
                meta, payload = self._recv_frame(remaining)
                kind = meta.get("kind")
                if kind == "time_probe":
                    self._send({"kind": "time_reply", "id": meta.get("id"),
                                "t2": net._wall_ns(), "t3": net._wall_ns()})
                elif kind == "sync_result":
                    self.synced = bool(meta.get("synced"))
                elif kind == "calibration_file":
                    calibrations.add_file(meta, payload)
                elif kind == "calibration_end":
                    entries = calibrations.finish(meta, payload)
                    self._send({"kind": "calibration_ack", "version": 1,
                                "files": entries})
                    self.calibrations = entries
                elif kind == "stream_start":
                    if not calibrations.complete:
                        raise ValueError("缺少完整标定文件,请检查头环 TCP 服务版本")
                    topics = meta.get("topics") or {}
                    if not isinstance(topics, dict):
                        raise ValueError("invalid stream_start topics")
                    self.advertised = dict(topics)
                    break
                elif kind == "error":
                    raise ValueError(f"server error: {meta.get('message')}")
                elif kind == "data":
                    raise ValueError("sensor data arrived before calibration/stream_start handshake")
                # 'status' 或其他:继续等。
            except (OSError, EOFError, ValueError) as exc:
                self.error = f"握手失败 — {type(exc).__name__}: {exc}"
                self._close_sock()
                return False
        self._jpeg_thread = threading.Thread(
            target=self._jpeg_worker, name="headband-check-jpeg", daemon=True)
        self._jpeg_thread.start()
        self._thread = threading.Thread(
            target=self._reader_loop, name="headband-check-reader", daemon=True)
        self._thread.start()
        return True

    # ---- 常驻读线程:只排空 socket + 分发,解码在各路独立线程 ----

    def _reader_loop(self) -> None:
        try:
            self._read_stream()
        except Exception as exc:
            self.error = self.error or f"读流失败 — {type(exc).__name__}: {exc}"
        finally:
            self._close_sock()

    def _read_stream(self) -> None:
        net = self._net
        while not self._stop.is_set():
            got = self._recv(0.2)
            if got is None:
                if not self._stop.is_set():
                    self.error = self.error or "连接已断开"
                return
            if got:
                for _ in range(net._DRAIN_MAX):
                    if not self._recv(0.0):
                        break
            while True:
                try:
                    frame = self._take_frame()
                except ValueError as exc:
                    self.error = f"流解析失败 — {exc}"
                    return
                if frame is None:
                    break
                meta, payload = frame
                kind = meta.get("kind")
                if kind == "error":
                    self.error = f"server error: {meta.get('message')}"
                    return
                elif kind == "time_probe":
                    # 服务器可能中途重新对时,照实回
                    try:
                        self._send({"kind": "time_reply", "id": meta.get("id"),
                                    "t2": net._wall_ns(), "t3": net._wall_ns()})
                    except OSError:
                        pass
                elif kind == "data":
                    topic = meta.get("topic")
                    if topic not in self.camera_topics or not payload:
                        continue
                    if self.advertised.get(topic) == "h264/annexb":
                        if meta.get("type") != "h264/annexb":
                            self._topic_error[topic] = "相机编码类型变更"
                            continue
                        self._dispatch_h264(topic, meta, payload)
                    else:
                        # JPEG 无参考帧依赖,只留最新一帧交给解码线程
                        self._jpeg_pending[topic] = payload

    def _dispatch_h264(self, topic: str, meta: dict, payload: bytes) -> None:
        """收包线程只做序号检查 + 入队,绝不在这里解码。"""
        seq = meta.get("stream_seq")
        if type(seq) is not int or seq < 0:
            self._topic_error[topic] = "H.264 缺少 stream_seq"
            return
        prev = self._last_seq.get(topic)
        keyframe = meta.get("keyframe") is True
        if prev is not None and seq != prev + 1:
            # 客户端消费慢时设备端会丢帧;预览允许跳帧,等下一个 IDR 重开
            self._resync_topic(topic, f"帧序号跳变 {prev}->{seq}")
        self._last_seq[topic] = seq
        if topic in self._h264_resync and not keyframe:
            return
        q = self._h264_q.get(topic)
        if q is None:
            q = queue.Queue(maxsize=self.H264_QUEUE_MAX)
            self._h264_q[topic] = q
            t = threading.Thread(
                target=self._h264_worker, args=(topic, q),
                name=f"headband-check-{topic}", daemon=True)
            self._h264_threads[topic] = t
            t.start()
        try:
            q.put_nowait((payload, keyframe))
        except queue.Full:
            self._resync_topic(topic, "解码积压,跳到下一个 IDR")
            if keyframe:
                # 被挤掉的这帧本身就是重开点:清空积压后单独留下它
                q.put_nowait((payload, keyframe))

    def _resync_topic(self, topic: str, reason: str) -> None:
        """丢光该路积压,标记等 IDR —— 只影响这一路,其余相机照常预览。"""
        self._h264_resync.add(topic)
        self._topic_error[topic] = reason
        q = self._h264_q.get(topic)
        if q is not None:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _h264_worker(self, topic: str, q: queue.Queue) -> None:
        import cv2
        from embodied_brain_collect.recorders.ego_headband.h264_decoder import (
            H264Decoder)
        decoder = None
        last_emit = 0.0
        interval = 1.0 / max(self.PREVIEW_FPS, 1e-6)
        while not self._decode_stop.is_set():
            try:
                payload, keyframe = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if topic in self._h264_resync:
                if not keyframe:
                    continue
                decoder = None      # 从 IDR 重建解码器
            if decoder is None:
                try:
                    decoder = H264Decoder()
                except Exception as exc:
                    self._h264_resync.add(topic)
                    self._topic_error[topic] = f"GPU 解码不可用 — {exc}"
                    continue
            try:
                frame = decoder.decode_frame(payload, keyframe)
            except Exception as exc:
                decoder = None
                self._h264_resync.add(topic)
                self._topic_error[topic] = f"解码失败,等下一个 IDR — {exc}"
                continue
            self._h264_resync.discard(topic)
            self._topic_error.pop(topic, None)
            # 逐帧解码保证参考链完整;GPU→CPU 拷贝 + 转 BGR 按预览帧率限流
            now = time.monotonic()
            if now - last_emit < interval:
                continue
            try:
                rgb = frame.to_ndarray(format="rgb24")
            except Exception as exc:
                self._topic_error[topic] = f"帧拷贝失败 — {exc}"
                continue
            with self._lock:
                self._latest[topic] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            last_emit = now

    def _jpeg_worker(self) -> None:
        import cv2
        import numpy as np
        while not self._decode_stop.is_set():
            got = False
            for topic in self.camera_topics:
                payload = self._jpeg_pending.pop(topic, None)
                if payload is None:
                    continue
                got = True
                img = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    with self._lock:
                        self._latest[topic] = img
                    self._topic_error.pop(topic, None)
            if not got:
                time.sleep(0.01)

    def error_for(self, topic: str) -> str:
        return self._topic_error.get(topic, "")

    # ---- 拼屏取帧 / 关闭 ----

    def read(self, topic: str):
        if self.error or self._stop.is_set():
            return None
        with self._lock:
            return self._latest.get(topic)

    def close(self) -> None:
        self._stop.set()
        self._decode_stop.set()
        # shutdown 先唤醒阻塞接收,线程退出后才能安全复用 feed。
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
            if not t.is_alive():
                self._thread = None
        for topic, worker in list(self._h264_threads.items()):
            worker.join(timeout=2.0)
            if not worker.is_alive():
                self._h264_threads.pop(topic, None)
        jpeg = self._jpeg_thread
        if jpeg is not None:
            jpeg.join(timeout=2.0)
            if not jpeg.is_alive():
                self._jpeg_thread = None
        # 活着的线程仍持有自己的队列,字典留给 open() 重置;这里只清待解数据
        self._jpeg_pending.clear()
        self._close_sock()

    def _close_sock(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ---- 帧格式(与 recorder 同一套,常量从 recorder 模块取)----

    def _send(self, meta: dict) -> None:
        raw = json.dumps(meta, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        self._sock.sendall(self._net._HEADER.pack(len(raw), 0) + raw)

    def _recv(self, timeout: float):
        sock = self._sock
        if sock is None:
            return None
        try:
            sock.settimeout(timeout)
            chunk = sock.recv(65536)
        except (socket.timeout, BlockingIOError):
            return False
        except OSError:
            return None
        if not chunk:
            if not self._stop.is_set():
                self.error = self.error or "设备关闭了连接"
            return None
        self._rx.extend(chunk)
        return True

    def _take_frame(self):
        buf = self._rx
        header = self._net._HEADER
        if len(buf) < header.size:
            return None
        m, n = header.unpack_from(buf, 0)
        if not 0 < m <= self._net._MAX_META or n > self._net._MAX_PAYLOAD:
            raise ValueError(f"invalid frame length (meta={m} payload={n})")
        total = header.size + m + n
        if len(buf) < total:
            return None
        meta = json.loads(bytes(buf[header.size:header.size + m]))
        if not isinstance(meta, dict):
            raise ValueError("frame metadata must be a JSON object")
        payload = bytes(buf[header.size + m:total])
        del buf[:total]
        return meta, payload

    def _recv_frame(self, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            frame = self._take_frame()
            if frame is not None:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("handshake timed out before a complete frame")
            r = self._recv(remaining)
            if r is None:
                raise EOFError("connection ended during handshake")


def _headband_slot_cfg() -> dict:
    """yaml 的 ego_headband 槽位;没启用(注释/缺失)返回 {}。"""
    try:
        from embodied_brain_collect.session.config import load_recorders
        return load_recorders().get("ego_headband") or {}
    except Exception:
        return {}


def _probe_headband() -> list[Source]:
    """EGO 头戴:一条 TCP 连接推四路鱼眼,每路建一个 Source(共用读线程)。"""
    try:
        import cv2  # noqa: F401  解码在读线程,这里只确认 cv2 可用
    except Exception:
        print("[headband] opencv 未安装,无法解码 JPEG 预览")
        return []
    try:
        from embodied_brain_collect.recorders.ego_headband.ego_headband_recorder_config import (
            EgoHeadbandRecorderConfig as _Cfg,
        )
    except Exception as exc:
        print(f"[headband] recorder 模块不可用 — {exc}")
        return []

    slot_cfg = _headband_slot_cfg()
    host = slot_cfg.get("host", _Cfg.host)
    port = int(slot_cfg.get("port", _Cfg.port))
    topics = tuple(slot_cfg.get("camera_topics") or _Cfg.camera_topics)
    names = tuple(slot_cfg.get("camera_names") or _Cfg.camera_names)
    # 体检工具不等 recorder 的 10s:设备不在线时快点放弃,别拖住整个扫描
    timeout = float(slot_cfg.get("connect_timeout", 5.0))

    print(f"  [headband] 正在连接 tcp://{host}:{port} ...")
    feed = _HeadbandFeed(host, port, topics, timeout)
    if not feed.open():
        print(f"  [headband] {feed.error}")
        return []
    print(f"  [headband] 已校验 {len(feed.calibrations)} 个标定文件(仅预览,保存在内存)")

    out = []
    for i, topic in enumerate(topics):
        if topic not in feed.advertised:
            print(f"  [headband] {topic}: 设备未发布(missing_publishers)")
            continue
        name = names[i] if i < len(names) else f"cam{i}"
        out.append(Source("headband", f"Headband {name}", None, _Cfg.cam_fps,
                          extra={"feed": feed, "topic": topic}))
    if not out:
        feed.close()
        return []
    # 等首帧,拿到真实分辨率(顺带确认流真的在出图)
    deadline = time.time() + 3.0
    while time.time() < deadline and any(s.res is None for s in out):
        for s in out:
            if s.res is None:
                img = feed.read(s.extra["topic"])
                if img is not None:
                    s.res = (img.shape[1], img.shape[0])
        time.sleep(0.05)
    for s in out:
        if s.res is None:
            print(f"  [headband] {s.label}: topic 在流里但 3s 内无首帧")
    return out


#: yaml kind → 探测 kind(体检探测与槽位对应共用一张表)
_KIND_OF = {"opencv_camera": "opencv", "realsense_camera": "realsense",
            "depthai_camera": "depthai", "net_ego_headband": "headband"}
_SLOT_OF_KIND = {v: k for k, v in _KIND_OF.items()}   # 反向:探测 kind → yaml


def probe_all(slots: dict) -> list[Source]:
    """只开 yaml 启用的相机槽位,yaml 没配的 kind 一律不碰。

    打开顺序仍是 realsense/depthai 在前、opencv 收尾:RealSense 的彩色头
    在 Windows 上就是一个可枚举的 UVC 摄像头,若 DSHOW 先抢到它,
    librealsense 会 start 成功却永远等不到帧 —— 同一台相机被两套栈打开。
    UVC 流独占,专用 SDK 先拿走后,opencv 打到该索引时读不到帧自动跳过。
    headband 走网络,与 UVC 无冲突,插在 depthai 之后开。
    """
    if not slots:
        print("[yaml] recorders.yaml 里没有启用的相机槽位,不开任何相机")
        return []
    by_kind: dict[str, list[tuple[str, dict]]] = {}
    for slot, c in slots.items():
        want = _KIND_OF.get(str(c.get("kind")))
        if want:
            by_kind.setdefault(want, []).append((slot, c))

    found: list[Source] = []
    if "realsense" in by_kind:
        try:
            import pyrealsense2 as rs
            found += _probe_realsense(rs, rs.context(), by_kind["realsense"])
        except Exception:
            print("[realsense] SDK 未安装")
    if "depthai" in by_kind:
        try:
            import depthai as dai
            found += _probe_depthai(dai, by_kind["depthai"])
        except Exception:
            print("[depthai] SDK 未安装")
    if "headband" in by_kind:
        found += _probe_headband()
    if "opencv" in by_kind:
        try:
            import cv2
        except Exception:
            cv2 = None
        if cv2 is None:
            print("[opencv] 未安装 (pip install opencv-python)")
        else:
            idxs: list[int] = []       # yaml 点名的索引,去重并保持槽位顺序
            for _, c in by_kind["opencv"]:
                idx = int(c.get("idx", 0))
                if idx not in idxs:
                    idxs.append(idx)
            has_rs = any(s.kind == "realsense" for s in found)
            for idx in idxs:
                s = _probe_opencv(idx, cv2, note_conflict=has_rs)
                if s:
                    found.append(s)
    return found


# =============================================================================
# 与 recorders.yaml 槽位对应
# =============================================================================

def _is_camera_slot(cfg: dict) -> bool:
    """是不是体检关心的相机槽位(net_ego_headband 不带 _camera 后缀但也是)。"""
    kind = str(cfg.get("kind", ""))
    return kind.endswith("_camera") or kind == "net_ego_headband"


def _camera_slots() -> dict:
    """recorders.yaml 里启用的相机槽位 {slot名: 配置dict};读不到返回空。

    只收 enabled 不为 false 的槽位 —— 与采集同口径:停用的槽位不开,
    也不参与槽位对应。
    """
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        return {}
    return {k: v for k, v in cfg.items()
            if v.get("enabled", True) and _is_camera_slot(v)}


def _disabled_camera_slots() -> list[str]:
    """yaml 里 enabled: false 的相机槽位名(提示操作员为什么没开)。"""
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        return []
    return [k for k, v in cfg.items()
            if not v.get("enabled", True) and _is_camera_slot(v)]


def slot_tag(src: Source, slots: dict, sources: list[Source]) -> str:
    """这台相机对应 yaml 里哪个槽位;对不上返回「no-slot」。

    opencv 按 idx 精确匹配(配重了显示成 a+b,配置错误一眼可见);
    realsense 按串号、depthai 按 mxid;headband 没有识别号,整台设备
    (四路共用一个连接)对上唯一槽位即默认对应;没配识别号且该类只有
    一台设备一个槽位时默认对应,否则留给人工核对。
    """
    want = _SLOT_OF_KIND[src.kind]
    hits = []
    for slot, c in slots.items():
        if c.get("kind") != want:
            continue
        if src.kind == "opencv" and c.get("idx") == src.extra.get("idx"):
            hits.append(slot)
        elif src.kind == "realsense" and c.get("serial") and \
                str(c["serial"]) == str(src.extra.get("serial")):
            hits.append(slot)
        elif src.kind == "depthai" and c.get("mxid") and \
                str(c["mxid"]) == str(src.extra.get("mxid")):
            hits.append(slot)
    if not hits:
        same_slots = [s for s, c in slots.items() if c.get("kind") == want]
        if src.kind == "headband":
            # 四路 Source 共用一个 TCP 连接,按连接数算设备台数
            n_dev = len({id(x.extra.get("feed")) for x in sources
                         if x.kind == "headband"})
        else:
            n_dev = sum(1 for x in sources if x.kind == src.kind)
        if len(same_slots) == 1 and n_dev == 1:
            hits = same_slots
    return "+".join(hits) if hits else "no-slot"


# =============================================================================
# 显示
# =============================================================================

def _read(src: Source, cv2) -> tuple[bool, object]:
    if src.kind == "opencv":
        return src.handle.read()
    if src.kind == "realsense":
        try:
            frames = src.pipeline.poll_for_frames()   # 非阻塞:单路挂了不拖死拼屏
        except Exception:
            return False, None
        f = frames.get_color_frame()
        if not f:
            return False, None
        import numpy as np
        return True, np.asanyarray(f.get_data())
    if src.kind == "depthai":
        try:
            frame = src.q.get()
            return True, frame.getCvFrame()
        except Exception:
            return False, None
    if src.kind == "headband":
        img = src.extra["feed"].read(src.extra["topic"])
        if img is not None and src.res is None:
            src.res = (img.shape[1], img.shape[0])
        return img is not None, img
    return False, None


def show(sources: list[Source], seconds: float, cv2) -> None:
    """所有画面拼成一张网格大图,单窗口显示。"""
    if not sources:
        print("\n没有找到任何相机。")
        return
    import math
    import numpy as np
    n = len(sources)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cell_w = min(640, 1600 // cols)
    cell_h = min(480, 900 // rows)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    print(f"\n拼屏显示 {n} 路(单窗口)— q / Esc 退出" +
          (f"(约 {seconds:g}s 后自动关闭)" if seconds else ""))
    t0 = time.time()
    reported = {i: False for i in range(n)}   # 失败只在状态变化时报一次
    last = {}                                 # index -> 最近一张成功帧
    last_ok = {}                              # index -> 该帧的单调时间
    scaled = {}                               # index -> (源帧, 缩放后帧)
    stale_after = 1.0
    while True:
        if seconds and time.time() - t0 >= seconds:
            break
        # 单次等待同时节流:不再按 150fps 反复 resize 整屏,给解码让出 CPU
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        now = time.monotonic()
        canvas[:] = 0
        for i, src in enumerate(sources):
            r, c = divmod(i, cols)
            x0, y0 = c * cell_w, r * cell_h
            lbl = src.label
            slot = src.extra.get("slot")
            if slot:
                lbl = f"{lbl} [{slot}]"
            ok, img = _read(src, cv2)
            if ok:
                last[i] = img
                last_ok[i] = now
                reported[i] = False
            show_img = last.get(i)
            if show_img is not None and now - last_ok.get(i, 0.0) < stale_after:
                cached = scaled.get(i)
                if cached is None or cached[0] is not show_img:
                    h, w = show_img.shape[:2]
                    scale = min(cell_w / w, (cell_h - 26) / h)
                    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
                    small = (cv2.resize(show_img, (tw, th))
                             if (tw, th) != (w, h) else show_img)
                    cached = (show_img, small)
                    scaled[i] = cached
                small = cached[1]
                th, tw = small.shape[:2]
                ox = x0 + (cell_w - tw) // 2
                oy = y0 + 26 + (cell_h - 26 - th) // 2
                canvas[oy:oy + th, ox:ox + tw] = small
                fps = f" @{src.fps:.0f}fps" if src.fps and src.fps > 0 else ""
                tag, color = f"{lbl}  {src.res[0]}x{src.res[1]}{fps}", \
                    (255, 255, 255)
            else:
                if not reported.get(i, False):
                    error = ""
                    if src.kind == "headband":
                        feed = src.extra["feed"]
                        error = (feed.error_for(src.extra["topic"])
                                 or feed.error)
                    print(f"  {lbl}: {error or '读帧失败,持续重试 ...'}")
                    reported[i] = True
                cv2.putText(canvas, f"{lbl}  no frame",
                            (x0 + 16, y0 + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tag, color = f"{lbl}  no frame", (0, 0, 255)
            cv2.rectangle(canvas, (x0 + 1, y0 + 1),
                          (x0 + cell_w - 2, y0 + 25), (40, 40, 44), -1)
            cv2.putText(canvas, tag, (x0 + 10, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                          (70, 70, 78), 1)
        cv2.imshow("check_cameras  [q/Esc quit]", canvas)
    cv2.destroyAllWindows()


def release(sources: list[Source]) -> None:
    for src in sources:
        try:
            if src.kind == "opencv":
                src.handle.release()
            elif src.kind == "realsense":
                src.pipeline.stop()
            elif src.kind == "depthai":
                try:
                    if src.pipeline is not None:
                        src.pipeline.stop()
                except Exception:
                    pass
                try:
                    if src.handle is not None:
                        src.handle.close()
                except Exception:
                    pass
            elif src.kind == "headband":
                src.extra["feed"].close()   # 幂等,四路共用、关四次无妨
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="只列出相机,不开画面")
    ap.add_argument("--idx", type=int, default=None,
                    help="忽略 yaml,手动只探测这一个 USB 索引(排查新相机用)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="显示 N 秒后自动退出(0 = 直到按 q)")
    args = ap.parse_args(argv)

    print("正在按 recorders.yaml 探测相机 ...")
    slots = _camera_slots()
    disabled = _disabled_camera_slots()
    if disabled:
        print(f"  [yaml] 以下相机槽位 enabled: false,本次不开: "
              f"{'、'.join(disabled)}")
    if args.idx is not None:
        import cv2
        sources = [s for s in [_probe_opencv(args.idx, cv2)] if s]
        for s in sources:
            s.extra["slot"] = slot_tag(s, slots, sources)
        print(f"  idx={args.idx}: " + ("无" if not sources else
              f"{sources[0].res[0]}x{sources[0].res[1]} @ "
              f"{sources[0].fps:.0f}fps  [{sources[0].extra.get('slot')}]"))
    else:
        sources = probe_all(slots)
        for s in sources:
            s.extra["slot"] = slot_tag(s, slots, sources)
        if sources:
            for s in sources:
                res = f"{s.res[0]}x{s.res[1]}" if s.res else "无首帧"
                print(f"  {s.kind:<9} {s.label:<22} "
                      f"{res} @ {s.fps:.0f}fps"
                      f"  [{s.extra.get('slot')}]")
        else:
            print("  (空)")
        for slot, c in slots.items():
            if c.get("kind") == "opencv_camera" and not any(
                    x.kind == "opencv" and x.extra.get("idx") == c.get("idx")
                    for x in sources):
                print(f"  [yaml] {slot} 配置的 idx={c.get('idx')} 本次未打开"
                      f"(没接 / 被占用 / DSHOW 枚举抖动)")

    if args.list:
        release(sources)
        return 0

    import cv2
    try:
        show(sources, args.seconds, cv2)
    finally:
        release(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
