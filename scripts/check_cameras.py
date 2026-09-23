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
    net_ego_headband_recorder 相同的握手协议;四路鱼眼各占一格

打开顺序 realsense/depthai 在前、opencv 收尾:RealSense 的彩色头在
Windows 上就是一个可枚举的 UVC 摄像头,若 DSHOW 先抢到它,librealsense
会 start 成功却永远等不到帧 —— 同一台相机被两套栈打开。UVC 流独占,
专用 SDK 先拿走后,opencv 打到该索引时读不到帧自动跳过。

显示时所有画面拼成一张网格大图(单窗口),每格标注来源、分辨率,以及
recorders.yaml 里对应的槽位名(对不上时显示 no-slot,配重了显示 a+b)。
按 q / Esc 退出;开不开画面对采集程序毫无影响,放心开着。

无法导入的 SDK 会标注「未安装」——那类相机在此机器上本来就录不了。
"""

from __future__ import annotations

import argparse
import json
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
    → stream_start 后,常驻一个读线程排空 socket,每路相机只保留最新一帧
    的 JPEG 解码结果 —— 拼屏循环按需来取,不落盘、不转码。
    """

    def __init__(self, host: str, port: int, camera_topics, connect_timeout: float):
        self.host = host
        self.port = port
        self.camera_topics = list(camera_topics)
        self.connect_timeout = connect_timeout
        self.synced: bool | None = None    # 设备时钟同步结果(仅展示用)
        self.advertised: dict = {}         # stream_start 报告的全部 topic
        self.error = ""
        self._net = None                   # net_ego_headband_recorder 模块
        self._sock: socket.socket | None = None
        self._rx = bytearray()
        self._latest: dict = {}            # topic -> 解码后的 BGR ndarray
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 握手(与 recorder 的 _handshake 同流程,不要求同步成功)----

    def open(self) -> bool:
        try:
            from embodied_brain_collect.recorders.ego_headband import \
                net_ego_headband_recorder as net
        except Exception as exc:
            self.error = f"recorder 模块导入失败 — {exc}"
            return False
        self._net = net
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 不带 capabilities:体检只要画面,不跟设备谈麦克风
            self._send({"kind": "hello", "version": net._VERSION,
                        "capabilities": []})
        except OSError as exc:
            self.error = f"tcp://{self.host}:{self.port} — {exc}"
            self._close_sock()
            return False
        deadline = time.time() + self.connect_timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self.error = "握手超时(没等到 stream_start)"
                self._close_sock()
                return False
            try:
                meta, _ = self._recv_frame(remaining)
            except (OSError, EOFError, ValueError) as exc:
                self.error = f"握手失败 — {type(exc).__name__}: {exc}"
                self._close_sock()
                return False
            kind = meta.get("kind")
            if kind == "time_probe":
                try:
                    self._send({"kind": "time_reply", "id": meta.get("id"),
                                "t2": net._wall_ns(), "t3": net._wall_ns()})
                except OSError:
                    pass
            elif kind == "sync_result":
                self.synced = bool(meta.get("synced"))
            elif kind == "stream_start":
                self.advertised = dict(meta.get("topics") or {})
                break
            elif kind == "error":
                self.error = f"server error: {meta.get('message')}"
                self._close_sock()
                return False
            # 'status' 或其他:继续等。
        self._thread = threading.Thread(
            target=self._reader_loop, name="headband-check-reader", daemon=True)
        self._thread.start()
        return True

    # ---- 常驻读线程:排空 socket,相机 topic 只留最新解码帧 ----

    def _reader_loop(self) -> None:
        import cv2
        import numpy as np
        net = self._net
        pending: dict = {}                 # topic -> 最新 jpeg,新的覆盖旧的
        while not self._stop.is_set():
            got = self._recv(0.2)
            if got is None:
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
                if kind == "time_probe":
                    # 服务器可能中途重新对时,照实回
                    try:
                        self._send({"kind": "time_reply", "id": meta.get("id"),
                                    "t2": net._wall_ns(), "t3": net._wall_ns()})
                    except OSError:
                        pass
                elif kind == "data":
                    topic = meta.get("topic")
                    if topic in self.camera_topics and payload:
                        pending[topic] = payload
            # 解码放在排空之后:1080p JPEG 逐帧解(~15ms/帧)追不上
            # 4x30fps 的入流,只解每路最新一帧才是能跟上的节奏
            for topic, payload in pending.items():
                img = cv2.imdecode(
                    np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with self._lock:
                        self._latest[topic] = img
            pending.clear()

    # ---- 拼屏取帧 / 关闭 ----

    def read(self, topic: str):
        with self._lock:
            return self._latest.get(topic)

    def close(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=2.0)
        self._close_sock()

    # ---- 帧格式(与 recorder 同一套,常量从 recorder 模块取)----

    def _send(self, meta: dict) -> None:
        raw = json.dumps(meta, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        self._sock.sendall(self._net._HEADER.pack(len(raw), 0) + raw)

    def _recv(self, timeout: float):
        if self._sock is None:
            return None
        try:
            self._sock.settimeout(timeout)
            chunk = self._sock.recv(65536)
        except (socket.timeout, BlockingIOError):
            return False
        except OSError:
            return None
        if not chunk:
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
        payload = bytes(buf[header.size + m:total])
        del buf[:total]
        return meta, payload

    def _recv_frame(self, timeout: float):
        deadline = time.time() + timeout
        while True:
            frame = self._take_frame()
            if frame is not None:
                return frame
            r = self._recv(max(0.01, deadline - time.time()))
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
    reported = {src.label: False for src in sources}  # 失败只在状态变化时报一次
    last = {}                                          # label -> 最近一张成功帧
    misses = {src.label: 0 for src in sources}         # 连续没取到新帧的次数
    while True:
        if seconds and time.time() - t0 >= seconds:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
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
                last[lbl] = img
                misses[lbl] = 0
                reported[lbl] = False
            else:
                misses[lbl] += 1      # poll 空拍是常态(帧率<循环率),不算失败
            show_img = last.get(lbl)
            if show_img is not None and misses[lbl] < 30:
                h, w = show_img.shape[:2]
                scale = min(cell_w / w, (cell_h - 26) / h)
                tw, th = max(1, int(w * scale)), max(1, int(h * scale))
                if (tw, th) != (w, h):
                    show_img = cv2.resize(show_img, (tw, th))
                ox = x0 + (cell_w - tw) // 2
                oy = y0 + 26 + (cell_h - 26 - th) // 2
                canvas[oy:oy + th, ox:ox + tw] = show_img
                fps = f" @{src.fps:.0f}fps" if src.fps and src.fps > 0 else ""
                tag, color = f"{lbl}  {src.res[0]}x{src.res[1]}{fps}", \
                    (255, 255, 255)
            else:
                if not reported[lbl]:
                    print(f"  {lbl}: 读帧失败,持续重试 ...")
                    reported[lbl] = True
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
        time.sleep(0.005)
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
        return 0

    import cv2
    try:
        show(sources, args.seconds, cv2)
    finally:
        release(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
