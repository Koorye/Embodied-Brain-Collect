"""2x2 live preview monitor for four camera thumbnails (ZMQ PULL).

Layout:
  cam0 (wrist L)  | cam1 (wrist R)
  scene (Neon)    | oak (OAK head)
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from typing import NamedTuple

import cv2
import numpy as np
import zmq

sys.stdout.reconfigure(line_buffering=True)

TOPICS = ("cam0", "cam1", "scene", "oak")
LABELS = {
    "cam0": "wrist cam0",
    "cam1": "wrist cam1",
    "scene": "Neon scene",
    "oak": "OAK head",
}
STALE_SEC = 2.0
DEFAULT_PULL = "tcp://127.0.0.1:9997"


class MonitorInfo(NamedTuple):
    index: int
    left: int
    top: int
    right: int
    bottom: int
    primary: bool


def list_monitors() -> list[MonitorInfo]:
    if sys.platform != "win32":
        return [MonitorInfo(0, 0, 0, 1920, 1080, True)]

    user32 = ctypes.windll.user32
    monitors: list[MonitorInfo] = []

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def _callback(_hmon, _hdc, lprect, _lparam):
        r = lprect.contents
        idx = len(monitors)
        monitors.append(MonitorInfo(
            idx, r.left, r.top, r.right, r.bottom, idx == 0,
        ))
        return 1

    user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_callback), 0)
    if not monitors:
        monitors.append(MonitorInfo(0, 0, 0, 1920, 1080, True))
    return monitors


def print_monitors(monitors: list[MonitorInfo]) -> None:
    print("Available monitors:")
    for m in monitors:
        w = m.right - m.left
        h = m.bottom - m.top
        tag = " PRIMARY" if m.primary else ""
        print(f"  [{m.index}] {w}x{h} at ({m.left},{m.top}){tag}")


def _monitor_origin(monitors: list[MonitorInfo], index: int) -> tuple[int, int]:
    if not monitors:
        return 0, 0
    if index < 0 or index >= len(monitors):
        print(f"[preview] monitor index {index} out of range; using 0")
        index = 0
    m = monitors[index]
    return m.left, m.top


def _blank_cell(cell_w: int, cell_h: int, label: str, stale: bool) -> np.ndarray:
    img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    status = "STALE" if stale else "NO SIGNAL"
    cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, status, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 220), 1)
    return img


def _fit_cell(bgr: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return _blank_cell(cell_w, cell_h, "", False)
    scale = min(cell_w / w, cell_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y0 = (cell_h - nh) // 2
    x0 = (cell_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _compose_grid(
    frames: dict[str, np.ndarray],
    last_ts: dict[str, float],
    cell_w: int,
    cell_h: int,
    now: float,
) -> np.ndarray:
    cells: list[np.ndarray] = []
    for topic in TOPICS:
        label = LABELS[topic]
        ts = last_ts.get(topic, 0.0)
        stale = (now - ts) > STALE_SEC if ts > 0 else True
        bgr = frames.get(topic)
        if bgr is None or stale:
            cells.append(_blank_cell(cell_w, cell_h, label, stale and ts > 0))
        else:
            cell = _fit_cell(bgr, cell_w, cell_h)
            cv2.putText(cell, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1)
            cells.append(cell)
    return np.vstack((np.hstack((cells[0], cells[1])),
                      np.hstack((cells[2], cells[3]))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pull-endpoint", default=DEFAULT_PULL)
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--list-monitors", action="store_true")
    ap.add_argument("--cell-width", type=int, default=320)
    ap.add_argument("--cell-height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args(argv)

    monitors = list_monitors()
    if args.list_monitors:
        print_monitors(monitors)
        return 0

    cell_w, cell_h = args.cell_width, args.cell_height
    win_name = "Record Live Preview (4 cam)"
    mx, my = _monitor_origin(monitors, args.monitor)

    pull = zmq.Context.instance().socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 8)
    pull.bind(args.pull_endpoint)
    print(f"[preview] PULL bind {args.pull_endpoint}  monitor={args.monitor} "
          f"origin=({mx},{my})")

    frames: dict[str, np.ndarray] = {}
    last_ts: dict[str, float] = {}
    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cell_w * 2, cell_h * 2)
    cv2.moveWindow(win_name, mx, my)

    min_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    last_draw = 0.0

    try:
        while True:
            for _sock, _ev in poller.poll(50):
                parts = pull.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    continue
                topic = parts[0].decode("ascii", errors="replace")
                buf = np.frombuffer(parts[1], dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is not None and topic in LABELS:
                    frames[topic] = bgr
                    last_ts[topic] = time.time()

            now = time.time()
            if now - last_draw >= min_interval:
                last_draw = now
                cv2.imshow(win_name, _compose_grid(
                    frames, last_ts, cell_w, cell_h, now))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print("\n[preview] Ctrl+C")
    finally:
        pull.close(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
