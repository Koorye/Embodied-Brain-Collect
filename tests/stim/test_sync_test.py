"""同步测试刺激程序 —— 用 SDL dummy driver 无头跑完整流程。"""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def test_fmt_clock():
    from embodied_brain_collect.stim.sync_test import _fmt
    assert _fmt(0.0) == "00:00.000"
    assert _fmt(12.3456) == "00:12.346"
    assert _fmt(61.5) == "01:01.500"


def test_cycle_layout():
    from embodied_brain_collect.stim import sync_test as S
    from embodied_brain_collect.stim.marker_codes import HAND_ACTIONS, make_hand_cue
    assert list(S._CYCLE) == ["抬左手", "放左手", "抬右手", "放右手"]
    assert list(HAND_ACTIONS) == ["LIFT_LEFT", "PUT_LEFT",
                                  "LIFT_RIGHT", "PUT_RIGHT"]
    # 每轮手势码唯一:第 0/1 轮的同一个动作码不同
    assert make_hand_cue(0, 0) != make_hand_cue(1, 0)


def test_full_headless_run(tmp_path, monkeypatch):
    """完整跑一遍(压缩参数),断言 marker 序列与动作节奏。"""
    import threading
    import time

    import numpy as np

    from embodied_brain_collect.recorders.marker import (
        UdpMarkerRecorder, MarkerRecorderConfig)
    from embodied_brain_collect.stim import sync_test as S

    # 端口 0 = 由操作系统分配空闲端口,避免与其他测试/进程冲突
    rec = UdpMarkerRecorder(MarkerRecorderConfig(
        session_dir=str(tmp_path), host="127.0.0.1", port=0))
    assert rec._open()
    actual_port = rec._sock.getsockname()[1]

    # 隔离的 configs 目录:stim.yaml 指向实际端口 + 压缩参数(sync_test 段)
    monkeypatch.setenv("EMBODIED_BRAIN_COLLECT_CONFIGS", str(tmp_path))
    (tmp_path / "stim.yaml").write_text(
        "udp_host: '127.0.0.1'\n"
        f"udp_port: {actual_port}\n"
        "serial: false\n"
        "sync_test:\n"
        "  fullscreen: false\n"
        "  imag_s: 0.6\n"
        "  cycle_s: 0.4\n"
        "  cycles: 2\n",
        encoding="utf-8")

    stop = threading.Event()

    def drain():
        while not stop.is_set():
            rec._poll(time.time())
            time.sleep(0.002)

    threading.Thread(target=drain, daemon=True).start()
    rc = S.main(["--font-size", "24"])   # 其余参数来自隔离的 stim.yaml
    time.sleep(0.2)
    stop.set()
    rec._close()
    rec._save()

    from embodied_brain_collect.stim.marker_codes import name_of
    z = np.load(tmp_path / "markers" / "markers.npz")
    codes = [name_of(int(c)) for c in z["marker_code"]]
    assert rc == 0
    assert codes[0] == "RUN_START" and codes[-1] == "RUN_END"
    assert codes.count("LIFT_LEFT_1") == 1          # 每轮码唯一
    assert codes.count("LIFT_LEFT_2") == 1
    assert codes.count("PUT_RIGHT_2") == 1
    # 相邻手势事件的间隔 ≈ cycle-s(0.4s),容差 25%
    t = z["marker_t_sent_pc"]
    hand_t = t[(z["marker_code"] >= 0xC0) & (z["marker_code"] <= 0xCF)]
    hand = np.diff(hand_t) * 1000
    assert np.all((hand > 300) & (hand < 500)), hand.tolist()
