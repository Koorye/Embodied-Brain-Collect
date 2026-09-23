"""Real EGO headband test — connect to the device, record (with microphone), verify.

``test_dummy_ego_headband.py`` exercises the *synthetic* recorder (color bars +
sinusoid IMU) so QC can run without hardware; it never touches the real
headband, and being a pytest module it does nothing when run with ``python``.
This script is the real-device counterpart: it connects to the wired-TCP
headband, records every stream for a few seconds — **including the microphone**
— saves the mp4/npz/wav to a fresh session dir, then prints an integrity report
(device-side stream summary + on-disk verification) and exits non-zero on any
problem.

Run from the repo root::

    python tests/ego_headband/test_net_ego_headband.py                 # 10s, audio on
    python tests/ego_headband/test_net_ego_headband.py --duration 20   # longer take
    python tests/ego_headband/test_net_ego_headband.py --host 192.168.55.9
    python tests/ego_headband/test_net_ego_headband.py --no-audio      # cameras+IMU only

Ctrl+C during the take still saves what was received.
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Run this file directly (``python tests/ego_headband/test_net_ego_headband.py``)
# even though the package is not pip-installed: prepend the src/ tree, mirroring
# pyproject's pytest ``pythonpath = ["src", "."]``. Inserting at the front also
# guarantees we exercise THIS working tree, not a stale editable-install copy.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from embodied_brain_collect.recorders.ego_headband import (  # noqa: E402
    EgoHeadbandRecorderConfig, NetEgoHeadbandRecorder)

# tests/sessions/ego_headband — same SESSION_DIR convention as tests/base.py,
# but computed from __file__ so the script also runs via a direct path.
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "sessions" / "ego_headband"


def _device_summary(rec: NetEgoHeadbandRecorder) -> None:
    """Print what the device actually sent, straight from the recorder's
    post-run internals (standalone run() keeps them all on the instance)."""
    print("\n=== 设备侧数据流 (来自录制器) ===")
    print(f"  时钟同步 synced = {rec._synced}"
          + (f"   open_error = {rec._open_error}" if rec._open_error else ""))
    if rec._topics:
        print(f"  stream_start 话题 ({len(rec._topics)}):")
        for topic, kind in sorted(rec._topics.items()):
            print(f"      {topic}  [{kind}]")
    if rec._unknown:
        print(f"  未配置的忽略话题: {sorted(rec._unknown)}")
    print(f"  相机收帧: {dict(sorted(rec._cam_count.items())) or '(无)'}")
    if rec._cam_decode_err:
        print(f"  相机解码失败: {dict(sorted(rec._cam_decode_err.items()))}")
    if rec._cam_empty:
        print(f"  相机空 payload: {dict(sorted(rec._cam_empty.items()))}")
    if rec._cam_decode_dropped:
        print(f"  相机解码队列丢帧: {rec._cam_decode_dropped}")
    print(f"  IMU 采样: {dict(sorted(rec._imu_count.items())) or '(无)'}")
    if rec._gaps:
        print(f"  stream_seq 断流: {dict(sorted(rec._gaps.items()))}")
    print(f"  麦克风可用 = {rec._audio_available}"
          + (f"   audio_error = {rec._audio_error}" if rec._audio_error else ""))


def _verify(out_dir: Path, cfg: EgoHeadbandRecorderConfig,
            duration: float) -> list[str]:
    """Check the on-disk output; return a list of problems (empty = all good)."""
    problems: list[str] = []
    npz_path = out_dir / "ego_headband.npz"
    if not npz_path.is_file():
        print(f"\n[FAIL] 没找到 NPZ: {npz_path}")
        return [f"NPZ 未生成 ({npz_path})"]
    z = np.load(npz_path)
    files = set(z.files)

    print("\n--- 相机 (mp4 + timestamps) ---")
    cam_ok = 0
    for name in cfg.camera_names:
        mp4 = out_dir / f"{name}.mp4"
        ts_key = f"{name}_timestamps"
        n_ts = int(z[ts_key].size) if ts_key in files else 0
        size = mp4.stat().st_size if mp4.is_file() else 0
        if size > 0 and n_ts > 0:
            cam_ok += 1
            print(f"  {str(name):<8} OK    {n_ts:>6} 帧   {size / 1e6:5.1f} MB")
        else:
            why = []
            if size == 0:
                why.append("无 mp4" if not mp4.is_file() else "mp4 为空")
            if n_ts == 0:
                why.append("无时间戳")
            print(f"  {str(name):<8} MISS  ({'/'.join(why)})")
            problems.append(f"相机 {name} 没有数据 ({'/'.join(why)})")
    if cam_ok == 0:
        problems.append("所有相机都没有数据")

    print("\n--- IMU (npz) ---")
    for j in range(cfg.n_imus):
        ts, gy, ac = f"imu{j}_ts", f"imu{j}_gyro", f"imu{j}_accel"
        if ts in files and gy in files and ac in files and z[ts].size > 0:
            print(f"  imu{j}    OK    ts={z[ts].size}  gyro={z[gy].shape}  "
                  f"accel={z[ac].shape}")
        else:
            print(f"  imu{j}    MISS")
            problems.append(f"imu{j} 没有数据")

    print("\n--- 录音 (microphone.wav) ---")
    if not cfg.audio_enabled:
        print("  (未启用录音)")
    else:
        wav = out_dir / "microphone.wav"
        err = str(z["microphone_error"]) if "microphone_error" in files else ""
        if wav.is_file() and wav.stat().st_size > 44:
            with wave.open(str(wav), "rb") as w:
                frames, rate = w.getnframes(), w.getframerate()
                ch, width = w.getnchannels(), w.getsampwidth()
            secs = frames / rate if rate else 0.0
            print(f"  OK    {secs:.2f}s   {frames} samples  "
                  f"@{rate}Hz {ch}ch {width * 8}bit")
            if err:
                problems.append(f"microphone_error: {err}")
            if secs < duration * 0.5:
                problems.append(
                    f"录音时长 {secs:.1f}s 远短于录制时长 {duration:.1f}s")
        else:
            print(f"  MISS  (无 microphone.wav 或为空)"
                  + (f"   error={err}" if err else ""))
            problems.append("没有录到音频 (microphone.wav 缺失/为空)"
                            + (f" — {err}" if err else ""))

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真机 EGO 头环录制+录音校验")
    ap.add_argument("--host", default="192.168.55.6", help="设备地址")
    ap.add_argument("--port", type=int, default=5577, help="设备推流端口")
    ap.add_argument("--duration", type=float, default=10.0, help="录制秒数")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="会话输出根目录")
    ap.add_argument("--no-audio", action="store_true", help="不启用录音")
    args = ap.parse_args(argv)

    audio = not args.no_audio
    # 每次跑落到带时间戳的新目录:MicrophoneWriter 用 'xb' 独占创建 wav,
    # 复用旧目录会因文件已存在而录音失败。
    session_dir = Path(args.out) / time.strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = session_dir / "ego_headband"

    cfg = EgoHeadbandRecorderConfig(
        session_dir=str(session_dir), duration=args.duration,
        host=args.host, port=args.port,
        audio_enabled=audio)

    print("=" * 60)
    print("EGO 头环真机录制校验")
    print(f"  设备:   tcp://{cfg.host}:{cfg.port}")
    print(f"  时长:   {cfg.duration:g}s   (Ctrl+C 提前停止并保存)")
    print(f"  录音:   {'开' if audio else '关'}")
    print(f"  输出:   {out_dir}")
    print("=" * 60)

    rec = NetEgoHeadbandRecorder(cfg)
    rc = rec.run()                      # open -> record(live heartbeat) -> save

    _device_summary(rec)
    problems = _verify(out_dir, cfg, cfg.duration)

    if rc != 0:
        problems.insert(0, f"录制器返回码 {rc} (open/record 失败,见上方日志)")
    if cfg.audio_enabled and not rec._audio_available:
        problems.append("设备未提供麦克风话题 /microphone/audio (audio/PCM)")

    ok = not problems
    print("\n=== 结果:", "PASS ===" if ok else "FAIL ===")
    for p in problems:
        print("  -", p)
    print(f"\n会话目录: {session_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
