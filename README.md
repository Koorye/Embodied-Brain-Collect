# embodied-brain-collect

多模态数据采集框架：Neon 眼动、RGB/深度相机、EMG、数据手套、位置跟踪、任务标记。
一个 launcher 并行管理所有传感器（每传感器独立进程），首条数据就绪后统一开录，
视频统一 libx265 帧精确编码，时间戳全为绝对 unix 时间。

详细设计见 [`REFACTOR_README.md`](REFACTOR_README.md)。

## 目录结构

```
src/embodied_brain_collect/
├── recorders/     # 各模态 recorder（eye/camera/emg/hand_pose/position/marker/tactile）
├── session/       # launcher + recorder 工厂（presets）
├── stim/          # 刺激程序（paradigm1）
├── config/        # 任务库配置
├── checkers/      # 数据检查工具
└── visualizers/   # 会话可视化
tests/             # 交互式硬件测试（GUI）
```

## 安装

依赖：Python ≥ 3.10，系统需安装 `ffmpeg`（含 libx265）。

```bash
conda create -n Embodied-Brain-Collect python=3.10
conda activate Embodied-Brain-Collect
pip install -r requirements.txt
pip install -e .          # 可编辑安装本包
```

## 使用

```bash
# 1. 全量 dummy 自测（模拟所有传感器）
python -m embodied_brain_collect.session.launcher \
    --dummy --session-dir ./data/dummy --duration 10

# 2. 生产采集（硬件按 production preset；Neon 自动发现）
python -m embodied_brain_collect.session.launcher \
    --session-dir ./data/session1 --recorders eye cam_third marker

# 3. 带刺激程序（刺激结束自动停录）
python -m embodied_brain_collect.session.launcher \
    --session-dir ./data/session1 --with-stim

# 4. 独立运行单个 recorder
python -c "
from embodied_brain_collect.recorders.eye import NeonEyeAsyncRecorder, EyeRecorderConfig
NeonEyeAsyncRecorder(EyeRecorderConfig(session_dir='./out', duration=60)).run()"

# 5. 硬件 GUI 测试（实时曲线 + scene 画面，Q 退出）
python -m tests.eye.test_neon_eye_async
```

自定义组合：`get_*` 工厂可传任意配置参数（见 `session/recorder_presets.py`）：

```python
from embodied_brain_collect.session.launcher import launch
from embodied_brain_collect.session.recorder_presets import (
    get_realsense_camera, get_neon_eye_async, get_udp_marker)

launch({
    "cam":    get_realsense_camera("./data/run1", depth=True, crf=20),
    "eye":    get_neon_eye_async("./data/run1"),
    "marker": get_udp_marker("./data/run1"),
})
```

## 输出格式（`{session_dir}/{slot}/`）

| 传感器 | 文件 |
|---|---|
| 相机 | `frames.mp4`（+`depth_frames.mp4`）、`*_timestamps.txt`、`camera_timestamps.npz` |
| 眼动 | `eye.npz`（gaze/imu/scene 时间戳）、`eye.mp4`、`*_timestamps.txt` |
| 其余 | `<slot>.npz`（数组 + 绝对时间戳） |

视频为 libx265 HEVC，`bframes=0` + 每秒关键帧——容器帧序号与时间戳数组下标严格 1:1。
