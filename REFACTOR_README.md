# 采集框架重构记录（2026-08-17）

本文档记录本次对 `src/recorders/` 与 `src/session/launcher.py` 的完整重构：
设计约定、修改清单、验证结果、已知事项。目标读者：第二天的自己。

---

## 1. 最终架构

```
python -m src.session.launcher --session-dir <dir> [--dummy] [--recorders ...] [--with-stim] [--duration N]
                              └─ launch(recorders, stim_cmd, duration)

父进程（launcher，只做编排）
  ├─ 为每个 recorder 创建独立子进程（multiprocessing/fork，真并行，绕开 GIL）
  ├─ 所有子进程【并行】执行 _open() 首条数据门（各自 open_timeout watchdog）
  │     └─ 任一失败 → 打印具体原因 + 完整 traceback → 中止，退出码 1，不采集
  ├─ 全部 ready → 启动 stim 子进程 → go_evt 放行
  └─ 采集期：聚合子进程心跳为一行状态；stop_event 优雅停止（15s 宽限 → 强杀兜底）

子进程（每个 recorder 一个）
  _open()  ──首条数据门──▶ ready 消息 ──go──▶ _record()
                                              ├─ _setup()（日志/信号/心跳基准）
                                              ├─ _loop()（poll 循环；duration/stop_event 停止）
                                              └─ _teardown()（_close + _save + 日志收尾）
```

- **所有 recorder 走同一条流程**（`_open` → `_record`）；asyncio 型（Neon async）只覆写 `_record()`
- **try/except 全部集中在 base**：`BaseRecorder.run()`（独立运行）与 launcher 子进程都只包一层，
  任何异常经 loguru `opt(exception=True)` 输出完整 traceback 到控制台 + 会话日志文件
- **时间戳约定：所有传感器一律保存绝对 unix 时间**（launcher 传 `time.time()`；相机用抓帧时刻
  主机墙钟；Neon 用手机 unix 时钟 + `pc_to_phone_offset_ms` 可换算）。不再有 0 起点的会话相对时间

---

## 2. 数据格式约定

| 传感器 | 落盘内容 |
|---|---|
| 相机 color | `{slot}/frames.mp4`（HEVC/libx265, yuv420p, CFR, bframes=0, keyint=fps）+ `frames_timestamps.txt` + `camera_timestamps.npz` |
| 相机 depth | `{slot}/depth_frames.mp4`（HEVC 12-bit gray12le）— ⚠️ 深度值**截断到 4095**（见 §6） |
| Neon eye | `{slot}/eye.npz`（gaze/imu/scene 时间戳等）+ `{slot}/eye.mp4`（HEVC/libx265, yuv420p, BGR 输入）+ 各 `*_timestamps.txt` |
| EMG / 手部 / 位置 / 标记 / 触觉 | `{slot}/{slot}.npz`（数组 + 绝对时间戳） |

**帧-时间戳 1:1 保证**：视频走统一的 `FFmpegWriter`（`src/recorders/ffmpeg_writer.py`）：
`-fps_mode cfr -video_track_timescale 90000 -x265-params bframes=0 -g fps -keyint_min fps`，
容器帧序号 == 写入顺序 == 时间戳数组下标。写入跟不上时**丢最旧帧**（有界队列，打印 drop 计数），
时间戳只记实际写入的帧——对齐永不漂移。

---

## 3. 修改清单

### 3.1 日志（loguru 0.7.3，已加入 requirements.txt）
- `src/recorders/base.py`：进程级 loguru 配置（彩色控制台 sink + 全局 excepthook/threading.excepthook）；
  每个 recorder 一个绑定 logger + `{session_dir}/{slot}/{slot}.log` 文件 sink
- `_log(msg, echo=)` 保留原 API（走 loguru）；新增 `_log_exc()`（带完整 traceback）
- launcher 子进程崩溃 → `rec.logger.opt(exception=True).error(...)`，traceback 进会话文件

### 3.2 BaseRecorder（`src/recorders/base.py`）
- 新增 `_record()`：setup → loop → teardown（launcher 子进程与独立 `run()` 共用）；asyncio 型覆写它
- `run()`：open（首条数据门）→ `_record()`，所有异常集中捕获 + traceback
- 新增 `_wait_first_sample(poll_fn, what, timeout)`：`_open` 首条数据门的通用实现
- 心跳统一：每个录制 key 显示 `长度(最新shape)(实时频率)`，频率 = 两次心跳间增量÷间隔；
  子进程经 `_hb_queue`（mp.Queue）上报，父进程聚合为一行
- 删除 `loop_style` 分支（launcher 不再区分 poll/run 型）
- `_loop()` 支持 `stop_event` 优雅停止

### 3.3 Launcher（`src/session/launcher.py`）— threading → multiprocessing
- 每 recorder 一个 fork 子进程：`_open()` 门 → ready → go → `_record()`
- 并行 open + 全就绪闸门 + 每 recorder 独立 open_timeout watchdog + 失败中止（退出码 1）
- 心跳聚合、stop_event 优雅停止、15s 宽限后强杀兜底

### 3.4 视频编码（`src/recorders/ffmpeg_writer.py`，新增）
- `FFmpegWriter(path, w, h, fps, input_pix_fmt, output_pix_fmt, crf, preset)`：
  ffmpeg/libx265 子进程，rawvideo stdin 裸流，帧精确（无 B 帧、CFR、每秒关键帧）
- 相机 color `rgb24→yuv420p`、depth `gray16le→gray12le`；eye scene `bgr24→yuv420p`
- 相机与 eye 共用同一个 writer

### 3.5 相机（`src/recorders/camera/`）
- `BaseCameraRecorder` 重写：`arr_video(key, ts, arr)` 唯一写帧入口——
  每流独立有界队列（maxsize=8）+ 独立 writer 线程（慢流不拖累快流），丢最旧帧保证同步
- **删除遗留 PNG 路径**（`_acc_arr`/`_PNG_SUBDIRS`/PNG 尾部重试）；dummy camera 也走 `arr_video`
- 三个实机实现（realsense/depthai/opencv）`_open` 均含首条数据门（10s）
- DepthAI：`cam_w/cam_h` 真正接线（≥1920 选 1080P，否则 720P），`cam_fps_hint` 生效；
  配置默认改为 1280×720
- `CameraRecorderConfig` 增加 `crf=23`、`preset="medium"`

### 3.6 Neon eye（`src/recorders/eye/`）
- async 版（生产路径）：
  - 删除 `run()` 覆写与 `loop_style`，改覆写 `_record()`（asyncio.run）
  - `_open()` 首条数据门：mDNS 发现 → 传感器连接等待 → 首 gaze 样本（8s）+ 首 scene 帧（15s，相机冷启动 ~5s）；
    发现的设备缓存复用，`_record()` 不再二次发现
  - scene 写盘：解码（worker 线程）→ 预览帧 + 有界队列（4）→ 独立 writer task → `FFmpegWriter`；
    时间戳在 writer 中记账（写进容器的帧才记），丢帧打印计数
  - 删除 `_marker_sub` 残留、`_offset_ms` 残留
- `EyeRecorderConfig`：**删除 `neon_ip`/`port`**（设备 mDNS 自动发现，无处用得到）；
  保留 `no_scene_video`/`crf`/`preset`

### 3.7 其余 recorder 的首条数据门（`_open` 拿到第一条数据才算成功）
| recorder | 门 | 超时 |
|---|---|---|
| weili EMG | 首个完整 EMG 帧 | 5s |
| Manus | 首份手套数据（ergo/skeleton） | 10s |
| RealSense | 首 frameset（含 depth 校验） | 10s |
| DepthAI | 首包 | 10s |
| OpenCV | 首个成功 read | 10s |
| OpenVR | 首个有效 pose | 10s |
| Neon async | 首 gaze 样本 + 首 scene 帧 | 8s + 15s |
| UDP marker | socket 绑定成功（stim 前无数据，豁免） | — |

门内样本在门通过后清空，会话时间线从采集起点干净开始。

### 3.8 配置与 presets
- `EyeRecorderConfig`：删 `neon_ip`/`port`；`PositionRecorderConfig`：删未用的 `device_serials`；
  `TactileRecorderConfig`：清空（TouchTronix 是 stub，参数留待实现时再加）；
  `DepthaiCameraConfig`：`cam_w/cam_h` 生效、默认 1280×720
- `recorder_presets.py`：**每个 `get_*` 透传对应 config 的全部参数**（含 `open_timeout`）；
  新增 `get_dummy_tactile` / `get_touchtronix_tactile`
- 生产 preset 的 eye 不再需要 IP；emg 端口改为自动检测（`port=""`）

### 3.9 测试
- `tests/eye/test_neon_eye*.py`：删除 `neon_ip` 参数
- GUI 测试循环此前已改为非阻塞（`flush_events` + 10Hz 重绘 + `draw_idle` + 4× 降采样 imshow）——
  阻塞式 `plt.pause` 会饿死 asyncio 事件循环导致 scene 流 0 帧（实测验证过）

---

## 4. 验证记录（本机，无实机硬件）

| 项 | 结果 |
|---|---|
| 全量 py_compile（src/tests/scripts） | ✅ |
| `--dummy --duration 6` 多进程 launcher | ✅ 全部 open OK → 采集 → 各传感器按标称频率（gaze 200/s、imu 110/s、scene 30/s、camera 29/s、emg 99/s、手部/位置 62/s）→ 干净保存，退出码 0 |
| 心跳格式 | ✅ `gaze_xy=301(2,)(200/s) scene_frames=46(480,640,3)(30/s)` 长度+shape+实时频率 |
| 相机 mp4 管线（90 帧 color+depth 双流 @30fps） | ✅ 解码帧数 == 时间戳数 == 90；ffprobe: hevc/yuv420p 与 hevc/gray12le；零掉帧 |
| mp4v 快速写入 | ✅ 50/50 无丢帧（x265 不丢帧，编码缓冲由 close 排空） |
| 失败路径（无 RealSense 硬件） | ✅ `open ERROR — RuntimeError: No device connected` + 完整 traceback → 中止，退出码 1 |
| eye async 导入/构造 | ✅ 无 `loop_style`、`_record` 就绪 |

---

## 5. 使用方法

```bash
# 全量 dummy 自测
python -m src.session.launcher --dummy --session-dir ./data/dummy --duration 10

# 生产（自动发现 Neon；其他硬件按 preset）
python -m src.session.launcher --session-dir ./data/session1 --recorders eye cam_third marker

# 带刺激程序
python -m src.session.launcher --session-dir ./data/session1 --with-stim

# 独立运行单个 recorder（不经 launcher）
from src.recorders.eye import NeonEyeAsyncRecorder, EyeRecorderConfig
NeonEyeAsyncRecorder(EyeRecorderConfig(session_dir="./out", duration=60)).run()

# GUI 硬件测试
python -m tests.eye.test_neon_eye_async
```

---

## 6. 已知事项 / 待办

1. **gray12le 截断**：相机 depth 视频为 12-bit，毫米深度值 > 4095（约 4 米）会被截断。
   如需全 16-bit：改 `base_camera_recorder._write_worker` 中 depth 的 `output_pix_fmt="gray16le"`（x265 支持，体积略大）
2. **TouchTronix 触觉手套是 stub**（`_open` 恒失败）——参数已在 config 中留好注释，实现时补
3. **`src/config/collection.py` 未接线**：只被 stim 读取；record 的启用/禁用配置尚未驱动 launcher（launcher 用 `--recorders` 过滤）。若要让 collection config 管理 recorder 列表，需在 `launcher.main()` 中接入
4. **`scripts/visualize.py` 的 eye scene 行已过时**：eye 场景视频现在在 `eye.mp4`（不在 npz 的 `scene_frames`），visualize 中该行恒为空
5. 生产 preset 中 emg 端口已改为自动检测；原 COM27/COM28（Windows）如仍需指定，用 `get_weili_emg(port=...)`
6. Neon scene 相机冷启动首帧约 5 秒——短于 5 秒的运行 scene 为 0 帧属正常
7. GUI 测试与 asyncio 的教训：matplotlib 阻塞泵会饿死事件循环（scene 流 0 帧）；测试必须用
   `flush_events()` + 节流重绘 + `draw_idle()`
