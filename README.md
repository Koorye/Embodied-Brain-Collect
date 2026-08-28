# embodied-brain-collect

**多模态具身数据采集框架** —— 眼动、RGB/深度相机、EEG、EMG、数据手套、位置跟踪、
任务标记，一台电脑同时采集。

一个 launcher 并行管理所有传感器（每传感器独立进程），所有设备**首条数据就绪后
统一开录**；视频统一 libx265 帧精确编码；每帧时间戳是重建后的绝对 unix 时间；
录完自动跑质量检查并生成一张自包含的网页报告，所有信号画在同一条时间轴上。

```
预检硬件 → 检查相机 → 打乱任务 → 逐任务录制+QC → 保留/重跑 → 会话汇总
   ↑                                                         ↓
   └──────────────  一条命令: python scripts/run_session.py  ─┘
```

---

## 硬件一览

| 模态 | 设备 | 采集实现 |
|---|---|---|
| 头戴相机 | OAK-D（DepthAI） | `depthai_camera` |
| 手腕相机 ×2 + 第三视角 | USB 相机（OpenCV DSHOW） | `opencv_camera` |
| 眼动 | Pupil Neon | `neon_eye_async` |
| EEG | Curry EEG（TCP） | `curry_eeg` |
| EMG ×2 | WAVELETECH 8 通道臂环（CP210x 串口） | `weili_emg` |
| 手部姿态 | MANUS 数据手套 | `manus_hand_pose` |
| 位置跟踪 | OpenVR tracker | `openvr_position` |
| 标记 | ParallelBox TTL + UDP | `udp_marker` / stim 的 `MarkerSender` |

## 安装

依赖：Python ≥ 3.10，系统需安装 `ffmpeg`（含 libx265）。

```bash
conda create -n Embodied-Brain-Collect python=3.10
conda activate Embodied-Brain-Collect
pip install -r requirements.txt
pip install -e .
```

> 本仓库的开发环境用 `Embodied-Brain-Collect` 这个 conda 环境；命令行脚本都自带
> `sys.path` 引导，直接 `python scripts/xxx.py` 即可，无需额外设置 `PYTHONPATH`。

## 快速开始

```bash
# 0. 确认所有传感器在线
python scripts/preflight.py
python scripts/check_cameras.py        # 每个相机开一个窗口,实时看画面

# 1. 一次完整采集会话(随机队列 → 逐任务录制+QC → n/r/q 确认 → 汇总)
python scripts/run_session.py
python scripts/run_session.py --seed 42    # 复现某次的随机队列
python scripts/run_session.py --auto-keep  # 无人值守(不逐条确认)
python scripts/run_session.py --dummy      # 假设备试跑整条链路

# 2. 想看某次录制的报告
python scripts/qc.py data/session-night/2026-08-21-14-30-00      # 控制台
python scripts/qc_report.py data/session-night/2026-08-21-14-30-00  # 生成 qc.html
```

## 配置

所有部署参数在仓库根目录 **`configs/`**（不在代码里）：

| 文件 | 内容 |
|---|---|
| `recorders.yaml` | 每个传感器 slot 用哪个实现 + 全部参数（相机 idx、COM 口、波特率、端口、`hz`）。YAML 注释里写着"为什么是这个值" |
| `tasks.yaml` | 任务库（task_id + 中文名）。**task_id 只用于标识，无编码范围限制**（v1.1.0 起任务身份写在 session 的 `meta.yaml`，不占 marker 码位），任务库可任意扩充 |
| `stim.yaml` | MarkerSender 传输参数 + 每个 stim 的参数（paradigm1 / sync_test 各自一段）。launcher 不传任何 stim 参数 |
| `checker.yaml` | QC 各检查的阈值（键 = 检查类名，如 `timestamp_gap.min_s`；缺项用代码默认值） |
| `meta.yaml` | 框架版本等元信息，每次采集时抄进 session 目录 |

环境变量 `EMBODIED_BRAIN_COLLECT_CONFIGS` 可指向其他配置目录。

## 标准工作流

### 1. 预检（确认所有传感器在线）

```bash
python scripts/preflight.py              # recorders.yaml 里全部
python scripts/preflight.py cam_head emg_left   # 只查某几个
```

用全新实例执行每个 recorder 的 `_open()` 首帧闸门，确认读到数据后关闭并出表，
任一失败退出码 1。预检实例用完即弃，不参与真实采集。

### 2. 相机检查

```bash
python scripts/check_cameras.py          # 枚举并开窗显示所有相机
python scripts/check_cameras.py --list   # 只列出,不开画面
python scripts/check_cameras.py --idx 2  # 只看 USB 索引 2
```

直接绕过 recorder 打开相机（与采集同一种打开方式），每个相机一个窗口，
左上角标注来源与分辨率。开不开画面对采集程序毫无影响。

### 3. 任务顺序

`configs/tasks.yaml` 是只读任务库(默认按 task_id 升序)。`run_session.py`
启动时以当前时间戳为 seed 在**内存中**随机采样本次执行队列,打印完整任务
列表后按 Enter 才开始 —— tasks.yaml 文件本身不会被改写,顺序乱了直接按
task_id 重排即可。

### 4. 逐任务录制 + QC（主控制脚本）

```bash
python scripts/run_session.py            # 完整会话
```

每个任务的实际录制由 launcher 完成（全部 recorder 多进程 + paradigm1 刺激程序），
录完自动跑 QC，然后问你：

```
  保留 [Enter] / 重跑 [r] / 退出 [q]:
```

* **保留** → 任务移到 tasks.yaml 队尾（轮转），继续下一个
* **重跑** → 删除本次录制目录，任务留在队首，马上重来
* **退出** → 结束会话，输出已完成的汇总

全部任务完成（或退出）后打印汇总并写入 `run_summary.json`：
**xx% 数据无误**、每种 QC 错误/警告的条数与涉及 session 占比、各 session 时长。
QC 只供参考，不替人做决定——有 ERROR 的录制也可以保留，由你拍板。

不用主脚本、单录一个任务时：

```bash
# 带刺激程序:自动取 tasks.yaml 第一个任务,完成后轮转
python -m embodied_brain_collect.session.launcher --session-dir data/session-night --stim paradigm1
# 同步测试(全自动想象/手势序列 + 计时画面,用于时钟校准)
python -m embodied_brain_collect.session.launcher --session-dir data/session-day --stim sync_test
# 只录部分传感器 / 限时 / dummy 自测
python -m embodied_brain_collect.session.launcher \
    --session-dir data/session-day --recorders eye cam_third marker --duration 60
python -m embodied_brain_collect.session.launcher --dummy --session-dir data/session-day --duration 10
```

`--session-dir` 是**班次根目录**：实际录制目录自动创建为
`{班次根}/yyyy-MM-dd-HH-mm-ss/`。

### 5. 自动 QC

采集结束（所有 recorder 保存完）后 launcher / run_session **自动**跑 QC：
控制台打印完整报告，并把 `qc_report.json` 与 `qc.html` 写进 session 目录。
`--skip-qc` 可跳过。QC 结论不改变录制结果。

手动复跑：

```bash
python scripts/qc.py data/session-day/2026-08-21-14-30-00            # 控制台 + 退出码
python scripts/qc.py <session> --json report.json
python scripts/qc_report.py data/session-day/2026-08-21-14-30-00     # 生成 qc.html
```

`qc.html` 是自包含单文件：所有传感器数据（相机画面缩略图 + 信号折线）、
统计数字、每条警告/错误都标注在统一时间轴的对应位置上，问题处会自动插帧。
**所有曲线全分辨率**（无降采样），放大到最深能看逐样本细节。

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `scripts/run_session.py` | **主控制脚本**：随机队列 → 逐任务录制+QC → n/r/q 确认 → 汇总 |
| `scripts/session_summary.py` | **现有数据汇总**：按 session-dir / date 统计已采数据（产量、任务覆盖、质量问题） |
| `scripts/preflight.py` | 预检：每个 recorder 打开 + 数据流探测，输出完整报告与分设备排查建议 |
| `scripts/check_cameras.py` | 相机体检：枚举 + 实时画面 |
| `scripts/qc.py` | 手动 QC（控制台 + 可选 JSON） |
| `scripts/qc_report.py` | 手动生成 qc.html |
| `scripts/rebuild_emg_timestamps.py` | 给旧 session 的 EMG 补重建时间戳（见下） |

## 目录结构

```
configs/                        # 部署配置（见上）
scripts/                        # 工作流脚本（见上）
src/embodied_brain_collect/
  recorders/                    # 各模态 recorder（含 dummy；hz 轮询上限在 base）
    emg/timestamp_rebuild.py    # EMG 逐帧时间戳重建(seq 号线性拟合)
  session/                      # launcher + recorder 工厂(presets 读 recorders.yaml)
  stim/                         # 刺激程序: base_stim 骨架 + paradigm1/sync_test 实现
  config/load.py                # configs/ 装载
  checkers/                     # 组合式 QC（阈值来自 configs/checker.yaml）
  visualizers/                  # QC 网页渲染
tests/                          # pytest（checkers/visualizers 纯软件）+ 硬件 GUI 测试
data/session-day/  data/session-night/   # 采集输出（git 忽略）
```

## 数据格式

### 采集输出（`{session_dir}/{slot}/`）

| 传感器 | 文件 |
|---|---|
| 相机 | `frames.mp4` + `<slot>.npz`（`frames_timestamps`） |
| 眼动 | `eye.npz`（gaze/imu/scene）、`eye.mp4` |
| 其余 | `<slot>.npz` + `<slot>.log` |

session 根下另有 `meta.yaml`（版本/任务/开始时间）、`qc_report.json`、`qc.html`。
视频为 libx265 HEVC，`bframes=0` + 每秒关键帧 —— 容器帧序号与时间戳数组下标严格 1:1。

### 时间戳语义

所有时间戳都是绝对 unix 秒（float64），各流对齐不需要任何换算。

### EMG 时间戳（臂环，重点）

臂环以 2000 Hz 输出 8 通道 EMG + ~113 Hz IMU，共用一个 8 位序列号。
Windows 下 `Serial.read(4096)` 会阻塞到攒满 4096 字节（约 66 ms，~140 帧），
若直接盖"读取到达时间"，**99.3% 的时间戳是重复的**、且每帧平均滞后 33 ms。

因此 v1.1.0 起，recorder 在**收尾时**用序列号重建逐帧时间戳：

1. 序列号按 8 位回绕展开成全局帧索引 k（EMG 与 IMU 共一个 k）
2. 每个读取批次的**最后一帧**作锚点（该帧到达延迟最小），线性拟合 t = a·k + b
3. 3σ 剔离群批次后重拟合；周期量化为整数纳秒
4. 输出 `emg_timestamps`（重建，严格单调）与 `emg_arrival_timestamps`（原始到达值，保留备查）

实测：拟合速率 1999.997 Hz（标称 2000 Hz，误差 3 ppm），锚点残差 ~0.44 ms，
零重复、零回退。帧索引预算检查确认**一帧未丢**（发出 54660 = 收到 54660）。

> 重建消掉了批量延迟与抖动，但"设备采样 → PC 读到"的固定传输延迟无法从数据
> 本身估出，跨模态毫秒级对齐需硬件同步脉冲（sync_test 就是为此准备的）。

**旧数据回填**：

```bash
python scripts/rebuild_emg_timestamps.py data/session-night/2026-08-24-18-09-17          # 预览
python scripts/rebuild_emg_timestamps.py data/session-night/2026-08-24-18-09-17 --write  # 写回
```

已含 `emg_arrival_timestamps` 的文件自动跳过；拟合被拒（数据不足以拟合）时原样保留。

## 刺激程序架构

```
base_stim.py        BaseStim —— 所有范式共享的骨架:
                    pygame 窗口/字体、MarkerSender(串口 TTL + UDP 双路)、
                    Esc 中止、SPACE 等待、时间压缩(--fast)、清理收尾、
                    公共 CLI 参数(--parallelbox/--fullscreen/--font-size ...)
paradigm1_pickplace.py   范式1:注视→指令→运动想象→pick & place(按键驱动)
sync_test.py             同步测试:全自动想象/手势序列 + 毫秒计时画面
factory.py               按 kind 构建子进程命令;launcher 只指定 kind,
                         其余参数全部来自 configs/stim.yaml
```

新写一个范式只需继承 `BaseStim` 并实现 `run_flow()`，约 50 行。
marker 码表在 `stim/marker_codes.py`（RUN_START/FIX_ON/TASK_ID/IMG_START/EXEC_END 等），
UDP 包带发送端 PC 时间戳，接收抖动不进入 marker 时间轴。

## 测试

```bash
pytest tests/checkers tests/visualizers       # 纯软件,50+ 用例
pytest tests/checkers tests/visualizers -m "not slow"   # 跳过解码视频的集成测试
python -m tests.eye.test_neon_eye_async       # 硬件 GUI 测试(Q 退出,需显示器)
```

## 常见问题

* **EMG 打不开 / 没数据**：先检查线缆是否按实——臂环的 USB 线接触不良是最常见原因，
  换根线再试，别急着改软件。
* **EMG 时间戳大量重复**：旧数据未重建（见上），跑一次 `rebuild_emg_timestamps.py`。
* **相机 idx 对不上**：`configs/recorders.yaml` 里 idx=3/2/1 是 USB 集线器物理口位。
  用 `scripts/check_cameras.py` 看每个索引实际是哪台相机再改。
* **qc.html 太大**（全分辨率 + 缩略图）：`python scripts/qc_report.py <session> --no-frames`
  或调低 `--fps` / `--thumb-width`。
* **Windows 下 Ctrl+C 不响应**：recorder 子进程用 spawn 启动，主脚本入口必须带
  `if __name__ == "__main__":` 保护——仓库内脚本都已处理。
* **任务顺序**：tasks.yaml 只读，随机顺序在 run_session 内存中（seed 启动时
  打印）；想恢复文件顺序按 task_id 升序重排即可。

## 版本历史

见 [CHANGELOG.md](CHANGELOG.md)。当前版本 **1.1.0**。

---

## v1.1.0 改动概览

1. **EMG 逐帧时间戳重建**（见上节）+ 旧数据回填脚本。
2. **QC 网页全分辨率**：任何曲线都不再降采样（100 点/秒包络已移除），
   放大可看逐样本；均匀时间轴以 stride 压缩避免体积膨胀。
3. **主控制脚本 `run_session.py`**：洗牌询问 → 逐任务录制+QC → 保留/重跑 →
   汇总报告（无误比例、错误分类与占比）。
4. **相机检查脚本 `check_cameras.py`**：枚举 + 实时画面。
5. **stim 基类重构**：`base_stim.BaseStim` 抽出两范式共用的窗口/marker/交互骨架，
   paradigm1 与 sync_test 只剩各自流程。
6. 中文 README（本文件）与 CHANGELOG。
