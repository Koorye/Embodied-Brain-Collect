# 修改记录

## 1.1.1 — 未发布

### 任务队列改为内存随机（不再改写 tasks.yaml）

* `configs/tasks.yaml` 恢复 task_id 升序并改为**只读任务库**：删除
  `shuffle_tasks.py` 与 `load.py` 的 `rewrite_tasks`/`consume_task`
  （此前每录一个任务都会重写该文件，顺序越滚越乱）。
* `run_session.py` 启动时以**当前时间戳为 seed** 在内存中随机采样本次执行
  队列（`--seed` 可复现），打印完整任务列表后按 Enter 才开始；队列推进、
  重采全部在内存中完成。
* 录制结束的确认改为防误触输入：必须输入 **n(下一条) / r(重采) / q(退出)**
  之一再回车，空回车与错误字母都要求重输（`--auto-keep` 仍可跳过）。

### 会话汇总改进（错误带流名、空录制不计入无误比例）

* 错误/警告按 **检查项 → 具体流（recorder 名）** 展示：每个检查项下多一行
  `流: marker 33 · eeg 6`，不再只给 StreamPresent/ClockAlign 这种看不出
  是哪台设备的名字；会话级 findings 标"会话级"。
* **空录制单独识别**：目录里只有 .log、没有任何 npz/mp4（典型：启动失败）
  的录制在明细行标 ∅ 并注明"空录制"，**不计入无误数据比例的分母**；有数据
  但缺 QC 报告（--skip-qc 等）的同样不计入。分母显式标注
  `分母 = 有数据且跑过 QC 的 N 条`。
* StreamPresent 附注一行解释：目录在但没有数据文件（未启动成功或没保存），
  具体原因看 `<slot>/<slot>.log` 与启动错误提示 —— 不再"迷惑"。
* `run_summary.json` 的 errors/warnings 结构随之变为
  `{检查项: {流: 条数}}`。

### 错误日志与分设备排查指引

* 新增 `session/troubleshooting.py`：启动错误（open 失败）与录制/保存错误
  分类；首次失败只提示重采，多次失败展开对应设备的排查方案
  （eeg 拔插同步盒 / emg 拔插 USB 按紧接线 / eye 拔插网线重启 app /
  hand_pose 拔插接收器确保手套全蓝常亮 / position 确保 app 连接 /
  cam 拔插对应 USB 口），并指出 traceback 所在的 `<slot>/<slot>.log`。
* `launcher.launch()` 返回 `LaunchResult`（int 兼容），携带
  `open_failures` / `runtime_errors` 明细；recorder 子进程录制中崩溃改为
  非零退出码，父进程据此归类为运行期错误。
* `BaseRecorder._teardown` 分级兜底：`_close` 崩溃不再连累 `_save` 落盘，
  两级异常都带完整 traceback 写进 recorder 自己的 .log。

### 预检重写（修复大量误报）

* 旧版用"open 后缓冲区样本数 > 0"判定成功，但相机在 open 阶段从不缓冲、
  EMG/手套/位置在闸门通过后会清空缓冲 —— 设备正常也被判
  "打开了设备但首帧数据为空"。改为三阶段检查：open 首帧闸门 →
  `probe_data_flow` 持续数据流探测（新增通用实现；neon 用 standby 队列、
  UDP marker 用端口绑定各自覆写）→ 关闭/落盘。
* 每个 slot 在**独立子进程**中检查（设备卡死不再拖垮整个预检，硬超时可配），
  探测数据写临时目录用完即删；报告含阶段、耗时、实测速率、错误、
  traceback、该 recorder 日志尾部与分设备排查建议，`--out` 可另存文件。
* OpenCV 相机后端按平台选择（`preferred_backend`）：Linux 下写死的
  `CAP_DSHOW` 会让在线相机直接打不开；`check_cameras.py` 同步修复。
* `recorders/__init__.py` / eye / position 的厂商类改为惰性导出，
  `recorder_presets` 对 neon/openvr 延迟导入 —— 缺个别 SDK 的机器不再
  拖垮整体导入，预检按 slot 单独报"依赖缺失"。
* `recorders.yaml`：eye `open_timeout: 120`、hand_pose `open_timeout: 60`
  （内部预热最坏 ~90s/~55s，默认 30s 看门狗会在正常冷启动时误判超时）。

### QC 页面 EEG/EMG 可视化滤波

* 新增 `visualizers/signal_filter.py`：零相位 SOS 级联（Butterworth 带通
  + 50 Hz 工频陷波及全部谐波），预设 EEG 0.5–70 Hz、EMG 20–450 Hz，
  参数可由 `configs/checker.yaml` 的 `filter:` 节覆盖。
* `qc_report.py` 渲染时对 eeg_data/emg_data 通道各嵌入一份滤波副本
  （`yf/flo/fhi` 字段，独立 int16 量化、共享时间轴），页面每个流卡片新增
  「原始/滤波」切换按钮，默认原始。原始 npz 数据永不修改；Trigger、IMU
  通道不滤波；`--no-filter` 可关闭以减小体积。
* scipy 为可选依赖（未安装时页面自动退化为纯原始曲线），已加入
  requirements.txt。
* 修复采样率估计 bug：EMG 重建后的时间戳间隔偏态（中位 0.473 ms、均值
  0.5 ms），按中位间隔会把 2000 Hz 估成 2113 Hz，滤波器设计频率整体偏移
  5.7%，50 Hz 陷波实际落在 47.3 Hz、工频直接漏过。改为按平均码率
  （样本数/跨度）估计，并加回归测试。

## 1.1.0 — 2026-08-24

### EMG 逐帧时间戳重建（核心）

臂环 2000 Hz 的数据在一台 Windows 机器上以 `Serial.read(4096)` 读取时，
每次读回 ~140 帧共享同一个到达时间戳（重复率 99.3%，每帧平均滞后 33 ms）。
本次改动在 recorder 收尾时用序列号重建逐帧时间戳，一帧不丢、零重复、严格单调。

* 新增 `recorders/emg/timestamp_rebuild.py`：8 位序列号展开成全局帧索引
  （EMG 与 IMU 共享）→ 每批最后一帧作锚点线性拟合 → 3σ 剔离群重拟合 →
  周期量化为整数纳秒。实测 1999.997 Hz（3 ppm）、残差 0.44 ms。
  带全套守卫：拟合被拒时原样保留到达时间戳，修时间戳绝不丢录制。
* `weili_emg_recorder.py`：`_close()` 时重建；`*_timestamps` 存重建值，
  原始到达值保留为 `*_arrival_timestamps`。
* 新增 `scripts/rebuild_emg_timestamps.py`：旧 session 回填（默认预览，
  `--write` 写回，`--force` 用保留的到达时间戳重拟合；已拟合文件自动跳过）。
* 已把 `data/session-night/2026-08-24-18-09-17` 的左右臂回填：
  99.25%/99.26% 重复 → 0，去掉 +33.2 ms 批量延迟。

### QC 与网页取消降采样

* `visualizers/qc_payload.py`：移除 100 点/秒 min/max 包络降采样 —— 所有
  曲线全分辨率，页面放大能看逐样本细节。`minmax_downsample` 保留但只在
  显式传 `max_pts` 时生效。
* 均匀时间轴以 stride 压缩传输（`tstride` 字段 + 前端展开），全分辨率下
  页面体积不失控；非均匀流自动回落完整数组。
* 移除 `--fine` 开关（全分辨率成为默认，开关无意义）。

### 主控制脚本

* 新增 `scripts/run_session.py`：询问洗牌 → 按 tasks.yaml 顺序逐任务录制
  （launcher 多进程 + paradigm1 stim + 自动 QC）→ 每任务询问保留/重跑/退出
  （重跑删目录重录，保留则任务轮转到队尾）→ 汇总报告（xx% 数据无误、每种
  QC 错误/警告的条数与涉及 session 占比、各 session 时长）+ `run_summary.json`。
  支持 `--auto-keep` / `--dummy` / `--skip-qc` / `--recorders` / `--shuffle-seed`。

### 相机检查脚本

* 新增 `scripts/check_cameras.py`：绕过 recorder 枚举 OpenCV USB（与采集同
  一种 DSHOW 打开方式）/ RealSense / DepthAI 相机，每个相机一个窗口显示
  实时画面（左上角标来源与分辨率）；`--list` 只列不显示，`--idx` 只看单台。

### stim 基类重构

* 新增 `stim/base_stim.py`：`BaseStim` 抽出两范式共用的骨架 —— pygame 窗口
  与字体、MarkerSender 构造（串口 TTL + UDP）、Esc 中止、SPACE 等待、时间
  压缩（`--fast`）、清理收尾、公共 CLI 参数（`add_common_args`）与
  stim.yaml 默认值合并（`stim_defaults`）。
* `paradigm1_pickplace.py` 与 `sync_test.py` 重写为继承 BaseStim，
  各自只剩 trial 流程（各 ~100 行），删除了重复的字体查找/窗口/收尾代码。

### 文档与版本

* README.md 全面重写：硬件一览、快速开始、完整工作流、脚本一览、数据格式
  （含 EMG 时间戳重建说明）、刺激程序架构、常见问题、v1.1.0 概览。
* 新增 CHANGELOG.md（本文件）。版本号 0.1.0 → **1.1.0**
  （`pyproject.toml`、`embodied_brain_collect.__version__`）。

### 移除 TASK_ID / SCENE_ID marker（任务身份不再占码位）

任务身份 marker 被移除：一次 session 只录一个任务，任务身份由 launcher 写进
session 的 `meta.yaml`（task_id + task_name），marker 流只承载事件时序。

* `marker_codes.py`：删除 `TASK_ID_BASE/LAST`、`SCENE_ID_BASE/LAST`、
  `make_task_id`（其越界检查此前被注释，task_id ≥ 32 会静默撞进 SCENE_ID 区）、
  `make_scene_id`；`name_of`/`is_known` 不再解析这两个区间。
* `paradigm1_pickplace.py`：不再发送 SCENE_ID 与 TASK_ID；`TrialSpec.scene` 字段删除。
* `shuffle_tasks.py`：删除 task_id ≤ 31 的 marker 槽位校验 —— 任务库规模不再
  受码表约束，可任意增长。
* 同步更新 `config/load.py` 与 `configs/tasks.yaml` 头注释、README、QC 页面注释。

### 现有数据汇总脚本

新增 `scripts/session_summary.py`：对已采集数据做统计分析。
`--session-dir` 选会话根目录，`--date` 用目录名前缀过滤（`2026-08-24`
当天 / `2026-08` 整月 / `2026` 全年），`-o` 写 JSON。输出数据量
（session 数、无误比例、时长）、每日产量、任务覆盖（任务库哪些已录/
缺失/重复）、各 QC 检查的条数与涉及 session 占比、逐 session 明细。
统计口径与 run_session 的会话汇总共用同一实现。

### 技术备注

* EMG 重建时间戳在 float64 下仍有 ±0.12 µs 的 epoch ulp 网格交替 —— 这是
  绝对 unix 时间用 float64 表示的物理极限（1.77e9 处 ulp = 0.238 µs），
  比拟合自身 ~0.44 ms 的不确定度小三个数量级，前端 stride 压缩已显式容忍。
* `qc.html` 因全分辨率体积变大（25 s 全模态会话约 54 MB），需要瘦身可用
  `qc_report.py --no-frames` 或调低缩略图参数。
