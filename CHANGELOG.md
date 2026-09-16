# 修改记录

## 1.2.1 — 未发布

### meta 附每个 episode 的 qc report,hardware 只列实际槽位(pack_daily)

* 打包结束时把每个 episode 对应源会话的 `qc_report.json` 原文汇总写进
  数据集 `meta/qc_reports.jsonl`(一行一个 episode:`episode_index`、
  `session` 源会话名、`level` 整体等级、`qc_report` 报告全文,
  findings/streams 明细全保留,约 30KB/集),QC 结论随数据走,不用回
  源目录查。`pack_daily_fast` 同步。
* `info.json` 的 `hardware` 原来用 `DEFAULT_RECORDER_NAMES` 全表预填,
  某天没采的模态(如 wristband、ego_headband)也带着默认显示名混进去。
  改为只列实际出现的槽位(会话里真实存在的槽位目录 ∪ meta.yaml
  recorders 显式声明),默认表仅作名字回退。已打包的
  `session-day/2026-09-16-08-57-22-11-10-32` 的 info.json 与
  qc_reports.jsonl 已就地补齐。

### `scripts/pack_daily_fast.py` — 打包多进程加速版

* 新增 `pack_daily.py` 的加速版,输出数据集一致,用法一致(另加
  `--workers N` 控制进程池宽,默认 4)。三处并行/省工:
  - 预扫与特征规格改用 `load_stream_index` 只读时间戳(原版每个会话
    要把全部 npz **完整解压三遍**:预扫、规格、写帧),并进程池并行;
    轻量索引抛异常时自动回退整段装载,语义与原版一致;
  - 全部视频截段(ffmpeg 重编码 + 首帧自验 + 时间戳 parquet)提前提交
    独立进程池,与主进程 add_frame 写帧循环完全重叠;
  - 每个会话的流装载在进程池预取(深度 1),与上一会话写帧重叠;
    去掉逐样本 tqdm,改为每流写完打印样本数。
* `add_frame` 逐样本 API 决定写帧循环仍在主进程串行;加速来自更轻的
  预扫 + 视频/流装载与写帧的流水线重叠。已用合成会话对拍:预扫特征集、
  特征规格、episode 事件序列、截段帧数、输出文件树与原版逐一一致。

### 指定 `--out` 也自动落 `<out>/<日期>-起-止`(pack_daily)

* 原来只有自动命名(`--out` 未指定)才按数据起止时刻追加
  `<日期>-<起>-<止>`,显式 `--out` 时名字完全固定。现在两种方式统一:
  显式 `--out` 时数据集自动进一层,落在 `<out>/<日期>-<起>-<止>/`
  子目录(起止口径不变:优先各会话 qc marker 窗口,缺报告回退目录名
  时刻);无可解析起止时保持原目录。`pack_daily_fast` 同步。

### 打包主时钟改为独立 30Hz 时间轴(pack_daily)

打包出来的数据集 marker 缺 241(RUN_START)、部分缺 17(FIX_ON):原实现
用相机首帧(窗口内)做主时钟起点,而 RUN_START 定义窗口起点、恒早于该帧,
`ts >= t0` 的对齐过滤把它必然切掉;FIX_ON 紧随其后,相机首帧落在它之后
的会话同样被切。修复:主时钟不再取任何 recorder 的时间戳,由 marker
窗口直接合成 —— `t_k = RUN_START + k/30`,末帧 `ceil((RUN_END-RUN_START)
×30)`,保证覆盖 RUN_END;`load_master_frames` 更名 `make_master_timeline`
(`pack_daily_fast` 同步),无 marker 窗口(--full 或缺 RUN_START/RUN_END)
回退用相机首末帧界定时长。所有流仍按窗口裁剪后 rebase 到 RUN_START,
RUN_START 钉在 t=0,起始事件不再丢失。

### eye 视频完整性修复(视频比时间戳少 1 帧 / 只有 1 帧)

两个症状对应两个缺陷,均在 `recorders/eye/neon_eye_async_recorder.py`:

* **差 1 帧**:`_scene_writer` 在 `write()` **之前**记 `scene_timestamps`,
  而 write 是丢进线程池异步执行的;停止时主协程 `_teardown()` 不等写盘
  任务排空就 `writer.close()`(阻塞调用),正在写的最后一帧被 close 掉 ——
  时间戳已 +1、视频没有这帧。修复:时间戳改到 **write 成功之后**记账
  (写失败的帧永不入账);停止后先经 `_finish_scene_video()` await 写盘
  任务排空队列(10s 超时)再关 ffmpeg,close 与在途 write 的竞态消除。
* **只有 1 帧**:写盘任务在 `_scene_task` 内部自建,不在主循环监督的
  tasks 列表里 —— ffmpeg 中途死掉(磁盘满/x265 崩溃)时写盘任务静默死亡,
  读帧任务继续收帧(队列 drop-oldest 全丢),rc 也不置 1,会话"正常"保存
  却只有第 1 帧。修复:写盘任务注册到 `self._scene_writer_task` 并纳入
  录制循环监督,死亡立刻 ERROR 日志 + rc=1 + stop,会话按失败处理。
* 读帧任务 finally 改为 `gather(return_exceptions=True)`,写盘异常的
  汇报权统一交给监督路径。
* `FFmpegWriter.close()` 收尸健壮化:ffmpeg 已死时 `stdin.close()` 的
  BrokenPipeError 不再抛出,失败统一走 rc 检查上报。

### 帧数-时间戳核对升级为严格相等(FrameCountMatch + reqc_all)

* 检查器 `FrameCountMatch`(eye/camera 全部视频流)从"超过
  max(2, 2%×N) 才 ERROR"改为**严格相等**:录制器逐帧 1:1 写时间戳,
  任何不等都是录制中断/丢帧,直接 ERROR(整文件与窗口两种比较口径一致;
  窗口模式两边同源同掩码,精确相等不误报)。
* `scripts/reqc_all.py` 重写:
  - 核对 session 内**所有视频**:`<slot>/frames.mp4` vs `<slot>.npz` 的
    `frames_timestamps`,eye 槽位 `eye.mp4` vs `scene_timestamps`(也兼容
    同目录 npz 里的 `{stem}_timestamps` 键),不再只查 eye;
  - 帧数用 `ffprobe -count_packets` 解封装计数,**不再 cv2 逐帧解码**
    (原来每条视频要解码几十秒,现在毫秒级,3 个 session 全量 0.2s);
  - 严格相等,帧数偏多同样 ERROR;有视频却找不到对应时间戳、ffprobe
    数帧失败,同样记 ERROR;
  - `--write` 把 ERROR finding 合并进 qc_report.json 对应流(流名 =
    槽位目录名)并重算流与会话等级;exit code 0 全部一致 / 1 有异常,
    可当流水线闸门。

### ffmpeg/ffprobe 工具统一封装(`embodied_brain_collect.utils.media`)

* 新增 `utils/media.py`:`media_tool(name)`(Windows 优先仓库 third_party 的
  `ffmpeg.exe` 与单文件 `ffprobe-win32-x64`,兼容旧目录布局,找不到再退
  系统 PATH;其余平台直接系统命令)+ `ffprobe_count(path)`
  (`-count_packets` 解封装计数)。
* `recorders/ffmpeg_writer.py`、`scripts/pack_daily.py`、
  `scripts/reqc_all.py` 三处各自的本地实现全部删除,统一走该模块 ——
  录制写盘、打包预扫、事后核对用的是同一套可执行文件、同一套计数口径。
* 顺带修正:`pack_daily._media_tool` 的 Windows/Linux 分支写反(Windows
  找无扩展名文件、Linux 反而兜底找 .exe);`ffmpeg_writer._find_ffmpeg`
  的系统 PATH 查找被注释掉(Linux 上 Exec format error),一并恢复。

## 1.2.0 — 未发布

### 图纸模式(采集队列由图纸驱动)

* 新增 `configs/environments` 图纸池:按场景目录存放 `config.yaml`
  (task/scene 定义)与图纸(`NNNN_comboNNN_rN.png`)。每次采集随机抽一张
  **未用过**的图纸全屏显示,采集员照图摆放实物,按 **n + Enter** 确认后才
  正式开录;其他按键无效,未确认直接关窗 = 取消本次(不开录)。
* 采集成功的图纸记入**该场景目录独立**的台账 `used.yaml`(键 = 文件名),
  之后不再被抽到;重采/退出不记账。删掉条目即可让图纸重新入池。目录里的
  总览图等非图纸文件不进抽取队列。
* 新增 `session/environment.py` 的 `Environment` 类:图纸池/台账/场景信息/
  全屏显示(matplotlib,中文字体自动选择,n+Enter 确认状态机)。
* `run_session.py` 支持 `--mode env/tasks` 与 `--stim` 切换,缺省读
  `configs/session.yaml`(新增,`mode`/`stim` 两个键)。
* **所有 stim** 经 `BaseStim` 支持 `--environment`:画面显示图纸对应
  场景/任务(paradigm1 指令屏、sync_test 标题)。
* 任务列表模式保留:`--mode tasks` 走 configs/tasks.yaml 队列。

### 生理腕带 recorder(wristband)

* 新增 BLE 协议 V1.5 recorder:压力 150 Hz / PPG 100 Hz / IMU 50 Hz /
  体温·血氧 1 Hz。open 时发 0x20 指令同步设备 UTC 钟(全部时间戳的来源),
  未同步的帧直接丢弃,保证没有坏时间戳入库。
* 确认闸门:launcher 的录制确认只认真正入库的 0x15 数据帧——设备只发
  状态帧不发数据帧时整场中止,不再留下空生理数据。

### 每日数据打包(pack_daily.py)

* 新增 `scripts/pack_daily.py`:按日期把班次根(如 data/session-day、
  data/session-night)下的会话打包成 **mf-lerobot 多频率数据集**
  (基于姊妹项目 Multi-Frequency-LeRobot 的 `mf_lerobot` 包)——每会话一个 episode,
  各传感器保持原生采样率独立存储,读取时按时间窗对齐。
* 视频用 **ffmpeg 直通截段**:不经逐帧解码/PNG 中间态,按 episode 窗口
  截段并归一化 30fps;**内容锚定**校正 CFR 重复帧导致的截段偏移
  (实测源数据存在 ±3 帧/100ms 的落点偏差,锚定后 ≤±1 帧,每路截段自验)。
* 时间戳统一 rebase 到 episode 主时钟首帧,负时间戳一律裁掉;只出现在
  部分会话的流整体舍弃,保证 meta 特征集在每个 episode 完整。
* 支持 `--source/--out/--force/--full/--max-episodes/--keep-images`。

### recorder 与 meta

* `recorders.yaml` 支持可选 `name`(设备显示名),随会话写入 meta.yaml 的
  `recorders` 字段({槽位: 显示名});输出目录/子进程身份仍用槽位键。
* 配置装载迁移:`config/load.py` → `session/config.py`;可选 SDK
  (openvr/depthai/pyrealsense2/pupil_labs/pycbsdk/serial)全部懒加载,
  装哪个 SDK 用哪个 recorder,缺 SDK 不再拖垮整体导入。
* `mf_lerobot` 为姊妹项目 Multi-Frequency-LeRobot(github.com/Koorye/
  Multi-Frequency-LeRobot)的 editable 依赖
  (打包功能所需)。

### EMG 时间戳拟合更抗失败

* 初估改用 **Theil–Sen 中位数斜率**(锚点等距抽样 ≤400 限制配对数):
  单个调度卡顿的锚点不再拖斜线 —— 旧的单轮流程里,3σ 剔离群的阈值恰恰
  被离群本身撑大,多离群时容易剔不干净而拒拟合。
* 剔离群从单轮改为**迭代 3 轮**,每轮用最小二乘在保留锚点上精修。
* 最低锚点数 8 → 5:短会话/稀疏读取不再因"read batches 不足"被直接拒拟合。
* 修复迭代剔离群的第 2 轮起用**剔剩掩码索引全量锚点**的长度错位
  (IndexError):凡是第 1 轮剔过、第 2 轮又要剔的会话整批 REFUSED
  (2026-09-15 夜班批量 rebuild,705 个文件中招);掩码改为只作用于
  保留集,705 个文件全部正常拟合。
* 拟合失败的上报级别 warn → **error**:recorder 收尾日志按 ERROR 记;
  QC 里 EMG 两个时间戳系列的重复时间戳(拟合被拒 = 仍是到达时间,
  ~99% 重复)从 WARN 升为 **ERROR**(`TimestampSanity` 新增 `dup_level`
  配置,其他模态仍为 WARN),session 级别随之 ERROR,qc.py 退出码从 1
  变 2,`pack_daily.py` 的 ERROR 排除闸门随之生效。拟合成功的会话严格
  递增、不产生任何提示。
* 新增回归测试:5 批次短会话可拟合、卡顿离群不倾斜不拒拟合、过少批次
  仍拒绝、两级离群触发第 2 轮剔除时照常拟合、emg 重复级别可配置。

### 文档

* README.md 全面重写(v1.2.0):Mermaid 架构/时序/流程图、两段式启动、
  图纸/任务双模式、采集员操作说明(**含白班/夜班收班打包**)、
  mf-lerobot 打包详解、腕带时间戳语义、常见问题扩充。

## 1.1.1 — 未发布(随 1.2.0 一并发布)

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
