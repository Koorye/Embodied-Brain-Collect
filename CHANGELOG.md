# 修改记录

## 1.4.3 — 2026-09-21

### 合并现场分支:Curry 阻抗门禁 / tracker 台数闸门 / 窗口边缘缺口检查 / 麦克风 ALSA 采集时间戳打包

* **Curry EEG 开录阻抗门禁**:curry_eeg_recorder 的 ``_open`` 发
  NetStream 请求 12/13 测一次阻抗(DATA_Impedances 码 4,每通道一个
  float32,单位 Ω),取若干帧有效包均值;除边缘通道
  (M1/P9/PO9/Cb1/Cb2 等耳后头围位,``impedance_edge_channels``)与
  Trigger 状态字槽位外,通过率(< ``impedance_max_kohm``)低于
  ``impedance_pass_rate`` 直接拒绝开录,超标通道清单写进报错文案。
  结果(阻抗值/检查掩码/通过率)写入 npz(``eeg_impedance_*``)。
  纠正协议注释:数据码 4 是阻抗不是 keepalive。配套独立体检工具
  ``scripts/impedance_check.py``(打印每通道阻抗表,``--save`` 存
  npz)。门禁开关 ``impedance_check``:**false = 完全绕过**(直连直录,
  不做任何阻抗动作);**true = 主动触发**(NetStream 12/13)。
  **触发式阻抗的两条时序红线**(现场实测违反会把放大器驱动打成
  Device Error,连 Curry 里点三角都失效,只能重启 Curry/放大器):
  ① 阻抗结束(13)后放大器回 connect 态、~10s 才恢复读数 —— 恢复前
  **绝不能断开连接**,门禁所有失败路径也都先等 EEG 恢复(两轮共 40s)
  再返回,失败后的断开因此落在正常的"数据在流"状态;② 不要发
  AmpConnect(10),与放大器自己的恢复过程相撞同样出错。eeg 槽位
  ``open_timeout`` 放宽到 180s(取数 + 等恢复最长 ~52s)。
  **已知副作用**:触发式阻抗会停掉放大器 DIN 事件通路 —— 触发后的
  会话 events=0,EEG↔marker 对齐必然失败(门禁前对齐 slope 残差
  ~1e-6、8 事件全命中);需要 marker 对齐的会话请把 ``impedance_check``
  设为 false,阻抗门禁与 marker 对齐二者目前不可兼得。
* **tracker 台数闸门**:checkers/position 新增 DeviceCount 检查;
  openvr_position_recorder 的 ``_open`` 在已连台数少于
  ``expected_devices`` 时直接失败(recorders.yaml position 段 = 3)
  —— 少连一台 SteamVR 不报错,只有台数能暴露。
* **TimestampGap 窗口首/尾边缘检查**:开录晚于 RUN_START / 提前于
  RUN_END 收尾不再漏报(checker.yaml 注释同步)。
* **麦克风打包只认 ALSA 采集时间戳**:pack_daily.load_microphone_index
  改按 microphone_stamp_ns 对齐(不再 read_complete 回推),缺/坏逐块
  时间戳的会话跳过音频;pack_daily_fast 补回麦克风探测与 audio 特征
  规格(此前 fast 版根本不打包音频)。MICROPHONE.md 补 capture basis。
* 测试:DeviceCount / 窗口边缘 / 麦克风打包与 commit / 头环真机脚本
  等新用例。
* **辅助员控制台(session/assist_console,纯被动)**:辅助员屏
  (``assist_console_display``,默认主屏)分屏显示 stim 屏幕镜像
  (``assist_mirror_port``,stim 每帧降采样 JPEG ~10fps 推流)+ 第三
  相机实时画面(cam_third ``preview_port`` 推流)—— 屏风遮挡下辅助员
  只看这块屏:左看 stim 阶段提示、右看采集员动作,直接按键盘空格切换
  阶段。**窗口带 WS_EX_NOACTIVATE 永不抢焦点**:键盘输入始终落在刺激
  窗口,误点控制台也不会断;Esc 只关控制台。采集轮自动启停:
  run_session_env 在图纸摆放完成后、stim 启动前拉起
  (``assist_console_display`` 不设则不启用),整轮结束自动终止。

### tracker 序列号角色绑定 + check_vive 体检工具 + EEG 排障三分支

* **角色→设备绑定(role_serial_map)**:npz 列序固定 = 映射书写顺序,
  会话间稳定,不再随开机先后漂移(此前左右手可能互换)。``_bind_roles``
  逐台校验:缺一台、多一台未绑定、两角色配同一设备、多台同标识均
  拒绝开录;npz 新增 ``roles`` 列。匹配值两种写法任选 —— **OpenVR
  序列号(61-BH…,推荐,永不变化)或 VIVE Hub 角色**
  (``vive_tracker_chest`` 等,经 Prop_ControllerType_String 读到,
  在 Hub 里改角色会跟着变);VIVE Hub 显示的设备 ID(FA61…)不是
  OpenVR 序列号,匹配不到时报错文案明确提示。connected-devices 日志
  带 vive_role 便于对照。recorders.yaml 已填入实测三台
  (left_wrist/right_wrist/chest)。
* **scripts/check_vive.py**:tracker 体检工具 —— 俯视(X-Z)/侧视
  (X-Y)双视图实时轨迹,0.5m 世界坐标网格等比例显示;依次拿起每台
  tracker,看哪个角色标签的轨迹跟着动即可核对左右手对齐。未绑定设备
  标 unbound,yaml 绑了未连上的顶部红字提醒;运行中开关 tracker 自动
  跟随(~2s 重扫);``--list`` / ``--seconds`` / ``--no-yaml``。
  只读位姿,与采集程序互不影响。
* **troubleshooting:EEG 排障按 error 关键词三分支** —— "阻抗检查
  未通过"→电极接触不良,整理电极/补导电膏(边缘位高阻属正常);
  "未收到有效阻抗数据"→放大器未采集(点蓝色三角);其余仍走同步盒
  /NetStream 老三样。recorder 报错文案含准确关键词供指引命中。

### 数据集列按角色命名(pack_daily / pack_daily_fast)

* 位姿特征 ``observation.device<i>_pose`` → **``observation.<角色>_pose``
  **(如 left_wrist_pose / right_wrist_pose / chest_pose),state/action
  通道 ``device<i>_x`` → ``<角色>_x``;角色来自 position.npz 的 roles
  (录制时按 role_serial_map 写入)。历史数据无 roles、长度不符、角色
  名非法/重复时逐台回退 device<i>(``_pose_names`` 纯函数,永不产生
  重复特征名)。
* **``_is_windowed`` 改按 ``*_pose`` 后缀判定**(原按
  ``observation.device`` 前缀)—— 否则角色命名的位姿特征会悄悄丢失
  滑窗对齐;pack_daily_fast 的 state 通道宽修正同步改为统计全部
  ``*_pose``(排除 hand_pose)。
* 验证:合成新/旧两种 session 实测两加载器特征键一致(state 58 维,
  通道名左腕/右腕/胸),``_pose_names`` 与 ``_is_windowed`` 新旧命名
  分支单测全过;真机三台 tracker 按序列号与 VIVE 角色两种写法均绑定
  成功,FA61 标识正确拒绝;check_vive 实机渲染核验(俯视/侧视/轨迹
  /状态行 ~86fps)。


## 1.4.2 — 2026-09-20

### 合并 prod 9-20 新改动:头环麦克风/多相机进数据集 + episode 输出路径重构

**EGO 头环麦克风进数据集(pack_daily + microphone_writer + recorder)**

* 新特征 ``observation.headband_microphone``(16kHz PCM):本体在
  ``ego_headband/microphone.wav``,npz 存逐块索引元数据(设备侧
  sample_index / wav 内采样位置 / read_complete_ns),打包按块对齐主时间轴;
  ``microphone_error`` 非空时安静跳过。
* ``MicrophoneWriter`` 换 **ALSA 捕获时间戳基准**:逐块元数据带
  capture_monotonic_ns / alsa_pointer_monotonic_ns / clock 映射不确定度,
  写入时按 ALSA 采样位置交叉校验(不符即拒),npz 记录
  ``microphone_timestamp_basis/reference`` —— 替代旧的"读完成时刻"基准。
* 录音 commit 前的**预热音频不落盘**:wav 是流式直写,预热段一旦写入便
  无法像视频缓冲那样在 commit 时丢弃(不拦则 wav 比视频多整段预热,
  时间轴无法对齐)。

**头环多相机/IMU 进数据集(pack_daily/pack_daily_fast)**

* ``discover_video_slots`` 升级为每槽位**多路**结构(dict[str, list]):
  ``ego_headband/{name}.mp4`` 按 npz 的 ``{name}_timestamps`` 动态发现
  (排序保证跨会话稳定),特征名 ``observation.images.headband_<name>_rgb``;
  ``observation.headband_imu<j>`` 六路 IMU 走窗口前缀规则。
* probe 适配多路结构;两脚本的**视频完整性检查保持只在 QC**(1.4.1
  原则,ContainerIntegrity/FrameCountMatch 把关,probe 仅发现),写循环
  保持异常中止。

**episode 输出路径重构(pack_episode)**

* 输出镜像改为 ``data/lerobot/<班次根>/<日期>/<collector_id>-<日期>-<时刻>``:
  日期-时刻形态的会话名拆成两级,叶子名带采集编号前缀(meta 快照的
  ``collector_id``,与 collect_info.jsonl 同源);``--out`` 显式指定时不改名。
* ``gated`` 参数(1.4.1)重新套回新结构:采集侧进程内调用跳过 QC/meta
  门槛,独立 CLI 自足。

**其他(自 prod)**

* ``session_summary`` 默认只统计**当天**,新增 ``--all`` 看全部;顺带修
  一个断链 —— 它还在 ``from run_session import _collect``(旧单文件私有
  函数),当前 run_session 已是转发壳,改从 ``session.run_base`` 导入
  (collect_runs/meta_task_label/print_summary)。
* 头环相机 QC trace 增加 **frame-diff** 曲线(qc_streams)。
* 清掉 prod 原生的两处未用导入(pack_daily_fast._hardware_names /
  pack_episode 的 pack_daily 别名)。

**验证**:真实会话上新结构 probe 完整发现 20 特征流;输出路径拆级与
  collector_id 正/缺分支实测;无 ego 数据的会话麦克风发现安静返回;
  launcher gated=False 调用链不变;168 测试全过。


### 深度检修:补齐 1.4.2 整合遗漏 + 消除重复提示/解码

* **pack_daily_fast 两个 probe 补齐去重**(_probe_session_index /
  _probe_session_full):1.4.2 只对 pack_daily 主 probe 去掉与 QC 重复的
  视频完整性判断,fast 直接拷自 prod 仍带两份 ffprobe 检查 —— 同一逻辑
  三处检查。现与 daily 同口径:probe 仅做发现,完整性只在 QC 把关。
  实测三个 probe 的特征集合逐项一致(真实会话 20 特征,零漂移)。
  顺带清掉随之失效的 ffprobe_count/write_info 未用导入。
* **ask_next 去掉重复的 QC 出错提示**:run_queue 在 QC ERROR 时已统一
  打印 hooks.qc_error_note + "确认无碍可按 n 保留数据",ask_next 里
  再打印一遍几乎相同的话 —— 删除(入参保留兼容)。
* **_FrameSource 首帧免二次解码**:构造时的后端探测本就解出一帧却丢
  掉,first_frame() 再开一遍 ffmpeg 管道 —— 现缓存探测首帧。AV1 素材
  的预检(run_session_video verify_decodable)与 stim 第一帧读取各少
  spawn 一次 ffmpeg。
* **未用导入清理**:ffmpeg_writer 的 pathlib.Path、pack_daily_fast 的
  write_info(prod 原生遗留)。
* 复核无虞的项:pack_episode 输出分支对重跑撞名目录(-2 后缀)正确落
  兜底;launcher 仅一个 run_pack_episode 定义(进程内版);launcher
  独立 CLI 与 run_base.record_one 为两个入口语义,不做合并(import
  环风险大于收益);168 测试全过,video dry-run(真实 AV1 素材)正常。

## 1.4.1 — 2026-09-18

### 合并复审:一个逻辑只检查一次 + 全量符号/文案核对(发布前自查)

* **去重 — 视频完整性只在 QC 检查**:pack 侧 `_probe_session` 删除与
  `ContainerIntegrity`/`FrameCountMatch` 重复的两处判断(ffprobe 容器
  读不出、帧数覆盖窗口末帧),probe 只保留"发现"职责(窗口内有视频
  时间戳的槽位进特征集合)。能过 QC 门槛的会话 probe 必过 — 不再判两次。
  写循环改为**异常中止**:未跑 QC 的历史数据带坏视频时,半写 episode
  继续追加会产出损坏数据集,中止并提示先补 QC 比硬跳过更安全。
* **去重 — 打包门槛只在入口执行一次**:`pack_episode.main` 新增
  ``gated`` 参数,采集侧进程内调用(launcher.run_pack_episode)传
  ``gated=False`` —— 触发条件(QC 无错且操作员保留)已由 run_queue 用
  同一信息源(qc_report.json/meta status)把关;独立 CLI 保持 gated=True
  自足。
* **符号/文案核对**:修复 launcher 与 configs/session.yaml 残留的
  "n + Enter" 文案(environment.show 实际要求 s 键);清理未用导入
  (run_base.random / factory.Callable / pack_episode 的 pack_daily 别名);
  pyflakes 全量过检(marker_codes 的动态码表注入为既有模式,运行时安全,
  168 测试含 dummy EEG 事件流验证)。
* **实测**:ContainerIntegrity 对截断容器(moov 前截断)正确 ERROR、好
  容器通过;真实历史会话 QC 回归(camera/eye 新检查无误报,空目录仍由
  StreamPresent 捕获);pack probe 对真实会话完整发现全部特征流;
  check_cameras/check_emg 的外部符号逐一存在;三个入口与 pack 两个
  脚本的 CLI 全部可用;recordings.yaml 自动插入的 enabled 标记不破坏
  yaml 结构、不触碰注释槽位。


### 合并 prod 生产线:单条打包 / EGO GPU 写盘 / 图纸新命名 / n/r/f/q(自 Embodied-Brain-Collect-prod 1.3.1 定向合并)

**单条打包管线(pack_episode)**

* 新增 `scripts/pack_episode.py`:一条会话打成一份独立 LeRobot 数据集
  (与 pack_daily 的"一天一数据集"相对),质量门槛与 pack_daily 完全一致
  (qc_report / meta status 非法拒收、marker 必须有 RUN_START/RUN_END 窗口、
  视频容器帧数必须覆盖窗口末帧)。输出镜像保存路径
  `data/lerobot/<班次根>/<会话名>`。
* launcher 集成:`run_pack_episode`(进程内执行,复用 `sys.modules`)+
  `warm_pack_dependencies`(开录前后台线程预载 mf_lerobot→torch 的 ~11s
  冷启动,之后每条打包零导入等待)+ `--pack-episode` CLI 与
  `session.yaml` 的 `pack_episode: true`。**触发条件 = QC 无错且操作员
  保留(n/q)**;error/failed 的目录留档但不出数据集。三个入口
  (env/tasks/video)经 run_base 同样生效;launcher 单发路径 rc==0 即打。
* pack_daily 配套:mf_lerobot 延迟导入(预扫不再静默卡十几秒)、
  `filter_meta_status`(meta status 非 success 整条排除)、probe 侧
  ffprobe RuntimeError 已有守卫(容器损坏跳过该条,不炸整次打包)。

**EGO 头戴 GPU 写盘**

* 新增 `recorders/ego_headband/ffmpeg_jpeg_writer.py`:四路鱼眼
  Motion-JPEG 不经 CPU 解码,管道直给 ffmpeg `mjpeg_cuvid` 解码 +
  `hevc_nvenc` 编码(实测 ~113fps/路 @RTX 3050;CPU 路线 17fps 会丢帧),
  无 NVIDIA 自动回退 CPU。`base_camera_recorder` 新增 `_make_writer`
  钩子,`net_ego_headband_recorder` 每相机独立队列+线程的 JPEG 旁路。
* `ffmpeg_writer`:ffmpeg 子进程进独立进程组/会话 —— 控制台 Ctrl+C
  绝不直达 ffmpeg,防止死在写 moov 之前容器永久没有索引;收尾只由
  stdin EOF 驱动。

**图纸新命名 + 读取缓存(environment 1.3.1)**

* `drawing_info` 新规则:文件名含 overview 的总览图优先排除;其余按
  `_` 分割取前 3 段,第 3 段全数字即 num(`场景_任务_num`,无 combo/rep);
  旧命名 `[前缀_]NNNN_comboNNN_rN` 仍兼容。`scene_title` 对缺失的
  combo/rep 不再显示 "None"。
* `Environment` 增加按场景目录的 config/used.yaml 进程内缓存:
  100 张图纸的队列构建 ~4.6s → ~0.1s(Windows 实测)。

**n/r/f/q 与 enabled 提示**

* 确认输入新增 **f = 当前采集失败,退出**:状态合成与重跑一致
  (QC ERROR → error,否则 failed),不记台账,汇总标"失败退出"。
* `run_session_env` 摆放确认键 n + Enter → **s + Enter**(自 prod;
  environment.show 同步)。
* `recorders/factory`:操作员点名但 yaml `enabled: false` 的槽位必须
  喊出来(少一路数据毫无线索);队列开始前打印全部停用槽位;
  `recorders.yaml` 每槽位显式 `enabled: true` 标记。
* 生产配置入库:`configs/environments/` 两场景 102 张图纸
  (餐桌_做三明治/餐桌_冲咖啡,含 config/placements/台账)、stim.yaml
  双屏(display:1 / drawing_display:0 / 图纸窗 2400×1200)、部署工具
  `sd_card_auto_uploader.exe` + `start_upload.bat`。

**视频容器损坏在 check 阶段标 ERROR(不进打包)**

* 新增 `checks.ContainerIntegrity`:ffprobe 解封装失败(容器没
  finalize/moov 缺失/损坏)直接 ERROR,注明"不会进入打包";接入
  camera/eye checker。pack 侧 probe 对 RuntimeError 跳过该条,
  单条坏视频不再能炸掉整次打包(media.ffprobe_count 的报错带 stderr
  诊断,自 prod 合并)。
* `session/config`:`SESSION_RUN_KEYS` 增加 `pack_episode`;
  launcher meta 随图纸固化 `notes`(config.yaml scene.notes)。
* 检查工具增强自 prod:`check_cameras`(槽位对应/Realsense 显式
  color 流/DepthAI 绑定设备/非阻塞 poll)、`check_emg`(IMU 面板/
  自适应幅度/文字缓存)。

**验证**:`_fit_eeg_to_pc` 向量化实现与 v1.3.0 朴素 O(n²) 参考实现的
等价性测试(规整+离群/1ms 抖动/1 万点密包三种输入,slope 1e-9 一致);
video_rate、BrainCo 接入与逐条录制保持不变,全套测试通过。

## 1.4.0 — 2026-09-18

### video_rate 改为逐条录制:每 trial 一次独立会话,产物自包含(run_session_video)

* 原来一次 launch 连续录完全部 trial(所有 trial 挤在同一份 npz 里,录完
  才一起落盘)。现在每条 trial = 一次独立录制会话(录 → 存 → 下一条),
  与图纸/任务入口对齐;打包侧单目录 = 单 episode,天然对齐 mf-lerobot。
* stim 经 `--result-dir <录制目录>` 把评分 JSON 直接写进本次录制目录
  (rating_<目录名>.json);录完后该条视频 **copy** 进录制目录 —— 一个
  数据目录自带 EEG/marker/评分/视频,不依赖外部路径。
* 台账按条记账:rc==0 且 QC 无错才 uses+1;rc!=0(刺激中止/崩溃)视频
  不拷贝、不记账,QC 有错视频拷贝(数据完整)但不记账。meta 状态合成
  新增 strict_rc:视频模式没录完整(rc!=0)一律 failed,不吃"QC 干净"
  的亏(run_base.run_queue 新参数;图纸/任务行为不变)。
* run_base 钩子签名随之演进:stim_cmd 增加 run_dir 参数,on_result 增加
  rc 参数 —— 模式需要把产物写进录制目录/按返回码记账时用。

### 素材按子集(任务)组织:configs/videos/<子集>/ + 任务指令作想象内容(run_session_video)

* 目录结构对齐图纸模式的场景目录:一个子目录 = 一个任务,必带
  `config.yaml`(与 `configs/environments/<场景>/` 同构),内含 `task`
  字段 = 任务指令 —— 开录时填进 stim.yaml `video_rate.instr1_text` 模板的
  `{task}` 占位符,作为被试观看第一帧时的想象内容;可选 `scene`(随
  meta 固化)/ `name`(meta task_name,缺省目录名)/ `instr1_text`、
  `instr2_text`(整体覆盖默认指令)。缺 config.yaml 或缺 task 在选子集时
  即报,不带病开录。
* 台账随子集走:`<子集>/used.yaml`(uses/last_session,成功才记账);
  抽取组合、fail/success 配平、可解性 fail-fast 均限定在该子集素材池内。
  `--subset` 显式指定;仅一个子集时自动选,多个时交互选(非交互环境
  如管道/CI 优雅降级为 SystemExit 提示改用 --subset)。
* launcher `_write_session_meta` 新增 `task_name`/`scene` 参数;
  `FRAMEWORK_KEYS` 补 `scene`(pack_daily 早已单独消费 meta.scene,
  不进 FRAMEWORK_KEYS 会让它重复漏进 collect_info)。
* 修 `load_stim()` 的 lru_cache 隐患:缓存键不含 configs 目录,
  同进程先读仓库配置再切隔离目录(headless 测试的
  EMBODIED_BRAIN_COLLECT_CONFIGS)会串读 —— headless 测试曾因此拿到
  仓库的 serial: true 去开不存在的串口。stim 进程一次只读一次,去掉缓存。

### 视频解码:cv2 读不出帧(AV1 等)时回退 ffmpeg 管道软解(paradigm_video_rate/run_session_video)

* 用户素材是 AV1 编码,opencv-python 自带的 FFmpeg 没编译 AV1 解码器 —— 且
  `isOpened()`/fps/帧数**全部正常**、`read()` 恒 False:第一帧直接报错,
  播放更是"0 帧播完"的静默失败。新增 `_FrameSource` 统一解码源:后端只认
  "能否真读出一帧",cv2 不行就 ffprobe 取参数 + ffmpeg 输出 rawvideo rgb24
  管道逐帧读(libdav1d 软解;ffmpeg/ffprobe 走 utils.media 与录制/打包同一
  套可执行解析),`frames()` 可从头重放,第一帧/整段播放共用。
* 实测用户素材(640x480@30, 1149 帧)回退后整片解码 1.4s(~830fps),
  远超实时,播放不掉帧;若台架软解跟不上,建议素材统一重编码 h265
  (configs/videos/README 编码说明)。
* `run_session_video` 开录前逐条试解第一帧(fail-fast):解不出的素材直接
  拦截并列出 + 给重编码命令,不再等 stim 中途死掉;`--dry-run` 同样过此校验。

### 修 launcher.run_qc 的 NameError:QC 一成功就崩,qc.html 永远出不来

* `run_qc` 把 `load_checker()` 内联传给 `qc_session(...)`,而下面的
  qc.html 开关又引用未定义的 `checker_cfg` —— 任何一次成功跑完检查的 QC
  都在 html 开关处 NameError(检查失败反而没事,走的是另一条 return)。
  改为先存 `checker_cfg` 再传入,并对齐 scripts/qc.py 的容错
  (checker.yaml 缺失时按空配置继续)。qc_batch/qc 两脚本本来就是对的,
  只有 launcher 这条路径错。回归测试:run_qc 跑空 session 必须完整走完
  并写出 qc_report.json。

### 入口按模式拆分:run_base 公共引擎 + 三个入口脚本

* 新增 `session/run_base.py`:三个入口共享的编排引擎 —— 会话汇总
  (collect_runs/print_summary/save_summary,原 run_session 私有实现下沉)、
  n/r/q 确认与 meta 状态合成(ask_next/mark_meta)、失败排查提示、
  `record_one`(recorder 工厂 → meta → stim 命令 → launch → QC)、
  `run_queue`(队列循环 + 中断/重跑/退出处理),以及公共 CLI
  (--session-dir/--seed/--duration/--dummy/--recorders/--skip-qc/
  --collector-id/--set)。模式差异收敛为 `ModeHooks` 四个缝:开录前准备
  (图纸摆放/Enter/无)、meta 字段(environment/task_id/task_name+scene)、
  stim 命令(--environment/--task-id/--trials-json)、成功记账(图纸台账/
  视频台账/无)。
* 入口拆成三个薄脚本,各自只做队列构建与钩子注入:
  `run_session_env.py`(图纸)、`run_session_tasks.py`(任务列表)、
  `run_session_video.py`(视频,固定 auto-keep,成功记子集台账,现在也
  打会话汇总)。`run_session.py` 保留为兼容转发壳(按 --mode 或
  session.yaml 的 mode 转发,旧命令不失效)。env/tasks 的 --stim 排除
  video_rate(需 trials-json,手选无意义);采集信息(--collector-id/
  --set)三个入口统一支持。
* 汇总的任务标签(meta_task_label)新增 meta.task_name 回退 —— 视频
  会话的 task_name 由此进汇总明细。

### 接入 video_rate RLHF 视频打分范式 + BrainCo BCIGo EEG(合回两个精简包的新东西)

**统一码表 —— video_rate 多 trial 阶段码段(markers.yaml + stim/marker_codes.py)**

* `markers.yaml` 新增可选键 `video_rate_base: 16`(null = 禁用):RLHF 视频打分
  的每 trial 阶段码(FIX/双段指令/图/视频/打分/重播共 14 相位)按
  `base + trial*14 + 相位` 编码,整场唯一 —— EEG 按码配对要求场内码不重复;
  段尾留 48 个重播打标码。trial 上限由 RUN_START 当前值反推(当前 12),
  导入时校验段顶不得触到 RUN_START。`name_of`/`is_known` 认识整段
  (`VR_INSTR2_ON_T06`/`REPLAY_MARK_07`);命名码显示优先(数值与 0x10-0xEF
  区的单 trial 命名码重叠 —— 一次采集只跑一种 stim,同场码仍唯一)。

**stim/paradigm_video_rate.py —— RLHF 视频打分范式(自 brainco_RLHF_Video 合回)**

* 单 trial:红点注视 → 指令1 → 图像(被试控) → 指令2 → 视频(真实时长) →
  动作质量打分(数字键+回车) → 重播打标(Shift=对/回车=错,可重做) → 播放
  质检;分数写 UDP tag(SCORE\*)与 `rating_<时间戳>.json`,不占 TTL 码位。
* 素材走 `--trials-json`(JSON 数组,每项 {video, image?, instr?}),
  **第一帧播放时直接从视频现读**(cv2 读首帧 → pygame surface,按 trial
  缓存),不落地 png;无 image 键时自动走现读路径。
* 配套:`MarkerSender.mark(..., ttl=False)` 只走 UDP(重播打标不打串口脉冲,
  免得 20ms 高电平卡播放线程);BaseStim 新增 `_wait_keys`/`_wait_enter`、
  `--sync-udp-port` 公共参数与 `font_banner`/`continue_color`。

**scripts/run_session_video.py —— video_rate 主控(取代老包的预生成场次方案)**

* 素材统一放 `configs/videos/`(`success_*`/`fail_*` 前缀命名,历史 sucess_
  拼写也识别;详见该目录 README)。每次运行时抽取组合:两池各取
  **使用次数最少**的视频,fail/success 尽量对半(fail 偏多),场内打乱;
  池子不足自动从另一池补并告警。`--dry-run` 只看组合,`--seed` 可复现。
* 台账 `configs/videos/used.yaml`:采集成功(rc==0)后本次用到的视频
  uses+1 并记录 last_session;失败不动台账。台账损坏按空处理不挡采集。
* 组合经 `--trials-json` 传给 stim 子进程;其余与 launcher.main 同构
  (recorders 工厂 → meta → launch → QC)。

**recorders/eeg/brainco_eeg_recorder.py —— BrainCo BCIGo 32 通道帽(软件同步)**

* `bcigo-sdk` 懒加载;mDNS 自动发现/手动 host:port/按序列号过滤;
  sample_rate/gain/signal 可配;32/33 通道(固件附 TRIG)自适应 reshape。
* 软件同步:SDK 回调即盖 `time.time()`、按 LSL 惯例"包内最后一样本≈包到达
  时刻"回推,每个 EEG 包是一个时钟观测点;收尾用与 Curry 相同的稳健直线
  拟合写 `eeg_timestamps_pc`,marker 的 `t_sent_pc` 经拟合映射回样本号写成
  `eeg_event_code/latency`(输出 schema 与 Curry 一致);包时钟拟合失败时
  回退在线软件 TTL 事件配对。`sync_udp_port` 在线心跳插值仅作显示,落盘以
  包时钟为准。
* `EegRecorderConfig` 新增 BrainCo 字段(sample_rate/gain/signal/
  scan_timeout_s/sync_udp_port/device_sn,Curry/dummy 忽略);
  recorders 工厂注册 `brainco_eeg`;recorders.yaml 留注释好的切换模板;
  requirements 注释 bcigo-sdk(按需安装)。

**base_eeg_recorder._fit_eeg_to_pc 向量化(自 brainco 包合回)**

* 成对斜率 O(n²) 纯 Python 双循环在 BrainCo 包时钟(一场 1 万+点)会卡几十秒,
  launcher join(15s) 杀进程丢 eeg.npz —— 改为均匀抽点(MAX_PAIRWISE_POINTS=
  400)矩阵化估斜率,内点/polyfit 仍全量。Curry/dummy/Intan 路径同享。

**launcher**

* 修 `--stim paradigm1` 直连路径的 NameError(task_id 未初始化即引用);
  stim kind 列表/帮助文案更新。

## 1.3.0 — 2026-09-17

### recorder 工厂下沉到 recorders 包,砍掉 28 个 get 样板(session/recorder_presets)

* 新增 `recorders/factory.py`:`REGISTRY`(kind → 子模块/Recorder 类/
  Config 类,24 个 kind)+ 通用 `build(kind, params, session_dir,
  duration)` —— **Config dataclass 的字段默认值就是默认配置**,yaml 写了
  的键覆盖、没写的走 dataclass 默认,不再需要每模态一个 `get_*` 函数把
  默认值抄一遍再透传。条目按需 import:neon/openvr/manus/pycbsdk 等 SDK
  只在对应 kind 被构造时才加载,没装 SDK 不影响其他模态。
* 这同时修掉一个**双层默认 bug**:旧 `get_intan_eeg` 会显式传
  `digital_map=None`,把 dataclass 的 `default_factory`(猴台架映射)吃
  掉 —— presets 路径下该默认从不生效。通用构建后 yaml 不写该键即走
  dataclass 默认;dict→tuple 归一化(ego topics)下沉到对应 Config 的
  `__post_init__`。
* `session/recorder_presets.py` 整个删除,编排逻辑(单槽位构建
  `build_recorder`、`get_production_recorders`、dummy 捆绑
  `get_dummy_recorders`)并入 `recorders/factory.py` —— device 层自带
  工厂,session 层不再留一份;launcher/run_session/preflight 只改 import
  路径,函数名与签名不变;`check_emg` 改走 factory。

### 数据集元数据:collect_info.jsonl 成为唯一边车,info.json 不再附加字段(pack_daily)

* info.json 保持 mf_lerobot 写下的 LeRobot 标准字段(fps/特征表)——
  pack_daily/pack_daily_fast 不再往里写 `collect_version`/`hardware`。
* `meta/collect_info.jsonl` 成为元数据边车,一行一个 episode,顶层平铺
  全部额外信息:`episode_index`、`session`、`collect_version` 采集程序
  版本、`hardware` 该会话实际采集的槽位→设备显示名(逐会话,不再是
  跨会话并集)、`collector_id` 等操作员维护键、`status` 录制结局
  (success/failed)、`task_name`(两种模式都有)、`scene`/`objects`
  图纸物体摆放(tasks 模式为 null)。
* 中间形态 objects.jsonl 撤销,内容并入本文件。

### 采集信息与图纸台账(run_session)

* `configs/session.yaml` 顶层维护操作员采集信息(`collector_id` 编号等;
  `mode`/`stim` 与框架保留键之外的顶层键都算,键不设 schema 可自由增减;
  避开 YAML 布尔词 yes/no/on/off —— `no:` 会被解析成 False)。
  **run_session 全部参数支持命令行覆盖**:`--collector-id` / `--set 键=值`
  (可多次;优先级 yaml < `--collector-id` < `--set`),启动横幅打印解析
  结果。开录时 launcher 把解析结果固化为该 session meta.yaml 的顶层
  字段 —— 录制时快照,之后改 session.yaml/换 CLI 参数不影响已录数据。
* 结算分支 q(成功并退出)原来漏记图纸台账:只有 n(保留)调
  `mark_used`,q 只标 `status: success`,成功采集的图纸下次还会被抽到。
  现在 q 与 n 一致:图纸模式记入该场景目录 `used.yaml` 并打印确认。

### session meta.yaml 新格式:task_name/scene/objects 顶层平铺(图纸模式)

* 开录固化字段重排:`task_name` **两种模式都有**(图纸模式取图纸
  config.yaml 的 `task`,任务模式仍取 tasks.yaml);`scene`(config.yaml
  的 `scene.name`)与 `objects`(**直接是物体列表**)仅图纸模式;
  num/combo/rep 不再重复存 —— 图纸号由 environment 文件名自带。
* 每个物体 = `name`/`color`/`shape`/`dims`(config.yaml)+ `cx`/`cy`/
  `ang`(placements.csv 中 num 匹配该图纸的行)+ `id` 与 `material`。
* **槽位组**:带 `slot.candidates` 的物体(如同名保鲜盒的 box/cyl 两种
  变体),顶层属性只是默认值 —— 按 placements 的 `{name}_obj` 列
  (`保鲜盒#<id>`)命中候选变体,shape/color/dims/material 一律以变体
  为准(未命中/无槽位回退物体自身默认;旧场景无该列时 id 为 null)。
* 打包兼容:旧会话的包壳快照(task/scene/num/combo/rep/objects)自动
  拆包,更旧的按 meta 里 environment 路径按当前 configs 原地重建。

### 图纸文件名容忍场景前缀(environment)

* 绘图工具的导出模板(`nameTpl: {scene}_{task}_...`)会给图纸与配置文件
  带场景名前缀(`餐桌_保鲜盒_0001_combo015_r2.png`、
  `餐桌_保鲜盒_config.yaml`、`餐桌_保鲜盒_placements.csv`),框架原来按
  裸名 `NNNN_comboNNN_rN.png`/`config.yaml`/`placements.csv` 查找,整个
  场景池直接为空。现在:`drawing_info` 的正则容忍任意非空前缀(总览图
  不含 combo 段仍被排除);`scene_config`/`scene_layout` 在精确名缺失时
  回退 `*_config.yaml`/`*_placements.csv`。台账键 = 实际文件名,不受影响。

### 简化总检:配置为唯一来源,砍掉全部静默回退

* **stim 参数唯一来源化**:base_stim/paradigm1/rgb/sync_test 的 argparse
  默认值不再存代码字面量(`_required` 从 stim.yaml 取,缺键启动即报缺
  哪个;只有语义默认留在代码:fast=1.0 不压缩、fullscreen 默认开)——
  代码与 yaml 双份默认早已漂移过(instr_s 代码 10s/yaml 5s),并当场
  抓到 rgb 段缺 font_size 的真实配置缺口。`parallelbox` 默认空串:serial
  开着没配口由 MarkerSender 拒绝(原来默认 "COM14" 会去开不存在的口);
  MarkerSender 自身的 port 默认 COM14 一并删除。
* **marker 码表补全必需项**:`hand_cue_base` 不再有 0xC0 内置默认,
  markers.yaml 必须显式给(数值或 null),与"码表无内置默认"策略一致。
* **QC 无窗口直接收场**:`find_run_window` 找不到 RUN_START/RUN_END 时
  qc_session 记 ERROR 后**立即返回,不再按全部数据跑各流检查** —— 全
  量结果只会误导(看着合格,实际不可对齐);print_report 的"会话起点"
  等回退分支随之删除。
* **pack 与 QC 同门槛**:非 `--full` 打包遇到无窗口会话直接跳过并说明
  (原来静默回退相机区间,产出看似正常实则不可对齐的数据);`--full`
  是显式全量模式,保持相机区间。
* **防御性 try 清理**:launcher run_qc / run_session html 开关 /
  environment.show 读 stim.yaml 的 FileNotFoundError→静默默认全部删除
  (configs 随仓库分发,缺文件即报);check_emg 槽位配置整体透传
  factory,不再手抄 port/baud(921600 字面量删除)。

### meta.status 三档合成:QC 优先于交互按键(run_session)

* 结局标记从"按键决定 success/failed"改为综合判定:**QC 有 ERROR →
  `error`(无论 n/r/q);QC 无错且按 r → `failed`;QC 无错且按 n/q →
  `success**。仅 `success` 记图纸台账(QC 有错按 n 保留数据也不再占图纸)。
  打包的 collect_info.jsonl 逐 episode 携带该状态,下游可按
  success/failed/error 分流。

### RUN_START/RUN_END 成为硬性 QC 门槛(marker checker)

* 一条数据的 marker.npz 必须含完整的 RUN_START→RUN_END 对,否则 QC 直接
  判 ERROR —— 不再是"按全部数据范围检查"的 WARN 回退(那种数据对 EEG
  对齐/打包裁剪都无从谈起,静默降级只会把问题推到打包端)。四种缺情形
  各有明确报错:未记录任何标记 / 缺 RUN_START / 缺 RUN_END / RUN_END 早于
  RUN_START(窗口无效);会话级 RunWindow finding 同步 WARN→ERROR。配合
  台账门控,这样的数据按 n/q 保留也不会把图纸记入台账。

### dummy 模式强制关闭 stim 串口(run_session/launcher)

* `--dummy` 是无硬件试跑,但 stim.yaml 的 `serial: true` 会让 stim 一启动
  就去打开 ParallelBox 串口 —— 机器上没有该设备时直接
  SerialException 崩掉,整条 marker 链(连 UDP 通路)全挂。现在 dummy
  模式给 stim 强制加 `--no-serial` 并打印提示:marker 走 UDP 通路
  (dummy 的 marker 槽位本来就是真实 UDP listener,marker 事件照样进
  数据),需要验证真实 TTL 时跑正式采集或手动运行 stim。launcher 独立
  CLI 的 `--dummy --with-stim` 路径同样处理。

### 展示屏幕可配置,图纸展示改用 pygame(environment + base_stim)

* `configs/stim.yaml` 公共段新增三个键:`display`(stim 程序启动屏幕,
  0=主屏)、`drawing_display`(图纸展示屏幕,缺省同 `display`)、
  `drawing_fullscreen`(图纸是否全屏,默认 true,false=窗口化方便主屏
  同时操作)。stim 侧有 `--display` CLI 覆盖;索引越界(配置的屏幕不
  存在)打印告警并回退主屏。
* 图纸展示从 matplotlib 改为 **pygame**(Environment.show 重写):消除
  多屏定位的后端难题(两套窗口体系统一用 SDL 的 display 参数),采集
  路径不再依赖 matplotlib。交互契约原样保留:n+Enter 确认开始、其他
  按键无效并提示、Esc/关窗=取消;图纸按比例缩放居中,标题/提示用中
  文字体渲染。

### 台账门控与最大采集量(run_session)

* **图纸台账只在 QC 无 ERROR 且按 n/q 时记录**:原来只要按 n/q 就记
  台账,QC 有 ERROR 的数据也会把图纸占掉。现在 QC 有 ERROR 时按 n/q
  仍保留数据(meta 标 success),但图纸**不记台账** —— 之后会重新抽到
  重采;提示语与 QC 判定行都会明示这一后果(输入提示同样带警告)。
  `--skip-qc` 时无 QC 信息,视为无 ERROR(操作员自行选择不检查)。
* **session.yaml 新增 `max_runs`(默认 50)**:设置了该值时,图纸/任务
  队列只取前 N 条,次数到即停;0 或不设 = 不限。CLI `--max-runs` 可覆盖
  (优先级 yaml < CLI),启动横幅打印本次计划条数。

### 新增 rgb 色块刺激(stim/rgb_cue)

* 纯颜色三阶段流程,**无任何文字提示**:黑底 + 5/4/3/2/1 倒数=等待期
  (每秒一跳,``wait_s`` 秒,数完自动切换)→ 蓝色=想象期(空格切换)→
  绿色=操作期(空格结束)。颜色 `#RRGGBB` 可配(stim.yaml `rgb:` 段,默认纯 RGB
  原色),阶段顺序与 marker 语法与 paradigm1 一致:RUN_START →
  INSTR_ON/OFF(红)→ IMG_START/END(蓝)→ EXEC_START/END(绿)→
  RUN_END;Esc 中止也补全阶段码与 RUN_END。已注册 stim 工厂
  (`--stim rgb`),无需 task-id,图纸/任务两种模式都可用;headless
  对拍验证发码序列逐值正确。

### marker 码表配置化,取消 P1 边界码(stim/eeg)

* 新增 `configs/markers.yaml`:全部具名码(RUN_START/RUN_END/FIX_ON/…,
  必须齐全)与手势码段基址 `hand_cue_base` 可按台架改。代码只认键名,
  数值导入时读取一次并强校验 —— 文件缺失/缺 codes 段/缺码名/未知键名/
  越界/码值重复,导入立即失败并给可操作报错,绝不带病发码(重复码会让
  EEG 按码配对失效);**没有内置默认表**,码表唯一来源是该文件。
* **取消 P1_RUN_START/P1_RUN_END**:simple_stim 与 paradigm1 统一发
  RUN_START/RUN_END。两线 TTL 接法的台架(码会掉位碰撞)直接在
  markers.yaml 把边界对设成单比特码(如 run_start: 16 / run_end: 32),
  intan 的 digital_map 默认映射目标也自动跟随码表当前值,代码层不再有
  第二套边界码。

### 新增 scripts/check_emg.py — EMG 左右手对应检查

* 实时显示左右两条 EMG 臂环(recorders.yaml 的 emg_left/emg_right 槽位,
  `--left-port/--right-port` 可临时指定串口)各 8 通道的滚动波形,绿=左
  橙=右,附活动值与帧率读数。操作流程全在窗口内:L 晃动左手 → R 晃动
  右手,各采一个窗口(`--seconds`,默认 3 s)自动对照两侧肌电活动量,
  给出"对应 ✓ / 疑似接反 ✗ / 活动不明显"提示;肉眼复核波形后 Y 确认
  退出(码 0),Q 放弃(码 1)。单事件循环,提示期窗口不冻结。
* 不经 recorder 落盘:直接驱动 weili_emg 的 open/poll(preflight 同款),
  临时目录不产生 npz。串口打不开时给可读报错并退出码 1(含
  SerialException 接住)。

### intan 修复与猴台架适配(eeg)

* **移除 open 服务器自愈**:"3s 无波形数据就 disconnect→connect 重启
  RHX TCP 波形服务器"的自愈段删除(连同 `_wait_first_block`):流确认
  统一归 launcher 的确认阶段(`_wait_data_flowing`),recorder 的 open
  只管连接与配置,与 blackrock/curry 同构。
* **`_data_connect` 方法名笔误修复**:open 调用的是不存在的
  `_connect_data_stream`,`except Exception` 把它吞成 "no attribute"
  —— intan 在本线上从未成功 open 过。已改回正确名字,失败路径给干净的
  连接错误。
* **digital_map 默认启用猴台架接线映射**:`IntanEegRecorderConfig.
  digital_map` 默认 `{"0x4000": 16, "0x2000": 32}`(实测接线 box
  bit4→DIN14、bit5→DIN13)映射到 markers.yaml 码表的边界对当前值
  (两线接法请在 markers.yaml 把 run_start/run_end 设为单比特码
  16/32),查表还原后才能进 EEG 对齐拟合;接线改动在 recorders.yaml
  覆盖。dataclass 字段
  经 `field(default_factory=…)` 给默认(dict 实例直接作默认值会在
  import 时抛 ValueError)。

### qc.html 支持设置降采样系数(qc_payload)

* 新增 checker.yaml 配置 ``html_max_pts_per_s``(点/秒,默认 0 = 不降采样,
  v1.1.0 全分辨率策略不变)。系数为正时,密度超过 ``时长 × 系数`` 的序列
  在编码前做 min/max 包络降采样 —— 尖峰不丢(孤立毛刺就是一两个样本宽,
  逐桶取极值都能存活),只影响页面,NPZ 原始数据从不改写。实测 30 kHz
  × 10 s 的序列:系数 0 为 2.3 MB/30 万点,系数 1000 为 78 KB/1 万点,
  尖峰无损。低于预算或不足 ``MIN_PTS``(600)的序列不受影响,逐样本
  缩放照旧;``uniform_ts`` 的 stride 只在未降采样时发(降采样后 x 间距
  不规则,series() 内强制,不再依赖调用方约定)。`series()` 的显式
  ``max_pts`` 参数仍是最高优先级,fine-EMG 等既有用法不受影响。

### 深度视频不再进黑屏检查(camera checker)

* 相机槽位的三个视频检查(BlackFrame/Freeze/FrameCountMatch)原来
  `video=""` 时按字典序取目录里第一个 mp4 —— 深度槽位(cam 目录里同时有
  `frames.mp4` 与 `depth_frames.mp4`)会选中深度视频:深度图天然趋黑
  (gray12le 原始深度值,室内场景均值亮度 ~7,全帧低于阈值 8),BlackFrame
  必然误报"100% 黑屏"ERROR,把整份 qc_report 拖成 ERROR(实际踩坑:
  monkey 线 2026-09-16 的 cam_head 会话)。修复:视频检查钉死
  `video="frames.mp4"`,深度流不做黑屏/冻结检查;FrameCountMatch 的比对
  对象(RGB 时间戳)与视频文件也从此一一对应,不再张冠李戴。RGB-only
  相机行为不变。

### 多 COM 口 TTL、simple_stim 接入、头环 config 接入(自 monkey 线合入)

* **MarkerSender 多 COM 口**:``port`` 接受逗号/分号分隔串或列表
  (``"COM5,COM14"``),同一个码同时写全部 TTL 口、共享同一保持窗口
  (时序与单口一致);单口打开/写入失败只响亮告警并摘除,其余口照常,
  全部失败才硬失败(与旧版单口行为一致)。stim.yaml ``parallelbox``
  与 ``--parallelbox`` 注释同步。多台放大器各自对齐时每台接一台
  ParallelBox 即可。
* **simple_stim 注册进 stim 工厂**:最小刺激流程(任务名 → 空格开跑 →
  空格结束 → 退出,只发一对 RUN_START/RUN_END 边界码,数值随
  markers.yaml 码表,Esc 中止也补 END 保住码对)。``marker_codes`` 补 P1 边界码(并入
  NAMED 反查表);``build_stim_cmd`` 对 simple 传 ``--task-id``
  (无 ``--once``,本来就是单次流程);stim.yaml 增 ``simple`` 参数段,
  session.yaml 的 stim 键注释同步。仅任务列表模式可用(需要 task-id)。
* **头环(EGO headband)config 接入**:``get_net_ego_headband`` 签名对齐
  新版 wired-TCP recorder(host 192.168.55.6:5577、connect_timeout、
  require_synced、camera_topics/camera_names/imu_topics、crf/preset/
  encoder、audio_enabled/require_audio/audio_topic),recorders.yaml 的
  ego_headband 块按新键更新(保持注释态,接好设备后启用);
  ``get_dummy_ego_headband`` 进 dummy 工厂映射;troubleshooting 提示去
  掉 UDP 旧说法。配套把 EgoHeadbandChecker 与 QC 页 ego_headband 提取
  器切到 ``{name}_timestamps`` + ``{name}.mp4`` 新 schema(每相机追加
  FrameCountMatch/BlackFrame/Freeze 视频检查),dummy headband 的
  schema/时钟/QC 三个测试按新契约更新。
* 不合入:YOLO 离线手部检测、任务库/任务流程改动。

### meta 附每个 episode 的 qc report(pack_daily)

* 打包结束时把每个 episode 对应源会话的 `qc_report.json` 原文汇总写进
  数据集 `meta/qc_reports.jsonl`(一行一个 episode:`episode_index`、
  `session` 源会话名、`level` 整体等级、`qc_report` 报告全文,
  findings/streams 明细全保留,约 30KB/集),QC 结论随数据走,不用回
  源目录查。`pack_daily_fast` 同步。

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


## 1.2.0 — 2026-09-16

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
