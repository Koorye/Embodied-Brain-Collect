# USB 相机配置（OpenCV / opencv_camera）

> 位置跟踪之外的所有普通 USB 相机（UVC 协议）都走 `opencv_camera` recorder：
> 经 OpenCV `VideoCapture` 逐帧采集。本项目现役的三个相机槽位（左腕、右腕、
> 第三视角）都是这一路。D455f 若要走 librealsense 按 serial 采集，
> 见 [realsense_camera.md](realsense_camera.md)。

OpenCV 相机只有**数字索引（idx）**、没有稳定身份，配错 idx 就会录错相机。
本文说明 idx 的机制、用 `scripts/setup/map_cameras.py` 人工指认的流程，以及
`configs/recorders.yaml` 的对应配置。

---

## 1 开始前的准备

- 相机经 USB 接入采集机， Windows/Linux 均可；装好系统驱动（能出画面的程度）。
- `pip install opencv-python`（采集机随 requirements 已装）。
- 指认工具：`pip install cv2-enumerate-cameras`（可选但强推，能显示设备名
  和 USB VID:PID；不装则脚本退化为盲扫索引）。

## 2 index 机制与漂移

`idx` 是后端**枚举顺序**，不绑定硬件：换 USB 口、换插拔顺序、系统更新，
都可能让同一台相机的 idx 变化。后端按平台固定（与采集 recorder 同款）：

| 平台 | 后端 |
|---|---|
| Windows | DirectShow（`CAP_DSHOW`） |
| Linux | V4L2（`CAP_V4L2`） |
| macOS | 自动（`CAP_ANY`） |

**对策**：

1. 相机插好后**固定 USB 口位不再动**，并把口位记在 yaml 注释里
   （模板即如此：`idx: 3  # 左腕相机，USB 集线器左起第二口`）。
2. 每次重新布置工位后，用 `map_cameras.py` 重新指认一遍。
3. 日常采集前用 `check_cameras.py` 体检，槽位对不上会当场暴露。

## 3 指认流程（map_cameras.py）

```bash
python scripts/setup/map_cameras.py               # 枚举 + 全部开窗,画面烙大字 idx=N
python scripts/setup/map_cameras.py --list        # 只列清单:idx / 设备名 / VID:PID / yaml槽位
python scripts/setup/map_cameras.py --backend msmf   # 换后端(排查用,一般不动)
```

步骤：

1. 运行脚本，**所有**能出图的相机同时开窗（拼成一张网格），每格画面正中烙着
   大号 `idx=N`，顶栏显示设备名、`VID:PID`、分辨率@帧率。
2. 依次对着每台相机挥手/遮挡：哪个画面跟着动，它就是 `idx=N`。
3. 顶栏方括号是 yaml 里已绑该 idx 的槽位名（`no-slot` = 没绑；`a+b` = 多个
   槽位配了同一 idx，**配置错误**，当场暴露）。
4. 把确认的 idx 抄进 `configs/recorders.yaml` 对应槽位，再跑
   `python scripts/check_cameras.py` 复核。

表格下方会提示 yaml 里配了但本次没打开的 idx（没接 / 被占用 / 漂移）。

> 注意：RealSense 若在线，其彩色头在 Windows 上也是一个可枚举的 UVC 相机，
> 可能占掉一个 idx 甚至出现在这里。录 D455f 请走 `realsense_camera` 槽位按
> serial 绑定（用 `scripts/setup/map_realsense.py` 指认），不要用 idx 录 RealSense。

## 4 yaml 配置

`configs/recorders.yaml` 槽位示例（摘自模板）：

```yaml
cam_left_wrist:
  enabled: true
  kind: opencv_camera
  name: "JIERUIWEITONG FH868"   # 设备显示名,写入 meta
  idx: 3                        # 左腕相机,USB 集线器左起第二口
  hz: 30
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | — | `opencv_camera` |
| `idx` | `0` | OpenCV 相机索引（本文主角） |
| `width` / `height` | `640` / `480` | 请求分辨率；谈不下来会在日志标 `DOWNGRADED!` |
| `fps` | `30` | 请求帧率 |
| `fourcc` | `MJPG` | 像素格式；高分辨率高帧率基本都要 MJPG |
| `hz` | `1000` | poll 上限；相机一般配 `30` |
| `crf` / `preset` | `23` / `medium` | libx265 编码质量/速度 |
| `preview_port` | `0` | 辅助员控制台 UDP 预览端口；`0` = 关闭 |
| `name` | — | 设备显示名，写入 meta |

## 5 验证

```bash
python scripts/check_cameras.py     # 按 yaml 开窗,每格标注对应槽位名
```

看到哪路画面 = 采集程序会录哪路；某槽位显示 no frame 或缺格，按
README「录制中的故障保护」与 troubleshooting 提示排查。

## 6 常见问题

| 现象 | 原因与处理 |
|---|---|
| 日志出现 `DOWNGRADED!` | 相机没谈下请求的分辨率/帧率:确认 `fourcc: MJPG`、换 USB3 口/短线;带宽不够就降 width/height |
| `map_cameras.py` 里 idx 能列出但打开失败 | 被 RealSense 彩色头等专用栈占用,或设备半死:拔插重试 |
| 两台相机画面互换 | idx 漂移了(动了 USB 口/插拔顺序):重新指认并固定口位 |
| 槽位顶栏显示 `a+b` | 多个槽位配了同一 idx,改 yaml 去重 |
| Windows 上枚举不到、设备管理器里有 | 试 `--backend msmf`;个别相机 DSHOW/MSMF 只有一个枚举得到 |

## 7 参考资料

- **cv2-enumerate-cameras**：<https://pypi.org/project/cv2-enumerate-cameras/>
  （枚举相机名/VID:PID 与 OpenCV index 的对应）。
- **OpenCV VideoCapture**：backend 编码进索引高位数字的说明见 OpenCV 官方文档。
- 本项目：`scripts/check_cameras.py`（体检）、`opencv_camera_recorder`
  （采集实现，后端选择见 `preferred_backend`）。
