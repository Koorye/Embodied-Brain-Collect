# Intel RealSense D455f 配置（realsense_camera）

> 第三视角相机走 `realsense_camera` recorder：经 librealsense（`pyrealsense2`）
> 采集彩色流。与 OpenCV 相机不同，RealSense 用 **serial（序列号）** 绑定设备
> —— serial 跨端口、跨重插稳定，多台同型号也靠它区分。
> 普通 USB 相机见 [opencv_camera.md](opencv_camera.md)。

---

## 1 开始前的准备

- RealSense（现役 D455f）经 **USB3 口**接入，线材合格（USB2 会上不了
  期望分辨率/帧率）。
- `pip install pyrealsense2`（采集机随 requirements 已装）。
- 可选：装 Intel 官方 `realsense-viewer` 交叉验证设备与固件
  （固件升级也用它）。

## 2 为什么用 serial 而不是 index

- RealSense 的彩色头在 Windows 上**同时是一个可枚举的 UVC 相机**，会出现在
  OpenCV 的索引空间里；用 idx 录 RealSense 容易和别的 UVC 相机混淆、且被
  专用 SDK/OpenCV 争抢（详见 check_cameras 的打开顺序说明）。
- 多台 D455f 长得一样，只有 serial 能区分；`recorders.yaml` 不配 serial 时
  recorder 默认拿枚举到的**第一台**，多台时等于抓阄 —— **多台务必配 serial**。

## 3 指认流程（map_realsense.py）

```bash
python scripts/setup/map_realsense.py               # 枚举 + 全部开窗,画面烙 serial 尾号
python scripts/setup/map_realsense.py --list        # 只列清单:serial / 设备名 / 固件 / yaml槽位
```

步骤：

1. 运行脚本，**所有** RealSense 同时开窗（拼成一张网格，640x480 彩色流，
   与 recorder 同款配置），每格画面正中烙着大号 serial 尾号，顶栏显示完整
   serial、设备名、固件版本。
2. 依次对着每台相机挥手/遮挡：哪个画面跟着动，它就是那台 serial。
3. 顶栏方括号是 yaml 里已绑该 serial 的槽位名（`no-slot` = 没绑；`a+b` =
   多个槽位配了同一 serial，**配置错误**）。
4. 把确认的 serial 抄进 `configs/recorders.yaml` 对应槽位，再跑
   `python scripts/check_cameras.py` 复核。

表格下方会提示 yaml 里配了但本次没打开的 serial，以及**没配 serial 的槽位**
（提醒补上）。某格显示 no frame：多半 UVC 流被其他程序占用（关掉占用者），
或设备还在热身。

> 多台 RealSense 同时开流吃 USB 带宽，指认窗口偶有掉帧不影响；正式采集请把
> 多台相机分接不同 USB 控制器（机箱前/后口往往各是一路）。

## 4 yaml 配置

`configs/recorders.yaml` 槽位示例：

```yaml
cam_third:
  enabled: true
  kind: realsense_camera
  name: "Intel RealSense D455f"   # 设备显示名,写入 meta
  serial: "317622075551"          # map_realsense.py 指认后填入;空 = 默认第一台
  hz: 30
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | — | `realsense_camera` |
| `serial` | `""` | 设备序列号；空 = 自动拿第一台（多台时必须配） |
| `width` / `height` | `640` / `480` | 彩色流分辨率 |
| `fps` | `30` | 彩色流帧率 |
| `depth` | `false` | 同时录 depth（对齐到彩色）；带宽翻倍，按需开 |
| `hz` | `1000` | poll 上限；一般配 `30` |
| `crf` / `preset` | `23` / `medium` | libx265 编码质量/速度 |
| `name` | — | 设备显示名，写入 meta |

recorder 只显式使能**彩色流**（与 `map_realsense.py` 一致）；裸开全部默认流
（depth/IR 一起上）会抢带宽导致不出帧。

## 5 验证

```bash
python scripts/check_cameras.py     # 按 yaml 开窗,每格标注对应槽位名
```

`check_cameras.py` 打开顺序是 RealSense 在前、OpenCV 收尾：专用 SDK 先拿走
UVC 流，OpenCV 打到彩色头所在索引时读不到帧会自动跳过 —— 这是设计行为，
不是故障。

## 6 常见问题

| 现象 | 原因与处理 |
|---|---|
| `map_realsense.py` start 成功但 no frame | UVC 流被别的程序占用(OpenCV 预览/realsense-viewer/浏览器):关掉占用者再试 |
| 枚举不到设备 | 换 USB3 口、换线;用 realsense-viewer 交叉验证;Linux 检查 udev 规则 |
| 画面/帧率不稳 | 多台挤同一个 USB 控制器:分接口;带宽不够就降 width/height 或 fps |
| 多台相机录到了同一台 | yaml 里 serial 没配(默认抓第一台):按第 3 节指认后补上 |
| 槽位顶栏显示 `a+b` | 多个槽位配了同一 serial,改 yaml 去重 |

## 7 参考资料

- **librealsense**：<https://github.com/IntelRealSense/librealsense>
  （SDK、realsense-viewer、udev 规则说明）。
- 本项目：`scripts/check_cameras.py`（体检，含打开顺序防争抢）、
  `realsense_camera_recorder`（采集实现）。
