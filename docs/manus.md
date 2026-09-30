# MANUS Metaglove 配置与标定（manus_hand_pose）

> 左右双手套为 MANUS Quantum Metaglove Pro Haptics，`manus_hand_pose`
> recorder 经 `manus_glove` SDK（见 `third_party/manus_glove`，封装
> ManusSDK 3.1.1）连接 **Manus Core** 采集：每手 20 通道手指关节角
> （ergonomics，共 40 通道）+ 骨架节点位置/四元数。

手套本身不直接出关节角 —— SDK 开录前要给每只手套下发 `.mcal` **标定文件**
（`LoadCalibrationFiles`），它决定传感器读数到手指关节角的换算。**不装、装错
左右或标定过期，手指数据都不可信。** 本文说明：安装 Manus Core → 按流程标定
→ 把标定文件存进 `~\.cache\manus_glove\`（SDK 读取目录，第 3 节）→ 验证。

---

## 1 开始前的准备

- **安装 Manus Core**（桌面端），双手套开机配对到 Core。版本与本仓库
  `third_party/MANUS_Core_3.1.1_SDK` 的 SDK 对应（3.1.1）。
- Windows：需 Microsoft Visual C++ Redistributable；Linux：安装 udev 规则
  （`third_party/manus_glove/README.md` 有现成的 `70-manus-hid.rules`，
  装完 `sudo udevadm control --reload-rules && sudo udevadm trigger`）。
- SDK 动态库自动解析：仓库带 `third_party/MANUS_Core_3.1.1_SDK/` 时直接用
   vendored 库，否则回退 `~/.cache/manus_glove/lib`（首次由 Manus 安装器
  下载）。
- `pip install cffi loguru`（manus_glove 的依赖）。

## 2 标定流程（Manus Core）

1. 双手套开机，确认 Manus Core 里**两只手套都在线**（recorder 开录要求
   2/2 双手齐，缺一只直接失败）。
2. 在 Manus Core 手套仪表盘按官方流程**逐只标定**（通常含静止参考 +
   全指屈伸活动；具体界面以所用 Core 版本为准）。
3. 标定完成后在 Core 里把标定存成 `.mcal`：**users → profile settings →
   calibration profile → save**，直接存进 `~\.cache\manus_glove\`（见第 3 节）。

> 换人佩戴、手套松紧变化、长时间漂移后都应**重新标定**，标定文件跟着人走。

## 3 把标定文件保存到 SDK 读取目录

SDK 默认在 `~\.cache\manus_glove\` 找 `LeftMetaglovePro.mcal` /
`RightMetaglovePro.mcal`；若 `recorders.yaml` 的 `hand_pose` 槽位改过
`calibration_dir` / `left_calibration` / `right_calibration`，以 yaml 为准
（存错目录 = 白装）。

标定完成后在 Manus Core 里手动保存（**users → profile settings →
calibration profile → save**）：

1. 保存目录选 `~\.cache\manus_glove\`（目录不存在就先新建）。
2. 左手存成 `LeftMetaglovePro.mcal`，右手存成 `RightMetaglovePro.mcal`
   —— 文件名必须一字不差；左右存反，手指数据互反。
3. 重新标定后重新 save 覆盖旧文件即可；想留底就先复制一份（如 `*.bak`，
   SDK 只认配置里指名的 `.mcal`，多余的文件不影响）。

## 4 验证

```bash
python scripts/preflight.py        # 采集前预检,hand_pose 打开 + 数据流探测
```

正常时采集/预检日志应看到：

```
[hand_pose:manus] gloves detected: 2/2 (ids=[...])
Calibration loaded successfully for Left glove (ID: ...)
Calibration loaded successfully for Right glove (ID: ...)
```

录一小段数据：做出抓握/张开的对应手势，检查 40 通道 `ergo_data` 数值随
动、左右手通道（左 0-19 / 右 20-39）各自变化 —— 左右标定装反在这里会暴露。

## 5 常见问题

| 现象 | 原因与处理 |
|---|---|
| 日志 `Calibration file not found` / `No left calibration path configured` | 标定文件没存进 SDK 读取目录:按第 3 节在 Manus Core 里重新 save 到 `~\.cache\manus_glove\`,核对目录与文件名 |
| `only 1/2 gloves found` 开录失败 | 一只手套没配对/没电:在 Manus Core 里配对齐两只再试 |
| `udev rules not found`(Linux) | 没装 udev 规则:按第 1 节安装 |
| 数据随动但左右手互反 | 左右标定存反:在 Manus Core 里把两个 `.mcal` 按正确的左右重新 save 覆盖 |
| 手指角度漂移/明显失真 | 标定过期:重新标定并重装;确认佩戴松紧与标定时一致 |
| 手套中途停止出数据 | recorder 原地抛错终止(设计行为):检查电量/USB 接收器,重连后重录 |

## 6 参考资料

- **manus_glove**（`third_party/manus_glove/README.md`）：SDK 用法、udev
  规则、标定目录约定。
- **MANUS Core / ManusSDK 3.1.1**：`third_party/MANUS_Core_3.1.1_SDK/`，
  标定操作以 Manus Core 官方文档为准。
- 本项目：`manus_hand_pose_recorder`（采集实现，ergonomics 40 通道布局）。
