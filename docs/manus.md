# MANUS Metaglove 配置与标定（manus_hand_pose）

> 左右双手套为 MANUS Quantum Metaglove Pro Haptics，`manus_hand_pose`
> recorder 经 `manus_glove` SDK（见 `third_party/manus_glove`，封装
> ManusSDK 3.1.1）连接 **Manus Core** 采集：每手 20 通道手指关节角
> （ergonomics，共 40 通道）+ 骨架节点位置/四元数。

手套本身不直接出关节角 —— SDK 开录前要给每只手套下发 `.mcal` **标定文件**
（`LoadCalibrationFiles`），它决定传感器读数到手指关节角的换算。**不装、装错
左右或标定过期，手指数据都不可信。** 本文说明：安装 Manus Core → 按流程标定
→ 一键把标定文件装进 `~/.cache/manus_glove/`（SDK 读取目录）→ 验证。

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
3. 标定完成后**导出标定文件**（`.mcal`），记住导出位置（如 `Downloads`）。

> 换人佩戴、手套松紧变化、长时间漂移后都应**重新标定**，标定文件跟着人走。

## 3 一键安装标定文件（install_manus_calib.py）

SDK 默认在 `~/.cache/manus_glove/` 找 `LeftMetaglovePro.mcal` /
`RightMetaglovePro.mcal`；若 `recorders.yaml` 的 `hand_pose` 槽位改过
`calibration_dir` / `left_calibration` / `right_calibration`，脚本以 yaml
为准（装错目录 = 白装）。

```bash
# 导出的文件名里含 left/right:直接给目录或文件,按文件名识别左右
python scripts/setup/install_manus_calib.py ~/Downloads/LeftMetaglovePro.mcal \
                                      ~/Downloads/RightMetaglovePro.mcal
python scripts/setup/install_manus_calib.py ~/Downloads          # 扫目录

# 文件名不含左右:显式指定
python scripts/setup/install_manus_calib.py --left 新标定L.mcal --right 新标定R.mcal

# 只重标了一只手:另一侧已装过会自动跳过
python scripts/setup/install_manus_calib.py --right 新标定R.mcal

# 只看当前安装状态 / 只预览
python scripts/setup/install_manus_calib.py --status
python scripts/setup/install_manus_calib.py ~/Downloads --dry-run
```

脚本行为：

- 同侧多个候选取**最新**的（如 `right_0930.mcal` 优于 `right_0929.mcal`），
  其余列出忽略；文件名同时含 left/right 的不自动识别，要求 `--left/--right`。
- 目标已存在时先备份为 `.bak`（已有 `.bak` 不覆盖，保留最早的原始标定）。
- 空文件（0 字节，多半导出失败）拒绝安装。
- 装完打印两侧状态，`--status` 随时可查。

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
| 日志 `Calibration file not found` / `No left calibration path configured` | 标定文件没装进 SDK 读取目录:跑第 3 节一键脚本,`--status` 核对目录与文件名 |
| `only 1/2 gloves found` 开录失败 | 一只手套没配对/没电:在 Manus Core 里配对齐两只再试 |
| `udev rules not found`(Linux) | 没装 udev 规则:按第 1 节安装 |
| 数据随动但左右手互反 | 左右标定装反:重新运行一键脚本,用 `--left/--right` 显式纠正(.bak 会保住原文件) |
| 手指角度漂移/明显失真 | 标定过期:重新标定并重装;确认佩戴松紧与标定时一致 |
| 手套中途停止出数据 | recorder 原地抛错终止(设计行为):检查电量/USB 接收器,重连后重录 |

## 6 参考资料

- **manus_glove**（`third_party/manus_glove/README.md`）：SDK 用法、udev
  规则、标定目录约定。
- **MANUS Core / ManusSDK 3.1.1**：`third_party/MANUS_Core_3.1.1_SDK/`，
  标定操作以 Manus Core 官方文档为准。
- 本项目：`manus_hand_pose_recorder`（采集实现，ergonomics 40 通道布局）、
  `scripts/setup/install_manus_calib.py`（本文第 3 节）。
