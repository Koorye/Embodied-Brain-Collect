# VIVE Ultimate Tracker 位置跟踪配置（SteamVR Null Driver 无头显）

> **适用场景**：不连接实体 VR 头显，使用 Null Driver 启动 SteamVR。本文补全原教程的
> Windows 和 Linux 路径说明，并提供 Windows 下 DexCap 与 VIVE Ultimate Tracker 的验证流程。
> **Linux 路径说明不代表整套 VIVE 与 DexCap 软件流程已在 Linux 上验证。**

SteamVR 默认要求实体头显在线。把 SteamVR 的 Null Driver（虚拟头显驱动）启用、
同时允许追踪器驱动运行，即可只用 tracker 采集位姿。完成配置后，还需要验证追踪器的
实际位姿数据，才能确认采集链路可用（见[第 6 节](#6-验证能否读取位姿)）。

---

## 目录

- [1 开始前的准备](#1-开始前的准备)
- [2 原理：四个配置项](#2-原理四个配置项)
- [3 一键配置](#3-一键配置)
- [4 在 Windows 中启动 SteamVR 并连接追踪器](#4-在-windows-中启动-steamvr-并连接追踪器)
- [5 SteamVR 角色设置](#5-steamvr-角色设置)
- [6 验证能否读取位姿](#6-验证能否读取位姿)
- [7 常见问题](#7-常见问题)
- [8 参考资料](#8-参考资料)

---

## 1 开始前的准备

- 已安装 Steam 和 SteamVR，并**至少启动过一次 SteamVR**（保证默认配置文件已生成）。
- 若继续执行本文的 Windows 追踪器验证流程：已安装对应的 VIVE Hub／VIVE Streaming Hub
  软件和 SteamVR 插件，并准备好 VIVE Wireless Dongle 和追踪器。
- **完全退出 SteamVR 后再配置**（系统托盘里的 SteamVR 图标也要退出），
  避免运行中的程序覆盖修改。

默认 SteamVR 目录：

| 系统 | 默认路径 |
|---|---|
| Windows | `C:\Program Files (x86)\Steam\steamapps\common\SteamVR` |
| Linux | `~/.local/share/Steam/steamapps/common/SteamVR`（`~/.steam/steam` 通常是其符号链接） |
| macOS | `~/Library/Application Support/Steam/steamapps/common/SteamVR` |

如果 Steam 安装在其他盘，在 Steam 中右键 SteamVR → 管理 → 浏览本地文件。

---

## 2 原理：四个配置项

| 配置项 | 设置值 | 作用 |
|---|---|---|
| `steamvr.requireHmd` | `false` | 不要求连接实体头显 |
| `steamvr.forcedDriver` | `"null"` | 使用 Null Driver |
| `steamvr.activateMultipleDrivers` | `true` | 允许 Null Driver 与追踪器等其他驱动同时运行 |
| `driver_null.enable` | `true` | 启用 Null Driver |

**注意**：`requireHmd`、`forcedDriver` 和 `activateMultipleDrivers` 放在 `"steamvr"`
对象下；Null Driver 的 `enable` 放在 `"driver_null"` 对象下。四项分属**两个不同的
配置文件**（SteamVR 目录下 `resources/settings/` 与 `drivers/null/resources/settings/`
各一个 `default.vrsettings`），不能把四项全部放在 `"steamvr"` 下。

---

## 3 一键配置

本项目脚本 `scripts/setup/configure_steamvr_null.py` 封装了上述四个配置项的全部修改：
输入 Steam 根目录即可完成两个文件的修改，自动备份、幂等可重跑、写后复检：

```bash
# Windows（Git Bash / PowerShell 均可，路径含空格要加引号）
python scripts/setup/configure_steamvr_null.py "C:\Program Files (x86)\Steam"

# Linux
python scripts/setup/configure_steamvr_null.py ~/.local/share/Steam

# 不给路径:自动探测常见安装位置(含 libraryfolders.vdf 里的其他 Steam 库)
python scripts/setup/configure_steamvr_null.py

# 常用选项
python scripts/setup/configure_steamvr_null.py <Steam目录> --dry-run      # 只预览改动,不写盘
python scripts/setup/configure_steamvr_null.py <Steam目录> --user-config  # 连用户配置一起改
python scripts/setup/configure_steamvr_null.py <Steam目录> --restore      # 从 .bak 恢复原文件
```

脚本行为：

- 写入前在同目录留 `default.vrsettings.bak`；已存在则不覆盖，始终保留最早的原始备份。
- 检测到 SteamVR 正在运行（`vrserver`/`vrmonitor`）会拒绝写入，`--force` 可跳过（不推荐）。
- 会检查用户配置 `<Steam目录>/config/steamvr.vrsettings`：其同名设置**优先于**默认配置，
  有冲突时提示；`--user-config` 把同样改动合并进去。
- 幂等：已是目标值则不动文件。**SteamVR 更新可能覆盖默认配置导致配置失效，
  重跑一次本脚本即可恢复。**

---

## 4 在 Windows 中启动 SteamVR 并连接追踪器

1. 运行一键配置（或确认配置已生效），重新启动 SteamVR。
2. 打开 VIVE Hub／VIVE Streaming Hub，连接无线接收器，开启并配对追踪器。
3. 按 VIVE 软件提示完成当前环境的建图与地图同步。
4. 确认所有追踪器显示 Ready，并保持 VIVE 软件和 SteamVR 运行。

此采集流程不需要实体头显或控制器。如果 VIVE 软件弹出头显配置引导，可关闭该引导，
继续追踪器配置。

> 原文中的 VIVE 私测入口与界面说明来自 2024 年版本；新版软件界面可能不同，
> 不应把旧版私测码视为当前必需步骤。

---

## 5 SteamVR 角色设置

为胸部、左手、右手安装位置的三个追踪器分别设置角色：
**Settings → Controllers → Manage Trackers**

| 安装位置 | SteamVR 角色 |
|---|---|
| 胸部 | `chest` |
| 左手（戴在左腕） | `left wrist` |
| 右手（戴在右腕） | `right wrist` |

> 原版 DexCap 教程使用的角色名是 `left elbow` / `right elbow`；本项目追踪器实际
> 佩戴在左/右手腕，角色应设为 **left wrist / right wrist**，与
> `configs/recorders.yaml` 里 `position.role_serial_map` 的 `left_wrist` /
> `right_wrist` 对应，**不要照抄 elbow**。设置后重启 SteamVR。

---

## 6 验证能否读取位姿

### 前置：在 configs/recorders.yaml 绑定序列号

`check_vive.py` 按 `position.role_serial_map` 给每台 tracker 显示角色标签，
**必须先把每台的 OpenVR 序列号配置进该映射**，才能按角色核对；没配置的设备只会
显示 `unbound:<序列号>`，验证无从判对错：

```yaml
position:
  enabled: true
  kind: openvr_position
  role_serial_map: {"left_wrist": "61-BH3700181", "right_wrist": "61-BH3702186", "chest": "61-BH3702202"}
```

序列号（ID）以 `check_vive.py --list` 的输出为准——VIVE Hub 里显示的设备 ID
（FA61…）不是 OpenVR 序列号，匹配不到。

### 验证步骤

```bash
python scripts/check_vive.py --list          # 只列出当前连接的 tracker 与 yaml 绑定
python scripts/check_vive.py                 # 开窗实时显示位置与轨迹,核对左右手对齐
```

依次拿起每台 tracker 走一小圈，窗口里哪个角色标签的轨迹跟着动，绑定就是对的；
标签不动或动的是 `unbound` 那台，说明 `role_serial_map` 绑错或没绑，当场暴露。
yaml 里绑了但没连上的设备会红字列在窗口顶部。

**DexCap 原版验证**：DexCap Hardware Tutorial 提供 `vive_test.py` 验证流程，
读到的位姿应随 tracker 移动连续变化。

> Linux 提示：Linux 路径说明不代表整套 VIVE 与 DexCap 软件流程已在 Linux 上验证；
> 生产采集流程仍以 Windows 为准。

---

## 7 常见问题

| 现象 | 原因与处理 |
|---|---|
| SteamVR 启动后仍提示找不到头显 | 默认配置未生效：重跑一键脚本；检查用户配置 `config/steamvr.vrsettings` 是否有相反的同名值（优先级更高，脚本会提示，`--user-config` 一并修改） |
| 之前配置过，更新 SteamVR 后失效 | SteamVR 更新可能覆盖默认配置文件，重跑一键脚本即可（幂等） |
| 追踪器不出现 / 不 Ready | 确认 VIVE Hub 已连接无线接收器并配对、SteamVR 插件已启用；按提示完成建图与地图同步 |
| `check_vive.py` 全是 unbound 标签 | `configs/recorders.yaml` 的 `role_serial_map` 没配或绑错;序列号以 `--list` 输出为准，VIVE Hub 里的设备 ID（FA61…）匹配不到 |
| `check_vive.py` 列不出某台 tracker | 确认已开机、在 VIVE Hub/SteamVR 里在线；见 README「配置文件」的 `role_serial_map` |

---

## 8 参考资料

- **DexCap Hardware Tutorial**：VIVE 追踪器配置、角色分配与 `vive_test.py` 验证流程。
- **SteamVRNoHeadset**：DexCap 引用的 Null Driver 无头显运行教程。
- **Valve OpenVR 驱动文档**：用户配置路径、配置优先级与驱动启用设置。
