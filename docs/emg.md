# EMG 臂环配置（WAVELETECH / weili_emg）

> 左右臂各一条 WAVELETECH 8 通道 EMG 臂环（WLEMGUV2A），经 USB-UART
> 串口芯片（CP210x / CH343）接入采集机，`weili_emg` recorder 逐帧采集。
> 两路槽位 `emg_left` / `emg_right` 在 `configs/recorders.yaml` 里按
> **COM 口（串口设备名）** 绑定。

臂环是同款设备、USB 特征完全一样，自动探测分不出左右；接反了左右手肌电就
互换了。本文说明如何**逐一插入设备**确认每台臂环的 COM 口、与 yaml 对齐，
最后用 `scripts/check_emg.py` 做左右手对应检查。

---

## 1 开始前的准备

- 两条臂环已充电开机，USB 线接采集机。
- Windows：装好 CP210x（Silicon Labs）或 CH343 驱动，设备管理器出现
  COM 口；Linux：无需驱动，出现 `/dev/ttyACM*` 或 `/dev/ttyUSB*`，
  当前用户需在 `dialout` 组。
- `pyserial`（采集机随 requirements 已装），列出串口用：

  ```bash
  python -m serial.tools.list_ports        # 基础列表
  python -m serial.tools.list_ports -v     # 含 hwid/序列号,信息更全
  ```

## 2 COM 口机制

`recorders.yaml` 里 `port` 留空 = **自动探测**：recorder 按 USB 特征
（`VID:PID=10C4` CP210x、`VID:PID=1A86:55D3` CH343 等，见
`weili_emg_recorder._auto_detect_port`）拿**第一台**匹配的设备。

单台臂环这样够用；但本项目左右臂各一条，两台特征一样，自动探测**分不出
左右**——两个槽位会抓到同一台。所以双臂环工位必须把两个 `port` 都固定：

| 平台 | 写法 | 稳定性 |
|---|---|---|
| Windows | `"COM31"` | 跟着设备+物理口走，口位不动基本不变 |
| Linux | `"/dev/ttyACM0"` | 按枚举顺序编号，重插可能互换；推荐 `/dev/serial/by-id/...`（含 USB 序列号） |

与相机 idx 一样：**口位固定下来不再动**，动了就重新指认。

## 3 逐一插入指认（确认左右）

### 指认脚本（推荐）

```bash
python scripts/setup/map_emg.py           # 交互指认:插一台,认一台
python scripts/setup/map_emg.py --list    # 只列当前串口,标记哪些带 EMG 特征
```

流程：开始前**拔掉所有臂环**按回车扫基线 → 脚本逐轮提示"插入臂环，绑到哪只
手（L/R）" → 边插边轮询，**新出现的 COM 口当场报告**（带 CP210x/CH343 特征
的会标注；连续两次扫描都在才算稳定，防枚举抖动；期间拔掉的口也会提示）→
左右认完自动汇总，打印 `recorders.yaml` 建议填法、与当前 yaml 的比对结果
（✓ 一致 / ✗ 不一致），最后提示用 `check_emg.py` 验证。等待插入时 Ctrl+C
随时退出，已认到的结果先打印。

### 手动指认（脚本不可用时）

1. **两条臂环全部拔掉**，跑一遍：

   ```bash
   python -m serial.tools.list_ports
   ```

   把当前列表记下来（基线）。

2. **只插入左手臂环**，再跑一遍：列表里**新出现的**那个 COM 口就是左臂环
   （如 `COM31`）。
3. 拔掉左臂环，**只插入右手臂环**，再跑一遍：新出现的就是右臂环
   （如 `COM30`）。
4. （加分项）若 `-v` 输出里两台臂环的 USB 序列号不同，可记下序列号，
   之后重插也能对上；Linux 由此可用 `/dev/serial/by-id/...` 固定路径。
5. 全部插回，确认两台同时在列。

> 也可以反过来利用此流程排查：某台臂环插上后列表没变化 → 线材/驱动问题，
> 先解决再继续。

## 4 与 recorders.yaml 对齐

把指认结果填进两个槽位（摘自模板）：

```yaml
emg_left:
  enabled: true
  kind: weili_emg
  name: "WLEMGUV2A"        # 设备显示名,写入 meta
  port: "COM31"            # 第 3 节指认的左臂环口;Linux 如 "/dev/ttyACM0"
  baud: 921600
  hz: 1000

emg_right:
  enabled: true
  kind: weili_emg
  name: "WLEMGUV2A"
  port: "COM30"            # 右臂环口
  baud: 921600
  hz: 1000
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | — | `weili_emg` |
| `port` | `""` | 空 = 自动探测（单台可用；双台必须固定，见第 2 节） |
| `baud` | `921600` | 波特率，与设备一致，一般不动 |
| `hz` | `1000` | poll 上限 |
| `link_timeout` | `0.1` | 出过字节后静默判死阈值（臂环断电/USB 松动检测） |

## 5 验证（check_emg.py）

```bash
python scripts/check_emg.py --list              # 只看左右槽位配置,不开窗
python scripts/check_emg.py                     # 开窗做左右手对应检查
```

窗口检查流程（不接也没关系，脚本按 yaml 打开左右两路）：

1. 按 **L** → 反复晃动**左手**数秒：上面板（绿，EMG 8 通道 + IMU 波形）
   应出现明显起伏，统计结论「对应 ✓」。
2. 按 **R** → 晃动**右手**：下面板（橙）起伏。
3. 结论为「疑似接反 ✗」→ 交换 yaml 里 `emg_left` / `emg_right` 的
   `port`，重跑复查。
4. 按 **Y** 确认退出（退出码 0）；**Q** 放弃（退出码 1）。

面板读数里的「延迟」应 <100 ms（>300 ms 变红 = 串口缓冲积压，读取跟不上，
不是设备慢）。`check_emg.py` 只驱动设备的 open/poll、不落盘，可放心反复跑。

## 6 常见问题

| 现象 | 原因与处理 |
|---|---|
| 打开失败：PermissionError / 拒绝访问 | 串口被占用（串口监视器、另一个脚本）或 Linux 无权限(`sudo usermod -aG dialout $USER` 后重登) |
| 两路都打开了但其实是同一台 | 两个槽位 `port` 留空或配了同一个口:按第 3 节逐一插入重新指认 |
| check_emg 结论「疑似接反」 | 交换 yaml 两个槽位的 `port` 后复查;指认阶段也可能标反了 |
| 昨天还能用,今天 COM 号变了 | 动了 USB 口位:重新指认并固定口位;Linux 可改用 `/dev/serial/by-id/...` |
| 延迟红字 / 链路判死 | 串口缓冲积压或臂环断电:换 USB 口/短线,确认臂环电量 |

## 7 参考资料

- **pyserial list_ports**：<https://pyserial.readthedocs.io/>（串口枚举工具）。
- 本项目：`scripts/check_emg.py`（左右手对应检查）、
  `weili_emg_recorder`（采集实现与自动探测的 VID:PID 表）、
  README「脚本一览」`preflight.py`（采集前逐设备预检）。
