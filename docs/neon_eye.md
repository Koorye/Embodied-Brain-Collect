# Pupil Neon 眼动仪配置（neon_eye_async）

> 眼动仪是 **Pupil Labs Neon**：眼镜形态的传感器 + 一台 Android 手机。
> 传感器直连手机，手机上运行 **Neon Companion app**；采集电脑经局域网
> 连接**手机**（不是传感器）取数。`neon_eye_async` recorder 通过
> `pupil-labs-realtime-api` 在网络上**自动发现**手机，采三路数据：
> gaze（注视点）、IMU、scene 场景视频（eye.mp4）。

因为自动发现走 mDNS 组播，**手机必须与采集电脑在同一网段** —— 这是本硬件
最常见的故障点。本文说明手机侧准备、同网段要求、每次采集的启动顺序，
以及故障排查。

---

## 1 开始前的准备（一次性）

1. **手机安装 app**：在 Android 手机上安装 Neon Companion app
   （Pupil Labs 应用商店/官网渠道）。
2. **配对**：传感器开机，打开 app，按 app 内引导把 Neon 传感器与手机配对
   （传感器画面在手机上正常显示即成功）。
3. **依赖**：采集机 `pip install pupil-labs-realtime-api`（requirements
   已含，`pupil-labs-realtime-api==1.8.0`）。

> 手机建议专用：开发者模式关闭电池优化（防止录制中被系统杀后台）、
> 录制时插电、关闭无关应用。

## 2 网络要求：手机与电脑在同一网段

recorder 不配 IP/端口 —— 设备靠 **mDNS 自动发现**（`Network().
wait_for_new_device`，超时 10 s）。组播发现要求手机和电脑彼此可达：

| 方式 | 说明 |
|---|---|
| 同一 Wi-Fi 路由器（推荐） | 手机与电脑连**同一个**路由器、同一网段;注意访客网络/AP 隔离会挡组播,发现不到 |
| 电脑接入手机热点 | 天然同网段;注意电脑端其他依赖外网的组件此时走手机流量 |

**同网段自查**：Windows `ipconfig`，手机 Wi-Fi 详情里看 IP —— 两者前三段
一致（如 `192.168.1.x`）即同网段。防火墙拦 mDNS（UDP 5353）也会导致发现
失败，排查时注意。

## 3 每次采集的启动顺序

1. 传感器开机、佩戴好。
2. **启动手机 app**（Neon Companion），确认传感器画面正常 —— **app 必须
   保持打开**，它是数据链路的一环，不是配对完就能关。
3. 确认手机与电脑在同一网段（第 2 节）。
4. 正常启动采集（`scripts/run_session_env.py` 或预检）。电脑端**自动发现**
   设备，无需任何 IP 配置。
5. 看启动日志，正常顺序是：

```
[eye:neon] <手机名> battery=xx%  pc_to_phone_offset=x.xx ms   ← 已发现手机并对时
[eye:neon] waiting for sensor(s): ...                        ← 等三路传感器就绪
[eye:neon] standby warm — streams running, waiting for go    ← 预热完成,可开录
```

预热时间线（写在 `open_timeout: 120` 里的原因）：发现设备 10 s + 传感器
就绪轮询 10 s + 三路首帧各 20 s，scene 摄像头冷启动还要 ~5 s。

## 4 yaml 配置

`configs/recorders.yaml` 槽位（摘自模板）：

```yaml
eye:
  enabled: true
  kind: neon_eye_async
  name: "Pupil Labs Neon Wearable Eye Tracker"  # 设备显示名,写入 meta
  hz: 1000
  link_timeout: 0.1             # 某路流这么久没新样本 → 判死(断电/断网)
  open_timeout: 120             # 预热最坏 ~90s,放宽到 120 防误判
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | — | `neon_eye_async`（异步全速率采集,防 simple-API 丢帧） |
| `hz` | `1000` | poll 上限 |
| `link_timeout` | `0.1` | 流静默判死阈值;gaze/imu 持续 100Hz+,静默即断;`0` = 关闭 |
| `open_timeout` | `120` | launcher 开录门禁的超时,冷启动时间线见第 3 节 |
| `no_scene_video` | `false` | `true` = 只记 scene 时间戳不写 mp4 |
| `crf` / `preset` | `23` / `medium` | eye.mp4（libx265）编码质量/速度 |

设备时间戳在手机时钟域，recorder 用 Core 的 time-echo 做对时
（`pc_to_phone_offset_ms` 落盘在 npz），眼动与其他模态同处电脑时钟轴。

## 5 验证

```bash
python scripts/preflight.py     # 采集前预检:打开设备 + 数据流探测
```

预检对眼动的判据是 **standby 队列持续进样**（gaze/imu 队列在增长）——
只要队列在动，链路就是活的。正式采集进度行会实时显示
`gaze=/ imu=/ scene=` 三个计数在涨，收工后 session 目录下应有
`eye/*.npz` + `eye.mp4`（时长与 scene_timestamps 1:1）。

## 6 常见问题

| 现象 | 原因与处理 |
|---|---|
| 10 s 内发现不到设备（open 超时第一步就卡住） | 手机与电脑不同网段 / 访客网络开了 AP 隔离 / 电脑防火墙拦 mDNS(UDP 5353) / 热点没开 —— 按第 2 节自查 |
| 日志停在 `waiting for sensor(s): gaze/imu/world` | **手机 app 没开或被切到后台**:打开 Neon Companion,传感器画面出来即恢复(recorder 会打印 open the Neon Camera app 提示) |
| open 超时(90 s) | 冷启动本来就慢(scene 摄像头 ~5 s),`open_timeout` 已放宽到 120;仍超时按上面两条排查 |
| 录制中流静默判死退出 | 手机被杀后台/锁屏断流/断网/断电:关电池优化、插电、保持 app 前台;传感器与手机距离别太远 |
| `pc_to_phone_offset` 异常大或波动 | Wi-Fi 拥塞:换 5 GHz、离路由器近点,或改用手机热点 |
| eye.mp4 缺帧(日志 scene writer behind) | 电脑解码跟不上:确认 ffmpeg 正常、降低 `crf` 负担;丢帧有时间戳记账,数据仍 1:1 |

## 7 参考资料

- **Pupil Labs Neon 文档**：<https://docs.pupil-labs.com/neon/>
  （Companion app 安装、配对与官方网络要求）。
- **pupil-labs-realtime-api**：<https://github.com/pupil-labs/pupil-labs-realtime-api>
  （实时 API,自动发现与三路流）。
- 本项目：`neon_eye_async_recorder`（采集实现:自动发现、对时、warm-standby、
  流监督）、`scripts/preflight.py`（预检）。
