# HAL 头环接收（2026-10-19）

兼容 `wired_tcp-windows-20261019.zip`：TCP framing 与标定握手保持 v1。
根据 `stream_start.topics` 自动选择旧 ROS JPEG 或 HAL Annex-B H.264。
HAL 模式只接收 `/imu_data_raw`，不会伪造滤波 IMU。

H.264 使用 PyAV 为每路保留独立解码状态，从连接首个 IDR 开始解码；
预热期间也解码，但不写入正式视频。每个排队包携带接收时的开录状态，
避免异步预热帧在 commit 后混入视频。正式输出继续使用现有 HEVC MP4
写盘器（GPU 优先），以及逐帧时间戳 NPZ；不假定 1920×1088，尺寸取自解码帧。
CPU 解码、RGB 转换和 HEVC 编码的四路持续吞吐需要实机验证。

首包要求 IDR + SPS/PPS，且每包必须恰好解码出一个无 B 帧的图像。
序号缺口、缺少采集时间戳、解码/写盘队列满会停止会话并记录错误。
HAL 停止录制时先排空队列，再关闭视频容器，避免丢弃尾帧。

每次会话仍必须完成三份标定文件的大小、SHA-256、保存和 ACK 校验。
标定存于 `<session>/ego_headband/calibrations/`，NPZ 保留文件名和哈希。
音频的 PCM/WAV 与时间戳保存方式不变。`stream_source` 和 `stream_topics_json`
记录在 NPZ，便于区分 HAL 与 ROS 数据源。

离线测试（不访问设备）：

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.ego_headband.test_hal_h264 tests.ego_headband.test_calibration_transfer -v
```

依赖：`av`（当前 stay 的 collect 环境已安装）。
