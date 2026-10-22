# HAL 头环接收（2026-10-21）

TCP framing、标定握手、音频/IMU 与输出文件名保持不变。
根据 stream_start.topics 区分旧 ROS JPEG 与 HAL Annex-B H.264。

## H.264 录制

四路 H.264 原始包通过 PyAV add_mux_stream 直接封装 MP4，不创建编码器或解码器。
压缩图像字节不变；输出编码为 H.264（原来为 HEVC），每包对应一帧。
保存原始采集时间戳到 NPZ；MP4 写**名义 CFR 时间戳**（`pts = i × 90000/cam_fps`，
时基 1/90000），容器时长 = 帧数/cam_fps。每帧的真实采集时刻只在 NPZ 的
`{name}_timestamps` 里——丢帧在 MP4 中表现为时长变短，不再是时间轴上的空洞。
录制端仍校验采集时间戳严格递增（乱序即数据损坏）。要求 `cam_fps` 为整数且能整除
90000（默认 30）。

CFR 容器的目的：让打包端 `pack_daily._smart_cut` 对头环也走"头段边角重编码 +
整 GOP 逐比特流拷贝"（实测 1243 帧截段 1.3s，全量重编码 5.5s），输出无二次编码
损失，mp4 tag 为 `avc3`（允许带内参数集）。历史 VFR 直存文件不满足该契约，
自动回退到按帧序号全量重编码，结果仍然正确。

正式录制从 commit 后每路首个 IDR + SPS/PPS 开始，丢弃之前依赖预热 GOP 的 P 帧；
当前 GOP 约 1 秒，因此各路开头最多可能等待约 1 秒，必须按 NPZ 时间戳对齐。
等待超过 5 秒仍无 IDR 或停止时没有可独立解码的视频会报错。
NPZ 增加 video_storage=h264_passthrough_cfr、各路 video_start_ns、
h264_start_skipped_frames 和 h264_start_wait_ns。
序号缺口、时间戳倒退、SPS 改变、写盘失败和队列溢出均停止并报告；停止时排空包队列。

## 预览

操作说明中的 check_cameras 和采集员控制台的 HAL 预览共享 H264Decoder，
使用 NVIDIA CUDA/NVDEC 硬件解码，allow_software_fallback=False。
未配置预览端口的录制不解码；控制台预览启用时才创建硬件解码器。
GPU 预览失败会明确记录并停用该路预览，不回退 CPU，不影响已经独立封装的视频。
GPU 解码后的 RGB 转换、缩放和控制台 UDP JPEG 输出仍使用 CPU；这不是 H.264 软件解码。

## 依赖和验收

stay collect 环境使用 PyAV 18.1.0、RTX 3050。直接封装需要 add_mux_stream 接口；
预览另需支持 CUDA 的 PyAV/FFmpeg 和 NVIDIA 驱动。运行中的旧采集不会自动切换，
结束当前会话后重新启动采集程序和预览程序，使新代码生效。

每次会话仍需三个标定文件的大小、SHA-256、落盘和 ACK 校验。
标定位于 <session>/ego_headband/calibrations/；源头环标定不变。
录制改为名义 CFR 的 H.264 后，`pack_daily._smart_cut` 同时支持 h264（头段用
libx264 并按源码流 profile 重编码，mp4 tag `avc3`），头环打包不再整段重编码；
不满足契约的历史文件仍走按帧序号的全量重编码回退。

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.ego_headband.test_hal_h264 tests.ego_headband.test_calibration_transfer tests.ego_headband.test_microphone_commit tests.ego_headband.test_preview_stream -v
```

测试使用合成数据，不连接头环；CUDA 预览测试需要 NVIDIA GPU。
