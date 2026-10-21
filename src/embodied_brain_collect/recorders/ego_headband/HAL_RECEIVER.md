# HAL 头环接收（2026-10-21）

TCP framing、标定握手、音频/IMU 与输出文件名保持不变。
根据 stream_start.topics 区分旧 ROS JPEG 与 HAL Annex-B H.264。

## H.264 录制

四路 H.264 原始包通过 PyAV add_mux_stream 直接封装 MP4，不创建编码器或解码器。
压缩图像字节不变；输出编码为 H.264（原来为 HEVC），每包对应一帧。
保存原始采集时间戳到 NPZ；MP4 的 PTS/DTS 为相对首个保存帧的采集时间，精度为微秒。
保留真实时间间隔，不补帧、不生成 CFR 时间戳。仅支持 camd baseline、无 B 帧。

正式录制从 commit 后每路首个 IDR + SPS/PPS 开始，丢弃之前依赖预热 GOP 的 P 帧；
当前 GOP 约 1 秒，因此各路开头最多可能等待约 1 秒，必须按 NPZ 时间戳对齐。
等待超过 5 秒仍无 IDR 或停止时没有可独立解码的视频会报错。
NPZ 增加 video_storage=h264_passthrough、各路 video_start_ns、
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
录制改为 H.264 后，pack_daily 的 HEVC 专用快速截段不适用，会走其既有重新编码回退路径。
本次未修改打包策略或其他设备。

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.ego_headband.test_hal_h264 tests.ego_headband.test_calibration_transfer tests.ego_headband.test_microphone_commit tests.ego_headband.test_preview_stream -v
```

测试使用合成数据，不连接头环；CUDA 预览测试需要 NVIDIA GPU。
