# 头环麦克风

默认在 TCP hello 中请求 `pcm_audio_v1`，接收已有四相机、两 IMU 以及 `/microphone/audio`。该音频通道由头环 `wired_tcp` 服务直接采集，不是 ROS topic。无需打开 Windows 本地麦克风。

配置新增：

- `audio_enabled: true`：请求并保存头环麦克风；设 false 保持原六路行为。
- `require_audio: true`：请求音频但服务端未提供时，打开失败。允许缺音频时可显式设 false。
- `audio_topic: /microphone/audio`。

输出仍在 `<session>/<output_dir>/`，默认 output_dir 为 `ego_headband`：

- 原有 `cam0.mp4`～`cam3.mp4` 与 IMU NPZ 字段保持原样。
- 新增 `microphone.wav`：16 kHz、单声道、16 位有符号小端 PCM，流式写入，关闭时补全 WAV 头；不覆盖已有同名文件。
- 同一个 NPZ 新增逐包 `microphone_read_complete_ns`（int64）、`microphone_sample_index`（设备本次连接的样本位置）、`microphone_wav_sample_offset`（此 WAV 中的位置）、`microphone_samples`、`microphone_stream_seq`。
- NPZ 另有 `microphone_sample_rate`、`microphone_channels`、`microphone_sample_width`、`microphone_timestamp_basis`、`microphone_error`。

每包 320 点/20 ms/640 字节。握手后等待录制期间的音频与图像一样被丢弃；因此第一包设备样本位置可能不为零，而 WAV 偏移从零开始。不要把两者混用。

音频时间保存服务端 `read_complete_ns`，不使用 `header.stamp_ns=0` 伪造采集时间，也不退化为 Windows 接收时间。它是服务端读完一包的时间，不是硬件采样时间；不能据此声称相机和音频硬件级同步。

音频格式、长度、时间来源或连续序号错误会停止接收，保存此前成功写入的 WAV/NPZ，并在日志和 `microphone_error` 留下原因；不补静音、不隐瞒间隙。注意框架原有 `_end_stream`/`run()` 不以退出码表示所有中途采集错误，验收时应读取日志和 NPZ 的错误字段。

Dummy 也通过同一路径生成 440 Hz 合成音。离线测试不需要 pytest：

```
python -m unittest discover -s tests/ego_headband -p test_microphone_recording.py -v
```

从项目根目录运行，确保 `src` 在 PYTHONPATH 中。


图像解码在每路独立线程中进行，避免同步 JPEG 解码阻塞音频和 IMU 接收。每路最多缓存 8 个压缩图像，过载时按现有视频实时性策略丢最旧图像并记录 `decode_queue_dropped`；原有视频编码写入队列仍可能报告丢帧，接收完整不等于视频文件无丢帧。

`tests/ego_headband/test_net_ego_headband.py` 的实时预览增加麦克风波形、包计数和峰值，不写 WAV；关闭预览会关闭头环连接并释放远端麦克风。
