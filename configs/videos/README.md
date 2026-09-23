# video_rate(RLHF 视频打分)素材目录 —— 按子集(任务)组织

一个**子集 = 一个任务**,与图纸模式的场景目录同构。所有遥操视频放在
子集目录里,由 `scripts/run_session_video.py` 运行时抽取组合,**不需要
预生成场次**,也**不需要离线提取第一帧**(stim 播放时直接从视频读)。

## 目录结构

    configs/videos/
      <子集>/                     # 一个任务,如 盖笔盖/
        config.yaml               # 必需:任务指令(task 字段,想象内容)
        success_<说明>.mp4        # 做对的示范视频 → success 池
        fail_<说明>.mp4           # 做错的示范视频 → fail 池
        used.yaml                 # 台账(自动维护,勿手改)

## config.yaml(每个子集必需)

```yaml
task: 双手分别拿起笔和笔盖，盖上笔盖后把笔放入笔筒   # 必填:任务指令
scene: 书桌            # 可选:场景名,随 session meta 固化
name: 盖笔盖           # 可选:显示名(meta task_name);缺省 = 目录名
# instr1_text: "…"     # 可选:完整覆盖默认指令(默认是 stim.yaml 的
# instr2_text: "…"     #   instr1_text 模板,{task} 占位符填上面的 task)
```

`task` 会填进 `configs/stim.yaml` 的 `video_rate.instr1_text` 模板的
`{task}` 占位符,成为被试看第一帧时的想象内容。

## 视频命名

文件名前缀决定池子(`success_` / `fail_`,大小写不敏感;历史素材的
`sucess_` 拼写也按 success 识别)。前缀之外的说明随意,如:

    fail_episode_000019.mp4
    success_episode_000074.mp4

## 台账 used.yaml(每子集一个)

`run_session_video.py` 自动维护:每次采集**成功**结束后,本次用到的视频
`uses` 计数 +1 并记录 `last_session`。抽取时优先给使用次数最少的视频,
保证覆盖均衡。删除本文件即清零该子集台账。

## 编码要求

解码走 cv2 优先、ffmpeg 软解回退:AV1 等 cv2 不带的编码能用系统 ffmpeg
播放,但**软件解码 1080p 可能跟不上 30fps**,播放会掉帧。
`run_session_video` 每次开录前会试解第一帧,解不出的直接拦截并提示。
素材量大的话建议统一重编码(仓库数据面同样是 h265):

    ffmpeg -i in.mp4 -c:v libx265 -crf 20 out.mp4
