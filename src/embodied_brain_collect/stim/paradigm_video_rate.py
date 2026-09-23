"""范式 video_rate —— 多 trial:注视 / 双段指令 / 看图 / 播视频 / 打分。

单 trial 流程::

    1. 红点注视              固定 ``--fix-pre-s``
    2. 文字指令 1            固定 ``--instr1-s``
    3. 呈现静止图像          被试控(回车继续)
    4. 文字指令 2            固定 ``--instr2-s``
    5. 播放视频              按视频真实时长(不受 --fast 压缩)
    6. 动作质量打分          被试控(数字键选分,回车确认)
    7. 重播准备              再看第一帧+提示,回车开始
    8. 重播视频              Shift=做对,回车=做错(写入 marker,按键分开记录)
    8b. 是否重做打标          是则再播一遍并重新打标;否则进入质检
    9. 播放质检              被试控(无问题 / 丢帧或遮挡 / 其他)

一整场 session 循环 ``n_trials`` 次(可在 stim.yaml / ``--n-trials`` 改;
上限由 markers.yaml 的 video_rate 码段反推)。每个 trial 的阶段码经
``make_vr_code(trial, phase)`` 编码,保证 TTL 码在整场唯一,EEG 对齐才能
成功。分数写在 UDP tag 与 ``rating_<时间戳>.json``,不单独占码位。

材料(按优先级)::

    --trials-json '[{"video": "configs/videos/fail_a.mp4"}, ...]'
        scripts/run_session_video.py 运行时抽取组合后这样传入;
        ``image`` 可省 —— 第一帧播放时直接从视频现读,不落地 png
    stim.yaml video_rate.trials 列表(手写场次)
    stim.yaml video_rate.image_path / video_path + --n-trials(单素材兜底)

Usage::

    python -m embodied_brain_collect.stim.paradigm_video_rate --n-trials 6 \\
        --image stimuli/ref.png --video stimuli/demo.mp4 \\
        --windowed --no-serial
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from embodied_brain_collect.stim.base_stim import BaseStim, stim_defaults
from embodied_brain_collect.stim import marker_codes as M

_SESSION_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

# 重播后的播放质量反馈(与动作质量打分分开;不占 TTL)
PLAY_QC_OPTIONS: tuple[tuple[int, str, str], ...] = (
    (0, "ok", "无问题"),
    (1, "dropped_or_occluded", "视频丢帧/动作有遮挡"),
    (2, "other", "其他"),
)


def _rating_stamp(out_dir: Path) -> str:
    """与 launcher 的 session 目录名对齐;单独跑 stim 时用当前时间。"""
    name = out_dir.name
    if _SESSION_STAMP.match(name):
        return name
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _ff_tool(name: str) -> str:
    """ffmpeg/ffprobe 可执行 —— 与录制/打包共用 utils.media 的解析约定
    (Windows third_party exe 优先,其余系统 PATH)。"""
    from embodied_brain_collect.utils.media import media_tool
    try:
        return media_tool(name)
    except RuntimeError as exc:
        raise SystemExit(f"[stim] {exc}") from exc


class _FrameSource:
    """整段视频的逐帧解码源:cv2 优先,读不出帧时回退 ffmpeg 管道软解。

    cv2 的坑:AV1 这类 cv2(自带 FFmpeg)没编译解码器的编码,``isOpened()``、
    fps、帧数**全都正常**,``read()`` 却恒为 False —— 所以选后端只认
    "能否真读出一帧",不信 isOpened。回退路径:ffprobe 取宽高/帧率/帧数,
    ffmpeg 输出 rawvideo rgb24 到管道逐帧读(libdav1d 等软件解码)。
    ffmpeg 可执行经 utils.media 与录制/打包同一套。
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.backend = ""
        self.fps = 0.0
        self.nframes = 0
        self.width = 0
        self.height = 0
        self._first: np.ndarray | None = None   # 探测时解出的首帧,免二次解码
        if not self._try_cv2():
            self._try_ffmpeg()
        if not self.backend:
            raise SystemExit(
                f"[stim] 无法解码视频: {self.path!r} — cv2 与 ffmpeg 都读"
                "不出帧(文件损坏,或编码无解码器且系统 ffmpeg 过旧)。"
                "可重编码为 h264/h265: ffmpeg -i in.mp4 -c:v libx265 "
                "-crf 20 out.mp4")

    # ---- 后端探测 -----------------------------------------------------------

    def _try_cv2(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False
        cap = cv2.VideoCapture(self.path)
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            cap.release()
            return False
        # 属性要在 release 之前取 —— release 后 get 返回 0
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.backend = "cv2"
        self.width, self.height = int(frame.shape[1]), int(frame.shape[0])
        self._first = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True

    def _probe_ffmpeg(self) -> bool:
        """ffprobe 拿宽高/帧率/帧数,填到 self 上;失败返回 False。"""
        ff = _ff_tool("ffprobe")
        try:
            out = subprocess.run(
                [ff, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-of", "json", self.path],
                capture_output=True, text=True, timeout=30,
                check=True).stdout
            stream = (json.loads(out).get("streams") or [None])[0]
        except (subprocess.SubprocessError, ValueError, OSError):
            return False
        if not stream:
            return False
        num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
        try:
            den_i = int(den or 1)
            self.fps = (int(num) / den_i) if den_i else 0.0
            self.width = int(stream.get("width", 0))
            self.height = int(stream.get("height", 0))
        except (TypeError, ValueError):
            return False
        if self.width <= 0 or self.height <= 0:
            return False
        return True

    def _try_ffmpeg(self) -> bool:
        if not self._probe_ffmpeg():
            return False
        frames = self._pipe_frames()
        try:
            first = next(iter(frames), None)
            if first is None:
                return False      # 编码解不动/文件空 — 管道立刻 EOF
            self._first = first   # 探测已解出的首帧,first_frame 免二次开管
        finally:
            frames.close()
        self.backend = "ffmpeg"
        # mp4 头里的 nb_frames 常缺失/不可信,按仓库口径数包(ffprobe_count)
        try:
            from embodied_brain_collect.utils.media import ffprobe_count
            self.nframes = ffprobe_count(Path(self.path))
        except Exception:
            pass
        return True

    # ---- 逐帧读取 -----------------------------------------------------------

    def frames(self):
        """从头开始的逐帧 RGB ndarray 迭代器;每次调用都重新开始。"""
        if self.backend == "cv2":
            return self._frames_cv2()
        return self._pipe_frames()

    def _frames_cv2(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def _pipe_frames(self):
        ff = _ff_tool("ffmpeg")
        try:
            proc = subprocess.Popen(
                [ff, "-v", "error", "-i", self.path, "-map", "0:v:0",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise SystemExit(f"[stim] ffmpeg 启动失败: {exc}") from exc
        nbytes = self.width * self.height * 3
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if not buf or len(buf) < nbytes:
                    return
                yield np.frombuffer(buf, np.uint8).reshape(
                    self.height, self.width, 3)
        finally:
            proc.stdout.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def first_frame(self):
        """一帧即弃:探测/素材体检用。构造时解出的首帧已缓存,不再开管道。"""
        if self._first is not None:
            return self._first
        frames = self.frames()
        try:
            return next(frames, None)
        finally:
            frames.close()


def _read_first_frame(video_path: str) -> "np.ndarray":
    """视频第一帧 → RGB ndarray(不落地 png;解不出 raise SystemExit)。"""
    frame = _FrameSource(video_path).first_frame()
    if frame is None:
        raise SystemExit(f"[stim] 无法从视频读出第一帧: {video_path!r}")
    return frame


def _wrap_text(font, text: str, max_width: int) -> list:
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        cur = ""
        for ch in para:
            trial = cur + ch
            if font.size(trial)[0] <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines or [""]


def _trial_spec(item, args: argparse.Namespace, idx: int) -> dict:
    """单条材料 → trial spec;``image`` 可空(第一帧从视频现读)。"""
    if not isinstance(item, dict):
        raise SystemExit(f"[stim] trials[{idx}] 必须是字典")
    video = str(item.get("video") or args.video or "")
    if not video:
        raise SystemExit(f"[stim] trials[{idx}] 缺 video 路径")
    if not Path(video).is_file():
        raise SystemExit(f"[stim] trial {idx + 1} 视频不存在: {video!r}")
    image = str(item.get("image") or "")
    if image and not Path(image).is_file():
        raise SystemExit(f"[stim] trial {idx + 1} 图像不存在: {image!r}")
    return {
        "image": image or None,
        "video": video,
        "instr1_text": str(item.get("instr1_text", args.instr1_text)),
        "instr2_text": str(item.get("instr2_text", args.instr2_text)),
    }


def _resolve_trials(args: argparse.Namespace, over: dict) -> list[dict]:
    """生成每 trial 的材料字典(--trials-json > yaml trials > 单素材)。"""
    raw = getattr(args, "trials_json", None)
    if raw:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[stim] --trials-json 不是合法 JSON: {exc}")
        if not isinstance(items, list) or not items:
            raise SystemExit("[stim] --trials-json 必须是非空 JSON 数组")
        return [_trial_spec(it, args, i) for i, it in enumerate(items)]

    listed = over.get("trials")
    if isinstance(listed, list) and listed:
        return [_trial_spec(it, args, i) for i, it in enumerate(listed)]

    n = int(args.n_trials)
    if n < 1:
        raise SystemExit("[stim] n_trials 必须 ≥ 1")
    if M.MAX_VR_TRIALS is None:
        raise SystemExit(
            "[stim] markers.yaml 已禁用 video_rate 码段(video_rate_base: "
            "null)— 无法发 trial 阶段码")
    if n > M.MAX_VR_TRIALS:
        raise SystemExit(
            f"[stim] n_trials={n} 超过上限 {M.MAX_VR_TRIALS} "
            "(markers.yaml video_rate 码位的 8-bit 限制)")
    return [_trial_spec({"video": args.video, "image": args.image},
                        args, i) for i in range(n)]


class VideoRateStim(BaseStim):
    """多 trial:注视 → 指令1 → 图像 → 指令2 → 视频 → 打分 → 重播标记。"""

    title = "Video Rate Paradigm"

    @staticmethod
    def add_args(ap: argparse.ArgumentParser, over: dict) -> None:
        ap.add_argument("--trials-json", default=None,
                        help="整场材料 JSON 数组(每项 {video, image?, "
                             "instr1_text?, instr2_text?});run_session_video "
                             "运行时抽取组合后传入,优先于 stim.yaml")
        ap.add_argument("--n-trials", type=int,
                        default=int(over.get("n_trials", 20)),
                        help=f"trial 数量(1..{M.MAX_VR_TRIALS});"
                             "若 stim.yaml 提供 trials 列表则以列表长度为准")
        ap.add_argument("--iti-s", type=float,
                        default=float(over.get("iti_s", 1.0)),
                        help="trial 间隔(秒);0=不自动间隔")
        ap.add_argument("--iti-space", action="store_true",
                        default=bool(over.get("iti_space", False)),
                        help="trial 之间等回车再继续(覆盖 iti-s)")
        ap.add_argument("--fix-pre-s", type=float,
                        default=float(over.get("fix_pre_s", 2.0)))
        ap.add_argument("--fix-radius", type=int,
                        default=int(over.get("fix_radius", 14)))
        ap.add_argument("--instr1-s", type=float,
                        default=float(over.get("instr1_s", 4.0)))
        ap.add_argument("--instr2-s", type=float,
                        default=float(over.get("instr2_s", 4.0)))
        ap.add_argument("--instr1-text",
                        default=str(over.get(
                            "instr1_text", "请认真观看接下来的参考图像")))
        ap.add_argument("--instr2-text",
                        default=str(over.get(
                            "instr2_text", "请观看示范视频，并准备评价动作质量")))
        ap.add_argument("--replay-prompt",
                        default=str(over.get(
                            "replay_prompt",
                            "关键节点做对用Shift标记，关键节点做错用回车标记。")))
        ap.add_argument("--image", default=str(over.get("image_path", "")),
                        help="默认静止图像(无 trials 列表时所有 trial 共用)")
        ap.add_argument("--video", default=str(over.get("video_path", "")),
                        help="默认视频(无 trials 列表时所有 trial 共用)")
        ap.add_argument("--rate-min", type=int,
                        default=int(over.get("rate_min", 0)))
        ap.add_argument("--rate-max", type=int,
                        default=int(over.get("rate_max", 5)))
        ap.add_argument("--rate-prompt",
                        default=str(over.get(
                            "rate_prompt", "请为动作质量打分")))
        ap.add_argument("--play-qc-prompt",
                        default=str(over.get(
                            "play_qc_prompt",
                            "本次遥操视频播放是否有问题？")))
        ap.add_argument("--replay-redo-prompt",
                        default=str(over.get(
                            "replay_redo_prompt",
                            "刚才这一遍打标需要重做吗？")))
        ap.add_argument("--max-replay-attempts", type=int,
                        default=int(over.get("max_replay_attempts", 3)),
                        help="同一 trial 最多重播打标几遍(含第一遍)")
        ap.add_argument("--result-dir",
                        default=str(over.get("result_dir", "data/ratings")),
                        help="打分 JSON 输出目录(文件名为 rating_<时间戳>.json)")

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        over = stim_defaults("video_rate")
        self.trials = _resolve_trials(args, over)
        if M.MAX_VR_TRIALS is not None \
                and len(self.trials) > M.MAX_VR_TRIALS:
            raise SystemExit(
                f"[stim] trials 数量 {len(self.trials)} > 上限 {M.MAX_VR_TRIALS}")
        for i, t in enumerate(self.trials):
            if not Path(t["video"]).is_file():
                raise SystemExit(f"[stim] trial {i + 1} 视频不存在: {t['video']!r}")
        if not (0 <= args.rate_min <= args.rate_max <= 9):
            raise SystemExit("[stim] rate_min/rate_max 须满足 0 ≤ min ≤ max ≤ 9")
        if int(args.max_replay_attempts) < 1:
            raise SystemExit("[stim] max_replay_attempts 须 ≥ 1")
        self.results: list[dict] = []
        self._replay_mark_i = 0
        self._first_frames: dict[int, object] = {}   # trial_idx -> Surface
        print(f"[stim] 将采集 {len(self.trials)} 个 trials "
              f"(码位上限 {M.MAX_VR_TRIALS})")

    def _code(self, trial_idx: int, phase: str) -> int:
        return M.make_vr_code(trial_idx, phase)

    def _tag(self, trial_idx: int, phase: str, extra: str = "") -> str:
        base = f"T{trial_idx + 1:02d}_{phase}"
        return f"{base}_{extra}" if extra else base

    # ---- 第一帧(不落地:播放时直接从视频现读) ------------------------------

    def _first_frame(self, trial_idx: int, video_path: str):
        """视频第一帧 → pygame Surface,每 trial 只读一次并缓存。"""
        if trial_idx in self._first_frames:
            return self._first_frames[trial_idx]
        frame = _read_first_frame(video_path)
        h, w = frame.shape[:2]
        surf = self.pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
        self._first_frames[trial_idx] = surf
        return surf

    # ---- 绘制 --------------------------------------------------------------

    def _draw_text_block(self, text: str, hint: str = "",
                         hint_color=None) -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        max_w = int(self.screen.get_width() * 0.85)
        lines = _wrap_text(self.font_instr, text, max_w)
        line_h = self.font_instr.get_linesize()
        total_h = line_h * len(lines)
        y0 = cy - total_h // 2 - (self.args.font_size if hint else 0) // 2
        for i, line in enumerate(lines):
            surf = self.font_instr.render(line, True, self.text_color)
            self.screen.blit(surf, surf.get_rect(center=(cx, y0 + i * line_h)))
        if hint:
            color = self.hint_color if hint_color is None else hint_color
            h = self.font_small.render(hint, True, color)
            self.screen.blit(
                h, h.get_rect(center=(cx, y0 + total_h + self.args.font_size)))

    def _blit_scaled(self, surf) -> None:
        sw, sh = self.screen.get_size()
        iw, ih = surf.get_size()
        scale = min(sw / max(iw, 1), sh / max(ih, 1))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        scaled = self.pygame.transform.smoothscale(surf, (nw, nh))
        self.screen.blit(scaled, scaled.get_rect(center=(sw // 2, sh // 2)))

    def _blit_top_banner(self, texts: list[str], *,
                         font=None, color=None, y0: int = 20) -> None:
        """正中上方多行提示(自动换行)。"""
        font = font or self.font_title
        color = self.continue_color if color is None else color
        cx = self.screen.get_width() // 2
        max_w = int(self.screen.get_width() * 0.92)
        y = y0
        for raw in texts:
            for line in _wrap_text(font, raw, max_w):
                surf = font.render(line, True, color)
                self.screen.blit(surf, surf.get_rect(midtop=(cx, y)))
                y += font.get_linesize()

    def _replay_prompt_lines(self) -> list[str]:
        text = str(self.args.replay_prompt or "")
        lines = [ln.strip() for ln in text.replace("\\n", "\n").splitlines()
                 if ln.strip()]
        return lines or [text]

    def _emit_replay_mark(self, trial_idx: int, t0: float, *,
                          kind: str, key: str, key_name: str) -> dict:
        """记录一次打标。``key`` 必须是 shift 或 enter,两者不能混写成同一类。"""
        idx = self._replay_mark_i
        self._replay_mark_i += 1
        t = time.time()
        if key == "shift":
            kind_tag = "SHIFT"
            label = "正确"
        elif key == "enter":
            kind_tag = "ENTER"
            label = "错误"
        else:
            raise ValueError(f"未知打标按键 {key!r},须为 shift 或 enter")
        rec: dict = {
            "index": idx + 1,
            "kind": kind,
            "key": key,
            "key_name": key_name,
            "label": label,
            "t_sent_pc": t,
            "video_time_s": t - t0,
        }
        extra = f"{kind_tag}_{idx + 1:02d}"
        if idx < M.MAX_REPLAY_MARKS:
            code = M.make_replay_mark_code(idx)
            tag = self._tag(trial_idx, "REPLAY_MARK", extra)
            rec["code"] = code
            rec["tag"] = tag
            self.marker.mark(code, tag, ttl=False)
        else:
            rec["code"] = None
            rec["tag"] = self._tag(trial_idx, "REPLAY_MARK", extra)
            print(f"[stim] 重播标记已达上限 {M.MAX_REPLAY_MARKS},仅写入 JSON")
        print(f"[stim] trial {trial_idx + 1}: {label}  {key}/{key_name}  "
              f"t={rec['video_time_s']:.3f}s  tag={rec['tag']}")
        return rec

    def _blit_replay_feedback(self, n_ok: int, n_bad: int,
                              flash_kind: str | None) -> None:
        """重播底部:正确/错误次数常驻;按键后约 0.6s 弹出对应文字。"""
        pg = self.pygame
        sw, sh = self.screen.get_size()
        gap = 48
        ok_s = self.font_title.render(f"正确 {n_ok}", True, (70, 220, 150))
        bad_s = self.font_title.render(f"错误 {n_bad}", True, (255, 150, 50))
        total_w = ok_s.get_width() + gap + bad_s.get_width()
        h = max(ok_s.get_height(), bad_s.get_height())
        pad_x, pad_y = 18, 10
        bar = pg.Surface((total_w + pad_x * 2, h + pad_y * 2), pg.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        bar_rect = bar.get_rect(midbottom=(sw // 2, sh - 28))
        self.screen.blit(bar, bar_rect)
        x0 = bar_rect.left + pad_x
        y0 = bar_rect.top + pad_y
        self.screen.blit(ok_s, (x0, y0))
        self.screen.blit(bad_s, (x0 + ok_s.get_width() + gap, y0))
        if not flash_kind:
            return
        label = "正确" if flash_kind == "node" else "错误"
        color = (70, 220, 150) if flash_kind == "node" else (255, 150, 50)
        fsurf = self.font_instr.render(label, True, color)
        frect = fsurf.get_rect(midbottom=(sw // 2, bar_rect.top - 18))
        pad2 = 18
        chip = pg.Surface((frect.width + pad2 * 2, frect.height + pad2),
                          pg.SRCALPHA)
        chip.fill((0, 0, 0, 175))
        self.screen.blit(chip, chip.get_rect(center=frect.center))
        self.screen.blit(fsurf, frect)

    # ---- 各阶段(带 trial_idx) ---------------------------------------------

    def _phase_fixation(self, trial_idx: int, timing: dict) -> None:
        self._draw_fixation()
        self._flip_and_mark(self._code(trial_idx, "FIX_ON"),
                            self._tag(trial_idx, "FIX_ON"))
        timing["fix_on"] = time.time()
        self._wait_seconds(self.args.fix_pre_s)

    def _phase_instr(self, trial_idx: int, text: str, seconds: float,
                     on_phase: str, off_phase: str, timing: dict,
                     key: str) -> None:
        self._draw_text_block(text)
        self._flip_and_mark(self._code(trial_idx, on_phase),
                            self._tag(trial_idx, on_phase))
        timing[f"{key}_on"] = time.time()
        self._wait_seconds(seconds)
        self.marker.mark(self._code(trial_idx, off_phase),
                         self._tag(trial_idx, off_phase))
        timing[f"{key}_off"] = time.time()

    def _trial_image(self, trial_idx: int, spec: dict):
        """trial 的静止图像:给了 image 文件用它;否则从视频现读第一帧。"""
        if spec.get("image"):
            return self.pygame.image.load(str(spec["image"])).convert()
        return self._first_frame(trial_idx, spec["video"]).convert()

    def _phase_image(self, trial_idx: int, spec: dict, timing: dict) -> None:
        img = self._trial_image(trial_idx, spec)
        hint = self.font_instr.render(
            "想象完毕之后按回车键继续", True, self.continue_color)

        def redraw():
            self._clear()
            self._blit_scaled(img)
            self.screen.blit(hint, hint.get_rect(
                midtop=(self.screen.get_width() // 2, 28)))

        redraw()
        self._flip_and_mark(self._code(trial_idx, "IMG_START"),
                            self._tag(trial_idx, "IMG_START"))
        timing["image_on"] = time.time()
        if not self._wait_enter(redraw):
            return
        self.marker.mark(self._code(trial_idx, "IMG_END"),
                         self._tag(trial_idx, "IMG_END"))
        timing["image_off"] = time.time()

    def _play_video(self, trial_idx: int, video_path: Path, timing: dict,
                    *, start_phase: str, end_phase: str, prefix: str,
                    collect_marks: bool = False,
                    overlay: list[str] | None = None,
                    send_ttl: bool = True) -> list[dict]:
        source = _FrameSource(str(video_path))
        if source.backend == "ffmpeg":
            print(f"[stim] cv2 读不出帧(AV1 等编码),走 ffmpeg 软解回退: "
                  f"{video_path.name} — 解码过慢时播放会掉帧,建议重编码")
        fps = source.fps
        nframes = source.nframes
        duration = (nframes / fps) if fps > 1e-3 and nframes > 0 else None
        frame_dt = (1.0 / fps) if fps > 1e-3 else (1.0 / 30.0)
        pg = self.pygame

        if send_ttl:
            self.marker.mark(self._code(trial_idx, start_phase),
                             self._tag(trial_idx, start_phase))
        t0 = time.time()
        timing[f"{prefix}_on"] = t0
        timing[f"{prefix}_meta_fps"] = fps
        timing[f"{prefix}_meta_nframes"] = nframes
        timing[f"{prefix}_meta_duration_s"] = duration

        marks: list[dict] = []
        n_node = n_bad = 0
        flash_kind: str | None = None
        flash_until = 0.0
        flash_s = 0.6
        frame_i = 0
        next_t = time.perf_counter()
        frames = source.frames()
        pg.key.set_repeat()  # 关闭按键重复,避免按住 Shift 连发
        down_keys: set[int] = set()
        while not self.aborted:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.aborted = True
                    break
                if ev.type == pg.KEYUP:
                    down_keys.discard(ev.key)
                    continue
                if ev.type != pg.KEYDOWN:
                    continue
                if ev.key == pg.K_ESCAPE:
                    self.aborted = True
                    break
                if ev.key in down_keys:
                    continue
                down_keys.add(ev.key)
                if collect_marks and ev.key in (pg.K_LSHIFT, pg.K_RSHIFT):
                    key_name = "LSHIFT" if ev.key == pg.K_LSHIFT else "RSHIFT"
                    marks.append(self._emit_replay_mark(
                        trial_idx, t0, kind="node",
                        key="shift", key_name=key_name))
                    n_node += 1
                    flash_kind = "node"
                    flash_until = time.perf_counter() + flash_s
                elif collect_marks and ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
                    key_name = ("RETURN" if ev.key == pg.K_RETURN
                                else "KP_ENTER")
                    marks.append(self._emit_replay_mark(
                        trial_idx, t0, kind="unsmooth",
                        key="enter", key_name=key_name))
                    n_bad += 1
                    flash_kind = "unsmooth"
                    flash_until = time.perf_counter() + flash_s
            if self.aborted:
                break
            frame = next(frames, None)
            if frame is None:
                break
            h, w = frame.shape[:2]
            surf = pg.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            self._clear()
            self._blit_scaled(surf)
            if overlay:
                self._blit_top_banner(overlay, font=self.font_title)
            if collect_marks:
                show = flash_kind if time.perf_counter() < flash_until else None
                self._blit_replay_feedback(n_node, n_bad, show)
            pg.display.flip()
            frame_i += 1
            next_t += frame_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

        frames.close()
        t1 = time.time()
        if send_ttl:
            self.marker.mark(self._code(trial_idx, end_phase),
                             self._tag(trial_idx, end_phase))
        timing[f"{prefix}_off"] = t1
        timing[f"{prefix}_played_s"] = t1 - t0
        timing[f"{prefix}_frames_shown"] = frame_i
        label = "重播" if collect_marks else "视频"
        extra = ""
        if collect_marks:
            n_node = sum(1 for m in marks if m.get("kind") == "node")
            n_bad = sum(1 for m in marks if m.get("kind") == "unsmooth")
            extra = (f"  正确(Shift) {n_node} 错误(回车) {n_bad}")
        print(f"[stim] trial {trial_idx + 1}: {label} "
              f"{(duration if duration else t1 - t0):.2f}s / {frame_i} 帧"
              + extra)
        return marks

    def _phase_video(self, trial_idx: int, video_path: Path,
                     timing: dict) -> None:
        self._play_video(trial_idx, video_path, timing,
                         start_phase="VIDEO_START", end_phase="VIDEO_END",
                         prefix="video")

    def _phase_replay_ready(self, trial_idx: int, spec: dict,
                            timing: dict, *, send_ttl: bool,
                            attempt: int) -> bool:
        img = self._trial_image(trial_idx, spec)
        prompt_lines = self._replay_prompt_lines()
        extra = "" if attempt <= 1 else f"（第 {attempt} 遍打标）"

        def redraw():
            self._clear()
            self._blit_scaled(img)
            self._blit_top_banner(prompt_lines, font=self.font_banner)
            y_hint = 28 + self.font_banner.get_linesize() * max(
                len(prompt_lines), 1)
            self._blit_top_banner(
                [f"按回车开始播放{extra}"],
                font=self.font_title, y0=y_hint)

        redraw()
        if send_ttl:
            self._flip_and_mark(self._code(trial_idx, "REPLAY_READY"),
                                self._tag(trial_idx, "REPLAY_READY"))
        else:
            self.pygame.display.flip()
        key = "replay_ready" if attempt <= 1 else f"replay_ready_{attempt}"
        timing[f"{key}_on"] = time.time()
        if not self._wait_enter(redraw):
            return False
        timing[f"{key}_off"] = time.time()
        return True

    def _phase_replay_redo(self, trial_idx: int, timing: dict,
                           attempt: int) -> bool | None:
        """重播结束后询问是否重做打标。True=再来一遍, False=进入下一环节。"""
        selected = None
        timing.setdefault("replay_redo", [])
        t_on = time.time()

        def draw():
            prompt = (
                f"Trial {trial_idx + 1}/{len(self.trials)}\n"
                f"{self.args.replay_redo_prompt}\n"
                "1  是，再看一遍并重新打标\n"
                "2  否，进入下一环节"
            )
            if selected is not None:
                prompt += f"\n当前选择: {'是，重做' if selected == 1 else '否，继续'}"
                hint = "请按回车键确认"
            else:
                hint = "请按 1 或 2"
            self._draw_text_block(prompt, hint, hint_color=self.continue_color)

        pg = self.pygame
        while not self.aborted:
            draw()
            pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.aborted = True
                    return None
                if ev.type != pg.KEYDOWN:
                    continue
                if ev.key == pg.K_ESCAPE:
                    self.aborted = True
                    return None
                if ev.key in (pg.K_BACKSPACE, pg.K_DELETE):
                    selected = None
                    continue
                val = None
                if pg.K_0 <= ev.key <= pg.K_9:
                    val = ev.key - pg.K_0
                elif pg.K_KP0 <= ev.key <= pg.K_KP9:
                    val = ev.key - pg.K_KP0
                if val in (1, 2):
                    selected = val
                if ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and selected is not None:
                    redo = selected == 1
                    timing["replay_redo"].append({
                        "after_attempt": attempt,
                        "redo": redo,
                        "t_on": t_on,
                        "t_off": time.time(),
                    })
                    print(f"[stim] trial {trial_idx + 1}: 打标"
                          f"{'重做' if redo else '确认完成'} (第 {attempt} 遍后)")
                    return redo
            time.sleep(0.01)
        return None

    def _phase_replay(self, trial_idx: int, spec: dict,
                      timing: dict) -> list[dict] | None:
        prompt_lines = self._replay_prompt_lines()
        max_n = int(self.args.max_replay_attempts)
        attempts: list[dict] = []
        marks: list[dict] = []
        for attempt in range(1, max_n + 1):
            if not self._phase_replay_ready(
                    trial_idx, spec, timing,
                    send_ttl=(attempt == 1), attempt=attempt):
                return None
            prefix = "replay" if attempt == 1 else f"replay_{attempt}"
            marks = self._play_video(
                trial_idx, Path(spec["video"]), timing,
                start_phase="REPLAY_START", end_phase="REPLAY_END",
                prefix=prefix, collect_marks=True,
                overlay=prompt_lines, send_ttl=(attempt == 1))
            if self.aborted:
                return None
            rec = {"attempt": attempt, "marks": marks, "redo": False}
            attempts.append(rec)
            if attempt >= max_n:
                print(f"[stim] trial {trial_idx + 1}: 已达重播上限 {max_n} 遍,"
                      "进入下一环节")
                break
            redo = self._phase_replay_redo(trial_idx, timing, attempt)
            if redo is None:
                return None
            rec["redo"] = redo
            if not redo:
                break
        timing["replay_marks"] = marks
        timing["replay_attempts"] = attempts
        return marks

    def _phase_rating(self, trial_idx: int, timing: dict) -> int | None:
        lo, hi = self.args.rate_min, self.args.rate_max
        selected = None
        self.marker.mark(self._code(trial_idx, "RATE_ON"),
                         self._tag(trial_idx, "RATE_ON"))
        timing["rate_on"] = time.time()

        def draw():
            prompt = (f"Trial {trial_idx + 1}/{len(self.trials)}\n"
                      f"{self.args.rate_prompt}\n"
                      f"分数越高，动作质量越高\n"
                      f"请按数字键 {lo}–{hi}")
            if selected is not None:
                prompt += f"\n当前选择: {selected}\n按 Delete 可清除后重新输入"
                hint = "请按回车键继续"
            else:
                hint = "请先输入分数"
            self._draw_text_block(prompt, hint, hint_color=self.continue_color)

        pg = self.pygame
        while not self.aborted:
            draw()
            pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.aborted = True
                    return None
                if ev.type != pg.KEYDOWN:
                    continue
                if ev.key == pg.K_ESCAPE:
                    self.aborted = True
                    return None
                if ev.key in (pg.K_BACKSPACE, pg.K_DELETE):
                    selected = None
                    continue
                val = None
                if pg.K_0 <= ev.key <= pg.K_9:
                    val = ev.key - pg.K_0
                elif pg.K_KP0 <= ev.key <= pg.K_KP9:
                    val = ev.key - pg.K_KP0
                if val is not None and lo <= val <= hi:
                    selected = val
                if ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and selected is not None:
                    # 分数只写在 tag / JSON,不另占 TTL 码(多 trial 会撞码)
                    self.marker.mark(
                        self._code(trial_idx, "RATE_OFF"),
                        self._tag(trial_idx, "RATE_OFF", f"SCORE{selected}"))
                    timing["rate_off"] = time.time()
                    timing["rating"] = selected
                    return selected
            time.sleep(0.01)
        return None

    def _phase_play_qc(self, trial_idx: int, timing: dict) -> dict | None:
        """重播结束后询问播放本身是否有问题。"""
        selected: int | None = None
        labels = {code: label for code, _key, label in PLAY_QC_OPTIONS}
        keys = {code: key for code, key, _label in PLAY_QC_OPTIONS}
        timing["play_qc_on"] = time.time()

        def draw():
            lines = [
                f"Trial {trial_idx + 1}/{len(self.trials)}",
                str(self.args.play_qc_prompt),
                "请按数字键选择，回车确认",
                "",
            ]
            for code, _key, label in PLAY_QC_OPTIONS:
                mark = "●" if selected == code else "○"
                lines.append(f"{mark}  {code}  {label}")
            if selected is not None:
                lines.append("")
                lines.append(f"当前选择：{labels[selected]}")
                hint = "请按回车键继续"
            else:
                hint = "请先选择一项"
            self._draw_text_block("\n".join(lines), hint,
                                  hint_color=self.continue_color)

        pg = self.pygame
        while not self.aborted:
            draw()
            pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.aborted = True
                    return None
                if ev.type != pg.KEYDOWN:
                    continue
                if ev.key == pg.K_ESCAPE:
                    self.aborted = True
                    return None
                if ev.key in (pg.K_BACKSPACE, pg.K_DELETE):
                    selected = None
                    continue
                val = None
                if pg.K_0 <= ev.key <= pg.K_9:
                    val = ev.key - pg.K_0
                elif pg.K_KP0 <= ev.key <= pg.K_KP9:
                    val = ev.key - pg.K_KP0
                if val is not None and val in labels:
                    selected = val
                if ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and selected is not None:
                    rec = {
                        "ok": selected == 0,
                        "codes": [selected],
                        "keys": [keys[selected]],
                        "labels": [labels[selected]],
                    }
                    timing["play_qc_off"] = time.time()
                    timing["play_qc"] = rec
                    print(f"[stim] trial {trial_idx + 1}: 播放质检 "
                          f"{'正常' if rec['ok'] else '有问题'} "
                          f"{rec['labels']}")
                    return rec
            time.sleep(0.01)
        return None

    def _run_one_trial(self, trial_idx: int, spec: dict) -> dict | None:
        timing: dict = {}
        self.marker.set_trial(trial_idx + 1)
        print(f"[stim] === trial {trial_idx + 1}/{len(self.trials)} ===")

        self._phase_fixation(trial_idx, timing)
        if self.aborted:
            return None

        self._phase_instr(trial_idx, spec["instr1_text"], self.args.instr1_s,
                          "INSTR_ON", "INSTR_OFF", timing, "instr1")
        if self.aborted:
            return None

        self._phase_image(trial_idx, spec, timing)
        if self.aborted:
            return None

        self._phase_instr(trial_idx, spec["instr2_text"], self.args.instr2_s,
                          "INSTR2_ON", "INSTR2_OFF", timing, "instr2")
        if self.aborted:
            return None

        self._phase_video(trial_idx, Path(spec["video"]), timing)
        if self.aborted:
            return None

        rating = self._phase_rating(trial_idx, timing)
        if self.aborted or rating is None:
            return None

        replay_marks = self._phase_replay(trial_idx, spec, timing)
        if self.aborted or replay_marks is None:
            return None

        play_qc = self._phase_play_qc(trial_idx, timing)
        if self.aborted or play_qc is None:
            return None

        return {
            "trial": trial_idx + 1,
            "rating": rating,
            "play_qc": play_qc,
            "image": spec["image"],
            "video": spec["video"],
            "instr1_text": spec["instr1_text"],
            "instr2_text": spec["instr2_text"],
            "replay_marks": replay_marks,
            "replay_attempts": timing.get("replay_attempts", []),
            "timing": timing,
        }

    def _iti(self, trial_idx: int) -> None:
        if trial_idx >= len(self.trials) - 1:
            return
        if self.args.iti_space:
            self._wait_enter(lambda: self._draw_text_block(
                f"Trial {trial_idx + 1} 完成",
                "请按回车键继续",
                hint_color=self.continue_color))
            return
        if self.args.iti_s > 0:
            self._draw_text_block(f"Trial {trial_idx + 1} 完成", "请稍候…")
            self.pygame.display.flip()
            self._wait_seconds(self.args.iti_s)

    def _save_results(self) -> None:
        out_dir = Path(self.args.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _rating_stamp(out_dir)
        payload = {
            "n_trials": len(self.trials),
            "n_completed": len(self.results),
            "rate_min": self.args.rate_min,
            "rate_max": self.args.rate_max,
            "session_dir": str(out_dir.resolve()),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "trials": self.results,
        }
        path = out_dir / f"rating_{stamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"[stim] 已写入 {path}  "
              f"({len(self.results)}/{len(self.trials)} trials)")

    def run_flow(self) -> None:
        self.marker.mark(M.RUN_START, "RUN_START")
        for i, spec in enumerate(self.trials):
            result = self._run_one_trial(i, spec)
            if result is None:
                break
            self.results.append(result)
            self._iti(i)
            if self.aborted:
                break
        self._save_results()
        if self.results:
            self.marker.mark(M.RUN_END, "RUN_END")


def main(argv: list[str] | None = None) -> int:
    over = stim_defaults("video_rate")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    BaseStim.add_common_args(ap, over)
    VideoRateStim.add_args(ap, over)
    args = ap.parse_args(argv)
    return VideoRateStim(args).run()


if __name__ == "__main__":
    sys.exit(main())
