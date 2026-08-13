"""Paradigm 1 (Pick & Place + motor imagery) -- pure Python stim using pygame.

Trial flow (per trial, experimenter-driven with SPACE):

    1. Press SPACE to start trial
    2. Red-dot fixation 2 s                         -> FIX_ON / FIX_OFF
    3. Instruction text (task_id + task_name) 10 s    -> TASK_ID + INSTR_ON / INSTR_OFF
    4. Close eyes, motor imagery (self-paced)         -> IMG_START
    5. Open eyes, press SPACE to begin execution      -> IMG_END
    6. Perform pick & place                           -> EXEC_START
    7. Press SPACE when done                          -> EXEC_END
    8. Press SPACE for next trial

Markers go through ParallelBox COM (EEG) and UDP (sync_hub), matching
record/sync/marker_codes.py.

Configuration:
    Edit config/collection.json (task library). Each session picks one task
    via --task-id; run number auto-increments (config/run_counters.json).

Example:
    python -m record.stim.paradigm1_pickplace --config record\\config\\collection.json --task-id 0

Smoke test:
    python -m record.stim.paradigm1_pickplace --config record\\config\\collection.json --task-id 0 \\
        --no-serial --fast 10 --windowed
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.config.collection import (
    CollectionConfig,
    ActiveSession,
    DEFAULT_COLLECTION_PATH,
    resolve_active_for_stim,
)
from src.stim.marker_sender import MarkerSender
from src.sync import marker_codes as M


@dataclass
class TrialSpec:
    task_id: int
    task_name: str
    scene: int


def _trial_from_active(active: ActiveSession) -> TrialSpec:
    return TrialSpec(
        task_id=active.task_id,
        task_name=active.task_name,
        scene=active.scene,
    )


def _find_font(preferred: list[str]) -> str | None:
    import platform
    if platform.system() == "Windows":
        dirs = [Path(r"C:\Windows\Fonts")]
    else:
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    for d in dirs:
        if not d.is_dir():
            continue
        for name in preferred:
            for f in d.rglob(name):
                return str(f)
    return None


class StimRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        trial: TrialSpec,
        active: ActiveSession,
    ) -> None:
        self.args = args
        self.trial = trial
        self.active = active

        import pygame
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mouse.set_visible(False)

        flags = pygame.FULLSCREEN if args.fullscreen else 0
        if args.windowed:
            flags = 0
        self.screen = pygame.display.set_mode((args.width, args.height), flags)
        pygame.display.set_caption("Paradigm 1 -- Pick & Place + MI")

        font_path = _find_font(["msyh.ttc", "simhei.ttf", "simsun.ttc",
                                 "NotoSansCJK*", "wqy-microhei.ttc",
                                 "DroidSansFallbackFull.ttf", "uming.ttc", "ukai.ttc"])
        if font_path is None:
            import platform
            if platform.system() != "Windows":
                font_path = pygame.font.get_default_font()
        self.font_instr = pygame.font.Font(font_path, args.font_size)
        self.font_small = pygame.font.Font(font_path, max(16, args.font_size // 3))
        self.font_title = pygame.font.Font(font_path, max(20, args.font_size // 2))

        self.marker = MarkerSender(
            port=args.parallelbox,
            baud=args.baud,
            udp_host=args.marker_host,
            udp_port=args.marker_port,
            hold_s=args.hold_s,
            enable_serial=not args.no_serial,
        )

        self.pygame = pygame
        self.bg = (0, 0, 0)
        self.fix_color = (220, 0, 0)
        self.text_color = (230, 230, 230)
        self.hint_color = (120, 120, 120)
        self.aborted = False

    # ----- drawing helpers --------------------------------------------------

    def _clear(self) -> None:
        self.screen.fill(self.bg)

    def _flip_and_mark(self, code: int, tag: str) -> float:
        self.pygame.display.flip()
        return self.marker.mark(code, tag)

    def _draw_fixation(self) -> None:
        self._clear()
        cx, cy = self.screen.get_width() // 2, self.screen.get_height() // 2
        self.pygame.draw.circle(self.screen, self.fix_color, (cx, cy), self.args.fix_radius)

    def _draw_instruction(self, task_id: int, task_name: str) -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        header = self.font_title.render(f"任务 #{task_id}", True, self.hint_color)
        body = self.font_instr.render(task_name, True, self.text_color)
        h_rect = header.get_rect(center=(cx, cy - self.args.font_size))
        b_rect = body.get_rect(center=(cx, cy + self.args.font_size // 4))
        self.screen.blit(header, h_rect)
        self.screen.blit(body, b_rect)

    def _draw_message(self, main: str, hint: str = "") -> None:
        self._clear()
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        main_surf = self.font_instr.render(main, True, self.text_color)
        main_rect = main_surf.get_rect(center=(cx, cy - (20 if hint else 0)))
        self.screen.blit(main_surf, main_rect)
        if hint:
            hint_surf = self.font_small.render(hint, True, self.hint_color)
            hint_rect = hint_surf.get_rect(center=(cx, cy + self.args.font_size))
            self.screen.blit(hint_surf, hint_rect)

    # ----- timing / input helpers -------------------------------------------

    def _scale(self, seconds: float) -> float:
        return seconds / max(1.0, self.args.fast)

    def _poll_abort(self) -> bool:
        pg = self.pygame
        for ev in pg.event.get():
            if ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE:
                self.aborted = True
                return True
        return False

    def _wait_seconds(self, seconds: float) -> None:
        end_t = time.perf_counter() + self._scale(seconds)
        while time.perf_counter() < end_t and not self.aborted:
            if self._poll_abort():
                return
            time.sleep(0.005)

    def _wait_space(self, draw_fn=None) -> bool:
        """Wait for SPACE. Returns False if aborted."""
        pg = self.pygame
        while not self.aborted:
            if draw_fn is not None:
                draw_fn()
                pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        self.aborted = True
                        return False
                    if ev.key == pg.K_SPACE:
                        return True
            time.sleep(0.01)
        return False

    # ----- main flow --------------------------------------------------------

    def _draw_status(self, trial_n: int, hint: str) -> None:
        header = (f"{self.active.subject}  task {self.trial.task_id}  "
                  f"run {self.active.run}  #{trial_n}")
        self._draw_message(header, hint)

    def run(self) -> int:
        run_started = False
        trial_n = 0
        trial = self.trial

        while not self.aborted:
            trial_n += 1
            hint = "按 [空格] 开始" if trial_n == 1 else "按 [空格] 开始下一次  |  Esc 结束"
            if not self._wait_space(lambda: self._draw_status(trial_n, hint)):
                break

            if not run_started:
                self.marker.mark(M.RUN_START, "RUN_START")
                run_started = True
                self.marker.mark(M.make_scene_id(trial.scene), "SCENE_ID")

            self.marker.set_trial(trial_n)
            print(f"[trial #{trial_n}] task_id={trial.task_id} run={self.active.run} "
                  f"name='{trial.task_name}'")

            self._draw_fixation()
            self._flip_and_mark(M.FIX_ON, "FIX_ON")
            self._wait_seconds(self.args.fix_pre_s)
            if self.aborted:
                break

            self.marker.mark(M.make_task_id(trial.task_id), "TASK_ID")
            self._draw_instruction(trial.task_id, trial.task_name)
            self._flip_and_mark(M.INSTR_ON, "INSTR_ON")
            self.marker.mark(M.FIX_OFF, "FIX_OFF")
            self._wait_seconds(self.args.instr_s)
            if self.aborted:
                break
            self.marker.mark(M.INSTR_OFF, "INSTR_OFF")

            self._draw_message("请闭眼，进行运动想象", "想象完成后睁眼，按 [空格] 继续")
            self._flip_and_mark(M.IMG_START, "IMG_START")
            if not self._wait_space(lambda: self._draw_message(
                    "请闭眼，进行运动想象", "想象完成后睁眼，按 [空格] 继续")):
                break
            self.marker.mark(M.IMG_END, "IMG_END")

            if not self._wait_space(lambda: self._draw_message(
                    "请执行 pick & place", "按 [空格] 开始实际动作")):
                break
            self._draw_message("执行中 ...", "完成后按 [空格] 结束")
            self._flip_and_mark(M.EXEC_START, "EXEC_START")
            if not self._wait_space(lambda: self._draw_message(
                    "执行中 ...", "完成后按 [空格] 结束")):
                break
            self.marker.mark(M.EXEC_END, "EXEC_END")

            if self.args.once:
                break

        if run_started:
            self.marker.mark(M.RUN_END, "RUN_END")
        return self._cleanup(0 if not self.aborted else 2)

    def _cleanup(self, rc: int) -> int:
        self.marker.close()
        self.pygame.mouse.set_visible(True)
        self.pygame.quit()
        return rc


def _apply_stim_config(args: argparse.Namespace, cfg: CollectionConfig) -> None:
    stim = cfg.stim
    args.parallelbox = stim.parallelbox
    args.baud = stim.baud
    args.hold_s = stim.hold_s
    args.marker_host = stim.marker_host
    args.marker_port = stim.marker_port
    args.fullscreen = stim.fullscreen
    args.fix_pre_s = stim.fix_pre_s
    args.instr_s = stim.instr_s
    args.font_size = stim.font_size
    args.fix_radius = stim.fix_radius


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="config/collection.json")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Task to run (must match launcher active_session)")
    ap.add_argument("--parallelbox", default="COM14")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--marker-host", default="127.0.0.1")
    ap.add_argument("--marker-port", type=int, default=9999)
    ap.add_argument("--hold-s", type=float, default=0.020,
                    help="TTL high duration on ParallelBox (s)")
    ap.add_argument("--no-serial", action="store_true", help="Disable ParallelBox writes")

    ap.add_argument("--fix-pre-s", type=float, default=2.0)
    ap.add_argument("--instr-s", type=float, default=10.0)
    ap.add_argument("--fast", type=float, default=1.0,
                    help="Time-compression factor for dry-runs (e.g. 10 = 10x faster).")

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fullscreen", action="store_true", default=True)
    ap.add_argument("--windowed", action="store_true",
                    help="Force windowed mode (overrides --fullscreen).")
    ap.add_argument("--font-size", type=int, default=64)
    ap.add_argument("--fix-radius", type=int, default=14)
    ap.add_argument("--once", action="store_true",
                    help="Run exactly one trial then exit")

    args = ap.parse_args(argv)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("[stim] pygame not installed. Run: pip install pygame", file=sys.stderr)
        return 2

    cfg_path = args.config or str(DEFAULT_COLLECTION_PATH)
    cfg, active = resolve_active_for_stim(args.task_id, config_path=cfg_path)
    _apply_stim_config(args, cfg)
    trial = _trial_from_active(active)
    print(f"[stim] subject={active.subject} task_id={active.task_id} "
          f"run={active.run} config={cfg_path}")
    print(f"[stim] task_name={active.task_name!r}  (Esc 结束，空格重复本任务)")

    runner = StimRunner(args, trial, active)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
