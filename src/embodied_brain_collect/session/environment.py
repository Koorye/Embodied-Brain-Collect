"""环境图纸 — 采集前的摆放准备阶段。

``configs/environments/`` 下按场景目录存放图纸:每个目录一个场景,内有
``config.yaml``(task/scene 定义)和一批 ``NNNN_comboNNN_rN.png`` 图纸。
每次采集随机抽一张**还没用过**的图纸全屏显示,采集员照着图纸摆放实物,
按 n + Enter 关闭图纸后才正式开录;采集成功的图纸由调用方记入**该场景目录
自己的台账** ``<场景目录>/used.yaml``,之后的抽取自动跳过 —— 重采/退出
的不记台账,图纸留在池里下次再抽。

台账是"已用"的唯一状态:删掉条目即可让该图纸重新进入抽取池。
目录里的非图纸文件(如 placements_overview.png)不进抽取队列。

所有操作封装在 :class:`Environment` 中;模块底部提供了绑定默认根目录
的同名便捷函数,调用方两种用法等价。
"""

from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path

import yaml

from .config import configs_dir

LEDGER_NAME = "used.yaml"



class Environment:
    """一个图纸池:场景目录下的抽取、台账与摆放显示,围绕 ``root`` 组织。"""

    ledger_name = LEDGER_NAME

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else configs_dir() / "environments"

    # ---- 路径 ---------------------------------------------------------------

    def path(self, rel: str) -> Path:
        return self.root / rel

    def _rel_key(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ---- 台账 ---------------------------------------------------------------

    def load_used(self, rel: str) -> dict[str, dict]:
        """该图纸所在场景目录的台账:{文件名: {session, at}};缺失/损坏 = 空。"""
        directory = self.path(rel).parent
        try:
            data = yaml.safe_load(
                (directory / self.ledger_name).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        used = (data or {}).get("used", {})
        return used if isinstance(used, dict) else {}

    def mark_used(self, rel: str, session: str = "") -> None:
        """采集成功后把图纸记入其场景目录的台账(幂等,键 = 文件名)。"""
        path = self.path(rel)
        used = self.load_used(rel)
        if path.name in used:
            return
        used[path.name] = {"session": session,
                           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        (path.parent / self.ledger_name).write_text(
            yaml.safe_dump({"used": used}, allow_unicode=True, sort_keys=False),
            encoding="utf-8")

    # ---- 抽取池 --------------------------------------------------------------

    def pool_unused(self) -> list[str]:
        """尚未采集的图纸(相对路径),按场景目录 + 文件名排序。

        只认 ``NNNN_comboNNN_rN.png`` 命名的图纸 —— 目录里的总览图
        (placements_overview.png 等)不进抽取队列。
        """
        if not self.root.is_dir():
            return []
        unused: list[str] = []
        for png in sorted(self.root.rglob("*.png")):
            rel = self._rel_key(png)
            if not self.drawing_info(rel):
                continue
            if png.name in self.load_used(rel):
                continue
            unused.append(rel)
        return unused

    def draw_unused(self, rng: random.Random | None = None) -> str:
        """随机抽一张未用过的图纸(相对路径);池空则 RuntimeError。"""
        unused = self.pool_unused()
        if not unused:
            total = sum(1 for p in self.root.rglob("*.png")
                        if self.drawing_info(self._rel_key(p)))
            raise RuntimeError(
                f"{self.root} 里没有未采集过的图纸 — "
                f"{total} 张图纸已全部采集,或目录为空")
        return (rng if rng is not None else random).choice(unused)

    # ---- 场景信息 ------------------------------------------------------------

    def scene_config(self, rel: str) -> dict:
        """图纸所在场景目录的 ``config.yaml``(容忍 ``*_config.yaml`` 前缀
        命名);缺失/损坏返回 {}。"""
        directory = self.path(rel).parent
        cfg_path = directory / "config.yaml"
        if not cfg_path.is_file():
            found = sorted(directory.glob("*_config.yaml"))
            if not found:
                return {}
            cfg_path = found[0]
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}

    @staticmethod
    def drawing_info(rel: str) -> dict:
        """文件名 ``0001_combo006_r1.png`` → {num, combo, rep};不匹配则 {}。

        绘图工具的导出模板可带场景前缀(``餐桌_保鲜盒_0001_combo015_r2``),
        正则容忍任意非空前缀 —— 总览图(placements_overview 等)不含
        combo 段,自然被排除。
        """
        m = re.match(r"(?:.+?_)?(\d+)_combo(\d+)_r(\d+)$", Path(rel).stem)
        return ({"num": int(m[1]), "combo": int(m[2]), "rep": int(m[3])}
                if m else {})

    def scene_layout(self, rel: str) -> dict:
        """图纸对应的物体摆放:config.yaml 的静态属性 + placements.csv 位姿。

        每个物体 = name/color/shape/dims(来自 config.yaml 的 objects 列表)
        + cx/cy/ang(来自 placements.csv 中 ``num`` 匹配该图纸的那一行,
        列名前缀为物体名)。返回 {task_name, scene, objects} —— 与开录时
        固化进 session meta.yaml 的顶层字段一一对应(num/combo/rep 由图纸
        文件名自带,不再重复存);
        图纸名不合法 / 缺 config.yaml objects / 缺 placements.csv 或对应行
        时返回 {} —— 调用方按空值跳过,不阻塞采集。
        """
        info = self.drawing_info(rel)
        if not info:
            return {}
        cfg = self.scene_config(rel)
        scene = cfg.get("scene") or {}
        objects_cfg = scene.get("objects") or cfg.get("objects") or []
        if not objects_cfg:
            return {}
        row: dict = {}
        directory = self.path(rel).parent
        csv_path = directory / "placements.csv"
        if not csv_path.is_file():
            found = sorted(directory.glob("*_placements.csv"))
            if not found:
                return {}
            csv_path = found[0]
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if str(r.get("num", "")).strip() == str(info["num"]):
                        row = r
                        break
        except OSError:
            return {}
        if not row:
            return {}

        def _pose(name: str, key: str) -> float:
            try:
                return float(row.get(f"{name}_{key}", "") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        objs = []
        for o in objects_cfg:
            name = str(o.get("name", ""))
            # 该图纸实际选中的物体变体 id:placements 的 ``{name}_obj`` 列,
            # 形如 "保鲜盒#144d11b9-…"(# 前是物体名,# 后是变体 id);
            # 旧场景没有这一列 → None
            raw_obj = (row.get(f"{name}_obj") or "").strip()
            obj_id = (raw_obj.split("#", 1)[1].strip() if "#" in raw_obj
                      else raw_obj or None)
            # 槽位组:shape/color/dims/material 以命中的 candidate 为准
            #(顶层对象只是默认值,同一名字下不同变体形状尺寸各不相同);
            # 没配槽位或 id 未命中 → 回退物体自身默认
            cand = next((c for c in (o.get("slot") or {}).get("candidates") or []
                         if obj_id and c.get("obj_id") == obj_id), {})

            def pick(key, default=None, _c=cand, _o=o):
                if _c.get(key) is not None:
                    return _c[key]
                if _o.get(key) is not None:
                    return _o[key]
                return default

            objs.append({
                "name": name,
                "id": obj_id,
                "color": pick("color", row.get(f"{name}_color") or None),
                "shape": pick("shape", "box"),
                "dims": dict(pick("dims") or {}),    # 键随 shape 走(box=w/l/h, cyl=d/h)
                "material": pick("material"),
                "cx": _pose(name, "cx"),
                "cy": _pose(name, "cy"),
                "ang": _pose(name, "ang"),
            })
        return {"task_name": str(cfg.get("task", "")),
                "scene": str(scene.get("name", "")),
                "objects": objs}

    def scene_title(self, rel: str) -> str:
        """指令屏标题:场景名 + 图纸编号,如 ``餐桌 · 图纸0001 (combo 6, rep 1)``。"""
        scene = (self.scene_config(rel).get("scene") or {}).get("name") or ""
        title = scene or Path(rel).parent.name
        info = self.drawing_info(rel)
        if info:
            title += (f" · 图纸{info['num']:04d}"
                      f" (combo {info['combo']}, rep {info['rep']})")
        return title

    def scene_task(self, rel: str) -> str:
        """config 里的任务名(``task`` 字段);缺失回退场景目录名。"""
        return str(self.scene_config(rel).get("task") or "") or Path(rel).parent.name

    # ---- 摆放显示 ------------------------------------------------------------

    def show(self, rel: str, *, fullscreen: bool | None = None,
             display: int | None = None) -> str:
        """展示一张图纸,按 ``n + Enter`` 确认后关闭(pygame 实现)。

        只有先按 ``n`` 再按 ``Enter`` 才算确认(返回 "done");其他按键
        一律无效(提示后原地等待),Esc/关窗未经确认视为取消(返回
        "abort")。屏幕与窗口取值优先级:参数 > stim.yaml 的
        ``drawing_display``(缺省回退 ``display``)/ ``drawing_fullscreen``
        (缺省全屏)/ ``drawing_width``+``drawing_height``(仅窗口化时
        生效,0 = 跟随屏幕;全屏模式始终跟随该屏分辨率)。
        """
        import pygame
        from ..stim.base_stim import _FONT_CANDIDATES, _find_font

        if display is None or fullscreen is None:
            from .config import load_stim
            stim_cfg = load_stim() or {}
            if display is None:
                display = int(stim_cfg.get(
                    "drawing_display", stim_cfg.get("display", 0)) or 0)
            if fullscreen is None:
                fullscreen = bool(stim_cfg.get("drawing_fullscreen", True))

        win_w = int(stim_cfg.get("drawing_width") or 0)
        win_h = int(stim_cfg.get("drawing_height") or 0)

        pygame.init()
        n_displays = pygame.display.get_num_displays()
        if display >= n_displays:
            print(f"[env] 配置的屏幕 {display} 不存在(共 {n_displays} 块)"
                  "— 用主屏")
            display = 0
        flags = pygame.FULLSCREEN if fullscreen else 0
        size = (win_w, win_h) if (not fullscreen and win_w > 0 and win_h > 0) \
            else (0, 0)
        screen = pygame.display.set_mode(size, flags, display=display)
        sw, sh = screen.get_size()
        pygame.display.set_caption(f"环境图纸 {rel}")

        img = pygame.image.load(str(self.path(rel)))
        iw, ih = img.get_size()
        scale = min(sw / iw, sh / ih)
        if scale < 1.0:                       # 只缩小不放大,保清晰
            img = pygame.transform.smoothscale(
                img, (max(1, int(iw * scale)), max(1, int(ih * scale))))
        img_rect = img.get_rect(center=(sw // 2, sh // 2))

        font = pygame.font.Font(
            _find_font(_FONT_CANDIDATES) or pygame.font.get_default_font(),
            max(20, sh // 50))
        caption = f"环境图纸 {rel} — 照图摆放,按 n + Enter 开始采集(Esc 取消)"

        pressed_n = False
        hint = ""
        verdict = "abort"
        clock = pygame.time.Clock()
        running = True
        print(f"[env] 图纸 {rel} ({iw}x{ih}) 显示于屏幕 {display} "
              f"({sw}x{sh}),摆放完成后按 n + Enter 开始采集")
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False           # 未经确认关窗 = 取消本次
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_n:
                        pressed_n = True
                        hint = "已按 n — 按 Enter 确认开始采集"
                    elif (ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                          and pressed_n):
                        verdict = "done"
                        running = False
                    else:
                        pressed_n = False
                        hint = "输入无效 — 请按 n + Enter 开始采集"

            screen.fill((24, 24, 24))
            screen.blit(img, img_rect)
            screen.blit(font.render(caption, True, (255, 255, 255)), (24, 12))
            if hint:
                screen.blit(font.render(hint, True, (255, 210, 80)),
                            (24, sh - 44))
            pygame.display.flip()
            clock.tick(30)

        pygame.display.quit()
        return verdict

# 默认实例 + 便捷函数:老调用写法(env.scene_task(rel) 等)保持不变
_environment = Environment()


def environments_dir() -> Path:
    return _environment.root


def load_used(rel: str) -> dict[str, dict]:
    return _environment.load_used(rel)


def mark_used(rel: str, session: str = "") -> None:
    _environment.mark_used(rel, session=session)


def pool_unused() -> list[str]:
    return _environment.pool_unused()


def draw_unused(rng: random.Random | None = None) -> str:
    return _environment.draw_unused(rng)


def scene_config(rel: str) -> dict:
    return _environment.scene_config(rel)


def drawing_info(rel: str) -> dict:
    return _environment.drawing_info(rel)


def scene_layout(rel: str) -> dict:
    return _environment.scene_layout(rel)


def scene_title(rel: str) -> str:
    return _environment.scene_title(rel)


def scene_task(rel: str) -> str:
    return _environment.scene_task(rel)


def show(rel: str, *, fullscreen: bool | None = None,
         display: int | None = None) -> str:
    return _environment.show(rel, fullscreen=fullscreen, display=display)
