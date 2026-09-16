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

import random
import re
import time
from pathlib import Path

import yaml

from .config import configs_dir

LEDGER_NAME = "used.yaml"

# 常见中文字体(按优先级);matplotlib 默认的 DejaVu 没有汉字字形
_CJK_FONTS = (
    "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "Source Han Sans CN", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "HYZhongYuanB5", "Arial Unicode MS",
)


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
        """图纸所在场景目录的 ``config.yaml``;缺失/损坏返回 {}。"""
        cfg_path = self.path(rel).parent / "config.yaml"
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}

    @staticmethod
    def drawing_info(rel: str) -> dict:
        """文件名 ``0001_combo006_r1.png`` → {num, combo, rep};不匹配则 {}。"""
        m = re.match(r"(\d+)_combo(\d+)_r(\d+)$", Path(rel).stem)
        return ({"num": int(m[1]), "combo": int(m[2]), "rep": int(m[3])}
                if m else {})

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

    def show(self, rel: str, *, fullscreen: bool = True) -> str:
        """全屏显示一张图纸,按 ``n + Enter`` 确认后关闭。

        只有先按 ``n`` 再按 ``Enter`` 才算确认(返回 "done");其他按键一律
        无效(提示后原地等待),未确认就关闭窗口视为取消(返回 "abort")。
        matplotlib 在函数内导入,用默认交互后端。
        """
        import matplotlib
        import matplotlib.pyplot as plt

        verdict = {"value": "abort"}          # 只有 n+Enter 确认才改成 done
        img = plt.imread(str(self.path(rel)))
        ow, oh = img.shape[1], img.shape[0]

        prop = self._cjk_font_prop()
        title_kw = {"fontproperties": prop} if prop is not None else {}
        caption = (f"环境图纸 {rel} — 照图摆放,按 n + Enter 开始采集"
                   if prop is not None else
                   f"Environment {rel} — arrange objects, then press n + Enter")

        fig = plt.figure(figsize=(16, 9), facecolor="#181818")
        ax = fig.add_axes([0, 0, 1, 1])          # 坐标区铺满窗口,imshow 保比例
        ax.imshow(img)
        ax.set_axis_off()
        box = dict(boxstyle="round", fc="#181818", ec="none", alpha=0.85)
        fig.text(0.5, 0.985, caption, ha="center", va="top",
                 color="white", fontsize=14, bbox=box, **title_kw)
        status = fig.text(0.5, 0.015, "", ha="center", va="bottom",
                          color="white", fontsize=13, bbox=box, **title_kw)
        try:
            mng = plt.get_current_fig_manager()
            if fullscreen:
                mng.full_screen_toggle()
        except Exception:
            pass  # 后端不支持全屏就按窗口大小显示

        state = {"n": False, "confirmed": False}

        def hint(msg: str) -> None:
            status.set_text(msg)
            fig.canvas.draw_idle()

        def on_key(event):
            if state["confirmed"]:
                return
            key = event.key or ""
            if key == "n":
                state["n"] = True
                hint("已按 n — 按 Enter 确认开始采集")
            elif key == "enter" and state["n"]:
                state["confirmed"] = True
                verdict["value"] = "done"
                plt.close(fig)
            else:
                state["n"] = False
                hint("输入无效 — 请按 n + Enter 开始采集")

        def on_close(event):
            if not state["confirmed"]:
                verdict["value"] = "abort"   # 未经确认关窗 = 取消本次

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)
        print(f"[env] 图纸 {rel} ({ow}x{oh}) 以 matplotlib 显示"
              f"(后端 {matplotlib.get_backend()}),摆放完成后按 n + Enter 开始采集")
        plt.show()                               # 阻塞到窗口被关闭
        return verdict["value"]

    @staticmethod
    def _cjk_font_prop():
        """本机已装的中文字体 FontProperties;一个都没有则 None。"""
        import matplotlib.font_manager as fm
        installed = {f.name for f in fm.fontManager.ttflist}
        for name in _CJK_FONTS:
            if name in installed:
                return fm.FontProperties(family=name)
        return None


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


def scene_title(rel: str) -> str:
    return _environment.scene_title(rel)


def scene_task(rel: str) -> str:
    return _environment.scene_task(rel)


def show(rel: str, *, fullscreen: bool = True) -> str:
    return _environment.show(rel, fullscreen=fullscreen)
