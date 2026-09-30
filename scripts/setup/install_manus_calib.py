#!/usr/bin/env python3
"""MANUS 标定文件安装 —— 把 Manus Core 标定得到的 .mcal 装进 SDK 读取目录。

    python scripts/setup/install_manus_calib.py ~/Downloads/LeftMetaglovePro.mcal \\
                                          ~/Downloads/RightMetaglovePro.mcal
    python scripts/setup/install_manus_calib.py ~/Downloads        # 扫目录,按文件名识别左右
    python scripts/setup/install_manus_calib.py --left L.mcal --right R.mcal
                                                          # 文件名不含左右时显式指定
    python scripts/setup/install_manus_calib.py --status        # 只看当前安装状态
    任意用法加 --dry-run                                  # 只预览,不写盘

为什么需要:manus_hand_pose recorder 经 manus_glove SDK 连 Manus Core 采集,
开录前 SDK 会 LoadCalibrationFiles 给每只手套下发 .mcal 标定 —— 不装或装错
左右,手指关节角数据不可信。SDK 默认在 ~/.cache/manus_glove/ 找
LeftMetaglovePro.mcal / RightMetaglovePro.mcal;若 recorders.yaml 的
hand_pose 槽位改过 calibration_dir / left_calibration / right_calibration,
本脚本以 yaml 为准(装错目录 = 白装)。

Manus Core 标定后导出的 .mcal 文件名五花八门,脚本按文件名里的
left/right 字样(不分大小写)识别左右,重命名成配置期望的名字装进目标
目录;同侧多个候选取最新的,其余列出忽略。目标文件已存在时先备份为
.bak(已有 .bak 不覆盖,保留最早的原始标定)。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

#: 与 HandPoseRecorderConfig / manus_data_publisher 的默认值一致
DEFAULT_CAL_DIR = Path("~/.cache/manus_glove").expanduser()
DEFAULT_LEFT = "LeftMetaglovePro.mcal"
DEFAULT_RIGHT = "RightMetaglovePro.mcal"


# =============================================================================
# 目标从哪来:recorders.yaml hand_pose 槽位 > SDK 默认
# =============================================================================

def cal_spec() -> tuple[Path, str, str]:
    """(标定目录, 左手文件名, 右手文件名);yaml 没配则用 SDK 默认。"""
    cal_dir, left, right = DEFAULT_CAL_DIR, DEFAULT_LEFT, DEFAULT_RIGHT
    try:
        from embodied_brain_collect.session.config import load_recorders
        cfg = load_recorders()
    except Exception:
        cfg = {}
    slot = next((v for v in cfg.values()
                 if isinstance(v, dict) and v.get("kind") == "manus_hand_pose"),
                None)
    if slot:
        if slot.get("calibration_dir"):
            cal_dir = Path(str(slot["calibration_dir"])).expanduser()
        left = str(slot.get("left_calibration") or left)
        right = str(slot.get("right_calibration") or right)
    return cal_dir, left, right


# =============================================================================
# 收集与识别
# =============================================================================

def collect_inputs(inputs: list[str]) -> list[Path]:
    """展开参数里的文件与目录(目录递归找 .mcal),去重排序。"""
    out: list[Path] = []
    for s in inputs:
        p = Path(s).expanduser()
        if p.is_dir():
            found = sorted(p.rglob("*.mcal"))
            if not found:
                print(f"[!] {p} 下没有 .mcal 文件")
            out += found
        elif p.is_file():
            out.append(p)
        else:
            print(f"[!] 跳过不存在的路径:{p}")
    seen: set[Path] = set()
    return [p for p in out
            if not (p.resolve() in seen or seen.add(p.resolve()))]


def classify(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """按文件名里的 left/right 分侧;返回 (左候选, 右候选, 两侧都沾的歧义)。"""
    left = [p for p in files if "left" in p.name.lower()]
    right = [p for p in files if "right" in p.name.lower()]
    ambiguous = [p for p in left if p in right]   # 文件名同时含 left/right
    return ([p for p in left if p not in ambiguous],
            [p for p in right if p not in ambiguous],
            ambiguous)


def _newest(paths: list[Path]) -> tuple[Path | None, list[Path]]:
    """候选里取 mtime 最新的一个,其余作为被忽略的返回。"""
    if not paths:
        return None, []
    ranked = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return ranked[0], ranked[1:]


# =============================================================================
# 安装
# =============================================================================

def _fmt_stat(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size} B, {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"


def install(src: Path, dst: Path, side: str, dry_run: bool) -> None:
    if src.stat().st_size == 0:
        raise SystemExit(f"[x] {src} 是空文件,多半导出失败,重新标定导出")
    bak = dst.with_name(dst.name + ".bak")
    print(f"  [{side}] {src}({_fmt_stat(src)})")
    print(f"      -> {dst}")
    if dst.exists():
        msg = (f"备份原标定 -> {bak.name}" if not bak.exists()
               else f"备份已存在,保留最早的:{bak.name}")
        print(f"      {msg}")
        if not dry_run and not bak.exists():
            shutil.copy2(dst, bak)
    print(f"      {'[预览] 将复制' if dry_run else '已安装'}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def show_status(cal_dir: Path, left: str, right: str) -> None:
    print(f"标定目录:{cal_dir}"
          + ("" if cal_dir.is_dir() else "  (目录不存在,还没装过)"))
    for side, name in (("左手", left), ("右手", right)):
        p = cal_dir / name
        state = f"已装({_fmt_stat(p)})" if p.is_file() else "未安装"
        print(f"  [{side}] {name}: {state}")
    others = sorted(cal_dir.glob("*.mcal")) if cal_dir.is_dir() else []
    extra = [p.name for p in others if p.name not in (left, right)]
    if extra:
        print(f"  目录里还有其他 .mcal(SDK 不读):{', '.join(extra)}")


# =============================================================================
# 主流程
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="教程见 docs/manus.md")
    ap.add_argument("inputs", nargs="*", default=[],
                    help=".mcal 文件或含 .mcal 的目录(可多个)")
    ap.add_argument("--left", default=None, help="显式指定左手 .mcal 文件")
    ap.add_argument("--right", default=None, help="显式指定右手 .mcal 文件")
    ap.add_argument("--status", action="store_true",
                    help="只显示当前安装状态,不安装")
    ap.add_argument("--dry-run", action="store_true",
                    help="只显示将做的复制,不写盘")
    args = ap.parse_args(argv)

    cal_dir, left_name, right_name = cal_spec()
    print(f"目标(按 recorders.yaml hand_pose 槽位):{cal_dir}  "
          f"左={left_name}  右={right_name}")
    if args.status:
        show_status(cal_dir, left_name, right_name)
        return 0

    files = collect_inputs(list(args.inputs))
    for flag, side_name in (("--left", "left"), ("--right", "right")):
        v = getattr(args, side_name)
        if v:
            p = Path(v).expanduser()
            if not p.is_file():
                raise SystemExit(f"[x] {flag} 指定的文件不存在:{p}")
            files.append(p)
    if not files:
        print("没有可安装的 .mcal —— Manus Core 标定后把导出的标定文件路径"
              "作为参数传入,或用 --status 查看当前状态。")
        return 1

    left, right, ambiguous = classify(files)
    for p in ambiguous:
        print(f"[!] 文件名同时含 left/right,不自动识别:{p.name}"
              " —— 用 --left/--right 显式指定")
    left_pick, left_ignored = _newest(left)
    right_pick, right_ignored = _newest(right)
    for side, ignored in (("左手", left_ignored), ("右手", right_ignored)):
        for p in ignored:
            print(f"[!] {side}同侧多个候选,取最新的,忽略:{p.name}"
                  f"({_fmt_stat(p)})")

    print("将安装:")
    for src, dst, side in ((left_pick, cal_dir / left_name, "左手"),
                           (right_pick, cal_dir / right_name, "右手")):
        if src:
            install(src, dst, side, args.dry_run)
        elif dst.is_file():
            # 只装一只手的常见场景:另一侧没传文件,但目标已装过 → 跳过
            print(f"  [{side}] 未传文件,沿用已安装的 {dst.name}"
                  f"({_fmt_stat(dst)});如需重装请一并传入该侧 .mcal")
        else:
            raise SystemExit(
                f"[x] {side}既没有传入标定文件,目标目录也没有已安装的。"
                "传入该侧的 .mcal,或用 --left/--right 显式指定。")
    if not args.dry_run:
        show_status(cal_dir, left_name, right_name)
    print("\n下一步:python scripts/preflight.py 复检;采集启动日志里应看到"
          " \"Calibration loaded successfully for Left/Right glove\"。"
          "\n详细流程见 docs/manus.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
