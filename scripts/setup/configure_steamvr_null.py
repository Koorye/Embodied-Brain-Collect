#!/usr/bin/env python3
"""SteamVR Null Driver 无头显配置 —— 输入 Steam 目录,自动改好两个 default.vrsettings。

    python scripts/setup/configure_steamvr_null.py "C:\\Program Files (x86)\\Steam"
    python scripts/setup/configure_steamvr_null.py                  # 不给路径则自动探测
    python scripts/setup/configure_steamvr_null.py --dry-run        # 只预览改动,不写盘
    python scripts/setup/configure_steamvr_null.py --user-config    # 连用户配置一起改
    python scripts/setup/configure_steamvr_null.py --restore        # 从 .bak 恢复原文件

背景:采集只需要 tracker 位姿,不需要实体头显。SteamVR 默认强制头显在线,改四个
字段即可用 Null Driver(虚拟头显)无头显启动,并允许 tracker 等其他驱动共存:

    drivers/null/resources/settings/default.vrsettings
        driver_null.enable = true
    resources/settings/default.vrsettings
        steamvr.requireHmd = false
        steamvr.forcedDriver = "null"
        steamvr.activateMultipleDrivers = true

注意:
- 写入前自动在同目录留 default.vrsettings.bak;已存在则不覆盖,始终保留最早备份。
- SteamVR 运行中会覆盖配置文件,检测到 vrserver/vrmonitor 在跑就拒绝写入(--force 跳过)。
- 用户配置 <Steam>/config/steamvr.vrsettings 的同名字段优先于默认配置;有冲突脚本会
  提示,--user-config 可把同样改动合并进去。
- SteamVR 更新可能覆盖默认配置,更新后重跑本脚本即可(幂等,已是目标值则不动文件)。

完整教程见 docs/vive_tracker.md;改完用 scripts/check_vive.py 验证位姿
(验证前需在 configs/recorders.yaml 的 role_serial_map 绑定每台序列号)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 相对 SteamVR 目录的两个目标文件,及各自要合并的字段(节 -> {键: 目标值})
TARGETS: list[tuple[Path, dict[str, dict]]] = [
    (Path("drivers") / "null" / "resources" / "settings" / "default.vrsettings",
     {"driver_null": {"enable": True}}),
    (Path("resources") / "settings" / "default.vrsettings",
     {"steamvr": {"requireHmd": False,
                  "forcedDriver": "null",
                  "activateMultipleDrivers": True}}),
]
USER_CONFIG = Path("config") / "steamvr.vrsettings"   # 相对 Steam 根目录
USER_CONFIG_WANTED = {"steamvr": {"requireHmd": False,
                                  "forcedDriver": "null",
                                  "activateMultipleDrivers": True},
                      "driver_null": {"enable": True}}

# 各平台常见 Steam 安装根目录(探测顺序即优先级)
_STEAM_CANDIDATES = {
    "nt": ["C:/Program Files (x86)/Steam", "C:/Program Files/Steam"],
    "posix": ["~/.local/share/Steam", "~/.steam/steam",
              "~/.var/app/com.valvesoftware.Steam/.local/share/Steam",
              "~/Library/Application Support/Steam"],
}


# =============================================================================
# 定位 Steam / SteamVR
# =============================================================================

def find_steam_root(arg: str | None) -> Path:
    """用户给的路径优先;没给就在常见位置里找装着 SteamVR 的那个。"""
    if arg:
        root = Path(arg).expanduser()
        if not root.is_dir():
            raise SystemExit(f"[x] 目录不存在:{root}")
        return root
    for cand in _STEAM_CANDIDATES.get(os.name, []):
        root = Path(cand).expanduser()
        if (root / "steamapps" / "common" / "SteamVR").is_dir():
            print(f"[i] 未指定路径,自动探测到 Steam:{root}")
            return root
    raise SystemExit("[x] 常见位置都没找到 Steam,请把 Steam 安装目录作为参数传入"
                     "(示例:python scripts/setup/configure_steamvr_null.py "
                     '"C:\\Program Files (x86)\\Steam")')


def resolve_steamvr_dir(root: Path) -> Path:
    """SteamVR 不在主库时,按 libraryfolders.vdf 里的其他 Steam 库逐个找。"""
    direct = root / "steamapps" / "common" / "SteamVR"
    if direct.is_dir():
        return direct
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(
                encoding="utf-8", errors="replace")):
            lib = Path(m.group(1).replace("\\\\", "\\")).expanduser()
            cand = lib / "steamapps" / "common" / "SteamVR"
            if cand.is_dir():
                print(f"[i] SteamVR 在其他 Steam 库:{cand}")
                return cand
    raise SystemExit(f"[x] 在 {root} 下没找到 steamapps/common/SteamVR —— "
                     "确认路径指向 Steam 根目录(而非 SteamVR 目录本身)")


def running_steamvr() -> list[str]:
    """正在跑的 SteamVR 进程名;探测失败视为没在跑(不阻塞配置)。"""
    try:
        if os.name == "nt":
            out = []
            for n in ("vrserver.exe", "vrmonitor.exe"):
                r = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {n}"],
                    capture_output=True, text=True)
                if r.returncode == 0 and n.lower() in (r.stdout or "").lower():
                    out.append(n)
            return out
        return [n for n in ("vrserver", "vrmonitor")
                if subprocess.run(["pgrep", "-x", n],
                                  capture_output=True).returncode == 0]
    except OSError:
        return []


# =============================================================================
# vrsettings 读写(JSON,容忍 // 注释与尾逗号)
# =============================================================================

def _strip_json_comments(text: str) -> str:
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            in_str = c != '"'
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and text[i + 1:i + 2] == "/":
            i = text.find("\n", i)
            if i < 0:
                break
        elif c == "/" and text[i + 1:i + 2] == "*":
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def load_vrsettings(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    tries = [raw,
             _strip_json_comments(raw),
             re.sub(r",(\s*[}\]])", r"\1", _strip_json_comments(raw))]
    err = ""
    for text in tries:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            err = str(e)
    raise SystemExit(f"[x] {path} 不是合法 JSON({err}),无法自动修改。"
                     "多半被 SteamVR 更新改写:先用 --restore 恢复备份,或重装/"
                     "校验 SteamVR 后重试;详见 docs/vive_tracker.md")


def merge_settings(data: dict, wanted: dict[str, dict]) -> list[tuple]:
    """把 wanted 合并进 data(缺节补节,其他键不动),返回 [(节, 键, 旧, 新)]。
    只列真实变化,重复运行零改动(幂等)。"""
    changes = []
    for section, kv in wanted.items():
        node = data.setdefault(section, {})
        for key, new in kv.items():
            old = node.get(key, "<缺省>")
            if old != new:
                node[key] = new
                changes.append((section, key, old, new))
    return changes


def save_vrsettings(path: Path, data: dict, dry_run: bool) -> None:
    bak = path.with_name(path.name + ".bak")
    if bak.exists():
        print(f"    备份已存在,保留最早的:{bak.name}")
    else:
        print(f"    备份原文件 -> {bak.name}")
        if not dry_run:
            shutil.copy2(path, bak)
    print(f"    {'[预览] 将写入' if dry_run else '写入'}:{path}")
    if not dry_run:
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def fmt(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def apply_file(steamvr: Path, rel: Path, wanted: dict[str, dict],
               dry_run: bool) -> bool:
    """改一个 default.vrsettings;返回是否修改了内容。"""
    path = steamvr / rel
    if not path.is_file():
        raise SystemExit(f"[x] 找不到 {path} —— SteamVR 安装可能不完整")
    print(f"  {rel}")
    data = load_vrsettings(path)
    changes = merge_settings(data, wanted)
    if not changes:
        print("    已是目标值,无需修改")
        return False
    for section, key, old, new in changes:
        print(f"    {section}.{key}: {fmt(old)} -> {fmt(new)}")
    save_vrsettings(path, data, dry_run)
    return True


# =============================================================================
# 主流程
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="完整教程见 docs/vive_tracker.md")
    ap.add_argument("steam_path", nargs="?", default=None,
                    help="Steam 根目录(如 C:\\Program Files (x86)\\Steam;"
                         "缺省则自动探测常见安装位置)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只显示将做的修改,不写盘、不留备份")
    ap.add_argument("--user-config", action="store_true",
                    help="把同样的改动合并进 <Steam>/config/steamvr.vrsettings"
                         "(默认只检查并提示冲突)")
    ap.add_argument("--restore", action="store_true",
                    help="用 default.vrsettings.bak 恢复两个文件")
    ap.add_argument("--force", action="store_true",
                    help="检测到 SteamVR 正在运行也照改(不推荐)")
    args = ap.parse_args(argv)

    if args.restore:
        return do_restore(find_steam_root(args.steam_path))

    root = find_steam_root(args.steam_path)
    steamvr = resolve_steamvr_dir(root)

    if not args.dry_run and not args.force:
        running = running_steamvr()
        if running:
            raise SystemExit(f"[x] SteamVR 正在运行({', '.join(running)}),"
                             "运行中改配置会被它覆盖。请先完全退出 SteamVR"
                             "(系统托盘里的 SteamVR 图标也要退出),"
                             "或用 --force 强行继续。")

    print(f"SteamVR 目录:{steamvr}")
    print("要做的修改:")
    changed = [apply_file(steamvr, rel, wanted, args.dry_run)
               for rel, wanted in TARGETS]
    if not any(changed):
        print("\n[i] 两个默认配置均已符合目标值,无需写入。")

    check_user_config(root, args)
    verify(steamvr, dry_run=args.dry_run)

    print("\n完成。重启 SteamVR 后生效;后续:"
          "\n  1. VIVE Hub 连接收发器并配对 tracker,按提示建图"
          "\n  2. Settings -> Controllers -> Manage Trackers 设角色"
          " (chest / left wrist / right wrist)"
          "\n  3. python scripts/check_vive.py --list   # 验证位姿链路"
          "\n详细步骤见 docs/vive_tracker.md")
    return 0


def check_user_config(root: Path, args) -> None:
    """用户配置同名键优先于默认配置;有冲突就提示,--user-config 才动手改。"""
    path = root / USER_CONFIG
    if not path.is_file():
        print(f"[i] 无用户配置 {path},默认配置即生效")
        return
    data = load_vrsettings(path)
    changes = merge_settings(data, USER_CONFIG_WANTED)
    if not changes:
        print("[i] 用户配置 steamvr.vrsettings 已符合目标值")
        return
    if not args.user_config:
        print(f"[!] 用户配置 {path} 有 {len(changes)} 个键与目标值不同,"
              "同名设置会覆盖默认配置,可能导致不生效。加 --user-config 一并修改:")
        for section, key, old, new in changes:
            print(f"    {section}.{key}: {fmt(old)} -> {fmt(new)}")
        return
    print(f"  {USER_CONFIG}(用户配置)")
    for section, key, old, new in changes:
        print(f"    {section}.{key}: {fmt(old)} -> {fmt(new)}")
    save_vrsettings(path, data, args.dry_run)


def verify(steamvr: Path, dry_run: bool) -> None:
    """写完重读一遍,确认四个键都是目标值。"""
    if dry_run:
        return
    for rel, wanted in TARGETS:
        data = load_vrsettings(steamvr / rel)
        for section, kv in wanted.items():
            for key, new in kv.items():
                got = data.get(section, {}).get(key)
                if got != new:
                    raise SystemExit(
                        f"[x] 校验失败:{rel} {section}.{key} = "
                        f"{fmt(got)}(应为 {fmt(new)})")
    print("[✓] 复检通过:两个文件四个键均为目标值")


def do_restore(root: Path) -> int:
    steamvr = resolve_steamvr_dir(root)
    ok = True
    for rel, _ in TARGETS:
        path = steamvr / rel
        bak = path.with_name(path.name + ".bak")
        print(f"  {rel}")
        if not bak.is_file():
            print(f"    没有 {bak.name},跳过")
            ok = False
            continue
        shutil.copy2(bak, path)
        print(f"    已从 {bak.name} 恢复")
    print("恢复完成;SteamVR 更新覆盖配置后也可重跑本脚本(不带 --restore)重新配置。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
