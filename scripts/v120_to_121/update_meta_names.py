#!/usr/bin/env python3
"""给已录制 session 的 meta.yaml 补齐/更新各槽位的设备显示名(recorders 字段)。

    python scripts/update_meta_names.py                       # 扫描 data/ 下全部 session
    python scripts/update_meta_names.py data/session-day data/session-night
    python scripts/update_meta_names.py data/session-night/2026-08-31-17-52-27/meta.yaml
    python scripts/update_meta_names.py --dry-run             # 只预览

名字来源是 configs/recorders.yaml 的 ``name`` 字段(未配 name 的槽位用槽位
键本身)。session meta 里已有的槽位更新为当前值,缺失的槽位补上 —— 使历史
数据与最新录制结果的 meta 对齐。meta 的其余字段(任务/图纸/QC 信息)不动。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from embodied_brain_collect.session.config import load_recorders  # noqa: E402


def recorder_names() -> dict[str, str]:
    """configs/recorders.yaml → {槽位: 设备显示名};未配 name 的槽位用槽位键。"""
    out: dict[str, str] = {}
    for slot, params in load_recorders().items():
        if not isinstance(params, dict):
            continue
        out[slot] = str(params.get("name") or slot)
    return out


def iter_meta_files(roots: list[Path]):
    """给目录时递归找 meta.yaml;给文件时只收 meta.yaml 本身。"""
    for root in roots:
        if root.is_file() and root.name == "meta.yaml":
            yield root
        elif root.is_dir():
            yield from sorted(root.glob("**/meta.yaml"))


def update_one(path: Path, names: dict[str, str],
               dry_run: bool = False) -> tuple[bool, list[str]]:
    """补齐/更新一个 meta 的 recorders 名。

    返回 (是否发生变化, 变化的槽位列表)。已有槽位更新为当前值,缺失的
    槽位补上;meta 的其余字段一律不动。
    """
    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    recorders = meta.get("recorders")
    recorders = dict(recorders) if isinstance(recorders, dict) else {}

    changed = [s for s, n in names.items() if recorders.get(s) != n]
    if not changed:
        return False, []

    for slot in changed:
        recorders[slot] = names[slot]
    meta["recorders"] = recorders
    if not dry_run:
        path.write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
    return True, changed


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="*",
                    default=[project_root / "data"],
                    help="班次根目录 / session 目录 / meta.yaml 文件"
                         "(默认 data/ 下全部)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只预览,不写回")
    args = ap.parse_args(argv)

    names = recorder_names()
    files = list(iter_meta_files(list(args.paths)))
    if not files:
        print("没有找到任何 meta.yaml", file=sys.stderr)
        return 1

    print(f"{len(files)} 个 session meta;设备名来源 = configs/recorders.yaml"
          + ("(dry-run)" if args.dry_run else ""))
    updated = 0
    for path in files:
        try:
            touched, changed = update_one(path, names, dry_run=args.dry_run)
        except (OSError, yaml.YAMLError) as exc:
            print(f"  ✗ {path}: 读写失败 — {exc}")
            continue
        if touched:
            detail = ", ".join(f"{s}={names[s]}" for s in changed)
            print(f"  ✓ {path}: 更新 {detail}")
            updated += 1
        else:
            print(f"  = {path}: 已对齐")

    print(f"\n完成: {updated}/{len(files)} 个 meta 更新"
          + ("(dry-run,未写回)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
