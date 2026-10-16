# -*- coding: utf-8 -*-
"""构建脚本:把任务族定义展开为候选任务,按类别均衡筛选 N 个(默认 100),
导出 YAML 设计库 + CSV 索引。

用法:
    python build.py            # 导出 100 个
    python build.py --total 500  # 扩量导出(候选池共 489,超出会取满池)
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import families_drinks
import families_table
import families_food
import families_tactile
import families_assemble
import families_clean
import families_store

MODULES = (families_drinks, families_table, families_food, families_tactile,
           families_assemble, families_clean, families_store)
ALL_FAMILIES = [f for m in MODULES for f in m.FAMILIES]

OUT_DIR = HERE.parent

# 现有 19 个任务的 task 名(新任务不得与其重名)
EXISTING_TASKS = {
    "倒水", "倒茶", "做三明治", "冲咖啡", "准备早餐", "夹花生米",
    "将吸管插入牛奶开口", "往抽纸盒补充抽纸", "往盘子里放糖果",
    "把水果放进盘子里面", "把水果装进饭盒袋", "把菜盘放在隔热垫上",
    "摆放碗筷", "擦咖啡渍", "收拾餐桌", "收纳餐具", "更换骨碟",
    "用勺子把糖舀进咖啡杯", "餐桌_舀汤到碗里",
}

# 变体未提供占位符时的族级默认值(即"基础版"的物品指代)
FAMILY_EXTRA = {
    "旋盖倒矿泉水": {"bottle": "矿泉水瓶", "cup": "水杯"},
    "冲奶粉": {"cup": "水杯", "kettle": "电热水壶"},
    "袋泡茶冲泡": {"tea": "红茶包", "cup": "水杯", "kettle": "电热水壶"},
    "手动榨橙汁": {"orange": "橙子", "cup": "水杯"},
    "调蜂蜜水": {"jar": "蜂蜜罐", "spoon": "蜂蜜勺", "kettle": "电热水壶",
             "cup": "水杯"},
    "冲藕粉": {"bag": "藕粉袋", "cup": "水杯", "kettle": "电热水壶"},
    "开罐倒汽水加冰": {"can": "易拉罐", "bowl": "冰碗", "cup": "水杯"},
    "手冲研磨咖啡": {"mill": "手摇磨豆机", "bag": "咖啡豆袋",
                 "pot": "手冲壶", "dripper": "滤杯"},
    "起子开啤酒": {"beer": "啤酒瓶", "cup": "水杯"},
    "辨温选壶倒水": {"hot": "保温壶", "cold": "凉水瓶", "cup": "水杯"},
    "糖浆调饮": {"syrup": "糖浆瓶", "water": "凉水瓶", "cup": "水杯"},
    "保温壶按压出水": {"pot": "按压式保温壶", "cup": "水杯"},
    "冲饮颗粒": {"bag": "冲剂条袋", "kettle": "电热水壶"},
    "冰格脱冰入杯": {"tray": "冰格", "cup": "水杯"},
    "温杯烫盏": {"pot": "茶壶", "cup": "茶杯", "basin": "水盂"},
    "公道杯分茶": {"fair": "公道杯", "cups": "两只茶杯", "tray": "茶盘"},
    "茶则量茶入壶": {"jar": "茶叶罐", "scoop": "茶则", "pot": "茶壶"},
    "盖碗出汤": {"bowl": "盖碗", "fair": "公道杯", "tray": "茶盘"},
    "茶巾擦拭茶盘": {"towel": "茶巾", "tray": "茶盘"},
    "剥橘子装盘": {"fruit": "橘子", "plate": "盘子"},
    "剥水煮蛋": {"egg": "水煮蛋", "plate": "盘子"},
    "磕鸡蛋入碗": {"egg": "鸡蛋", "bowl": "碗"},
    "剥香蕉切段": {"fruit": "香蕉", "plate": "盘子", "knife": "塑料刀"},
    "掰馒头摆盘": {"bun": "馒头", "plate": "盘子"},
    "简易早餐全套": {"plate": "盘子"},
    "纸巾折花摆盘": {"napkin": "餐巾纸", "cup": "水杯"},
    "冰袋冷敷准备": {"pack": "冰袋", "towel": "毛巾", "timer": "定时器"},
}

# 虽然写有触觉检查点、但任务成功主要靠视觉的族,标注为"视觉主导"
VISUAL_ONLY = {
    "桌旗铺设", "摆筷枕与筷子", "纸巾折花摆盘", "两物换位不碰撞",
    "刻度线接水", "桌面文具归位", "漏斗分装盐", "调料归位转盘",
}

# 触觉要素归类(用于统计,关键词匹配 tactile 字段)
TACTILE_CLASSES = [
    ("力控/握持", ["握", "捏", "滑", "力度", "轻拿", "轻放", "轻按"]),
    ("重量/掂重", ["掂", "重量", "坠手", "重杯", "轻重"]),
    ("纹理/质感", ["纹", "织", "绒", "粗", "光滑", "光面", "颗粒"]),
    ("软硬/干湿", ["软", "硬挺", "湿度", "干湿", "回弹"]),
    ("温度感知", ["温", "热", "凉", "冰", "烫"]),
    ("盲辨/触觉定位", ["盲", "闭眼", "袋里", "袋内", "帘"]),
    ("旋拧/插入到位", ["拧", "旋", "插", "扣", "锁", "咔哒", "咬合", "螺", "卡"]),
]

SKIP_KEYS = ("suffix", "objects", "extra", "background")


def family_defaults(fam):
    subs = {}
    for v in fam["variants"]:
        for k, val in v.items():
            if k not in SKIP_KEYS:
                subs.setdefault(k, val)
    subs.update(FAMILY_EXTRA.get(fam["family"], {}))
    return subs


def resolve_notes(fam, var):
    subs = family_defaults(fam)
    for k, val in var.items():
        if k not in SKIP_KEYS:
            subs[k] = val
    lines = ["背景摆放:" + (var.get("background") or fam["background"])]
    lines += fam["notes"].format(**subs).split("\n")
    extra = var.get("extra")
    if extra:
        extra = extra.format(**subs)
        m = re.match(r"^(\d+)b\.", extra)
        n = m.group(1) if m else "1"
        idx = 0
        for i, ln in enumerate(lines):
            if ln.startswith(n + ".") or ln.startswith(n + "b."):
                idx = i
        lines.insert(idx + 1, extra)
    return "\n".join(lines)


def build_task(fam, var):
    name = "餐桌_" + fam["family"] + var.get("suffix", "")
    tactile = fam.get("tactile") or ""
    modalities = ["视觉"]
    if tactile and fam["family"] not in VISUAL_ONLY:
        modalities.append("触觉")
    return {
        "name": name,
        "category": fam["category"],
        "modalities": modalities,
        "tactile": tactile,
        "difficulty": fam["difficulty"],
        "background": var.get("background") or fam["background"],
        "objects": var.get("objects") or fam["objects"],
        "notes": resolve_notes(fam, var),
    }


def candidate_queues():
    """每个类别一条候选队列:按轮次交错各族(第1轮取每族第1变体,依次类推)。"""
    by_cat = {}
    for fam in ALL_FAMILIES:
        by_cat.setdefault(fam["category"], []).append(fam)
    queues = {}
    for cat, fams in by_cat.items():
        q = []
        for p in range(max(len(f["variants"]) for f in fams)):
            for f in fams:
                if p < len(f["variants"]):
                    q.append((f, f["variants"][p]))
        queues[cat] = q
    return queues


def select(total):
    queues = candidate_queues()
    selected = []
    while len(selected) < total:
        avail = [(sum(1 for t in selected if t["category"] == c), c)
                 for c, q in queues.items() if q]
        if not avail:
            break
        _, cat = min(avail)
        fam, var = queues[cat].pop(0)
        selected.append(build_task(fam, var))
    return selected


def validate(tasks):
    errors = []
    names = [t["name"] for t in tasks]
    if len(tasks) != TOTAL:
        errors.append(f"数量 {len(tasks)} != {TOTAL}(候选池不足)")
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        errors.append(f"重名: {dup}")
    clash = [n for n in names if n[len('餐桌_'):] in EXISTING_TASKS]
    if clash:
        errors.append(f"与现有任务重名: {clash}")
    allowed_letters = set("O K P L N G F C R V B H J W S T X A E I M Q U D b ml".split())
    for t in tasks:
        if "{" in t["notes"] or "}" in t["notes"]:
            errors.append(f"{t['name']}: notes 残留占位符")
        if not t["notes"].startswith("背景摆放:"):
            errors.append(f"{t['name']}: notes 缺少背景摆放行")
        for w in re.findall(r"[A-Za-z]+", t["notes"]):
            if w not in allowed_letters:
                print(f"  [提醒] {t['name']}: 英文片段 {w!r}")
    return errors


def tactile_stats(tasks):
    stats = {}
    for label, kws in TACTILE_CLASSES:
        stats[label] = sum(
            1 for t in tasks
            if t["tactile"] and any(k in t["tactile"] for k in kws))
    return stats


def main():
    global TOTAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=100)
    args = ap.parse_args()
    TOTAL = args.total

    tasks = select(TOTAL)
    print(f"[1] 展开+筛选: {len(tasks)}/{TOTAL}")

    errors = validate(tasks)
    if errors:
        print("[2] 校验失败:")
        for e in errors:
            print("   -", e)
        sys.exit(1)
    print("[2] 校验通过: 名称唯一、无占位符残留、无重名冲突")

    for i, t in enumerate(tasks, 1):
        t["id"] = i
    tasks = [{"id": t["id"], **t} for t in tasks]

    by_cat, by_mod, by_diff = {}, {}, {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        key = "+".join(t["modalities"])
        by_mod[key] = by_mod.get(key, 0) + 1
        by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1

    doc = {
        "meta": {
            "标题": "餐桌任务设计库",
            "生成时间": time.strftime("%Y-%m-%d %H:%M"),
            "任务总数": len(tasks),
            "类别分布": by_cat,
            "模态分布": by_mod,
            "难度分布": by_diff,
            "触觉要素分布": tactile_stats(tasks),
            "说明": ("每个任务 = 一个 configs/environments/餐桌_<任务名>/ 目录的设计蓝本;"
                     "notes 可直接粘贴进图纸设计工具;触觉任务在 tactile 字段注明触觉要素"),
        },
        "tasks": tasks,
    }
    out_yaml = OUT_DIR / f"餐桌任务设计_{len(tasks)}.yaml"
    with io_open(out_yaml) as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False,
                       width=100, default_flow_style=False)

    out_csv = OUT_DIR / f"餐桌任务设计_{len(tasks)}_index.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "任务名", "类别", "模态", "触觉要素", "难度",
                    "背景摆放", "核心物品", "步骤数"])
        for t in tasks:
            objs = "、".join(o["name"] for o in t["objects"])
            steps = sum(1 for ln in t["notes"].split("\n")
                        if re.match(r"^\d+b?\.", ln))
            w.writerow([t["id"], t["name"], t["category"],
                        "+".join(t["modalities"]), t["tactile"],
                        t["difficulty"], t["background"], objs, steps])

    print(f"[3] 导出: {out_yaml.name}, {out_csv.name}")
    print("    类别分布:", by_cat)
    print("    模态分布:", by_mod)
    print("    难度分布:", by_diff)


def io_open(path):
    import io
    return io.open(path, "w", encoding="utf-8")


if __name__ == "__main__":
    main()
