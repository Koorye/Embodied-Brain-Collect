# -*- coding: utf-8 -*-
"""餐桌任务设计库 —— 任务族定义的公共工具。

每个任务族(family)是一类同质活动的模板,通过 variants(变体)展开成
多个具体任务。变体通过占位符替换 + 可选的追加步骤改写 notes,保证
每个任务的可执行细节(物品/手规则/目标格/数量/触觉检查点)真实不同。
"""

# 网格引用惯例(与现有图纸 notes 保持一致):
#   面前/正前方     = O/K/P/L 格
#   桌面中心        = N/G 格(或 H/K 格)
#   右下角          = F/C 格
#   左下角/左前方   = R/V 格
#   右前方/桌边     = B/C 格

def ob(name, shape, material, weight, d=None, h=None, l=None, w=None):
    """构造一个物品描述。cyl 用 d/h,box 用 h/l/w(单位:cm / g)。"""
    if shape == "cyl":
        dims = {"d": d, "h": h}
    else:
        dims = {"h": h, "l": l, "w": w}
    return {"name": name, "shape": shape, "material": material,
            "weight": weight, "dims": dims}


def F(family, category, difficulty, objects, notes, variants,
      background="花、水杯架", tactile=None):
    """任务族定义。tactile 非空 ⇒ 该族默认为 触觉+视觉 任务。"""
    return {
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "background": background,
        "tactile": tactile,
        "objects": objects,
        "notes": notes,
        "variants": variants,
    }


def V(suffix="", **subs):
    """普通变体:替换 notes 占位符。"""
    return {"suffix": suffix, **subs}
