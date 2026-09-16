"""ffmpeg / ffprobe 可执行文件解析 + 视频容器帧数统计 — 全仓库统一入口。

约定:Windows 用仓库 third_party 自带的 exe(``ffmpeg.exe`` 与
``ffprobe-win32-x64``,版本固定,采集机不依赖系统安装),找不到再退系统
PATH;Linux/其他平台直接用系统命令。录制端(recorders.ffmpeg_writer)与
数据端(scripts/pack_daily、scripts/reqc_all)都从这里取,保证写盘、打包
预扫与事后核对用的是同一套可执行文件、同一套计数口径。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32" or os.name == "nt"

# 源码 checkout 的仓库根(utils/media.py → parents[3])+ Windows 部署机的
# 固定安装位置
_PROJECT_ROOTS = (
    Path(__file__).resolve().parents[3],
    Path("C:/Projects/Embodied-Brain-Collect"),
)


def media_tool(name: str) -> str:
    """返回 ``ffmpeg`` / ``ffprobe`` 的可执行文件路径,找不到抛 RuntimeError。

    Windows:先 third_party 的 exe,再系统 PATH。ffprobe 在仓库里是无
    扩展名的单文件 ``ffprobe-win32-x64``,同时兼容旧的目录布局
    ``ffprobe-win32-x64/ffprobe.exe``。
    其余平台:系统 PATH。
    """
    if _IS_WINDOWS:
        candidates: list[Path] = []
        for root in _PROJECT_ROOTS:
            candidates.append(root / "third_party" / f"{name}.exe")
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    found = shutil.which(name)
    if found:
        return str(found)
    if _IS_WINDOWS:
        raise RuntimeError(
            f"Windows 下未找到 {name}:third_party(ffmpeg.exe / "
            f"ffprobe-win32-x64)与系统 PATH 均无")
    raise RuntimeError(f"系统 PATH 里找不到 {name},请先安装 ffmpeg")


def ffprobe_count(path: Path) -> int:
    """容器里视频包的真实个数(解封装计数,不解码像素,毫秒级)。

    比 ``nb_frames`` 头字段可信:它数的是文件里实际存在的包,与
    ``scripts/reqc_all`` 的核对、``scripts/pack_daily`` 的完整性门同口径。
    """
    out = subprocess.run(
        [media_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip() or 0)
