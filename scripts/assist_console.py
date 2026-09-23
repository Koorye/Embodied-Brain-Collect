#!/usr/bin/env python3
"""辅助员控制台启动入口 —— 配置见 configs/assist_console.yaml。

    python scripts/assist_console.py
    python scripts/assist_console.py --windowed   # 调试:窗口化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_brain_collect.session.assist_console import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
