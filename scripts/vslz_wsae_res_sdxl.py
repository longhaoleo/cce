"""脚本入口：转调模块化实现。"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """补齐导入路径，确保可从任意工作目录运行。"""
    repo_root = Path(__file__).resolve().parents[1]
    scripts_root = repo_root / "scripts"
    for p in (str(repo_root), str(scripts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> None:
    _bootstrap_path()
    from sdxl_wsae.cli import main as real_main

    real_main()


if __name__ == "__main__":
    main()
