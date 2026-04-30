#!/usr/bin/env python3
"""SharedSAE 批量概念擦除入口。"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """保证从任意工作目录运行时能导入仓库内模块。"""
    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "scripts"
    for path in (str(repo_root), str(scripts_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_path()

from sdxl_wsae.shared_sae.batch_erase import main  # noqa: E402


if __name__ == "__main__":
    main()

