"""未实现实验的占位执行器。"""

from __future__ import annotations

import os

from ..utils import ensure_dir


def run_placeholder_experiment(*, exp_id: str, title: str, output_dir: str, notes: str) -> None:
    """把实验标题和实现要点写入占位文档，便于后续逐步填充。"""
    ensure_dir(output_dir)
    path = os.path.join(output_dir, f"{exp_id}_TODO.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {exp_id}: {title}\n\n")
        f.write("## 当前状态\n")
        f.write("占位模块（框架已接入，可在此文件对应模块中补全实现）。\n\n")
        f.write("## 实现要点\n")
        f.write(notes.strip() + "\n")
    print(f"{exp_id} 目前为占位模块，说明文件已输出: {path}")
