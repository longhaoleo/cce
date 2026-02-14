"""实验 2.5：跨轨迹嫁接（Trajectory Grafting）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp25_trajectory_grafting(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp25",
        title="跨轨迹嫁接（构图源 + 风格源）",
        output_dir=output_dir,
        notes="""
1. 分别采集 Source A（构图）与 Source B（风格）的特征轨迹。
2. 按时间段拼接轨迹：
   - 早期用 A；
   - 晚期用 B。
3. 注入混合轨迹并生成结果。
4. 评估构图一致性与风格转移强度。
""",
    )
