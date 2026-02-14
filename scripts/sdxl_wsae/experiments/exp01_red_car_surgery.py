"""实验 1：组合概念外科手术（Red Car Surgery）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp01_red_car_surgery(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp01",
        title="组合概念外科手术（Red Car）",
        output_dir=output_dir,
        notes="""
1. 准备 Red Car / Red Apple / Blue Car 对照组。
2. 用全时段积分差分法定位“汽车红色涂装”特征。
3. 在全时段或早期时段做 clamp-to-zero / negative steering。
4. 评估：
   - Red Car 颜色或对象变化；
   - Red Apple 与 Blue Car 基本不变。
""",
    )
